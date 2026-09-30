# =============================================================================
# Heston calibration  (Stage 4/5)
# =============================================================================
# Loss: vega-weighted price residuals,
#     loss(p) = mean_i [ (C_model_i(p) − mid_i) / vega_i ]²
# Since dC = vega·dσ to first order, (ΔC)/vega ≈ Δσ — this IS the implied-vol
# RMSE, without pushing AD through the iterative inverter on every objective
# call (the implicit-function shortcut dσ/dC = 1/vega, applied at the loss
# level). vega_i is evaluated once at the MARKET iv and held constant.
#
# Bounds via smooth reparametrization, not box constraints:
#     κ, θ, ξ, v0 = exp(x)   (positive),   ρ = tanh(x)  (−1, 1)
# so the optimizer runs unconstrained LBFGS and ForwardDiff differentiates
# straight through the transform. This same shape gains H = 0.5·logistic(x)
# when rough Heston arrives.
#
# The Feller condition 2κθ ≥ ξ² is deliberately NOT enforced: index surfaces
# routinely calibrate to Feller-violating parameters; we report it instead.
#
# Two optimizers share one residual vector r(x) = (C_model − mid)/vega:
#   :lbfgs — LBFGS on mean(r²), ForwardDiff gradient (the original path).
#   :lm    — Levenberg–Marquardt on r itself, ForwardDiff Jacobian. The loss
#            is a sum of squares, so J'J is the Gauss–Newton Hessian for free;
#            LM typically needs far fewer iterations than LBFGS here.
# And two pricers for classical Heston:
#   :adaptive — per-quote adaptive Gil-Pelaez (quadgk), the original path.
#   :batch    — per-expiry fixed-node batch pricer (shared CF evaluations),
#               the same engine rough Heston uses.
# =============================================================================

using Optim
using ForwardDiff
using LinearAlgebra: Diagonal, diag, Symmetric

_pack(p::HestonParams) = [log(p.κ), log(p.θ), log(p.ξ), atanh(p.ρ), log(p.v0)]
_unpack(x) = HestonParams(exp(x[1]), exp(x[2]), exp(x[3]), tanh(x[4]), exp(x[5]))

"""
    heston_residuals(x, quotes, S; price_rtol=1e-7)

Vega-normalized price residuals `(C_model − mid)/vega` (≈ IV errors) for
classical Heston at transformed parameters `x`, one per quote, priced by
adaptive quadrature. Type-generic in `x` (ForwardDiff Duals flow through).
"""
function heston_residuals(x, quotes, S; price_rtol = 1e-7)
    p = _unpack(x)
    return map(quotes) do qt
        c = heston_price(S, qt.K, qt.r, qt.q, p, qt.T;
                         call = qt.side === :call, rtol = price_rtol)
        (c - qt.mid) / qt.vega
    end
end

"""
    heston_loss(x, quotes, S; price_rtol=1e-7)

Mean squared vega-normalized price residual (≈ mean squared IV error) at
transformed parameter vector `x`. `quotes` need fields
`(T, K, r, q, side, mid, vega)` — `prepare_chain` output qualifies.
Type-generic in `x` so ForwardDiff Duals flow through.
"""
function heston_loss(x, quotes, S; price_rtol = 1e-7)
    p = _unpack(x)
    s = zero(eltype(x))
    for qt in quotes
        c = heston_price(S, qt.K, qt.r, qt.q, p, qt.T;
                         call = qt.side === :call, rtol = price_rtol)
        s += ((c - qt.mid) / qt.vega)^2
    end
    return s / length(quotes)
end

# Residuals for any model whose CF comes from `mkcf(T)` (u -> ψ(u)), batch
# priced per expiry group. Shared by classical and rough Heston.
function _batch_residuals(mkcf, groups)
    R = typeof(real(mkcf(groups[1].T)(0.0)))
    out = R[]
    for g in groups
        cs = batch_call_prices(mkcf(g.T), g.F, g.disc, g.Ks, g.T; iv_hint = g.iv_atm)
        @inbounds for i in eachindex(cs)
            push!(out, (cs[i] - g.cmids[i]) / g.vegas[i])
        end
    end
    return out
end

"""
    heston_batch_residuals(x, groups)

Classical-Heston residuals batch-priced per expiry from `group_quotes`
output — the same fixed-node engine as rough Heston, ~all strikes of an
expiry for the price of one CF sweep.
"""
heston_batch_residuals(x, groups) =
    (p = _unpack(x); _batch_residuals(T -> (u -> heston_cf(u, T, p)), groups))

"""
    heston_batch_loss(x, groups)

Mean of `heston_batch_residuals(x, groups)²`.
"""
heston_batch_loss(x, groups) = sum(abs2, heston_batch_residuals(x, groups)) /
                               sum(g -> length(g.Ks), groups)

# ---------------------------------------------------------------------------
# Levenberg–Marquardt (Marquardt diagonal scaling, ForwardDiff Jacobian)
# ---------------------------------------------------------------------------

"""
    levenberg_marquardt(rfun, x0; maxiter=100, λ0=1e-3, gtol=1e-10,
                        xtol=1e-10, ftol=1e-14)

Minimize `sum(abs2, rfun(x))` by Levenberg–Marquardt. `rfun` must be generic
in `x` (the Jacobian is `ForwardDiff.jacobian(rfun, x)`). Each iteration
solves `(J'J + λ·diag(J'J)) δ = −J'r`; λ shrinks on accepted steps and grows
on rejected ones. Converged when the gradient, the step, or the relative
decrease of the objective falls below its tolerance.

Returns `(minimizer, minimum, iterations, converged)` where `minimum` is the
sum of squares.
"""
function levenberg_marquardt(rfun, x0::AbstractVector; maxiter::Int = 100,
                             λ0 = 1e-3, gtol = 1e-10, xtol = 1e-10, ftol = 1e-14)
    x = float.(collect(x0))
    r = rfun(x)
    f = sum(abs2, r)
    λ = λ0
    iters = 0
    converged = false
    while iters < maxiter
        iters += 1
        J = ForwardDiff.jacobian(rfun, x)
        g = J' * r
        if maximum(abs, g) < gtol
            converged = true; break
        end
        A = Symmetric(J' * J)
        D = Diagonal(max.(diag(A), 1e-12))
        accepted = false
        while λ < 1e16
            δ = -((A + λ * D) \ g)
            xn = x + δ
            rn = rfun(xn)
            fn = sum(abs2, rn)
            if isfinite(fn) && fn < f
                small_step = maximum(abs, δ) < xtol * (1 + maximum(abs, x))
                small_gain = (f - fn) < ftol * max(f, eps())
                x, r, f = xn, rn, fn
                λ = max(λ / 3, 1e-12)
                accepted = true
                converged = small_step || small_gain
                break
            end
            λ *= 4
        end
        # No descent step exists at any damping: x is a (numerical) minimum.
        accepted || (converged = true; break)
        converged && break
    end
    return (minimizer = x, minimum = f, iterations = iters, converged = converged)
end

# Shared driver: run the chosen optimizer on residual function `rfun`.
function _fit(rfun, x0, n, method, maxiter, g_tol)
    if method === :lm
        res = levenberg_marquardt(rfun, x0; maxiter = maxiter)
        return res.minimizer, sqrt(res.minimum / n), res.converged, res.iterations
    elseif method === :lbfgs
        f(x) = sum(abs2, rfun(x)) / n
        od = OnceDifferentiable(f, x0; autodiff = :forward)
        res = optimize(od, x0, LBFGS(), Optim.Options(iterations = maxiter, g_tol = g_tol))
        return Optim.minimizer(res), sqrt(Optim.minimum(res)),
               Optim.converged(res), Optim.iterations(res)
    else
        throw(ArgumentError("method must be :lbfgs or :lm, got $(repr(method))"))
    end
end

"""
    calibrate_heston(quotes, S; p0=HestonParams(2.0, 0.04, 0.5, -0.5, 0.04),
                     method=:lbfgs, pricer=:adaptive, maxiter=300,
                     price_rtol=1e-7)

Fit classical Heston to a prepared chain with ForwardDiff derivatives.
`method` is `:lbfgs` (LBFGS on the mean squared residual) or `:lm`
(Levenberg–Marquardt on the residual vector). `pricer` is `:adaptive`
(per-quote quadgk, `price_rtol` applies) or `:batch` (per-expiry fixed-node
batch pricer; quotes without an `F` field get `F = S·e^{(r−q)T}`).
Returns `(params, rmse, converged, iterations)`; `rmse` is in implied-vol
units (multiply by 1e4 for bps).
"""
function calibrate_heston(quotes, S;
                          p0::HestonParams = HestonParams(2.0, 0.04, 0.5, -0.5, 0.04),
                          method::Symbol = :lbfgs, pricer::Symbol = :adaptive,
                          maxiter::Int = 300, price_rtol = 1e-7)
    isempty(quotes) && throw(ArgumentError("calibrate_heston: empty quote set"))
    rfun = if pricer === :adaptive
        x -> heston_residuals(x, quotes, S; price_rtol = price_rtol)
    elseif pricer === :batch
        groups = group_quotes(quotes; S = S)
        x -> heston_batch_residuals(x, groups)
    else
        throw(ArgumentError("pricer must be :adaptive or :batch, got $(repr(pricer))"))
    end
    x, rmse, conv, iters = _fit(rfun, _pack(p0), length(quotes), method, maxiter, 1e-9)
    return (params = _unpack(x), rmse = rmse, converged = conv, iterations = iters)
end

# ---------------------------------------------------------------------------
# Rough Heston calibration (Stage 8): same loss philosophy, batch pricing.
# H ∈ (0.01, 0.49) via a scaled logistic — same smooth-reparametrization
# pattern as the other bounds, AD flows straight through.
# ---------------------------------------------------------------------------

_logistic(z) = 1 / (1 + exp(-z))
_logit(p) = log(p / (1 - p))

_pack_rough(p::RoughHestonParams) =
    [log(p.κ), log(p.θ), log(p.ξ), atanh(p.ρ), log(p.v0), _logit((p.H - 0.01) / 0.48)]
_unpack_rough(x) = RoughHestonParams(exp(x[1]), exp(x[2]), exp(x[3]), tanh(x[4]),
                                     exp(x[5]), 0.01 + 0.48 * _logistic(x[6]))

"""
    group_quotes(quotes; S=nothing)

Group `prepare_chain`-style quotes by expiry for batch pricing. Put mids are
converted to call-equivalent mids via exact parity (`C = P + disc·(F − K)`;
vega is identical either side). Returns one NamedTuple per expiry with
`(T, F, r, q, disc, iv_atm, Ks, cmids, vegas)`.

Quotes without an `F` field (e.g. hand-built synthetic sets) need the spot
`S`; their forward is then `S·e^{(r−q)T}`.
"""
function group_quotes(quotes; S = nothing)
    fwd(qt) = hasproperty(qt, :F) ? qt.F :
              S === nothing ? throw(ArgumentError("group_quotes: quotes lack F; pass S")) :
              S * exp((qt.r - qt.q) * qt.T)
    byT = Dict{Float64,Vector{Int}}()
    for (i, qt) in enumerate(quotes)
        push!(get!(byT, qt.T, Int[]), i)
    end
    groups = @NamedTuple{T::Float64, F::Float64, r::Float64, q::Float64,
                         disc::Float64, iv_atm::Float64, Ks::Vector{Float64},
                         cmids::Vector{Float64}, vegas::Vector{Float64}}[]
    for T in sort!(collect(keys(byT)))
        idx = byT[T]
        q1 = quotes[idx[1]]
        disc = exp(-q1.r * T)
        Ks = Float64[]; cmids = Float64[]; vegas = Float64[]
        for i in idx
            qt = quotes[i]
            push!(Ks, qt.K)
            push!(cmids, qt.side === :call ? qt.mid : qt.mid + disc * (fwd(qt) - qt.K))
            push!(vegas, qt.vega)
        end
        F1 = fwd(q1)
        atm = quotes[idx[argmin([abs(log(quotes[i].K / F1)) for i in idx])]]
        push!(groups, (; T, F = F1, r = q1.r, q = q1.q, disc,
                       iv_atm = atm.iv, Ks, cmids, vegas))
    end
    return groups
end

"""
    rough_heston_residuals(x, groups; N=96)

Vega-normalized residuals for rough Heston at transformed parameters `x`,
batch-priced per expiry group.
"""
rough_heston_residuals(x, groups; N::Int = 96) =
    (p = _unpack_rough(x); _batch_residuals(T -> make_rough_cf(T, p; N = N), groups))

"""
    rough_heston_loss(x, groups; N=96)

Mean squared vega-normalized residual for rough Heston at transformed
parameters `x`, batch-priced per expiry group.
"""
function rough_heston_loss(x, groups; N::Int = 96)
    p = _unpack_rough(x)
    s = zero(eltype(x)); n = 0
    for g in groups
        ψ = make_rough_cf(g.T, p; N = N)
        cs = batch_call_prices(ψ, g.F, g.disc, g.Ks, g.T; iv_hint = g.iv_atm)
        @inbounds for i in eachindex(cs)
            s += ((cs[i] - g.cmids[i]) / g.vegas[i])^2
            n += 1
        end
    end
    return s / n
end

"""
    calibrate_rough_heston(quotes; p0=RoughHestonParams(1.5,0.05,0.35,-0.65,0.011,0.12),
                           method=:lbfgs, maxiter=60, N=96, S=nothing)

Fit rough Heston (6 parameters incl. H) to a prepared chain with ForwardDiff
derivatives through the fractional solver. `method` is `:lbfgs` or `:lm`, as
in `calibrate_heston`; `S` is only needed for quotes without an `F` field.
Same return shape as `calibrate_heston`.
"""
function calibrate_rough_heston(quotes;
                                p0::RoughHestonParams = RoughHestonParams(1.5, 0.05, 0.35, -0.65, 0.011, 0.12),
                                method::Symbol = :lbfgs, maxiter::Int = 60,
                                N::Int = 96, S = nothing)
    isempty(quotes) && throw(ArgumentError("calibrate_rough_heston: empty quote set"))
    groups = group_quotes(quotes; S = S)
    rfun = x -> rough_heston_residuals(x, groups; N = N)
    x, rmse, conv, iters = _fit(rfun, _pack_rough(p0), length(quotes), method, maxiter, 1e-8)
    return (params = _unpack_rough(x), rmse = rmse, converged = conv, iterations = iters)
end
