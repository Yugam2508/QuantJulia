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
# =============================================================================

using Optim
using ForwardDiff

_pack(p::HestonParams) = [log(p.κ), log(p.θ), log(p.ξ), atanh(p.ρ), log(p.v0)]
_unpack(x) = HestonParams(exp(x[1]), exp(x[2]), exp(x[3]), tanh(x[4]), exp(x[5]))

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

"""
    calibrate_heston(quotes, S; p0=HestonParams(2.0, 0.04, 0.5, -0.5, 0.04),
                     maxiter=300, price_rtol=1e-7)

Fit classical Heston to a prepared chain by LBFGS with ForwardDiff gradients.
Returns `(params, rmse, converged, iterations)`; `rmse` is in implied-vol
units (multiply by 1e4 for bps).
"""
function calibrate_heston(quotes, S;
                          p0::HestonParams = HestonParams(2.0, 0.04, 0.5, -0.5, 0.04),
                          maxiter::Int = 300, price_rtol = 1e-7)
    isempty(quotes) && throw(ArgumentError("calibrate_heston: empty quote set"))
    f(x) = heston_loss(x, quotes, S; price_rtol = price_rtol)
    x0 = _pack(p0)
    od = OnceDifferentiable(f, x0; autodiff = :forward)
    res = optimize(od, x0, LBFGS(),
                   Optim.Options(iterations = maxiter, g_tol = 1e-9))
    return (params = _unpack(Optim.minimizer(res)),
            rmse = sqrt(Optim.minimum(res)),
            converged = Optim.converged(res),
            iterations = Optim.iterations(res))
end
