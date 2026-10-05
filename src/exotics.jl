# =============================================================================
# Path-dependent options — American, barrier, Asian
# =============================================================================
# Everything else in the package prices European payoffs. This file adds the
# three standard path-dependent families, each with a closed form or lattice
# reference AND a Monte Carlo pricer that works on any simulated path matrix
# — so a payoff can be priced under Black-Scholes and Heston from the same
# code, and the Black-Scholes case is always checkable against an exact value.
#
#   Paths:     simulate_gbm_paths, simulate_heston_paths (QE scheme). Matrices
#              of spot prices, one row per path, columns t₀ = 0, …, t_n = T.
#   American:  crr_price — Cox-Ross-Rubinstein binomial tree (European or
#              American); lsm_american_price — Longstaff-Schwartz regression
#              Monte Carlo on any paths.
#   Barrier:   barrier_price — Reiner-Rubinstein closed forms for all eight
#              single-barrier options under continuous monitoring;
#              barrier_mc_price — discrete monitoring on the path grid.
#              bgk_barrier — Broadie-Glasserman-Kou shift H·e^{±0.5826σ√Δ}
#              that maps a discretely monitored barrier to the continuous
#              formula.
#   Asian:     geometric_asian_price — closed form for the discretely sampled
#              geometric average; asian_mc_price — arithmetic average, with
#              the geometric average as a control variate on GBM paths.
# =============================================================================

using LinearAlgebra: qr

# -----------------------------------------------------------------------------
# Path simulation
# -----------------------------------------------------------------------------

"""
    simulate_gbm_paths(S, r, q, σ, T, nsteps, npaths; rng=nothing)

Exact geometric Brownian motion paths: an `npaths × (nsteps+1)` matrix of
spot prices on t_k = kT/nsteps, column 1 = `S`.
"""
function simulate_gbm_paths(S, r, q, σ, T, nsteps::Int, npaths::Int; rng = nothing)
    Δ = T / nsteps
    drift = (r - q - σ^2 / 2) * Δ
    vol = σ * sqrt(Δ)
    P = Matrix{Float64}(undef, npaths, nsteps + 1)
    for i in 1:npaths
        x = log(S)
        P[i, 1] = S
        for k in 1:nsteps
            x += drift + vol * _randn(rng)
            P[i, k+1] = exp(x)
        end
    end
    return P
end

"""
    simulate_heston_paths(S, r, q, p::HestonParams, T, nsteps, npaths; rng=nothing)

Heston spot paths by Andersen's QE scheme (see `simulate_heston`): an
`npaths × (nsteps+1)` matrix of spot prices, column 1 = `S`.
"""
function simulate_heston_paths(S, r, q, p::HestonParams, T, nsteps::Int, npaths::Int;
                               rng = nothing)
    Δ = T / nsteps
    qe = _qe_consts(p, Δ)
    P = Matrix{Float64}(undef, npaths, nsteps + 1)
    for i in 1:npaths
        x = 0.0; v = p.v0
        P[i, 1] = S
        for k in 1:nsteps
            v, dx = _qe_step(qe, v, rng)
            x += dx
            P[i, k+1] = S * exp((r - q) * k * Δ + x)
        end
    end
    return P
end

_payoff(s, K, call) = call ? max(s - K, 0.0) : max(K - s, 0.0)

# -----------------------------------------------------------------------------
# American options
# -----------------------------------------------------------------------------

"""
    crr_price(S, K, r, q, σ, T; call=true, american=true, N=1000)

Cox-Ross-Rubinstein binomial-tree price, European or American. Error is
O(1/N) (oscillating with N's parity).
"""
function crr_price(S, K, r, q, σ, T; call::Bool = true, american::Bool = true, N::Int = 1000)
    dt = T / N
    u = exp(σ * sqrt(dt)); d = 1 / u
    disc = exp(-r * dt)
    pu = (exp((r - q) * dt) - d) / (u - d)
    0 < pu < 1 || throw(DomainError(pu, "CRR risk-neutral probability outside (0,1); increase N"))
    V = [_payoff(S * u^j * d^(N - j), K, call) for j in 0:N]
    for i in N-1:-1:0
        for j in 0:i
            cont = disc * (pu * V[j+2] + (1 - pu) * V[j+1])
            V[j+1] = american ? max(cont, _payoff(S * u^j * d^(i - j), K, call)) : cont
        end
    end
    return V[1]
end

"""
    lsm_american_price(paths, K, r, T; call=false, degree=3)

Longstaff-Schwartz American option price from a spot-path matrix (rows =
paths, columns t₀ … t_n = T, exercise allowed at t₁ … t_n). At each date the
discounted future cash flow of in-the-money paths is regressed on
polynomials of S/K up to `degree`; exercise happens where the payoff beats
the fitted continuation value. Returns `(price, stderr)`. Like all
regression estimators it is slightly LOW-biased (a suboptimal rule).
"""
function lsm_american_price(paths::AbstractMatrix, K, r, T; call::Bool = false, degree::Int = 3)
    npaths, ncols = size(paths)
    n = ncols - 1
    dt = T / n
    df = exp(-r * dt)
    cf = [_payoff(paths[i, end], K, call) for i in 1:npaths]   # cash flow, valued at current date
    for k in n:-1:2                                              # dates t_{k-1}, columns k
        cf .*= df                                                # bring to t_{k-1}
        itm = [i for i in 1:npaths if _payoff(paths[i, k], K, call) > 0]
        length(itm) > degree + 1 || continue
        x = [paths[i, k] / K for i in itm]
        A = [xi^j for xi in x, j in 0:degree]
        β = qr(A) \ cf[itm]
        cont = A * β
        for (m, i) in enumerate(itm)
            ex = _payoff(paths[i, k], K, call)
            ex > cont[m] && (cf[i] = ex)
        end
    end
    cf .*= df                                                    # t₁ → t₀
    m = sum(cf) / npaths
    se = sqrt(sum(abs2, cf .- m) / (npaths - 1) / npaths)
    return (price = m, stderr = se)
end

# -----------------------------------------------------------------------------
# Barrier options
# -----------------------------------------------------------------------------

"""
    barrier_price(S, K, H, r, q, σ, T; kind=:down_out, call=true)

Reiner-Rubinstein closed form for a single-barrier European option under
Black-Scholes with CONTINUOUS monitoring and no rebate. `kind` is one of
`:down_out`, `:down_in`, `:up_out`, `:up_in`. Knock-in + knock-out = vanilla.
A barrier already breached at t = 0 gives 0 (out) or the vanilla (in).
"""
function barrier_price(S, K, H, r, q, σ, T; kind::Symbol = :down_out, call::Bool = true)
    kind in (:down_out, :down_in, :up_out, :up_in) ||
        throw(ArgumentError("kind must be :down_out, :down_in, :up_out or :up_in"))
    vanilla = bs_price(S, K, r, q, σ, T; call)
    down = kind in (:down_out, :down_in)
    out = kind in (:down_out, :up_out)
    if (down && S <= H) || (!down && S >= H)
        return out ? zero(vanilla) : vanilla
    end
    η = down ? 1 : -1
    φ = call ? 1 : -1
    sT = σ * sqrt(T)
    μ = (r - q - σ^2 / 2) / σ^2
    x1 = log(S / K) / sT + (1 + μ) * sT
    x2 = log(S / H) / sT + (1 + μ) * sT
    y1 = log(H^2 / (S * K)) / sT + (1 + μ) * sT
    y2 = log(H / S) / sT + (1 + μ) * sT
    Sq, Kr = S * exp(-q * T), K * exp(-r * T)
    N = normal_cdf
    A = φ * Sq * N(φ * x1) - φ * Kr * N(φ * x1 - φ * sT)
    B = φ * Sq * N(φ * x2) - φ * Kr * N(φ * x2 - φ * sT)
    C = φ * Sq * (H / S)^(2(μ + 1)) * N(η * y1) - φ * Kr * (H / S)^(2μ) * N(η * y1 - η * sT)
    D = φ * Sq * (H / S)^(2(μ + 1)) * N(η * y2) - φ * Kr * (H / S)^(2μ) * N(η * y2 - η * sT)
    hi = K > H
    knock_in = if call && down
        hi ? C : A - B + D
    elseif call                                  # up, call
        hi ? A : B - C + D
    elseif down                                  # down, put
        hi ? B - C + D : A
    else                                         # up, put
        hi ? A - B + D : C
    end
    return out ? vanilla - knock_in : knock_in
end

"""
    bgk_barrier(H, σ, T, nsteps; down=true)

Broadie-Glasserman-Kou continuity correction: a barrier monitored at
`nsteps` equally spaced dates prices like a continuously monitored one at
H·e^{−0.5826σ√Δ} (down) or H·e^{+0.5826σ√Δ} (up), Δ = T/nsteps.

It is an asymptotic approximation, not exact. Checked with 1M paths (50
dates, σ = 0.2, T = 1): a down-and-out call matches to < 0.01%, but up-and-out
calls — whose payoff is LARGE at the barrier — are off by 1–2% (BGK
overstates them), the known weak spot of the correction.
"""
bgk_barrier(H, σ, T, nsteps::Int; down::Bool = true) =
    H * exp((down ? -1 : 1) * 0.5826 * σ * sqrt(T / nsteps))

"""
    barrier_mc_price(paths, K, H, r, T; kind=:down_out, call=true)

Monte Carlo price of a single-barrier option monitored DISCRETELY at the
path dates t₁ … t_n. Returns `(price, stderr)`.
"""
function barrier_mc_price(paths::AbstractMatrix, K, H, r, T; kind::Symbol = :down_out,
                          call::Bool = true)
    kind in (:down_out, :down_in, :up_out, :up_in) ||
        throw(ArgumentError("kind must be :down_out, :down_in, :up_out or :up_in"))
    down = kind in (:down_out, :down_in)
    out = kind in (:down_out, :up_out)
    npaths = size(paths, 1)
    disc = exp(-r * T)
    pay = map(1:npaths) do i
        row = @view paths[i, 2:end]
        hit = down ? any(<=(H), row) : any(>=(H), row)
        alive = out ? !hit : hit
        alive ? disc * _payoff(paths[i, end], K, call) : 0.0
    end
    m = sum(pay) / npaths
    return (price = m, stderr = sqrt(sum(abs2, pay .- m) / (npaths - 1) / npaths))
end

# -----------------------------------------------------------------------------
# Asian options
# -----------------------------------------------------------------------------

"""
    geometric_asian_price(S, K, r, q, σ, T, n; call=true)

Closed form for an Asian option on the GEOMETRIC average of the spot at the
`n` dates t_i = iT/n, i = 1…n, under Black-Scholes: log G is normal with
mean log S + (r − q − σ²/2)·T(n+1)/(2n) and variance σ²T(n+1)(2n+1)/(6n²).
"""
function geometric_asian_price(S, K, r, q, σ, T, n::Int; call::Bool = true)
    μG = log(S) + (r - q - σ^2 / 2) * T * (n + 1) / (2n)
    σG = σ * sqrt(T * (n + 1) * (2n + 1) / (6 * n^2))
    d2 = (μG - log(K)) / σG
    d1 = d2 + σG
    disc = exp(-r * T)
    EG = exp(μG + σG^2 / 2)
    c = disc * (EG * normal_cdf(d1) - K * normal_cdf(d2))
    return call ? c : c - disc * (EG - K)
end

"""
    asian_mc_price(paths, K, r, T; call=true, control=nothing)

Monte Carlo price of an Asian option on the ARITHMETIC average of the spot
at the path dates t₁ … t_n. Pass `control = (S, q, σ)` for GBM paths to use
the geometric-average option as a control variate (its exact price from
`geometric_asian_price`), which removes most of the variance. Returns
`(price, stderr)`.
"""
function asian_mc_price(paths::AbstractMatrix, K, r, T; call::Bool = true, control = nothing)
    npaths, ncols = size(paths)
    n = ncols - 1
    disc = exp(-r * T)
    arith = [disc * _payoff(sum(@view paths[i, 2:end]) / n, K, call) for i in 1:npaths]
    if control === nothing
        m = sum(arith) / npaths
        return (price = m, stderr = sqrt(sum(abs2, arith .- m) / (npaths - 1) / npaths))
    end
    S, q, σ = control
    geo = [disc * _payoff(exp(sum(log, @view paths[i, 2:end]) / n), K, call) for i in 1:npaths]
    exact = geometric_asian_price(S, K, r, q, σ, T, n; call)
    ga, gg = arith .- sum(arith) / npaths, geo .- sum(geo) / npaths
    β = sum(ga .* gg) / max(sum(abs2, gg), eps())
    adj = arith .- β .* (geo .- exact)
    m = sum(adj) / npaths
    return (price = m, stderr = sqrt(sum(abs2, adj .- m) / (npaths - 1) / npaths))
end
