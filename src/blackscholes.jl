# =============================================================================
# Black-Scholes-Merton warm-up  (Week 1)
# =============================================================================
# You implement the bodies. Each function has a contract (what it returns, the
# convention) and the standard building blocks named — but the actual
# expressions are yours to write. The test suite in test/test_blackscholes.jl
# is the ground truth: implement, run `] test`, watch red turn green.
#
# STYLE RULE (matters later): keep these functions GENERIC in their numeric
# type. Do NOT annotate arguments as `::Float64` and do NOT hard-code
# `zeros(Float64, …)` inside. When calibration comes, ForwardDiff will call
# these with `Dual` numbers to get gradients; a stray `Float64` silently kills
# differentiability. Write `bs_price(S, K, r, q, σ, T)` with no type walls.
#
# Convention used throughout:
#   S  = spot price
#   K  = strike
#   r  = risk-free rate (continuously compounded)
#   q  = continuous dividend / carry yield
#   σ  = volatility (annualized)
#   T  = time to maturity in years
#   call::Bool — true for a call, false for a put
# =============================================================================

"""
    normal_cdf(x)

Standard normal cumulative distribution function Φ(x).

Convention hint: Φ(x) = (1 + erf(x / √2)) / 2.  `erf` is already imported into
this module (see QuantJulia.jl). Keep it one line and type-generic.
"""
function normal_cdf(x)
    error("TODO(you): implement Φ(x) from erf")
end

"""
    bs_price(S, K, r, q, σ, T; call::Bool=true)

Black-Scholes-Merton price of a European option with continuous carry yield q.

Building blocks (standard notation):
    d₁ = (log(S/K) + (r - q + σ²/2)·T) / (σ·√T)
    d₂ = d₁ - σ·√T
The call is assembled from S·e^(−qT)·Φ(d₁) and K·e^(−rT)·Φ(d₂); the put follows
from the same pieces (or from put-call parity — the test checks parity, so use
whichever you like as long as it's consistent).

Think about the degenerate limits before you code: what should this return when
T = 0? when σ = 0? The tests probe those.
"""
function bs_price(S, K, r, q, σ, T; call::Bool=true)
    error("TODO(you): implement the BSM price")
end

"""
    bs_vega(S, K, r, q, σ, T)

∂price/∂σ. Same for calls and puts. You'll reuse this as the derivative in the
Newton step of `implied_vol`, so get it right here first.

Building block: the standard normal PDF φ(d₁) = e^(−d₁²/2) / √(2π).
"""
function bs_vega(S, K, r, q, σ, T)
    error("TODO(you): implement vega")
end

"""
    bs_delta(S, K, r, q, σ, T; call::Bool=true)

∂price/∂S. The test checks this against a finite-difference of `bs_price`, so it
also validates that your price and delta are mutually consistent.
"""
function bs_delta(S, K, r, q, σ, T; call::Bool=true)
    error("TODO(you): implement delta")
end

"""
    implied_vol(price, S, K, r, q, T; call::Bool=true,
                σ0=0.2, tol=1e-8, maxiter=100)

Invert `bs_price` for the volatility σ that reproduces the given market `price`.
This is the single most important routine to get solid — the entire Heston
calibration later inverts model AND market prices into implied-vol space to
compare them, and reuses exactly this idea.

Suggested method: Newton on f(σ) = bs_price(σ) − price, with f'(σ) = bs_vega(σ).
Newton is fast because vega is available in closed form. Watch the failure
modes the tests will throw at you:
  • deep OTM / very short T → vega ≈ 0, Newton steps explode. Guard it.
  • an arbitrage-violating price (below intrinsic / above the bound) has NO
    solution. Decide what to return (error? NaN?) and make it deliberate.
A robust implementation often falls back to bisection when Newton misbehaves;
start with plain Newton, then harden it once the round-trip test passes.
"""
function implied_vol(price, S, K, r, q, T; call::Bool=true,
                     σ0=0.2, tol=1e-8, maxiter=100)
    error("TODO(you): implement implied-vol inversion")
end
