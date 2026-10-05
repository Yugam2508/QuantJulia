# =============================================================================
# Market data: discount curves, dividend schedules, forwards
# =============================================================================
# Until now every pricer took flat r and q. Real books need term structures
# and discrete cash dividends. This layer turns them into the only two numbers
# a European pricer needs per expiry — the forward F(T) and the discount
# factor D(T) — so every model in the package (Black, any characteristic
# function, the trees) prices off real market data unchanged.
#
# Curves (continuous compounding, time in years):
#   FlatCurve(r);  ZeroCurve(times, zero_rates) with zero rates interpolated
#   LINEARLY in t (flat extrapolation) — the convention of QuantLib's
#   ZeroCurve(…, Linear(), Continuous), which the tests check against.
#
# Dividends — the escrowed-dividend model: the spot is a dividend-free GBM
# part S* plus an escrow that exactly funds the cash dividends paid before
# expiry. With a continuous yield q as well, the escrow grows at r − q like
# the stock, so a dividend D_i at t_i needs D_i·P_r(t_i)/P_q(t_i) today:
#   S* = S − Σ_{t_i ≤ T} D_i·P_r(t_i)/P_q(t_i),     F(T) = S*·P_q(T)/P_r(T),
# with P_r, P_q the discount factors of the rate and dividend-yield curves.
# S* carries the volatility. This is QuantLib's model — the P_r/P_q escrow
# was identified by matching AnalyticDividendEuropeanEngine to 1e-14 (a plain
# P_r escrow is off by ~0.7% of the dividend PV when q ≠ 0) — and the same
# model as its FD engine's Escrowed cash-dividend option for Americans.
# =============================================================================

abstract type AbstractCurve end

"""
    FlatCurve(r)

Flat continuously compounded rate `r`.
"""
struct FlatCurve{T<:Real} <: AbstractCurve
    r::T
end

"""
    ZeroCurve(times, rates)

Continuously compounded zero rates at pillar `times` (years, increasing),
interpolated linearly in time and extrapolated flat.
"""
struct ZeroCurve{T<:Real} <: AbstractCurve
    times::Vector{Float64}
    rates::Vector{T}
    function ZeroCurve(times::AbstractVector, rates::AbstractVector)
        length(times) == length(rates) >= 1 || throw(ArgumentError("need matching, non-empty pillars"))
        issorted(times) && allunique(times) || throw(ArgumentError("pillar times must be increasing"))
        R = promote_type(eltype(rates), Float64)
        new{R}(collect(Float64, times), collect(R, rates))
    end
end

"""
    zero_rate(curve, t)

Continuously compounded zero rate to time `t`.
"""
zero_rate(c::FlatCurve, t) = c.r
function zero_rate(c::ZeroCurve, t)
    ts, rs = c.times, c.rates
    t <= ts[1] && return rs[1]
    t >= ts[end] && return rs[end]
    j = searchsortedlast(ts, t)
    w = (t - ts[j]) / (ts[j+1] - ts[j])
    return (1 - w) * rs[j] + w * rs[j+1]
end

"""
    discount(curve, t)

Discount factor P(0, t) = exp(−z(t)·t).
"""
discount(c::AbstractCurve, t) = exp(-zero_rate(c, t) * t)

"""
    forward_rate(curve, t1, t2)

Continuously compounded forward rate between `t1` and `t2`.
"""
forward_rate(c::AbstractCurve, t1, t2) = log(discount(c, t1) / discount(c, t2)) / (t2 - t1)

"""
    DividendSchedule(times, amounts)

Discrete cash dividends `amounts` paid at `times` (years).
"""
struct DividendSchedule
    times::Vector{Float64}
    amounts::Vector{Float64}
    function DividendSchedule(times::AbstractVector, amounts::AbstractVector)
        length(times) == length(amounts) || throw(ArgumentError("times and amounts differ in length"))
        o = sortperm(times)
        new(collect(Float64, times)[o], collect(Float64, amounts)[o])
    end
end
DividendSchedule() = DividendSchedule(Float64[], Float64[])

"""
    MarketData(S, rates, divyield=FlatCurve(0.0), dividends=DividendSchedule())

Spot, discount curve, continuous dividend-yield curve and cash dividends.
"""
struct MarketData{T<:Real,C1<:AbstractCurve,C2<:AbstractCurve}
    S::T
    rates::C1
    divyield::C2
    dividends::DividendSchedule
end
MarketData(S, rates::AbstractCurve, divyield::AbstractCurve = FlatCurve(0.0),
           dividends::DividendSchedule = DividendSchedule()) =
    MarketData(S, rates, divyield, dividends)

"""
    pv_dividends(md, T)

Escrowed value today of the cash dividends paid in `(0, T]`:
Σ D_i·P_r(t_i)/P_q(t_i) (the amount that, growing at r − q like the stock,
pays each dividend; equals the plain PV when the yield curve is zero).
"""
function pv_dividends(md::MarketData, T)
    s = zero(md.S)
    for (t, d) in zip(md.dividends.times, md.dividends.amounts)
        0 < t <= T && (s += d * discount(md.rates, t) / discount(md.divyield, t))
    end
    return s
end

"""
    forward(md, T)

Forward price to `T` under the escrowed-dividend model:
(S − escrowed dividends paid by T)·P_q(T)/P_r(T).
"""
forward(md::MarketData, T) =
    (md.S - pv_dividends(md, T)) * discount(md.divyield, T) / discount(md.rates, T)

"""
    discount(md, T)

Discount factor to `T` from the market's rate curve.
"""
discount(md::MarketData, T) = discount(md.rates, T)

"""
    black_price(F, K, disc, σ, T; call=true)

Black-76 price on forward `F` with discount factor `disc`. With `F` and `disc`
from `forward(md, T)` and `discount(md, T)`, this is the escrowed-dividend
European price under term-structure rates.
"""
function black_price(F, K, disc, σ, T; call::Bool = true)
    sT = σ * sqrt(T)
    d1 = (log(F / K) + sT^2 / 2) / sT
    d2 = d1 - sT
    c = disc * (F * normal_cdf(d1) - K * normal_cdf(d2))
    return call ? c : c - disc * (F - K)
end

"""
    market_price(md, K, T, model; call=true)

European price of strike `K`, expiry `T` under `model` with market data `md`.
`model` is a Black volatility (a `Real`) or anything with a characteristic
function in the package convention — `HestonParams`, `BatesParams`,
`MertonParams`, `VGParams`, `RoughHestonParams` — or a function
`T -> (u -> ψ(u))`. Forwards and discounting come from `md` (curves and
escrowed cash dividends).
"""
function market_price(md::MarketData, K, T, model; call::Bool = true)
    F = forward(md, T)
    disc = discount(md, T)
    model isa Real && return black_price(F, K, disc, model, T; call)
    ψ = _cf_of(model, T)
    c = cos_call_prices(ψ, F, disc, (K,), T)[1]
    return call ? c : c - disc * (F - K)
end

_cf_of(p::HestonParams, T) = u -> heston_cf(u, T, p)
_cf_of(p::BatesParams, T) = u -> bates_cf(u, T, p)
_cf_of(p::MertonParams, T) = u -> merton_cf(u, T, p)
_cf_of(p::VGParams, T) = u -> vg_cf(u, T, p)
_cf_of(p::RoughHestonParams, T) = make_rough_cf(T, p)
_cf_of(f::Function, T) = f(T)

"""
    crr_price(md::MarketData, K, σ, T; call=true, american=true, N=1000)

Cox-Ross-Rubinstein tree on market data: term-structure rates enter through
per-step forward rates, and cash dividends through the escrowed model — the
tree runs on S* = S − PV(dividends to T) and the exercise value at time t adds
back the PV (at t) of the dividends still to be paid by T. This is the same
model as QuantLib's FD engine with the Escrowed cash-dividend model.
"""
function crr_price(md::MarketData, K, σ, T; call::Bool = true, american::Bool = true,
                   N::Int = 1000)
    dt = T / N
    u = exp(σ * sqrt(dt)); d = 1 / u
    Sstar = md.S - pv_dividends(md, T)
    Sstar > 0 || throw(DomainError(Sstar, "dividends exceed the spot"))
    Pr(t) = discount(md.rates, t)
    Pq(t) = discount(md.divyield, t)
    G(t) = Pq(t) / Pr(t)                           # risk-neutral growth of S* to t
    # escrow at t for the cash dividends paid in (t, T]: what S − S* is at t
    divpv(t) = sum((dd * (Pr(ti) / Pr(t)) / (Pq(ti) / Pq(t))
                    for (ti, dd) in zip(md.dividends.times, md.dividends.amounts)
                    if t < ti <= T); init = 0.0)
    payoff(s) = call ? max(s - K, 0.0) : max(K - s, 0.0)
    node(i, j) = Sstar * u^(2j - i)                # dividend-free part at step i
    V = [payoff(node(N, j)) for j in 0:N]
    for i in N-1:-1:0
        t = i * dt
        g = G(t + dt) / G(t)                       # one-step forward growth
        pu = (g - d) / (u - d)
        0 < pu < 1 || throw(DomainError(pu, "CRR probability outside (0,1); increase N"))
        df = Pr(t + dt) / Pr(t)
        for j in 0:i
            cont = df * (pu * V[j+2] + (1 - pu) * V[j+1])
            V[j+1] = american ? max(cont, payoff(node(i, j) + divpv(t))) : cont
        end
    end
    return V[1]
end
