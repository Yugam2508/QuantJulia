using Dates

@testset "nyse_holidays / business_days / year_fraction" begin
    h26 = nyse_holidays(2026)
    for d in (Date(2026, 1, 1), Date(2026, 1, 19), Date(2026, 2, 16), Date(2026, 4, 3),
              Date(2026, 5, 25), Date(2026, 6, 19), Date(2026, 7, 3), Date(2026, 9, 7),
              Date(2026, 11, 26), Date(2026, 12, 25))
        @test d in h26
    end
    @test length(h26) == 10
    @test Date(2027, 12, 24) in nyse_holidays(2027)        # Sat Christmas → Fri
    @test Date(2021, 12, 31) ∉ nyse_holidays(2021)         # Sat New Year: not observed
    @test Date(2022, 12, 26) in nyse_holidays(2022)        # Sun Christmas → Mon
    @test !any(d -> Date(2022, 1, 1) <= d <= Date(2022, 1, 3), nyse_holidays(2022))
    @test Date(2021, 6, 19) ∉ nyse_holidays(2021)          # Juneteenth from 2022
    # The Stage-8 snapshot: Thu 2 Jul 2026 close → Mon 6 Jul is ONE trading day
    # (Fri 3 Jul is the observed Independence Day, then the weekend).
    @test business_days(Date(2026, 7, 2), Date(2026, 7, 6)) == 1
    @test year_fraction(Date(2026, 7, 2), Date(2026, 7, 6); daycount = :business) ≈ 1 / 252
    @test year_fraction(Date(2026, 7, 2), Date(2026, 7, 6)) ≈ 4 / 365
    @test business_days(Date(2026, 7, 2), Date(2026, 7, 6); holidays = Date[]) == 2
    @test business_days(Date(2026, 7, 6), Date(2026, 7, 2)) == 0
    @test business_days(Date(2026, 1, 1), Date(2027, 1, 1)) == 251
    @test_throws ArgumentError year_fraction(Date(2026, 1, 1), Date(2026, 2, 1); daycount = :act360)
end

@testset "prepare_chain — business-day T conserves total variance" begin
    S, r, q, σ = 100.0, 0.03, 0.01, 0.2
    vd = Date(2026, 7, 2)
    expiry = Date(2026, 10, 2)
    Tc = Dates.value(expiry - vd) / 365
    row(K) = begin
        c = bs_price(S, K, r, q, σ, Tc; call = true)
        p = bs_price(S, K, r, q, σ, Tc; call = false)
        "Fri Oct 02 2026,SYN,0,0,$(c-0.05),$(c+0.05),0,0,0,0,0,$K,SYN,0,0,$(p-0.05),$(p+0.05),0,0,0,0,0"
    end
    lines = ["", "SYNTH INDEX,Last: $S,Change: 0.00",
             "\"Date: July 2, 2026 at 4:15 PM EDT\",Bid: 99,Ask: 101,Size: 1*1,Volume: 0",
             "header", [row(K) for K in (90.0, 95.0, 100.0, 105.0, 110.0)]...]
    path = joinpath(mktempdir(), "synth_bd.csv")
    write(path, join(lines, "\n"))
    raw = read_cboe(path)
    cal = prepare_chain(raw)
    bus = prepare_chain(raw; daycount = :business)
    Tb = business_days(vd, expiry) / 252
    @test bus.expiries[1].T ≈ Tb
    @test bus.expiries[1].F ≈ cal.expiries[1].F rtol = 1e-10    # same prices, same forward
    for (qc, qb) in zip(cal.quotes, bus.quotes)
        @test qb.iv^2 * qb.T ≈ qc.iv^2 * qc.T rtol = 1e-4     # total variance conserved
    end
end

# Synthetic rough-Heston snapshot with quotes in prepare_chain shape.
function _rough_snapshot(p, Ts; N = 32, S = 100.0, r = 0.02, qy = 0.0)
    quotes = @NamedTuple{T::Float64, K::Float64, F::Float64, r::Float64, q::Float64,
                         side::Symbol, mid::Float64, iv::Float64, vega::Float64}[]
    for T in Ts
        F = S * exp((r - qy) * T); disc = exp(-r * T)
        Ks = F .* exp.(0.25 * sqrt(T) .* (-1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0))
        cs = batch_call_prices(make_rough_cf(T, p; N), F, disc, Ks, T; iv_hint = 0.2)
        for (K, c) in zip(Ks, cs)
            side = K >= F ? :call : :put
            mid = side === :call ? c : c - disc * (F - K)
            iv = implied_vol(mid, S, K, r, qy, T; call = side === :call)
            push!(quotes, (; T, K, F, r, q = qy, side, mid, iv, vega = bs_vega(S, K, r, qy, iv, T)))
        end
    end
    return quotes
end

@testset "ATM skew estimators" begin
    # Flat Black-Scholes smile → zero skew.
    S, r, q, σ, T = 100.0, 0.01, 0.0, 0.2, 0.25
    F = S * exp((r - q) * T)
    Ks = F .* exp.(range(-0.1, 0.1, length = 9))
    g = (T = T, F = F, r = r, q = q, disc = exp(-r * T), iv_atm = σ, Ks = collect(Ks),
         cmids = [bs_price(S, K, r, q, σ, T) for K in Ks], vegas = ones(9))
    @test abs(market_atm_skew(g)) < 1e-8
    @test market_atm_skew(merge(g, (; Ks = g.Ks[1:3], cmids = g.cmids[1:3]))) === nothing

    # Linearized model skew vs exact-IV finite difference on a rough surface.
    p = RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.1)
    for T in (0.02, 0.25, 1.0)
        F = S * exp(r * T); disc = exp(-r * T)
        ψ = make_rough_cf(T, p; N = 64)
        c0 = batch_call_prices(ψ, F, disc, (F,), T; iv_hint = 0.2)[1]
        gT = (T = T, F = F, r = r, q = 0.0, disc = disc,
              iv_atm = implied_vol(c0, S, F, r, 0.0, T))    # linearize at the true ATM vol
        lin = model_atm_skew(ψ, gT)
        Km, Kp = F * exp(-0.01), F * exp(0.01)
        cs = batch_call_prices(ψ, F, disc, (Km, Kp), T; iv_hint = 0.2)
        exact = (implied_vol(cs[2], S, Kp, r, 0.0, T) - implied_vol(cs[1], S, Km, r, 0.0, T)) / 0.02
        @test lin ≈ exact rtol = 5e-3
        @test lin < 0
    end

    # Power law: exact T^{H−1/2} data returns H.
    Ts = [0.01, 0.03, 0.1, 0.3, 1.0]
    fit = skew_powerlaw_H(Ts, -0.8 .* Ts .^ (0.12 - 0.5))
    @test fit.H ≈ 0.12
    @test fit.c ≈ 0.8
end

@testset "joint rough fit — shared H across two snapshots, per-date v0" begin
    Ts = (0.04, 0.2, 0.8)
    base = RoughHestonParams(1.5, 0.04, 0.4, -0.7, 0.03, 0.15)
    p1 = base
    p2 = RoughHestonParams(1.5, 0.04, 0.4, -0.7, 0.05, 0.15)     # v0 moved
    sets = [_rough_snapshot(p1, Ts), _rough_snapshot(p2, Ts)]

    # Loss is ~0 at the truth, with and without the skew term (the residual
    # floor is the quadrature node set, steered by iv_hint, differing from the
    # one that generated the quotes).
    ds = [prepare_identification_set(qs) for qs in sets]
    @test all(st -> st !== nothing, ds[1].stencils)
    xt = QuantJulia._pack_joint(base, [p1.v0, p2.v0])
    mp, ms = rough_joint_loss(xt, ds; N = 32, parts = true)
    @test mp < 1e-8
    @test ms < 1e-8
    # Away from the truth (ρ changed → smile tilted) the skew term registers.
    xb = QuantJulia._pack_joint(RoughHestonParams(1.5, 0.04, 0.4, -0.5, 0.03, 0.15), [p1.v0, p2.v0])
    _, msb = rough_joint_loss(xb, ds; N = 32, parts = true)
    @test msb > 1e-6

    p0 = RoughHestonParams(1.5, 0.05, 0.5, -0.6, 0.04, 0.3)
    for w in (0.0, 1.0)
        res = calibrate_rough_heston_joint(sets; p0, skew_weight = w, N = 32, maxiter = 80)
        @test length(res.params) == 2
        @test res.rmse < 2e-4
        @test res.params[1].H ≈ 0.15 atol = 0.05
        @test res.params[1].H == res.params[2].H           # shared by construction
        @test res.params[1].v0 ≈ p1.v0 rtol = 0.1
        @test res.params[2].v0 ≈ p2.v0 rtol = 0.1
    end
    @test_throws ArgumentError calibrate_rough_heston_joint([])
end
