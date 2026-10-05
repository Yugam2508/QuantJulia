# Market-data layer: curve algebra, consistency with the flat-rate pricers,
# the early-exercise effect of cash dividends, and QuantLib 1.43 references
# for curves, escrowed cash dividends and Heston on term structures.

@testset "curves — interpolation, extrapolation, forward rates" begin
    c = ZeroCurve([0.0, 1.0, 2.0], [0.02, 0.03, 0.05])
    @test zero_rate(c, 0.5) ≈ 0.025
    @test zero_rate(c, 1.5) ≈ 0.04
    @test zero_rate(c, 5.0) == 0.05 && zero_rate(c, -1.0) == 0.02      # flat extrapolation
    @test discount(c, 2.0) ≈ exp(-0.1)
    @test forward_rate(c, 1.0, 2.0) ≈ (0.05 * 2 - 0.03) / 1
    @test discount(FlatCurve(0.04), 3.0) ≈ exp(-0.12)
    @test_throws ArgumentError ZeroCurve([1.0, 0.5], [0.01, 0.02])
    @test_throws ArgumentError ZeroCurve([0.0, 1.0], [0.01])
end

@testset "flat market data reproduces the flat-rate pricers" begin
    S, r, q, σ, T = 100.0, 0.03, 0.01, 0.25, 0.75
    md = MarketData(S, FlatCurve(r), FlatCurve(q))
    @test forward(md, T) ≈ S * exp((r - q) * T)
    for K in (85.0, 100.0, 120.0), call in (true, false)
        @test market_price(md, K, T, σ; call) ≈ bs_price(S, K, r, q, σ, T; call)
        p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
        @test market_price(md, K, T, p; call) ≈ heston_price(S, K, r, q, p, T; call, rtol = 1e-12) atol = 1e-10
    end
    @test crr_price(md, 100.0, σ, T; call = false, N = 500) ≈
          crr_price(S, 100.0, r, q, σ, T; call = false, N = 500)
end

@testset "cash dividends — escrow, forward, early exercise of calls" begin
    md = MarketData(100.0, FlatCurve(0.03), FlatCurve(0.0), DividendSchedule([0.25, 0.75], [2.0, 2.0]))
    @test pv_dividends(md, 0.5) ≈ 2exp(-0.03 * 0.25)
    @test pv_dividends(md, 1.0) ≈ 2exp(-0.0075) + 2exp(-0.0225)
    @test forward(md, 1.0) ≈ (100 - pv_dividends(md, 1.0)) * exp(0.03)
    # with cash dividends an American call can be worth exercising early
    am = crr_price(md, 90.0, 0.2, 1.0; call = true, N = 1000)
    eu = crr_price(md, 90.0, 0.2, 1.0; call = true, american = false, N = 1000)
    @test am > eu + 0.05
    @test eu ≈ market_price(md, 90.0, 1.0, 0.2) atol = 5e-3
    @test_throws DomainError crr_price(MarketData(1.0, FlatCurve(0.0), FlatCurve(0.0),
                                                  DividendSchedule([0.5], [2.0])), 1.0, 0.2, 1.0)
end

@testset "QuantLib 1.43 — curves, escrowed cash dividends, Heston on curves" begin
    pt = [0, 30, 182, 365, 730, 1825] ./ 365
    rc = ZeroCurve(pt, [0.030, 0.032, 0.036, 0.040, 0.042, 0.045])
    qc = ZeroCurve(pt, [0.010, 0.010, 0.012, 0.015, 0.015, 0.016])
    divs = DividendSchedule([60, 150, 240, 330] ./ 365, [0.8, 0.8, 0.9, 0.9])
    md0, mdd = MarketData(100.0, rc, qc), MarketData(100.0, rc, qc, divs)
    hp = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    tol = Dict("discount_r" => 1e-14, "discount_q" => 1e-14, "bs_curves" => 1e-12,
               "bs_cashdiv_european" => 1e-12, "bs_cashdiv_american_put" => 2e-3,
               "heston_curves" => 1e-9)
    lines = readlines(joinpath(@__DIR__, "..", "validation", "quantlib_marketdata.csv"))
    @test lines[1] == "case,call,K,T,value"
    for l in lines[2:end]
        f = split(l, ','); case = f[1]; call = f[2] == "1"
        K, T, v = parse.(Float64, f[3:5])
        o = case == "discount_r" ? discount(rc, T) :
            case == "discount_q" ? discount(qc, T) :
            case == "bs_curves" ? market_price(md0, K, T, 0.22; call) :
            case == "bs_cashdiv_european" ? market_price(mdd, K, T, 0.22; call) :
            case == "bs_cashdiv_american_put" ? crr_price(mdd, K, 0.22, T; call = false, N = 2000) :
            market_price(md0, K, T, hp; call)
        @test isapprox(o, v; atol = tol[case])
    end
    @test length(lines) - 1 == 33
end
