# Rough Heston Monte Carlo vs the fractional-Riccati Fourier price: two code
# paths sharing nothing but the model definition.
using Random

@testset "rough MC — H = 1/2 reduces to classical Euler" begin
    S, K, r, q, T = 100.0, 100.0, 0.02, 0.0, 0.5
    p = RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.5)
    mc = rough_heston_mc_price(S, K, r, q, p, T; nsteps = 200, npaths = 40_000,
                               rng = MersenneTwister(11))
    f = heston_price(S, K, r, q, HestonParams(2.0, 0.04, 0.5, -0.7, 0.04), T)
    @test abs(mc.price - f) < 4 * mc.stderr
    @test_throws DomainError simulate_rough_heston(
        RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.6), T, 10, 10)
end

@testset "rough MC — matches the Fourier price at H = 0.1" begin
    # ξ small enough that the positivity floor rarely binds (see
    # src/montecarlo.jl on the floor bias at high vol-of-vol).
    S, r, q, T = 100.0, 0.0, 0.0, 0.5
    p = RoughHestonParams(2.0, 0.04, 0.15, -0.7, 0.04, 0.1)
    npaths = 20_000
    X = simulate_rough_heston(p, T, 400, npaths; rng = MersenneTwister(1))
    ST = S .* exp.(X)
    # martingale: E[S_T] = F
    se_m = sqrt(sum(abs2, ST .- sum(ST) / npaths) / (npaths - 1) / npaths)
    @test abs(sum(ST) / npaths - S) < 4 * se_m
    ψ = make_rough_cf(T, p; N = 512)
    for K in (90.0, 100.0, 110.0)
        pay = max.(ST .- K, 0.0)
        m = sum(pay) / npaths
        se = sqrt(sum(abs2, pay .- m) / (npaths - 1) / npaths)
        f = batch_call_prices(ψ, S, 1.0, (K,), T; iv_hint = 0.2)[1]
        @test abs(m - f) < 4 * se
    end
end

@testset "rough MC — H = 0.2 pricer with control variate, call and put" begin
    S, r, q, T = 100.0, 0.01, 0.0, 0.5
    p = RoughHestonParams(2.0, 0.04, 0.3, -0.7, 0.04, 0.2)
    F = S * exp((r - q) * T); disc = exp(-r * T)
    ψ = make_rough_cf(T, p; N = 512)
    for (K, call) in ((105.0, true), (95.0, false))
        mc = rough_heston_mc_price(S, K, r, q, p, T; call, nsteps = 400, npaths = 20_000,
                                   rng = MersenneTwister(7))
        c = batch_call_prices(ψ, F, disc, (K,), T; iv_hint = 0.2)[1]
        f = call ? c : c - disc * (F - K)
        @test abs(mc.price - f) < 4 * mc.stderr
        @test mc.stderr < 0.05
    end
end
