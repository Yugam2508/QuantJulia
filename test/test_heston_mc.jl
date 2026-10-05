# Heston QE Monte Carlo: unbiased with few steps where Euler is not, checked
# against the Fourier price on a Feller-violating parameter set (the regime
# index calibrations live in).
using Random

const PQE = HestonParams(1.0, 0.04, 1.0, -0.9, 0.04)        # 2κθ/ξ² = 0.08

@testset "QE — matches Fourier with 8 steps, Feller badly violated" begin
    S, r, q, T = 100.0, 0.02, 0.0, 1.0
    for (K, call) in ((80.0, false), (100.0, true), (120.0, true))
        mc = heston_mc_price(S, K, r, q, PQE, T; call, nsteps = 8, npaths = 200_000,
                             rng = MersenneTwister(11))
        @test abs(mc.price - heston_price(S, K, r, q, PQE, T; call)) < 4 * mc.stderr
    end
end

@testset "Euler — the baseline QE fixes is visibly biased at 8 steps" begin
    S, K, r, q, T = 100.0, 100.0, 0.0, 0.0, 1.0
    f = heston_price(S, K, r, q, PQE, T)
    eu = heston_mc_price(S, K, r, q, PQE, T; nsteps = 8, npaths = 100_000,
                         scheme = :euler, rng = MersenneTwister(3))
    qe = heston_mc_price(S, K, r, q, PQE, T; nsteps = 8, npaths = 100_000,
                         rng = MersenneTwister(3))
    @test (eu.price - f) > 20 * eu.stderr
    @test abs(qe.price - f) < 4 * qe.stderr
end

@testset "QE — martingale, non-negative variance path, argument checks" begin
    X = simulate_heston(PQE, 1.0, 16, 100_000; rng = MersenneTwister(5))
    ST = exp.(X)
    se = sqrt(sum(abs2, ST .- sum(ST) / length(ST)) / (length(ST) - 1) / length(ST))
    @test abs(sum(ST) / length(ST) - 1) < 4 * se              # E[S_T]/F = 1
    @test all(isfinite, X)
    @test_throws ArgumentError simulate_heston(PQE, 1.0, 8, 10; scheme = :milstein)
    @test_throws ArgumentError simulate_heston(PQE, 1.0, 0, 10)
end

@testset "QE — Feller-satisfying parameters too (the squared-Gaussian branch)" begin
    p = HestonParams(3.0, 0.05, 0.3, -0.5, 0.05)             # 2κθ/ξ² ≈ 3.3
    S, K, r, q, T = 100.0, 105.0, 0.01, 0.0, 0.5
    mc = heston_mc_price(S, K, r, q, p, T; nsteps = 8, npaths = 100_000,
                         rng = MersenneTwister(9))
    @test abs(mc.price - heston_price(S, K, r, q, p, T)) < 4 * mc.stderr
end
