# Rough Bergomi: the Volterra process has the exact variance, η = 0 is Black-
# Scholes, the price process is a martingale, and the ATM skew follows the
# T^{H−1/2} power law that defines roughness.
using Random

@testset "rBergomi — Volterra variance Var(log V_T) = η² T^{2H}" begin
    for H in (0.1, 0.3), T in (0.1, 1.0)
        p = RBergomiParams(0.04, 1.0, -0.9, H)
        lv = log.(simulate_rbergomi(p, T, 200, 20_000; rng = MersenneTwister(3)).V)
        m = sum(lv) / length(lv)
        @test sum(abs2, lv .- m) / (length(lv) - 1) ≈ T^(2H) rtol = 0.03
    end
    @test_throws DomainError simulate_rbergomi(RBergomiParams(0.04, 1.0, -0.9, 0.6), 1.0, 10, 10)
end

@testset "rBergomi — η = 0 is Black-Scholes; martingale" begin
    S, r, q, T = 100.0, 0.02, 0.0, 0.5
    p0 = RBergomiParams(0.04, 0.0, -0.9, 0.1)
    Ks = [90.0, 100.0, 110.0]
    pr = rbergomi_mc_prices(S, Ks, r, q, p0, T; nsteps = 50, npaths = 50_000,
                            rng = MersenneTwister(4))
    for (K, c, se) in zip(Ks, pr.prices, pr.stderrs)
        @test abs(c - bs_price(S, K, r, q, 0.2, T)) < 4se
    end
    X = simulate_rbergomi(RBergomiParams(0.04, 1.5, -0.9, 0.1), 1.0, 100, 50_000;
                          rng = MersenneTwister(5)).X
    ST = exp.(X)
    se = sqrt(sum(abs2, ST .- sum(ST) / length(ST)) / (length(ST) - 1) / length(ST))
    @test abs(sum(ST) / length(ST) - 1) < 4se
end

@testset "rBergomi — short-dated ATM skew follows T^{H−1/2}" begin
    S = 100.0
    Ts = [0.005, 0.01, 0.02, 0.04, 0.08]
    fitH(H) = begin
        p = RBergomiParams(0.04, 1.9, -0.9, H)
        sk = map(Ts) do T
            δ = 0.05 * sqrt(T)
            Ks = [S * exp(-δ), S * exp(δ)]
            c = rbergomi_mc_prices(S, Ks, 0.0, 0.0, p, T; nsteps = 50, npaths = 100_000,
                                   rng = MersenneTwister(2)).prices
            (implied_vol(c[2], S, Ks[2], 0.0, 0.0, T) - implied_vol(c[1], S, Ks[1], 0.0, 0.0, T)) / 2δ
        end
        @test all(<(0), sk)                                # ρ < 0: downward skew
        @test issorted(abs.(sk); rev = true)               # steepening as T ↓
        skew_powerlaw_H(Ts, sk).H
    end
    h1, h3 = fitH(0.1), fitH(0.3)
    @test h1 ≈ 0.1 atol = 0.05
    @test h3 ≈ 0.3 atol = 0.05
    @test h3 - h1 > 0.15                                   # roughness is identified
end
