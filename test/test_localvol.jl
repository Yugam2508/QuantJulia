# Dupire local volatility: closed-form special cases, the Durrleman identity,
# and the round trip — local-vol Monte Carlo reprices the surface it came from.
using Random

@testset "dupire_local_variance — flat and term-structure-only surfaces" begin
    @test dupire_local_variance((k, T) -> 0.04T, 0.3, 0.7) ≈ 0.04
    a, b = 0.03, 0.01                               # w = aT + bT²  →  σ² = a + 2bT
    for T in (0.2, 1.0), k in (-0.4, 0.0, 0.25)
        @test dupire_local_variance((k, T) -> a * T + b * T^2, k, T) ≈ a + 2b * T
    end
end

@testset "Dupire denominator is Durrleman's g (local variance ⇔ no butterfly arbitrage)" begin
    p = SVIParams(0.02, 0.1, -0.5, 0.05, 0.2)
    T0 = 0.5
    w(k, T) = svi_total_variance(p, k) + (T - T0)  # ∂_T w = 1 at the slice
    for k in (-0.6, -0.1, 0.0, 0.4)
        @test 1 / dupire_local_variance(w, k, T0) ≈ svi_butterfly_g(p, k)
    end
end

@testset "ssvi_surface — θ interpolation and calendar guard" begin
    p = SSVIParams(-0.7, 1.1, 0.4)
    w = ssvi_surface(p, [0.25, 0.5, 1.0], [0.01, 0.02, 0.05])
    @test w(0.0, 0.5) ≈ 0.02                         # θ is the ATM total variance
    @test w(0.0, 0.75) ≈ 0.035                       # linear between nodes
    @test w(0.0, 0.125) ≈ 0.005                      # linear from (0, 0)
    @test_throws ArgumentError ssvi_surface(p, [0.25, 0.5], [0.02, 0.01])
end

@testset "round trip — local-vol MC reprices the SSVI surface" begin
    p = SSVIParams(-0.7, 1.1, 0.4)
    w = ssvi_surface(p, [0.25, 0.5, 1.0], [0.01, 0.02, 0.04])
    S, r, q = 100.0, 0.0, 0.0
    for T in (0.25, 1.0)
        Ks = S .* exp.(0.2 * sqrt(T) .* (-1.5:0.5:1.5))
        pr = local_vol_mc_prices(S, Ks, r, q, w, T; nsteps = 400, npaths = 100_000,
                                 rng = MersenneTwister(1))
        for (K, c, se) in zip(Ks, pr.prices, pr.stderrs)
            iv_in = sqrt(w(log(K / S), T) / T)
            @test abs(c - bs_price(S, K, r, q, iv_in, T)) < 4se
            @test abs(implied_vol(c, S, K, r, q, T) - iv_in) < 1.5e-3   # < 15 bps
        end
    end
end
