# Lifted Heston: classical recovery, kernel approximation, CF identities,
# convergence to the fractional-Riccati rough CF, and AD through H.
using SpecialFunctions: gamma

@testset "lifted_kernel — positive weights, increasing rates, converging integral" begin
    H = 0.1; α = H + 0.5
    errs = map((10, 20, 80)) do n
        c, x = lifted_kernel(H, n)
        @test all(>(0), c) && all(>(0), x) && issorted(x)
        t = 0.1
        abs(sum(c .* (1 .- exp.(-x .* t)) ./ x) / (t^α / gamma(α + 1)) - 1)
    end
    @test errs[1] > errs[2] > errs[3]                # ∫₀ᵗ K converges with n
    @test_throws DomainError lifted_kernel(0.5, 10)
    @test_throws DomainError lifted_kernel(-0.1, 10)
end

@testset "lifted solver — one factor with x = 0 is classical Heston" begin
    T = 0.5
    ph = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    ψ1 = QuantJulia._lifted_cf(T, ph.κ, ph.θ, ph.ξ, ph.ρ, ph.v0, [1.0], [0.0], 800)
    for u in (0.5, 2.0, 10.0, -1im)
        @test ψ1(u) ≈ heston_cf(u, T, ph) rtol = 1e-5
    end
end

@testset "lifted CF — normalization and martingale" begin
    p = RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.1)
    for T in (0.05, 1.0)
        ψ = make_lifted_cf(T, p)
        @test ψ(0.0) ≈ 1
        @test ψ(-1im) ≈ 1 rtol = 1e-10
        @test abs(ψ(5.0)) <= 1 + 1e-12
    end
end

@testset "lifted → rough: implied vols converge to the fractional solver" begin
    S, r, q, T = 100.0, 0.0, 0.0, 0.5
    Ks = [80.0, 90.0, 100.0, 110.0, 120.0]
    iv(ψ) = [implied_vol(c, S, K, r, q, T)
             for (K, c) in zip(Ks, batch_call_prices(ψ, S, 1.0, Ks, T; iv_hint = 0.2))]
    for H in (0.1, 0.3)
        p = RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, H)
        ivr = iv(make_rough_cf(T, p; N = 512))
        gap = [maximum(abs.(iv(make_lifted_cf(T, p; n, N = 400)) .- ivr)) for n in (10, 20, 80)]
        @test gap[1] > gap[2] > gap[3]
        @test gap[2] < 7e-4                          # n = 20: within 7 bps
        @test gap[3] < 2e-4                          # n = 80: within 2 bps
    end
end

@testset "lifted CF — ForwardDiff gradient incl. H vs finite differences" begin
    S, K, T = 100.0, 105.0, 0.25
    price(x) = batch_call_prices(make_lifted_cf(T, RoughHestonParams(x...); n = 20, N = 100),
                                 S, 1.0, (K,), T; iv_hint = 0.2)[1]
    x0 = [2.0, 0.04, 0.5, -0.7, 0.04, 0.15]
    g = ForwardDiff.gradient(price, x0)
    @test all(isfinite, g)
    for i in 1:6
        h = 1e-6 * max(1.0, abs(x0[i]))
        xp = copy(x0); xp[i] += h; xm = copy(x0); xm[i] -= h
        @test g[i] ≈ (price(xp) - price(xm)) / 2h rtol = 1e-4 atol = 1e-8
    end
end
