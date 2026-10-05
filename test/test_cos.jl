# COS pricer: exact on Black-Scholes, agrees with Gil-Pelaez on Heston and
# rough Heston, converges in N, and is differentiable.

@testset "COS — Black-Scholes to machine precision" begin
    S, r, q, σ = 100.0, 0.03, 0.01, 0.25
    for T in (0.02, 0.5, 3.0)
        ψ = u -> exp(-(im * u + u^2) * σ^2 * T / 2)
        F = S * exp((r - q) * T); disc = exp(-r * T)
        Ks = F .* exp.(σ * sqrt(T) .* (-3:1.0:3))
        cs = cos_call_prices(ψ, F, disc, Ks, T)
        @test maximum(abs.(cs .- [bs_price(S, K, r, q, σ, T) for K in Ks])) < 1e-11
        for K in (80.0, 120.0)
            @test cos_price(ψ, S, K, r, q, T; call = false) ≈ bs_price(S, K, r, q, σ, T; call = false) atol = 1e-11
        end
    end
end

@testset "COS — cumulants read off the CF" begin
    T = 0.8
    c = cf_cumulants(u -> exp(-(im * u + u^2) * 0.04 * T / 2))
    @test c.c1 ≈ -0.02 * T atol = 1e-8                    # E[X] = −σ²T/2
    @test c.c2 ≈ 0.04 * T rtol = 1e-6
end

@testset "COS — matches Gil-Pelaez for Heston across maturities and strikes" begin
    S, r, q = 100.0, 0.02, 0.01
    p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    for T in (0.02, 0.25, 1.0, 5.0)
        F = S * exp((r - q) * T); disc = exp(-r * T)
        ψ = u -> heston_cf(u, T, p)
        Ks = F .* exp.(0.3 * sqrt(T) .* (-2:1.0:2))
        cs = cos_call_prices(ψ, F, disc, Ks, T)
        for (K, c) in zip(Ks, cs)
            @test c ≈ price_from_cf(ψ, S, K, r, q, T; rtol = 1e-13) atol = 1e-10
        end
    end
end

@testset "COS — rough Heston agrees with the batch Gil-Pelaez pricer" begin
    S, r, q, T = 100.0, 0.02, 0.0, 0.25
    p = RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.1)
    F = S * exp((r - q) * T); disc = exp(-r * T)
    ψ = make_rough_cf(T, p; N = 256)
    Ks = collect(85.0:5.0:115.0)
    @test cos_call_prices(ψ, F, disc, Ks, T) ≈
          batch_call_prices(ψ, F, disc, Ks, T; iv_hint = 0.2) rtol = 1e-5 atol = 1e-6
end

@testset "COS — error falls with N; truncation governs the floor" begin
    S, r, q, T = 100.0, 0.0, 0.0, 1.0
    p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    ψ = u -> heston_cf(u, T, p)
    ref = price_from_cf(ψ, S, 110.0, r, q, T; rtol = 1e-13)
    err(N, L) = abs(cos_call_prices(ψ, S, 1.0, (110.0,), T; N, L)[1] - ref)
    @test err(32, 24) > err(64, 24) > err(128, 24)
    @test err(512, 24) < 1e-10
    @test err(512, 8) > 1e3 * err(512, 24)                  # interval too narrow
end

@testset "COS — gradient through the pricer matches finite differences" begin
    S, K, r, q, T = 100.0, 105.0, 0.02, 0.0, 0.5
    price(x) = cos_price(u -> heston_cf(u, T, HestonParams(x...)), S, K, r, q, T)
    x0 = [2.0, 0.04, 0.5, -0.7, 0.04]
    g = ForwardDiff.gradient(price, x0)
    for i in 1:5
        h = 1e-6 * max(1.0, abs(x0[i]))
        xp = copy(x0); xp[i] += h; xm = copy(x0); xm[i] -= h
        @test g[i] ≈ (price(xp) - price(xm)) / 2h rtol = 1e-5 atol = 1e-8
    end
end
