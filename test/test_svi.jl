@testset "SVI — closed-form pieces" begin
    p = SVIParams(0.02, 0.1, -0.5, 0.05, 0.2)
    # w'' and w' inside g match AD.
    for k in (-0.5, 0.0, 0.3)
        w(x) = svi_total_variance(p, x)
        w1 = ForwardDiff.derivative(w, k)
        w2 = ForwardDiff.derivative(x -> ForwardDiff.derivative(w, x), k)
        g = (1 - k * w1 / (2w(k)))^2 - (w1^2 / 4) * (1 / w(k) + 1 / 4) + w2 / 2
        @test svi_butterfly_g(p, k) ≈ g
    end
    @test svi_butterfly_free(p)
    # Gatheral–Jacquier (2014) Example 3.1: a raw-SVI slice WITH butterfly
    # arbitrage — g dips below zero.
    bad = SVIParams(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    @test !svi_butterfly_free(bad)
    @test minimum(k -> svi_butterfly_g(bad, k), range(-1.5, 1.5, length = 601)) < 0
end

@testset "fit_svi — recovers a known slice, fits a Heston smile" begin
    T = 0.5
    truth = SVIParams(0.01, 0.08, -0.6, 0.02, 0.15)
    ks = collect(range(-0.4, 0.3, length = 15))
    ivs = [svi_iv(truth, k, T) for k in ks]
    fit = fit_svi(ks, ivs, T)
    @test fit.rmse < 1e-6
    for k in (-0.35, 0.0, 0.25)
        @test svi_total_variance(fit.params, k) ≈ svi_total_variance(truth, k) rtol = 1e-4
    end
    @test_throws ArgumentError fit_svi(ks[1:4], ivs[1:4], T)

    # A Heston smile: SVI should fit it to well under a vol point.
    S, r, q = 100.0, 0.0, 0.0
    ph = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    ks2 = collect(range(-0.3, 0.2, length = 13))
    ivh = map(ks2) do k
        K = S * exp(k)
        call = k >= 0
        implied_vol(heston_price(S, K, r, q, ph, T; call), S, K, r, q, T; call)
    end
    fh = fit_svi(ks2, ivh, T)
    @test fh.rmse < 1e-3                          # < 10 bps
    @test svi_butterfly_free(fh.params)
end

@testset "SSVI — slice map, arbitrage-free fit" begin
    p = SSVIParams(-0.7, 1.1, 0.4)                  # η(1+|ρ|) = 1.87 ≤ 2
    for θ in (0.005, 0.04, 0.2), k in (-0.5, 0.0, 0.4)
        @test svi_total_variance(ssvi_slice(p, θ), k) ≈ ssvi_total_variance(p, θ, k)
    end
    @test ssvi_total_variance(p, 0.04, 0.0) ≈ 0.04     # θ IS the ATM total variance
    θs = [0.004, 0.012, 0.03, 0.06]
    @test ssvi_arbitrage_free(p, θs)
    @test !ssvi_arbitrage_free(SSVIParams(-0.7, 1.5, 0.4), θs)   # η(1+|ρ|) > 2
    @test all(θ -> svi_butterfly_free(ssvi_slice(p, θ)), θs)
    @test isempty(calendar_violations([k -> ssvi_total_variance(p, θ, k) for θ in θs],
                                      [0.1, 0.3, 0.75, 1.5]))

    # Recover known SSVI from synthetic slices.
    Ts = [0.1, 0.3, 0.75, 1.5]
    slices = map(zip(Ts, θs)) do (T, θ)
        ks = collect(range(-0.3, 0.3, length = 11))
        (T = T, ks = ks, ivs = [sqrt(ssvi_total_variance(p, θ, k) / T) for k in ks], θ = θ)
    end
    fit = fit_ssvi(slices)
    @test fit.rmse < 1e-6
    @test fit.params.ρ ≈ p.ρ rtol = 1e-3
    @test fit.params.η ≈ p.η rtol = 1e-3
    @test fit.params.γ ≈ p.γ rtol = 1e-3
    @test ssvi_arbitrage_free(fit.params, sort(fit.θs))
end

@testset "calendar_violations flags crossing slices" begin
    w1(k) = 0.04 + 0.1k^2
    w2(k) = 0.05 + 0.02k^2                         # crosses w1 for |k| > 0.35
    v = calendar_violations([w1, w2], [0.5, 1.0])
    @test !isempty(v)
    @test all(t -> abs(t[3]) > 0.35, v)
    # maturities are sorted internally: swapping them makes k = 0 violate
    @test !isempty(calendar_violations([w1, w2], [1.0, 0.5]; ks = [0.0]))
end

@testset "fit_svi_surface — end to end on a Heston surface" begin
    S, r, q = 100.0, 0.01, 0.0
    ph = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    quotes = @NamedTuple{T::Float64, K::Float64, F::Float64, iv::Float64}[]
    for T in (0.1, 0.25, 0.5, 1.0)
        F = S * exp((r - q) * T)
        for z in range(-2, 1.5, length = 9)
            K = F * exp(z * 0.2 * sqrt(T))
            call = K >= F
            iv = implied_vol(heston_price(S, K, r, q, ph, T; call), S, K, r, q, T; call)
            push!(quotes, (; T, K, F, iv))
        end
    end
    s = fit_svi_surface(quotes)
    @test length(s.slices) == 4
    @test s.svi_rmse < 1e-3                         # per-slice SVI: < 10 bps
    @test s.ssvi_rmse < 5e-3                        # 3-number SSVI: < 50 bps
    @test s.svi_rmse <= s.ssvi_rmse                 # SVI has 20 params vs 3
    @test all(s.butterfly_free)
    @test isempty(s.calendar_violations)
    @test ssvi_arbitrage_free(s.ssvi.params, sort(s.ssvi.θs))
end
