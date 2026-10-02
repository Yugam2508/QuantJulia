# Jump models: structural CF identities, an independent closed form (Merton's
# Poisson series), nesting limits, and generic calibration.

const PM = MertonParams(0.15, 0.8, -0.12, 0.10)
const PB = BatesParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.8, -0.12, 0.10)
const PVG = VGParams(0.18, 0.25, -0.14)

# Second cumulant (variance of X_T) from the CF: −d²/du² log ψ at 0, by a
# central difference (independent of the CF code's own structure).
_c2(ψ; h = 1e-3) = -real(log(ψ(h)) - 2log(ψ(0.0)) + log(ψ(-h))) / h^2

@testset "jump CFs — normalization, martingale, conjugate symmetry" begin
    for (name, cf) in (("Merton", (u, T) -> merton_cf(u, T, PM)),
                       ("Bates", (u, T) -> bates_cf(u, T, PB)),
                       ("VG", (u, T) -> vg_cf(u, T, PVG)))
        for T in (0.1, 1.0, 3.0)
            @test cf(0.0, T) ≈ 1
            @test cf(-1im, T) ≈ 1 rtol = 1e-10              # E[S_T] = F
            @test cf(-2.3, T) ≈ conj(cf(2.3, T))
            @test abs(cf(4.0, T)) <= 1 + 1e-12
        end
    end
    @test_throws DomainError vg_cf(1.0, 1.0, VGParams(0.5, 5.0, 0.3))
end

@testset "jump CFs — second cumulants match the model variances" begin
    T = 0.7
    k = exp(PM.μJ + PM.δJ^2 / 2) - 1
    @test _c2(u -> merton_cf(u, T, PM)) ≈ (PM.σ^2 + PM.λ * (PM.μJ^2 + PM.δJ^2)) * T rtol = 1e-5
    @test _c2(u -> vg_cf(u, T, PVG)) ≈ (PVG.σ^2 + PVG.ν * PVG.θ^2) * T rtol = 1e-5
end

@testset "Merton — Fourier price matches the closed-form Poisson series" begin
    S, r, q = 100.0, 0.03, 0.01
    for T in (0.1, 0.5, 2.0), K in (75.0, 95.0, 100.0, 110.0, 140.0), call in (true, false)
        f = price_from_cf(u -> merton_cf(u, T, PM), S, K, r, q, T; call)
        @test f ≈ merton_price(S, K, r, q, PM, T; call) rtol = 1e-7 atol = 1e-9
    end
    # λ = 0 is Black-Scholes
    p0 = MertonParams(0.2, 0.0, -0.1, 0.1)
    @test merton_price(100.0, 105.0, r, q, p0, 1.0) ≈ bs_price(100.0, 105.0, r, q, 0.2, 1.0)
end

@testset "Bates — nests Heston (λ = 0) and Merton (ξ → 0, v0 = θ)" begin
    T = 0.75
    ph = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    pb0 = BatesParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.0, -0.1, 0.1)
    for u in (0.5, 3.0, 12.0)
        @test bates_cf(u, T, pb0) ≈ heston_cf(u, T, ph)
    end
    σ = 0.2
    pb = BatesParams(1.0, σ^2, 1e-4, 0.0, σ^2, 0.8, -0.12, 0.10)
    pm = MertonParams(σ, 0.8, -0.12, 0.10)
    for u in (0.5, 3.0, 12.0)
        @test bates_cf(u, T, pb) ≈ merton_cf(u, T, pm) rtol = 1e-5
    end
end

@testset "Variance Gamma — ν → 0 is Black-Scholes; skew sign follows θ" begin
    S, K, r, q, T = 100.0, 105.0, 0.02, 0.0, 0.5
    near_bs = price_from_cf(u -> vg_cf(u, T, VGParams(0.2, 1e-6, 0.0)), S, K, r, q, T)
    @test near_bs ≈ bs_price(S, K, r, q, 0.2, T) rtol = 1e-5
    iv(p, K) = (c = price_from_cf(u -> vg_cf(u, T, p), S, K, r, q, T);
                implied_vol(c, S, K, r, q, T))
    @test iv(PVG, 90.0) > iv(PVG, 110.0)                       # θ < 0: left skew
    pos = VGParams(0.18, 0.25, 0.14)
    @test iv(pos, 90.0) < iv(pos, 110.0)                       # θ > 0: right skew
end

@testset "short-dated wings — jumps fatten them, Heston can't" begin
    # The classic motivation: at T = 1 week, Bates' downside wing (the 90
    # put) sits far above ATM, while Heston with the same diffusion barely
    # lifts it — a diffusion can't move 10% in a week.
    S, r, q, T = 100.0, 0.0, 0.0, 7 / 365
    ph = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    iv(cf, K) = implied_vol(batch_call_prices(cf, S, 1.0, (K,), T; iv_hint = 0.2)[1],
                            S, K, r, q, T)
    wing(cf) = iv(cf, 90.0) - iv(cf, 100.0)
    @test wing(u -> bates_cf(u, T, PB)) > 3 * wing(u -> heston_cf(u, T, ph))
end

@testset "calibrate_cf_model — recovers Merton parameters from a synthetic surface" begin
    S, r, qy = 100.0, 0.02, 0.0
    truth = MertonParams(0.15, 0.8, -0.12, 0.10)
    quotes = @NamedTuple{T::Float64, K::Float64, r::Float64, q::Float64,
                         side::Symbol, mid::Float64, iv::Float64, vega::Float64}[]
    for T in (0.05, 0.25, 1.0), z in -2:0.5:2
        F = S * exp((r - qy) * T); K = F * exp(z * 0.2 * sqrt(T)); side = K >= F ? :call : :put
        mid = merton_price(S, K, r, qy, truth, T; call = side === :call)
        iv = implied_vol(mid, S, K, r, qy, T; call = side === :call)
        push!(quotes, (; T, K, r, q = qy, side, mid, iv, vega = bs_vega(S, K, r, qy, iv, T)))
    end
    # x = [log σ, log λ, μJ, log δJ]
    mk(x, T) = (p = MertonParams(exp(x[1]), exp(x[2]), x[3], exp(x[4])); u -> merton_cf(u, T, p))
    res = calibrate_cf_model(mk, [log(0.2), log(0.3), -0.05, log(0.2)], quotes; S)
    @test res.converged
    @test res.rmse < 1e-4
    p = MertonParams(exp(res.x[1]), exp(res.x[2]), res.x[3], exp(res.x[4]))
    @test p.σ ≈ truth.σ rtol = 0.02
    @test p.λ ≈ truth.λ rtol = 0.1
    @test p.μJ ≈ truth.μJ rtol = 0.1
    @test p.δJ ≈ truth.δJ rtol = 0.1
end
