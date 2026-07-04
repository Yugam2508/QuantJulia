# Gates for the Heston characteristic function. No literature price values are
# pinned yet (that gate is required before Stage 6 per the rough-Heston spec);
# these are *structural* identities that a wrong CF cannot fake.

const P_STD = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)

@testset "heston_cf — normalization & martingale" begin
    for T in (0.1, 0.5, 1.0, 3.0)
        @test heston_cf(0.0, T, P_STD) ≈ 1
        # E[S_T]/F = 1 ⇔ ψ(−i) = 1: the discounted stock is a martingale.
        @test heston_cf(-1im, T, P_STD) ≈ 1 rtol = 1e-10
    end
end

@testset "heston_cf — CF properties on the real line" begin
    T = 1.0
    for u in (0.3, 1.0, 2.7, 5.0, 10.0)
        ψ = heston_cf(u, T, P_STD)
        @test abs(ψ) ≤ 1 + 1e-12                       # |E e^{iuX}| ≤ 1
        @test heston_cf(-u, T, P_STD) ≈ conj(ψ)        # X is real
    end
end

@testset "heston_cf — deterministic-variance collapse (ξ → 0)" begin
    # With ξ tiny and ρ = 0, v(t) = θ + (v0−θ)e^{−κt} is deterministic, so
    # X_T ~ N(−w/2, w) with w = θT + (v0−θ)(1−e^{−κT})/κ and
    # ψ(u) = exp(−(iu+u²)w/2). Exercises κ, θ, v0 handling in closed form —
    # the same role the α→1 gate will play for rough Heston.
    ξ = 1e-3
    for (κ, θ, v0, T) in ((1.0, 0.04, 0.04, 1.0),
                          (2.0, 0.04, 0.09, 0.5),
                          (0.5, 0.09, 0.02, 2.0))
        p = HestonParams(κ, θ, ξ, 0.0, v0)
        w = θ * T + (v0 - θ) * (1 - exp(-κ * T)) / κ
        for u in (0.5, 1.0, 2.0, 5.0)
            @test heston_cf(u, T, p) ≈ exp(-(im * u + u^2) * w / 2) rtol = 1e-4
        end
    end
end

@testset "heston_cf — long-maturity branch continuity (the little trap)" begin
    # The g₁ form of the CF develops 2πi phase jumps at long maturity when the
    # complex log crosses its branch cut; the g₂ form must sweep smoothly.
    T = 10.0
    us = range(0.01, 30.0, length = 2000)
    vals = [heston_cf(u, T, P_STD) for u in us]
    @test maximum(abs.(diff(vals))) < 0.05
end
