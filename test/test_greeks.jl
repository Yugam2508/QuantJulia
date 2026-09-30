# Model Greeks by AD: checked against finite differences, against the
# Black-Scholes closed forms in the ξ → 0 limit, and rough ↔ classical at H = 1/2.

@testset "heston_greeks — finite differences" begin
    S, K, r, q, T = 100.0, 105.0, 0.03, 0.01, 0.75
    p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    for call in (true, false)
        g = heston_greeks(S, K, r, q, p, T; call)
        P(s, x, t) = heston_price(s, K, x, q, p, t; call, rtol = 1e-12)
        @test g.price ≈ P(S, r, T)
        h = 1e-3
        @test g.delta ≈ (P(S + h, r, T) - P(S - h, r, T)) / 2h rtol = 1e-6
        @test g.gamma ≈ (P(S + h, r, T) - 2P(S, r, T) + P(S - h, r, T)) / h^2 rtol = 1e-3
        h = 1e-5
        @test g.theta ≈ -(P(S, r, T + h) - P(S, r, T - h)) / 2h rtol = 1e-5
        @test g.rho ≈ (P(S, r + h, T) - P(S, r - h, T)) / 2h rtol = 1e-5
        pv(v0) = heston_price(S, K, r, q, HestonParams(p.κ, p.θ, p.ξ, p.ρ, v0), T; call, rtol = 1e-12)
        @test g.sens.v0 ≈ (pv(p.v0 + h) - pv(p.v0 - h)) / 2h rtol = 1e-5
        @test g.vega ≈ 2sqrt(p.v0) * g.sens.v0
    end
    # Parity: call and put share gamma and parameter sensitivities.
    gc = heston_greeks(S, K, r, q, p, T; call = true)
    gp = heston_greeks(S, K, r, q, p, T; call = false)
    @test gc.gamma ≈ gp.gamma rtol = 1e-6
    @test gc.delta - gp.delta ≈ exp(-q * T) rtol = 1e-8
    @test gc.sens.ρ ≈ gp.sens.ρ rtol = 1e-6
end

@testset "heston_greeks → Black-Scholes as ξ → 0 with v0 = θ" begin
    S, K, r, q, T, σ = 100.0, 95.0, 0.02, 0.0, 0.5, 0.2
    p = HestonParams(1.0, σ^2, 1e-3, 0.0, σ^2)
    for call in (true, false)
        g = heston_greeks(S, K, r, q, p, T; call)
        @test g.delta ≈ bs_delta(S, K, r, q, σ, T; call) rtol = 1e-3
        @test g.gamma ≈ bs_gamma(S, K, r, q, σ, T) rtol = 1e-3
        @test g.theta ≈ bs_theta(S, K, r, q, σ, T; call) rtol = 1e-3
        @test g.rho ≈ bs_rho(S, K, r, q, σ, T; call) rtol = 1e-3
    end
end

@testset "rough_heston_greeks — H = 1/2 matches classical, FD at H = 0.1" begin
    S, K, r, q, T = 100.0, 100.0, 0.02, 0.0, 0.5
    gc = heston_greeks(S, K, r, q, HestonParams(2.0, 0.04, 0.5, -0.7, 0.04), T)
    gr = rough_heston_greeks(S, K, r, q, RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.5), T; N = 256)
    for k in (:price, :delta, :gamma, :theta, :rho, :vega)
        @test getfield(gr, k) ≈ getfield(gc, k) rtol = 2e-3
    end

    p = RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.1)
    g = rough_heston_greeks(S, K, r, q, p, T; N = 96)
    ivh = sqrt(p.v0)
    P(s, x, t) = QuantJulia._rough_price(s, K, x, q, p, t, true, 96, ivh)
    h = 1e-3
    @test g.delta ≈ (P(S + h, r, T) - P(S - h, r, T)) / 2h rtol = 1e-5
    @test g.gamma ≈ (P(S + h, r, T) - 2P(S, r, T) + P(S - h, r, T)) / h^2 rtol = 1e-3
    h = 1e-5
    @test g.theta ≈ -(P(S, r, T + h) - P(S, r, T - h)) / 2h rtol = 1e-4
    @test g.rho ≈ (P(S, r + h, T) - P(S, r - h, T)) / 2h rtol = 1e-5
    PH(H) = QuantJulia._rough_price(S, K, r, q, RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, H),
                                     T, true, 96, ivh)
    @test g.sens.H ≈ (PH(0.1 + h) - PH(0.1 - h)) / 2h rtol = 1e-4
    @test g.delta > 0 && g.gamma > 0 && g.theta < 0
end
