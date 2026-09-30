# Ground-truth tests for the Black-Scholes warm-up.
# Implement the functions in src/blackscholes.jl until every set here is green.
# The tests are ordered roughly in the order you should implement:
#   normal_cdf  →  bs_price  →  bs_vega  →  bs_delta  →  implied_vol

@testset "normal_cdf" begin
    @test normal_cdf(0.0) ≈ 0.5
    @test normal_cdf(-1.3) ≈ 1 - normal_cdf(1.3)      # symmetry Φ(-x)=1-Φ(x)
    @test normal_cdf(-8.0) < 1e-10                     # left tail → 0
    @test normal_cdf(8.0)  > 1 - 1e-10                 # right tail → 1
    @test normal_cdf(-0.5) < normal_cdf(0.5)           # monotone increasing
end

# Canonical case: S=100, K=100, r=0.05, q=0, σ=0.2, T=1 (computed to 1e-15).
const GOLDEN_CALL = 10.450583572185565
const GOLDEN_PUT  =  5.573526022256971

@testset "bs_price — golden value" begin
    @test bs_price(100, 100, 0.05, 0.0, 0.2, 1.0; call=true)  ≈ GOLDEN_CALL
    @test bs_price(100, 100, 0.05, 0.0, 0.2, 1.0; call=false) ≈ GOLDEN_PUT
end

@testset "bs_price — put-call parity" begin
    # C - P = S·e^{-qT} - K·e^{-rT}, must hold for every strike.
    S, r, q, σ, T = 100.0, 0.03, 0.01, 0.25, 0.75
    for K in (60.0, 90.0, 100.0, 110.0, 150.0)
        C = bs_price(S, K, r, q, σ, T; call=true)
        P = bs_price(S, K, r, q, σ, T; call=false)
        @test C - P ≈ S*exp(-q*T) - K*exp(-r*T)
    end
end

@testset "bs_price — no-arbitrage bounds & monotonicity" begin
    S, K, r, q, T = 100.0, 105.0, 0.04, 0.0, 0.5
    lo = max(0.0, S*exp(-q*T) - K*exp(-r*T))          # discounted intrinsic
    hi = S*exp(-q*T)                                    # can't be worth more than spot
    for σ in (0.05, 0.2, 0.5, 1.0)
        C = bs_price(S, K, r, q, σ, T; call=true)
        @test lo ≤ C ≤ hi
    end
    # price is strictly increasing in σ (vega > 0)
    @test bs_price(S, K, r, q, 0.1, T) < bs_price(S, K, r, q, 0.3, T) < bs_price(S, K, r, q, 0.6, T)
end

@testset "bs_price — degenerate limits" begin
    S, K, r, q = 100.0, 90.0, 0.05, 0.0
    # σ → 0 : price → discounted intrinsic of the forward
    @test bs_price(S, K, r, q, 1e-8, 1.0; call=true) ≈ max(0.0, S*exp(-q) - K*exp(-r)) atol=1e-4
    # T → 0 : price → intrinsic
    @test bs_price(S, K, r, q, 0.2, 1e-10; call=true)  ≈ max(0.0, S - K) atol=1e-4
    @test bs_price(S, K, r, q, 0.2, 1e-10; call=false) ≈ max(0.0, K - S) atol=1e-4
end

@testset "bs_vega — matches finite difference" begin
    S, K, r, q, σ, T = 100.0, 110.0, 0.03, 0.01, 0.22, 1.5
    h = 1e-6
    fd = (bs_price(S, K, r, q, σ+h, T) - bs_price(S, K, r, q, σ-h, T)) / (2h)
    @test bs_vega(S, K, r, q, σ, T) ≈ fd rtol=1e-5
    @test bs_vega(S, K, r, q, σ, T) > 0                 # vega always positive
end

@testset "bs_delta — matches finite difference" begin
    S, K, r, q, σ, T = 100.0, 95.0, 0.03, 0.0, 0.3, 1.0
    h = 1e-4
    fd_call = (bs_price(S+h, K, r, q, σ, T; call=true)  - bs_price(S-h, K, r, q, σ, T; call=true))  / (2h)
    fd_put  = (bs_price(S+h, K, r, q, σ, T; call=false) - bs_price(S-h, K, r, q, σ, T; call=false)) / (2h)
    @test bs_delta(S, K, r, q, σ, T; call=true)  ≈ fd_call rtol=1e-4
    @test bs_delta(S, K, r, q, σ, T; call=false) ≈ fd_put  rtol=1e-4
    # call delta ∈ (0,1), put delta ∈ (-1,0)
    @test 0 < bs_delta(S, K, r, q, σ, T; call=true)  < 1
    @test -1 < bs_delta(S, K, r, q, σ, T; call=false) < 0
end

@testset "implied_vol — round trip" begin
    S, K, r, q, T = 100.0, 100.0, 0.03, 0.0, 1.0
    for σ_true in (0.08, 0.15, 0.30, 0.60, 1.0)
        for call in (true, false)
            px = bs_price(S, K, r, q, σ_true, T; call=call)
            @test implied_vol(px, S, K, r, q, T; call=call) ≈ σ_true rtol=1e-6
        end
    end
    # off-the-money strikes too, where vega is smaller
    for K in (70.0, 130.0)
        σ_true = 0.28
        px = bs_price(S, K, r, q, σ_true, T; call=true)
        @test implied_vol(px, S, K, r, q, T; call=true) ≈ σ_true rtol=1e-6
    end
end

@testset "bs_gamma — matches finite difference, same for calls and puts" begin
    S, K, r, q, σ, T = 100.0, 105.0, 0.03, 0.01, 0.25, 0.8
    h = 1e-3
    for call in (true, false)
        fd = (bs_price(S+h, K, r, q, σ, T; call) - 2bs_price(S, K, r, q, σ, T; call) +
              bs_price(S-h, K, r, q, σ, T; call)) / h^2
        @test bs_gamma(S, K, r, q, σ, T) ≈ fd rtol=1e-4
    end
    @test bs_gamma(S, K, r, q, σ, T) > 0
    @test bs_gamma(S, K, r, q, σ, T) ≈
          ForwardDiff.derivative(s -> bs_delta(s, K, r, q, σ, T), S)
    @test bs_gamma(S, K, r, q, 0.0, T) == 0
    @test bs_gamma(S, K, r, q, σ, 0.0) == 0
end

@testset "bs_theta — matches −∂/∂T, parity" begin
    S, K, r, q, σ, T = 100.0, 95.0, 0.04, 0.015, 0.3, 1.2
    h = 1e-6
    for call in (true, false)
        fd = -(bs_price(S, K, r, q, σ, T+h; call) - bs_price(S, K, r, q, σ, T-h; call)) / (2h)
        @test bs_theta(S, K, r, q, σ, T; call) ≈ fd rtol=1e-6
    end
    # Θ_C − Θ_P = −∂/∂T (S·e^{−qT} − K·e^{−rT}) = q·S·e^{−qT} − r·K·e^{−rT}
    @test bs_theta(S, K, r, q, σ, T; call=true) - bs_theta(S, K, r, q, σ, T; call=false) ≈
          q*S*exp(-q*T) - r*K*exp(-r*T)
    # σ → 0 limit agrees with the degenerate branch
    for call in (true, false), K in (80.0, 120.0)
        @test bs_theta(S, K, r, q, 1e-6, T; call) ≈ bs_theta(S, K, r, q, 0.0, T; call) atol=1e-8
    end
    @test bs_theta(S, K, r, q, σ, 0.0) == 0
end

@testset "bs_rho — matches finite difference, parity" begin
    S, K, r, q, σ, T = 100.0, 110.0, 0.035, 0.0, 0.2, 2.0
    h = 1e-6
    for call in (true, false)
        fd = (bs_price(S, K, r+h, q, σ, T; call) - bs_price(S, K, r-h, q, σ, T; call)) / (2h)
        @test bs_rho(S, K, r, q, σ, T; call) ≈ fd rtol=1e-6
    end
    @test bs_rho(S, K, r, q, σ, T; call=true) - bs_rho(S, K, r, q, σ, T; call=false) ≈
          K*T*exp(-r*T)
    @test bs_rho(S, K, r, q, σ, T; call=true) > 0
    @test bs_rho(S, K, r, q, σ, T; call=false) < 0
    for call in (true, false), K in (80.0, 120.0)
        @test bs_rho(S, K, r, q, 1e-6, T; call) ≈ bs_rho(S, K, r, q, 0.0, T; call) atol=1e-8
    end
end

@testset "Greeks — agree with ForwardDiff through bs_price" begin
    S, K, r, q, σ, T = 100.0, 102.0, 0.02, 0.01, 0.35, 0.6
    for call in (true, false)
        @test bs_delta(S, K, r, q, σ, T; call) ≈ ForwardDiff.derivative(s -> bs_price(s, K, r, q, σ, T; call), S)
        @test bs_vega(S, K, r, q, σ, T)        ≈ ForwardDiff.derivative(v -> bs_price(S, K, r, q, v, T; call), σ)
        @test bs_rho(S, K, r, q, σ, T; call)   ≈ ForwardDiff.derivative(x -> bs_price(S, K, x, q, σ, T; call), r)
        @test bs_theta(S, K, r, q, σ, T; call) ≈ -ForwardDiff.derivative(t -> bs_price(S, K, r, q, σ, t; call), T)
    end
end
