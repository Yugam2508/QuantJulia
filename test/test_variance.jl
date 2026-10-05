# Variance swaps: the CF route against closed-form expected variance
# (Heston; rough Heston via Mittag-Leffler), and model-free replication
# against the model strike.
using SpecialFunctions: gamma

# Two-parameter Mittag-Leffler function by its power series (fine for |z| ≲ 5).
_mittag_leffler(z, α, β; n = 200) = sum(z^k / gamma(α * k + β) for k in 0:n)

@testset "variance_swap_strike — Black-Scholes, Heston, rough Heston closed forms" begin
    for T in (0.1, 1.0)
        @test variance_swap_strike(u -> exp(-(im * u + u^2) * 0.04 * T / 2), T) ≈ 0.04 rtol = 1e-7
    end
    p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.09)
    for T in (0.05, 0.5, 3.0)
        exact = (p.θ * T + (p.v0 - p.θ) * (1 - exp(-p.κ * T)) / p.κ) / T
        @test variance_swap_strike(u -> heston_cf(u, T, p), T) ≈ exact rtol = 1e-6
    end
    # Rough Heston: E[v_t] = θ + (v0−θ)·E_α(−κt^α), so
    # ∫₀ᵀ E[v_t]dt = θT + (v0−θ)·T·E_{α,2}(−κT^α). Independent of the solver.
    pr = RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.09, 0.1)
    α = pr.H + 0.5
    for T in (0.1, 0.5, 1.0)
        exact = pr.θ + (pr.v0 - pr.θ) * _mittag_leffler(-pr.κ * T^α, α, 2.0)
        @test variance_swap_strike(make_rough_cf(T, pr; N = 512), T) ≈ exact rtol = 1e-4
    end
end

@testset "replicate_variance — Black-Scholes strip recovers σ²" begin
    S, r, q, σ, T = 100.0, 0.03, 0.01, 0.25, 0.5
    F = S * exp((r - q) * T); disc = exp(-r * T)
    Ks = collect(range(F * exp(-8σ * sqrt(T)), F * exp(8σ * sqrt(T)), length = 801))
    otm = [K < F ? bs_price(S, K, r, q, σ, T; call = false) : bs_price(S, K, r, q, σ, T) for K in Ks]
    @test replicate_variance(Ks, otm, F, disc, T) ≈ σ^2 rtol = 1e-3
    # A narrow strip misses the wings: it must UNDER-state the strike.
    nar = findall(K -> abs(log(K / F)) < 0.5σ * sqrt(T), Ks)
    @test replicate_variance(Ks[nar], otm[nar], F, disc, T) < 0.8σ^2
    @test_throws ArgumentError replicate_variance(Ks[[2, 1, 3]], otm[1:3], F, disc, T)
end

@testset "replication matches the model strike on a Heston chain" begin
    S, r, q, T = 100.0, 0.02, 0.0, 0.5
    p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.06)
    F = S * exp((r - q) * T); disc = exp(-r * T)
    sd = sqrt(0.05 * T)
    Ks = collect(range(F * exp(-9sd), F * exp(6sd), length = 601))
    cs = batch_call_prices(u -> heston_cf(u, T, p), F, disc, Ks, T; iv_hint = 0.25)
    otm = [K < F ? c - disc * (F - K) : c for (K, c) in zip(Ks, cs)]
    @test replicate_variance(Ks, otm, F, disc, T) ≈
          variance_swap_strike(u -> heston_cf(u, T, p), T) rtol = 5e-3
end

@testset "market_variance_term_structure and model_vix" begin
    S, r, q, σ = 100.0, 0.01, 0.0, 0.2
    quotes = @NamedTuple{T::Float64, K::Float64, F::Float64, r::Float64, side::Symbol, mid::Float64}[]
    for T in (0.1, 0.5)
        F = S * exp((r - q) * T)
        for K in range(F * exp(-8σ * sqrt(T)), F * exp(8σ * sqrt(T)), length = 401)
            side = K < F ? :put : :call
            push!(quotes, (; T, K, F, r, side, mid = bs_price(S, K, r, q, σ, T; call = side === :call)))
        end
    end
    ts = market_variance_term_structure(quotes)
    @test length(ts) == 2
    @test all(t -> isapprox(t.vol, σ; rtol = 1e-3), ts)
    @test model_vix(T -> (u -> exp(-(im * u + u^2) * σ^2 * T / 2))) ≈ 20.0 rtol = 1e-6
end
