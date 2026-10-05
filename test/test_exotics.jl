# Path-dependent options: every Monte Carlo pricer is checked against an exact
# reference (closed form or a converged lattice) under Black-Scholes.
using Random

@testset "CRR — European converges to BS; American put benchmark; no early call exercise" begin
    S, K, r, q, σ, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0
    for call in (true, false)
        @test crr_price(S, K, r, q, σ, T; call, american = false, N = 2000) ≈
              bs_price(S, K, r, q, σ, T; call) atol = 2e-3
    end
    am = crr_price(S, K, r, q, σ, T; call = false, N = 5000)
    @test am ≈ 6.0904 atol = 2e-3                     # standard American put benchmark
    @test am > bs_price(S, K, r, q, σ, T; call = false)
    # q = 0: early exercise of a call is never optimal
    @test crr_price(S, K, r, 0.0, σ, T; call = true, N = 1000) ≈
          crr_price(S, K, r, 0.0, σ, T; call = true, american = false, N = 1000)
end

@testset "paths — GBM and Heston are martingales after discounting" begin
    S, r, q, T = 100.0, 0.03, 0.01, 1.0
    for P in (simulate_gbm_paths(S, r, q, 0.25, T, 20, 100_000; rng = MersenneTwister(1)),
              simulate_heston_paths(S, r, q, HestonParams(2.0, 0.04, 0.5, -0.7, 0.04), T, 20,
                                    100_000; rng = MersenneTwister(2)))
        @test size(P) == (100_000, 21)
        @test all(==(S), P[:, 1])
        ST = P[:, end]
        se = sqrt(sum(abs2, ST .- sum(ST) / length(ST)) / (length(ST) - 1) / length(ST))
        @test abs(sum(ST) / length(ST) - S * exp((r - q) * T)) < 4se
    end
end

@testset "LSM — American put vs the CRR tree; Heston early-exercise premium" begin
    S, K, r, q, σ, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0
    P = simulate_gbm_paths(S, r, q, σ, T, 50, 100_000; rng = MersenneTwister(1))
    l = lsm_american_price(P, K, r, T)
    ref = crr_price(S, K, r, q, σ, T; call = false, N = 5000)
    @test abs(l.price - ref) < 4l.stderr + 0.02       # LSM is slightly low-biased
    @test l.price > bs_price(S, K, r, q, σ, T; call = false)
    ph = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    Ph = simulate_heston_paths(S, r, q, ph, T, 50, 100_000; rng = MersenneTwister(2))
    lh = lsm_american_price(Ph, K, r, T)
    @test lh.price - heston_price(S, K, r, q, ph, T; call = false) > 4lh.stderr
end

@testset "barrier_price — in/out parity, limits, breached barriers" begin
    S, r, q, σ, T = 100.0, 0.05, 0.02, 0.25, 0.75
    for call in (true, false), K in (90.0, 100.0, 110.0)
        van = bs_price(S, K, r, q, σ, T; call)
        for H in (80.0, 95.0)
            @test barrier_price(S, K, H, r, q, σ, T; kind = :down_out, call) +
                  barrier_price(S, K, H, r, q, σ, T; kind = :down_in, call) ≈ van
        end
        for H in (105.0, 125.0)
            @test barrier_price(S, K, H, r, q, σ, T; kind = :up_out, call) +
                  barrier_price(S, K, H, r, q, σ, T; kind = :up_in, call) ≈ van
        end
        @test barrier_price(S, K, 1e-6, r, q, σ, T; kind = :down_out, call) ≈ van rtol = 1e-8
        @test barrier_price(S, K, 1e6, r, q, σ, T; kind = :up_out, call) ≈ van rtol = 1e-8
        @test barrier_price(S, K, 120.0, r, q, σ, T; kind = :down_out, call) == 0
        @test barrier_price(S, K, 120.0, r, q, σ, T; kind = :down_in, call) == van
    end
    @test_throws ArgumentError barrier_price(S, 100.0, 90.0, r, q, σ, T; kind = :double)
end

@testset "barrier_mc_price — discrete monitoring vs the BGK-shifted closed form" begin
    # BGK is an asymptotic correction: exact to < 0.01% for the down-and-out
    # call here, but 1–2% off for up-and-out calls (payoff large at the
    # barrier; measured with 1M paths). Tolerance = MC noise + 2.5% of price.
    S, K, r, q, σ, T, n = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 50
    P = simulate_gbm_paths(S, r, q, σ, T, n, 100_000; rng = MersenneTwister(1))
    for (kind, H, call) in ((:down_out, 90.0, true), (:down_in, 90.0, false),
                            (:up_out, 120.0, true), (:up_in, 120.0, true))
        mc = barrier_mc_price(P, K, H, r, T; kind, call)
        Hc = bgk_barrier(H, σ, T, n; down = kind in (:down_out, :down_in))
        ref = barrier_price(S, K, Hc, r, q, σ, T; kind, call)
        @test abs(mc.price - ref) < 4mc.stderr + 0.025ref
    end
    # where BGK is sharp, it is decisive: the down-and-out call matches the
    # shifted formula to MC noise, while the unshifted one is far off
    mc = barrier_mc_price(P, K, 90.0, r, T; kind = :down_out)
    @test abs(mc.price - barrier_price(S, K, bgk_barrier(90.0, σ, T, n), r, q, σ, T)) < 4mc.stderr
    @test abs(mc.price - barrier_price(S, K, 90.0, r, q, σ, T)) > 8mc.stderr
end

@testset "Asian — geometric closed form, arithmetic ≥ geometric, control variate" begin
    S, K, r, q, σ, T, n = 100.0, 100.0, 0.05, 0.01, 0.2, 1.0, 50
    P = simulate_gbm_paths(S, r, q, σ, T, n, 100_000; rng = MersenneTwister(3))
    disc = exp(-r * T)
    for call in (true, false)
        geo = [disc * QuantJulia._payoff(exp(sum(log, @view P[i, 2:end]) / n), K, call) for i in 1:size(P, 1)]
        m = sum(geo) / length(geo); se = sqrt(sum(abs2, geo .- m) / (length(geo) - 1) / length(geo))
        @test abs(m - geometric_asian_price(S, K, r, q, σ, T, n; call)) < 4se
    end
    plain = asian_mc_price(P, K, r, T)
    cv = asian_mc_price(P, K, r, T; control = (S, q, σ))
    @test abs(plain.price - cv.price) < 4plain.stderr
    @test cv.stderr < plain.stderr / 10                 # the control variate earns its keep
    @test cv.price > geometric_asian_price(S, K, r, q, σ, T, n)   # AM ≥ GM
end
