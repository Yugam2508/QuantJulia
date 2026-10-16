using Test, QuantJulia, Random

# The reproducibility contract of src/parallel.jl: a seed fixes every path,
# whatever the thread count, and paths are a deterministic function of
# (seed, block), so a shorter run is a prefix of a longer one.

@testset "parallel Monte Carlo" begin
    ph = HestonParams(1.5, 0.04, 0.5, -0.7, 0.04)
    pr = RoughHestonParams(1.5, 0.04, 0.15, -0.7, 0.04, 0.2)
    pb = RBergomiParams(0.04, 1.5, -0.7, 0.1)
    w = (k, T) -> 0.04 * T + 0.01 * k^2
    sims = [
        "heston_qe"     => (n; kw...) -> simulate_heston(ph, 0.5, 8, n; kw...),
        "heston_euler"  => (n; kw...) -> simulate_heston(ph, 0.5, 8, n; scheme = :euler, kw...),
        "rough_heston"  => (n; kw...) -> simulate_rough_heston(pr, 0.5, 8, n; kw...),
        "rbergomi"      => (n; kw...) -> simulate_rbergomi(pb, 0.5, 8, n; kw...).X,
        "gbm_paths"     => (n; kw...) -> vec(simulate_gbm_paths(100.0, 0.03, 0.01, 0.2, 0.5, 4, n; kw...)[:, end]),
        "heston_paths"  => (n; kw...) -> vec(simulate_heston_paths(100.0, 0.03, 0.01, ph, 0.5, 4, n; kw...)[:, end]),
    ]
    n = 2MC_BLOCK + 300                      # three blocks, the last one partial

    @testset "$name" for (name, sim) in sims
        a = sim(n; seed = 42)
        @test length(a) == n
        @test sim(n; seed = 42) == a                           # same seed, same paths
        @test sim(n; seed = 43) != a                           # different seed, different paths
        @test sim(MC_BLOCK + 10; seed = 42) == a[1:MC_BLOCK+10] # prefix property
        @test sim(n; rng = MersenneTwister(7)) == sim(n; rng = MersenneTwister(7))
        Random.seed!(11); b = sim(n)
        Random.seed!(11); @test sim(n) == b                    # global seeding reproduces
        @test length(unique(a)) == n                           # no repeated streams across blocks
    end

    @testset "local vol prices" begin
        r1 = local_vol_mc_prices(100.0, [90.0, 100.0], 0.0, 0.0, w, 0.25; nsteps = 8, npaths = n, seed = 5)
        r2 = local_vol_mc_prices(100.0, [90.0, 100.0], 0.0, 0.0, w, 0.25; nsteps = 8, npaths = n, seed = 5)
        @test r1 == r2
    end

    @testset "pricers forward the seed" begin
        @test heston_mc_price(100.0, 100.0, 0.0, 0.0, ph, 0.5; nsteps = 8, npaths = n, seed = 3) ==
              heston_mc_price(100.0, 100.0, 0.0, 0.0, ph, 0.5; nsteps = 8, npaths = n, seed = 3)
        @test rough_heston_mc_price(100.0, 100.0, 0.0, 0.0, pr, 0.5; nsteps = 8, npaths = n, seed = 3) ==
              rough_heston_mc_price(100.0, 100.0, 0.0, 0.0, pr, 0.5; nsteps = 8, npaths = n, seed = 3)
        @test rbergomi_mc_prices(100.0, [100.0], 0.0, 0.0, pb, 0.5; nsteps = 8, npaths = n, seed = 3) ==
              rbergomi_mc_prices(100.0, [100.0], 0.0, 0.0, pb, 0.5; nsteps = 8, npaths = n, seed = 3)
    end

    @testset "independent of the thread count" begin
        code = """
            using QuantJulia
            p = HestonParams(1.5, 0.04, 0.5, -0.7, 0.04)
            x = simulate_heston(p, 0.5, 8, $(4MC_BLOCK + 17); seed = 2026)
            y = simulate_rough_heston(RoughHestonParams(1.5, 0.04, 0.15, -0.7, 0.04, 0.2),
                                      0.5, 8, $(3MC_BLOCK); seed = 2026)
            print(Threads.nthreads(), " ", hash((x, y)))
            """
        here = hash((simulate_heston(ph, 0.5, 8, 4MC_BLOCK + 17; seed = 2026),
                     simulate_rough_heston(pr, 0.5, 8, 3MC_BLOCK; seed = 2026)))
        for t in (1, 3)
            cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t $t -e $code`
            nt, h = split(read(cmd, String))
            @test parse(Int, nt) == t
            @test parse(UInt, h) == here
        end
    end
end
