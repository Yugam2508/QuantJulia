# =============================================================================
# Reproducible multithreaded Monte Carlo
# =============================================================================
# Every simulator splits its paths into fixed blocks of `MC_BLOCK` paths.
# Block b draws from its own Xoshiro256++ stream, seeded from (master seed, b)
# through a SplitMix64 mix, and the blocks run under `Threads.@threads`. Which
# thread runs which block, and in what order, does not change a single draw, so
#
#   * a given seed gives bit-identical paths with 1 thread or 64;
#   * blocks never share an RNG, so there is no lock and no race;
#   * paths are independent across blocks (Xoshiro seeding hashes its seed
#     with SHA-256, so neighbouring block seeds give unrelated streams).
#
# The master seed comes from the `seed` keyword when given, otherwise from one
# draw of `rng` (the task-local default RNG when `rng = nothing`). So
# `Random.seed!(k)` before a call, `rng = MersenneTwister(k)` and `seed = k`
# all reproduce a run exactly. The streams are those of the Julia version in
# use; Random does not promise identical streams across Julia releases.
# =============================================================================

"""
    MC_BLOCK

Number of paths per independently seeded block in every Monte Carlo
simulator (see `src/parallel.jl`). Part of the reproducibility contract:
changing it changes the paths a seed produces.
"""
const MC_BLOCK = 1024

function _splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

_master_seed(rng, seed) =
    seed !== nothing ? seed % UInt64 :
    rand(rng === nothing ? Random.default_rng() : rng, UInt64)

_block_rng(s::UInt64, b::Integer) = Xoshiro(_splitmix64(s ⊻ _splitmix64(UInt64(b))))

# Run `f(paths, rng)` for each block of path indices, in parallel. `f` must
# write only to entries indexed by its own `paths` and draw only from its own
# `rng`.
function _foreach_block(f, npaths::Integer, rng, seed)
    s = _master_seed(rng, seed)
    nb = cld(npaths, MC_BLOCK)
    Threads.@threads for b in 1:nb
        f(((b - 1) * MC_BLOCK + 1):min(b * MC_BLOCK, npaths), _block_rng(s, b))
    end
    return nothing
end
