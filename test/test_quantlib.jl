# Cross-validation against QuantLib 1.43. The reference values are generated
# by validation/quantlib_reference.py and committed as
# validation/quantlib_reference.csv, so CI checks agreement without QuantLib.
# Tolerances are per family: closed forms agree to machine precision; the
# Fourier pricers to ~1e-11; the American put compares two different
# approximations (our CRR tree vs QuantLib's finite-difference grid).

const QL_CSV = joinpath(@__DIR__, "..", "validation", "quantlib_reference.csv")

function _ql_ours(case, kind, call, S, K, r, q, T, e)
    if case == "bs"
        σ = e[1]
        kind == "price" ? bs_price(S, K, r, q, σ, T; call) :
        kind == "delta" ? bs_delta(S, K, r, q, σ, T; call) :
        kind == "gamma" ? bs_gamma(S, K, r, q, σ, T) :
        kind == "vega" ? bs_vega(S, K, r, q, σ, T) :
        kind == "theta" ? bs_theta(S, K, r, q, σ, T; call) :
        kind == "rho" ? bs_rho(S, K, r, q, σ, T; call) :
        implied_vol(bs_price(S, K, r, q, σ, T; call), S, K, r, q, T; call)
    elseif case == "heston"
        heston_price(S, K, r, q, HestonParams(e...), T; call, rtol = 1e-12)
    elseif case == "bates"
        price_from_cf(u -> bates_cf(u, T, BatesParams(e...)), S, K, r, q, T; call, rtol = 1e-12)
    elseif case == "american"
        crr_price(S, K, r, q, e[1], T; call = false, N = 5000)
    elseif case == "barrier"
        barrier_price(S, K, e[2], r, q, e[1], T; kind = Symbol(kind), call)
    elseif case == "asian_geometric"
        geometric_asian_price(S, K, r, q, e[1], T, Int(e[2]); call)
    else
        error("unknown case $case")
    end
end

const QL_TOL = Dict("bs" => 1e-10, "heston" => 1e-9, "bates" => 1e-9, "american" => 2e-3,
                    "barrier" => 1e-10, "asian_geometric" => 1e-10)

@testset "QuantLib 1.43 cross-validation" begin
    lines = readlines(QL_CSV)
    @test lines[1] == "case,model,kind,call,S,K,r,q,T,extra,value"
    counts = Dict{String,Int}()
    for l in lines[2:end]
        f = split(l, ',')
        case, kind, call = f[1], f[3], f[4] == "1"
        S, K, r, q, T = parse.(Float64, f[5:9])
        e = parse.(Float64, split(f[10], ';'))
        ql = parse(Float64, f[11])
        @test isapprox(_ql_ours(case, kind, call, S, K, r, q, T, e), ql; atol = QL_TOL[case])
        counts[case] = get(counts, case, 0) + 1
    end
    @test sum(values(counts)) == 119                  # the whole reference set is exercised
    @test Set(keys(counts)) == Set(keys(QL_TOL))
end
