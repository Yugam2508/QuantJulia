# Calibration diagnostics: exact on linear least squares, the known sloppy
# direction for Heston, and H identifiability driven by short maturities.
using LinearAlgebra: diag, inv

@testset "calibration_diagnostics — exact OLS covariance, transform, noise" begin
    A = [1.0 0.0; 1.0 1.0; 1.0 2.0; 1.0 3.0; 1.0 4.5]
    b = [0.1, 1.2, 1.9, 3.2, 4.4]
    x̂ = A \ b
    rf(x) = A * x - b
    d = calibration_diagnostics(rf, x̂; noise = 0.3)
    @test d.cov ≈ 0.09 * inv(A' * A)
    @test d.stderr ≈ sqrt.(diag(0.09 * inv(A' * A)))
    @test d.corr[1, 1] ≈ 1 && d.corr[1, 2] ≈ d.cov[1, 2] / prod(d.stderr)
    @test issorted(d.eigvals) && d.condition ≈ d.eigvals[2] / d.eigvals[1]
    # default σ: residual RMS with the n − m correction
    @test calibration_diagnostics(rf, x̂).sigma ≈ sqrt(sum(abs2, rf(x̂)) / 3)
    # delta method: g(x) = 2x doubles the standard errors
    @test calibration_diagnostics(rf, x̂; noise = 0.3, transform = x -> 2x).stderr ≈ 2d.stderr
    # standard errors scale linearly with the noise level
    @test calibration_diagnostics(rf, x̂; noise = 0.6).stderr ≈ 2d.stderr
    @test_throws ArgumentError calibration_diagnostics(x -> [x[1] - 1], [0.0, 0.0])
end

# synthetic quote set (prepare_chain shape) from any CF maker
function _diag_surface(mkcf, Ts; S = 100.0)
    qs = @NamedTuple{T::Float64, K::Float64, F::Float64, r::Float64, q::Float64,
                     side::Symbol, mid::Float64, iv::Float64, vega::Float64}[]
    for T in Ts, z in -2:0.5:2
        K = S * exp(z * 0.2 * sqrt(T)); side = K >= S ? :call : :put
        c = batch_call_prices(mkcf(T), S, 1.0, (K,), T; iv_hint = 0.2)[1]
        mid = side === :call ? c : c - (S - K)
        iv = implied_vol(mid, S, K, 0.0, 0.0, T; call = side === :call)
        push!(qs, (; T, K, F = S, r = 0.0, q = 0.0, side, mid, iv, vega = bs_vega(S, K, 0.0, 0.0, iv, T)))
    end
    return qs
end

@testset "heston_diagnostics — κ is the least determined parameter" begin
    p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    d = heston_diagnostics(_diag_surface(T -> (u -> heston_cf(u, T, p)), [0.1, 0.25, 0.5, 1.0, 2.0]),
                           100.0, p)
    @test d.names == ["κ", "θ", "ξ", "ρ", "v0"]
    rel = d.stderr ./ abs.([p.κ, p.θ, p.ξ, p.ρ, p.v0])
    @test argmax(rel) == 1
    @test all(>(0), d.stderr)
end

@testset "rough_heston_diagnostics — short maturities identify H" begin
    p = RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.1)
    long = rough_heston_diagnostics(_diag_surface(T -> make_rough_cf(T, p; N = 64), [0.5, 1.0, 2.0]), p; N = 64)
    full = rough_heston_diagnostics(_diag_surface(T -> make_rough_cf(T, p; N = 64),
                                                  [0.02, 0.05, 0.1, 0.5, 1.0, 2.0]), p; N = 64)
    @test long.names[6] == "H"
    @test full.stderr[6] < long.stderr[6] / 5           # the short end is where H lives
    @test abs(long.eigvecs[6, 1]) > 0.8                 # sloppiest direction is mostly H
    @test full.condition < long.condition
end
