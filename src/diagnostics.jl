# =============================================================================
# Calibration identifiability diagnostics
# =============================================================================
# A calibration returns a point; this says how well the data pins it down.
# Every residual in the package is (≈) an implied-vol error, so if quotes
# carry IV noise of size σ_ε (e.g. 1 bp), the Gauss-Newton approximation
# gives the parameter covariance at the fit,
#     Cov(x) ≈ σ_ε² (JᵀJ)⁻¹,   J = ∂r/∂x  (ForwardDiff),
# and by the delta method, for natural parameters p = g(x),
#     Cov(p) ≈ G Cov(x) Gᵀ,   G = ∂g/∂x.
# The eigen-decomposition of JᵀJ exposes "sloppy" directions — parameter
# combinations the data barely constrains (small eigenvalues). That is the
# quantitative form of docs/rough_heston.md's finding that H was
# unidentified on one weekend snapshot. On a synthetic rough surface with
# 1 bp IV noise: maturities 0.5–2y only give SE(H) ≈ 0.017 and a sloppiest
# direction that is 96% H; adding 0.02–0.1y expiries cuts SE(H) twelvefold
# (≈ 0.0014) — the short end is where H lives.
#
# σ_ε defaults to the fit's own residual RMS (with the degrees-of-freedom
# correction); pass `noise` to use a quote-noise level instead — necessary on
# synthetic, perfectly fitted data where the residuals are ~0.
# =============================================================================

using LinearAlgebra: Symmetric, eigen, inv, diag, I

"""
    calibration_diagnostics(rfun, x; noise=nothing, transform=identity, names=nothing)

Identifiability report for a least-squares fit at `x`, where `rfun(x)` is the
residual vector (ForwardDiff-generic). Returns a NamedTuple with

- `stderr`: standard errors of the (transformed) parameters,
- `corr`: their correlation matrix,
- `cov`: their covariance,
- `eigvals`, `eigvecs`: eigen-decomposition of JᵀJ in the fitting
  coordinates `x` (ascending — the first column is the sloppiest direction),
- `condition`: λ_max/λ_min of JᵀJ,
- `sigma`: the IV noise level used, `names`: parameter labels.

`noise` sets σ_ε (default: residual RMS with an n − m correction);
`transform(x)` maps fitting coordinates to natural parameters for the
delta-method covariance.
"""
function calibration_diagnostics(rfun, x::AbstractVector; noise = nothing,
                                 transform = identity, names = nothing)
    r = rfun(x)
    J = ForwardDiff.jacobian(rfun, x)
    n, m = size(J)
    n > m || throw(ArgumentError("need more residuals ($n) than parameters ($m)"))
    σ = noise === nothing ? sqrt(sum(abs2, r) / (n - m)) : float(noise)
    JtJ = Symmetric(J' * J)
    E = eigen(JtJ)
    covx = σ^2 * inv(JtJ)
    G = transform === identity ? Matrix{Float64}(I, m, m) : ForwardDiff.jacobian(transform, x)
    cov = G * covx * G'
    se = sqrt.(max.(diag(cov), 0.0))
    corr = cov ./ (se * se')
    return (stderr = se, corr = corr, cov = cov, eigvals = E.values, eigvecs = E.vectors,
            condition = E.values[end] / E.values[1], sigma = σ,
            names = names === nothing ? ["p$i" for i in 1:length(se)] : names)
end

const _HESTON_NAMES = ["κ", "θ", "ξ", "ρ", "v0"]
const _ROUGH_NAMES = ["κ", "θ", "ξ", "ρ", "v0", "H"]

"""
    heston_diagnostics(quotes, S, p::HestonParams; noise=1e-4)

`calibration_diagnostics` for classical Heston at parameters `p` on a quote
set (batch-priced), reported in natural parameters (κ, θ, ξ, ρ, v0).
`noise` is the per-quote IV noise (default 1 bp).
"""
function heston_diagnostics(quotes, S, p::HestonParams; noise = 1e-4)
    groups = group_quotes(quotes; S = S)
    calibration_diagnostics(x -> heston_batch_residuals(x, groups), _pack(p); noise,
                            transform = x -> (q = _unpack(x); [q.κ, q.θ, q.ξ, q.ρ, q.v0]),
                            names = _HESTON_NAMES)
end

"""
    rough_heston_diagnostics(quotes, p::RoughHestonParams; noise=1e-4, N=96, S=nothing)

`calibration_diagnostics` for rough Heston, in natural parameters
(κ, θ, ξ, ρ, v0, H) — in particular, the standard error of H a quote set
supports.
"""
function rough_heston_diagnostics(quotes, p::RoughHestonParams; noise = 1e-4, N::Int = 96,
                                  S = nothing)
    groups = group_quotes(quotes; S = S)
    calibration_diagnostics(x -> rough_heston_residuals(x, groups; N), _pack_rough(p); noise,
                            transform = x -> (q = _unpack_rough(x); [q.κ, q.θ, q.ξ, q.ρ, q.v0, q.H]),
                            names = _ROUGH_NAMES)
end
