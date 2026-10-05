# =============================================================================
# Lifted Heston — a Markovian multi-factor approximation of rough Heston
# =============================================================================
# Abi Jaber (2019), "Lifting the Heston model". The rough kernel
# K(t) = t^{α−1}/Γ(α), α = H + 1/2, is a completely monotone function, i.e. a
# mixture of exponentials. Truncating that mixture to n terms,
#     K(t) ≈ Σᵢ cᵢ e^{−xᵢ t},
# turns rough Heston into an n-factor MARKOVIAN model — classical Heston is
# the n = 1, x = 0, c = 1 case. Two payoffs:
#   * the characteristic function solves n ORDINARY Riccati ODEs, cost O(nN)
#     instead of the fractional solver's O(N²);
#   * as n → ∞ it converges to rough Heston, giving an independent check on
#     the fractional solver from a completely different discretization.
#
# Weights and mean-reversions: K(t) = ∫₀^∞ e^{−γt} μ(dγ) with
# μ(dγ) = γ^{−α} dγ / (Γ(α)Γ(1−α)). Partition γ-space geometrically,
# η₀ = 0 < η₁ < … < ηₙ with ηᵢ = rₙ^{i − n/2}, and on each cell keep the mass
# and the mean rate:
#   cᵢ = ∫ μ = (ηᵢ^{1−α} − ηᵢ₋₁^{1−α}) / (Γ(α)Γ(2−α))
#   xᵢ = ∫ γμ / cᵢ = (1−α)/(2−α) · (ηᵢ^{2−α} − ηᵢ₋₁^{2−α}) / (ηᵢ^{1−α} − ηᵢ₋₁^{1−α})
# Starting the first cell at η₀ = 0 keeps ALL the low-rate mass, which
# matters as H → 1/2 (α → 1), where μ piles up near γ = 0. The mass above ηₙ
# (the kernel's t → 0 singularity) is what n → ∞ recovers.

# Characteristic function (same convention as src/heston.jl):
#   log ψ(u) = ∫₀ᵀ F(u, φ(s)) g₀(T − s) ds,   φ = Σ cᵢψᵢ,
#   ψᵢ' = −xᵢψᵢ + F(u, φ),  ψᵢ(0) = 0,
#   F(u, v) = −(u² + iu)/2 + (iuρξ − κ)v + (ξ²/2)v²,
#   g₀(t) = v0 + κθ Σ cᵢ(1 − e^{−xᵢt})/xᵢ.
# (n = 1, x = 0, c = 1 integrates by parts to the classical exp(v0·φ(T) +
# κθ∫φ) — the same quadratic F as both other Heston solvers.)
#
# Time stepping: exponential integrator for the −xᵢψᵢ term (the xᵢ span many
# orders of magnitude) with F treated by the trapezoid rule IMPLICITLY. As in
# the fractional solver, F is quadratic, so each step is one scalar complex
# quadratic in φ, solved in closed form (root nearest the previous φ).
# Second order in Δ, unconditionally stable in the linear part.
# =============================================================================

"""
    lifted_kernel(H, n; rn=10^(8/n))

Weights `c` and mean-reversion speeds `x` of the n-term exponential
approximation Σ cᵢ e^{−xᵢt} to the rough kernel t^{H−1/2}/Γ(H+1/2)
(Abi Jaber's geometric partition with ratio `rn`). Requires 0 < H < 1/2.
"""
function lifted_kernel(H, n::Int; rn = 10.0^(8 / n))
    0 < H < 0.5 || throw(DomainError(H, "lifted_kernel needs 0 < H < 1/2"))
    α = H + one(H) / 2
    η = [rn^(i - n / 2) for i in 1:n]
    # η₀ = 0 terms written out as zero: 0^p has a NaN p-derivative under AD.
    p1(e) = e^(1 - α)
    p2(e) = e^(2 - α)
    d1 = [i == 1 ? p1(η[1]) : p1(η[i]) - p1(η[i-1]) for i in 1:n]
    d2 = [i == 1 ? p2(η[1]) : p2(η[i]) - p2(η[i-1]) for i in 1:n]
    c = d1 ./ (gamma(α) * gamma(2 - α))
    x = (1 - α) / (2 - α) .* d2 ./ d1
    return (c = c, x = x)
end

"""
    make_lifted_cf(T, p::RoughHestonParams; n=20, N=200, rn=10^(8/n))

Characteristic function `u -> ψ(u)` of the n-factor lifted Heston model that
approximates rough Heston with parameters `p` (H from `p.H`), on `N` time
steps. Same convention as `heston_cf` / `make_rough_cf`, so it plugs into
every pricer. ForwardDiff-safe in all parameters, including H.
"""
function make_lifted_cf(T, p::RoughHestonParams; n::Int = 20, N::Int = 200,
                        rn = 10.0^(8 / n))
    c, x = lifted_kernel(p.H, n; rn)
    return _lifted_cf(T, p.κ, p.θ, p.ξ, p.ρ, p.v0, c, x, N)
end

# ∫₀ᵗ e^{−xs} ds, with its x → 0 limit (the classical, x = 0 factor).
_expint(x, t) = x == 0 ? t * one(x) : (1 - exp(-x * t)) / x

# CF for an arbitrary exponential kernel Σ cᵢe^{−xᵢt}.
function _lifted_cf(T, κ, θ, ξ, ρ, v0, c, x, N::Int)
    Δ = T / N
    E = exp.(-x .* Δ)
    W = _expint.(x, Δ)
    B = sum(c .* W)
    g0(t) = v0 + κ * θ * sum(c .* _expint.(x, t))
    gs = [g0(T - k * Δ) for k in 0:N]               # g₀(T − s_k)
    c2 = ξ^2 / 2
    return function (u)
        c0 = -(u^2 + im * u) / 2
        c1 = im * u * ρ * ξ - κ
        Fv(v) = c0 + (c1 + c2 * v) * v
        ψs = zeros(typeof(c0 * B * c1), length(c))
        φ = zero(eltype(ψs))
        Fk = Fv(φ)
        acc = Fk * gs[1] / 2                        # trapezoid for ∫ F g₀
        for k in 1:N
            A = sum(c .* E .* ψs) + B * Fk / 2
            # φ = A + (B/2)·F(φ):  (B c2/2)φ² + (B c1/2 − 1)φ + (A + B c0/2) = 0
            qa = B * c2 / 2
            qb = B * c1 / 2 - 1
            qc = A + B * c0 / 2
            sq = sqrt(qb^2 - 4 * qa * qc)
            s = real(conj(qb) * sq) ≥ 0 ? 1 : -1
            qq = -(qb + s * sq) / 2
            r1 = qq / qa
            r2 = qc / qq
            φn = abs2(r1 - φ) < abs2(r2 - φ) ? r1 : r2
            Fn = Fv(φn)
            @. ψs = E * ψs + W * (Fk + Fn) / 2
            φ = φn
            Fk = Fn
            acc += (k == N ? Fk / 2 : Fk) * gs[k+1]
        end
        return exp(acc * Δ)
    end
end
