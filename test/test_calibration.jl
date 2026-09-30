using Dates

@testset "cboe parser + chain prep — synthetic file round trip" begin
    # Build a CBOE-format file from Black-Scholes prices with KNOWN
    # (S, F, r, q, σ), then check the pipeline recovers all of them.
    S, r, q, σ = 100.0, 0.03, 0.01, 0.2
    vd = Date(2026, 7, 4)
    expiry = Date(2026, 10, 2)                     # Fri, 90 days out
    T = 90 / 365
    F = S * exp((r - q) * T)

    row(K) = begin
        c = bs_price(S, K, r, q, σ, T; call = true)
        p = bs_price(S, K, r, q, σ, T; call = false)
        "Fri Oct 02 2026,SYN,0,0,$(c-0.05),$(c+0.05),0,0,0,0,0,$K,SYN,0,0,$(p-0.05),$(p+0.05),0,0,0,0,0"
    end
    lines = [
        "",
        "SYNTH INDEX,Last: $S,Change: 0.00",
        "\"Date: July 4, 2026 at 12:23 AM EDT\",Bid: 99,Ask: 101,Size: 1*1,Volume: 0",
        "Expiration Date,Calls,Last Sale,Net,Bid,Ask,Volume,IV,Delta,Gamma,Open Interest,Strike,Puts,Last Sale,Net,Bid,Ask,Volume,IV,Delta,Gamma,Open Interest",
        # already-expired expiry → must be dropped
        "Thu Jul 02 2026,SYN,0,0,10.0,11.0,0,0,0,0,0,100.00,SYN,0,0,10.0,11.0,0,0,0,0,0",
        [row(K) for K in (90.0, 95.0, 100.0, 105.0, 110.0)]...,
        # zero-bid OTM call → must be rejected, and excluded from the forward fit
        "Fri Oct 02 2026,SYN,0,0,0,0.05,0,0,0,0,0,140.00,SYN,0,0,39.0,41.0,0,0,0,0,0",
    ]
    path = joinpath(mktempdir(), "synth_cboe.csv")
    write(path, join(lines, "\n"))

    raw = read_cboe(path)
    @test raw.spot == S
    @test raw.quote_date == vd
    @test length(raw.rows) == 7

    chain = prepare_chain(raw; valuation_date = vd)
    @test chain.rejects[:expired] == 1
    @test chain.rejects[:zero_bid] == 1
    @test length(chain.expiries) == 1
    e = chain.expiries[1]
    @test e.F ≈ F rtol = 1e-3                      # parity regression recovers F
    @test e.r ≈ r atol = 1e-3                      # ... and the discount rate
    @test e.q ≈ q atol = 1e-3                      # ... and the implied carry
    @test length(chain.quotes) == 5
    for qt in chain.quotes
        @test qt.side === (qt.K >= F ? :call : :put)   # OTM side selection
        @test qt.iv ≈ σ atol = 1e-3                    # inverter recovers σ
    end
end

@testset "calibrate_heston — recovers known parameters from synthetic surface" begin
    S, r, qy = 100.0, 0.02, 0.01
    truth = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    quotes = @NamedTuple{T::Float64, K::Float64, r::Float64, q::Float64,
                         side::Symbol, mid::Float64, iv::Float64, vega::Float64}[]
    for T in (0.1, 0.5, 1.5), K in (85.0, 92.5, 100.0, 107.5, 115.0)
        F = S * exp((r - qy) * T)
        side = K >= F ? :call : :put
        mid = heston_price(S, K, r, qy, truth, T; call = side === :call)
        iv = implied_vol(mid, S, K, r, qy, T; call = side === :call)
        vega = bs_vega(S, K, r, qy, iv, T)
        push!(quotes, (; T, K, r, q = qy, side, mid, iv, vega))
    end

    res = calibrate_heston(quotes, S; p0 = HestonParams(1.2, 0.06, 0.8, -0.4, 0.06))
    @test res.rmse < 1e-4                          # sub-basis-point fit
    @test res.params.v0 ≈ truth.v0 rtol = 5e-2
    @test res.params.ρ ≈ truth.ρ rtol = 5e-2
    @test res.params.θ ≈ truth.θ rtol = 1e-1
    @test res.params.ξ ≈ truth.ξ rtol = 1e-1
    @test res.params.κ ≈ truth.κ rtol = 2e-1       # κ is the sloppy direction
end

# Synthetic classical-Heston quote set shared by the optimizer/pricer tests.
function _synthetic_heston_quotes(S, r, qy, truth)
    quotes = @NamedTuple{T::Float64, K::Float64, r::Float64, q::Float64,
                         side::Symbol, mid::Float64, iv::Float64, vega::Float64}[]
    for T in (0.1, 0.5, 1.5), K in (85.0, 92.5, 100.0, 107.5, 115.0)
        F = S * exp((r - qy) * T)
        side = K >= F ? :call : :put
        mid = heston_price(S, K, r, qy, truth, T; call = side === :call)
        iv = implied_vol(mid, S, K, r, qy, T; call = side === :call)
        push!(quotes, (; T, K, r, q = qy, side, mid, iv, vega = bs_vega(S, K, r, qy, iv, T)))
    end
    return quotes
end

@testset "levenberg_marquardt — Rosenbrock and linear least squares" begin
    rosen(x) = [10 * (x[2] - x[1]^2), 1 - x[1]]
    res = levenberg_marquardt(rosen, [-1.2, 1.0]; maxiter = 200)
    @test res.converged
    @test res.minimizer ≈ [1.0, 1.0] atol = 1e-6
    @test res.minimum < 1e-12
    # Linear residuals: LM must land on the normal-equations solution.
    A = [1.0 2.0; 3.0 4.0; 5.0 7.0]; b = [1.0, -1.0, 2.0]
    lin = levenberg_marquardt(x -> A * x - b, zeros(2))
    @test lin.minimizer ≈ A \ b rtol = 1e-8
end

@testset "group_quotes — forward from S when quotes lack F" begin
    S, r, qy = 100.0, 0.02, 0.01
    quotes = _synthetic_heston_quotes(S, r, qy, HestonParams(2.0, 0.04, 0.5, -0.7, 0.04))
    @test_throws ArgumentError group_quotes(quotes)
    gs = group_quotes(quotes; S = S)
    @test length(gs) == 3
    for g in gs
        @test g.F ≈ S * exp((r - qy) * g.T)
        @test length(g.Ks) == 5
    end
end

@testset "batch vs adaptive residuals agree (classical Heston)" begin
    S, r, qy = 100.0, 0.02, 0.01
    quotes = _synthetic_heston_quotes(S, r, qy, HestonParams(2.0, 0.04, 0.5, -0.7, 0.04))
    x = QuantJulia._pack(HestonParams(1.5, 0.05, 0.6, -0.5, 0.03))
    # group order = sorted T, quotes built in sorted-T order with K inner
    ra = heston_residuals(x, quotes, S)
    rb = heston_batch_residuals(x, group_quotes(quotes; S = S))
    # Residuals are IV errors: agree to 0.5 bp (the worst case is the
    # short-dated far-OTM quote, where tiny vega magnifies the batch pricer's
    # ~1e-7 absolute price error).
    @test maximum(abs, rb .- ra) < 5e-5
    @test heston_batch_loss(x, group_quotes(quotes; S = S)) ≈
          heston_loss(x, quotes, S) rtol = 1e-3
end

@testset "calibrate_heston — LM and batch pricer recover known parameters" begin
    S, r, qy = 100.0, 0.02, 0.01
    truth = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    quotes = _synthetic_heston_quotes(S, r, qy, truth)
    p0 = HestonParams(1.2, 0.06, 0.8, -0.4, 0.06)
    for (method, pricer) in ((:lm, :adaptive), (:lm, :batch), (:lbfgs, :batch))
        res = calibrate_heston(quotes, S; p0, method, pricer)
        @test res.rmse < 1e-4
        @test res.params.v0 ≈ truth.v0 rtol = 5e-2
        @test res.params.ρ ≈ truth.ρ rtol = 5e-2
        @test res.params.θ ≈ truth.θ rtol = 1e-1
    end
    lm = calibrate_heston(quotes, S; p0, method = :lm)
    @test lm.converged
    @test lm.iterations < 60                        # LBFGS needs several times more
    @test_throws ArgumentError calibrate_heston(quotes, S; method = :newton)
    @test_throws ArgumentError calibrate_heston(quotes, S; pricer = :fft)
end

@testset "calibrate_rough_heston — LM recovers a synthetic rough surface" begin
    S, r, qy = 100.0, 0.02, 0.0
    truth = RoughHestonParams(1.5, 0.04, 0.4, -0.7, 0.03, 0.15)
    N = 32
    quotes = @NamedTuple{T::Float64, K::Float64, F::Float64, r::Float64, q::Float64,
                         side::Symbol, mid::Float64, iv::Float64, vega::Float64}[]
    for T in (0.05, 0.25, 1.0)
        F = S * exp((r - qy) * T); disc = exp(-r * T)
        Ks = F .* exp.(0.2 * sqrt(T) .* (-1.0, -0.5, 0.0, 0.5, 1.0))
        cs = batch_call_prices(make_rough_cf(T, truth; N), F, disc, Ks, T; iv_hint = 0.2)
        for (K, c) in zip(Ks, cs)
            side = K >= F ? :call : :put
            mid = side === :call ? c : c - disc * (F - K)
            iv = implied_vol(mid, S, K, r, qy, T; call = side === :call)
            push!(quotes, (; T, K, F, r, q = qy, side, mid, iv, vega = bs_vega(S, K, r, qy, iv, T)))
        end
    end
    p0 = RoughHestonParams(1.5, 0.05, 0.5, -0.6, 0.035, 0.25)
    res = calibrate_rough_heston(quotes; p0, method = :lm, maxiter = 40, N)
    @test res.rmse < 2e-4                           # ≲ 2 bps on the synthetic set
    @test res.params.v0 ≈ truth.v0 rtol = 0.1
    @test res.params.ρ ≈ truth.ρ rtol = 0.1
    @test res.params.H ≈ truth.H atol = 0.05
end
