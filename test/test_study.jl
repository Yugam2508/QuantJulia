using Test, QuantJulia, Dates

# Synthetic CBOE snapshot (read_cboe shape) priced by rough Heston with known H,
# with a proportional bid-ask spread.
function synthetic_snapshot(p; S = 5000.0, r = 0.04, q = 0.012, vd = Date(2026, 7, 2),
                            days = (7, 14, 30, 60, 91, 182, 365), N = 64, spread = 0.002)
    rows = @NamedTuple{expiry::Date, K::Float64, cbid::Float64, cask::Float64, pbid::Float64, pask::Float64}[]
    for dd in days
        T = dd / 365; F = S * exp((r - q) * T); disc = exp(-r * T)
        Ks = round.(F .* exp.(0.2 * sqrt(T) .* range(-3, 2, length = 41)) ./ 5) .* 5
        cs = batch_call_prices(make_rough_cf(T, p; N), F, disc, Ks, T; iv_hint = 0.2)
        for (K, c) in zip(Ks, cs)
            pp = c - disc * (F - K)
            h(x) = max(spread * x, 0.05)
            push!(rows, (; expiry = vd + Day(dd), K, cbid = max(c - h(c), 0.0), cask = c + h(c),
                          pbid = max(pp - h(pp), 0.0), pask = pp + h(pp)))
        end
    end
    return (spot = S, quote_date = vd, rows = rows)
end

@testset "study" begin
    @testset "last_trading_day" begin
        @test last_trading_day(Date(2026, 7, 4)) == Date(2026, 7, 2)   # Sat; Fri Jul 3 observed holiday
        @test last_trading_day(Date(2026, 7, 6)) == Date(2026, 7, 6)   # Monday
        @test last_trading_day(Date(2026, 1, 1)) == Date(2025, 12, 31) # New Year's Day
    end

    @testset "invert_powerlaw_H" begin
        Hs = [0.1, 0.2, 0.3]
        @test invert_powerlaw_H(0.15, Hs, [0.0, 0.1, 0.2]) ≈ 0.25
        @test invert_powerlaw_H(0.5, Hs, [0.0, 0.1, 0.2]) === nothing     # outside the map
        @test invert_powerlaw_H(0.05, Hs, [0.0, 0.1, 0.05]) === nothing   # not monotone
    end

    @testset "power-law H: exact only in the short, narrow-window limit" begin
        # Narrow windows and T ≤ 0.008: the estimator recovers H.
        m = powerlaw_bias_map(0.3, 0.02, 0.3, -0.7, 0.02, [0.001, 0.002, 0.004, 0.008];
                              Hs = (0.3,), width = 0.1, minwin = 0.0, N = 192)
        @test m.Heff[1, 1] ≈ 0.3 atol = 0.01
        # Market window on quoted maturities: monotone in H but biased low.
        Ts = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
        m = powerlaw_bias_map(0.3, 0.02, 0.3, -0.7, 0.02, Ts; Hs = (0.1, 0.3, 0.5), N = 128)
        @test issorted(m.Heff[:, 1])
        @test all(m.Heff[:, 1] .< m.Hs .- 0.05)
        # Heston (H = 0.5) row matches a direct classical-CF skew.
        ψ = u -> heston_cf(u, 0.25, HestonParams(0.3, 0.02, 0.3, -0.7, 0.02))
        @test m.skews[3][4] ≈ model_window_skew(ψ, 0.25) rtol = 1e-10
    end

    @testset "identification_study recovers H on a synthetic snapshot" begin
        p = RoughHestonParams(1.5, 0.04, 0.4, -0.7, 0.03, 0.1)
        raw = synthetic_snapshot(p)
        r = identification_study(raw; valuation_date = raw.quote_date, N = 64, maxiter = 40)
        @test r.converged
        @test r.n_expiries == 7
        @test r.rmse < 1e-3                              # within the 0.2% spread
        @test abs(r.H_fit - 0.1) < 3 * r.se_H + 0.01
        @test r.se_H < 0.02
        # The power-law estimator on the same snapshot is far off.
        @test r.H_modelfree < 0.05
    end
end
