# What does the power-law H measure? Two experiments.
#
#   A. Bias map (runs from committed results): for rough Heston at the SPX
#      calibration (results/rough_fit.csv) and at a moderate reference set,
#      price the model ATM skew on the market's 50-expiry grid for true
#      H ∈ {0.05, …, 0.5}, apply the market power-law estimator per window,
#      and read the market's H_eff back through the map.
#   B. Snapshot study (needs CBOE files in data/snapshots/): run
#      `identification_study` on every snapshot under calendar and business
#      time, without and with the skew-slope term.
#
#     julia --project=. scripts/05_identify_H.jl

using QuantJulia
using Dates, Printf

const ROOT = joinpath(@__DIR__, "..")

readkv(path) = Dict(l[1] => l[2] for l in (split(x, ',') for x in readlines(path)[2:end]))

function bias_map()
    rows = [parse.(Float64, split(l, ',')) for l in readlines(joinpath(ROOT, "results", "skew_term_structure.csv"))[2:end]]
    Ts = [r[1] for r in rows]; mkt = [r[2] for r in rows]
    windows = ["full" => T -> true, "front" => T -> T <= 0.16,
               "belly" => T -> 0.03 <= T <= 1.04, "back" => T -> T >= 0.5]
    Hmkt = [skew_powerlaw_H(Ts[w.(Ts)], mkt[w.(Ts)]).H for w in last.(windows)]
    fit = readkv(joinpath(ROOT, "results", "rough_fit.csv"))
    sets = ["spx_fit" => Tuple(parse(Float64, fit[k]) for k in ("kappa", "theta", "xi", "rho", "v0")),
            "reference" => (0.3, 0.02, 0.3, -0.7, 0.02)]
    io = open(joinpath(ROOT, "results", "powerlaw_bias.csv"), "w")
    println(io, "params,H_true,", join(first.(windows), ','))
    println(io, "market,NA,", join(round.(Hmkt; digits = 4), ','))
    for (name, (κ, θ, ξ, ρ, v0)) in sets
        m = powerlaw_bias_map(κ, θ, ξ, ρ, v0, Ts; windows)
        @printf("\n%s  (κ=%.3g θ=%.3g ξ=%.3g ρ=%.3g v0=%.3g)\n  %-6s %s\n", name, κ, θ, ξ, ρ, v0, "H",
                join([@sprintf("%8s", w) for w in m.windows]))
        for i in eachindex(m.Hs)
            @printf("  %-6.2f %s\n", m.Hs[i], join([@sprintf("%8.3f", h) for h in m.Heff[i, :]]))
            println(io, name, ',', m.Hs[i], ',', join(round.(m.Heff[i, :]; digits = 4), ','))
        end
        @printf("  %-6s %s\n  %-6s %s\n", "market", join([@sprintf("%8.3f", h) for h in Hmkt]), "→ H",
                join([(h = invert_powerlaw_H(Hmkt[j], m.Hs, m.Heff[:, j]);
                       h === nothing ? @sprintf("%8s", "none") : @sprintf("%8.3f", h)) for j in eachindex(Hmkt)]))
    end
    close(io)
    println("\nSaved → results/powerlaw_bias.csv")
end

function snapshot_study()
    dir = joinpath(ROOT, "data", "snapshots")
    files = isdir(dir) ? filter(endswith(".csv"), readdir(dir; join = true)) : String[]
    isempty(files) && return println("\nNo snapshots in data/snapshots/ — skipping experiment B.")
    out = open(joinpath(ROOT, "results", "identification_study.csv"), "w")
    println(out, "file,quote_date,valuation_date,daycount,skew_weight,H_modelfree,n_skew,H_fit,se_H,rmse,skew_rmse,n_quotes,n_expiries,converged")
    for f in files, daycount in (:calendar, :business), sw in (0.0, 1.0)
        raw = read_cboe(f)
        vd = last_trading_day(raw.quote_date)
        r = identification_study(raw; valuation_date = vd, daycount, skew_weight = sw)
        @printf("%-28s %-9s w=%.0f  H_pl=%.3f  H_fit=%.3f ± %.3f  rmse=%.1fbp\n",
                basename(f), daycount, sw, r.H_modelfree, r.H_fit, r.se_H, 1e4 * r.rmse)
        println(out, join((basename(f), raw.quote_date, vd, daycount, sw, r.H_modelfree, r.n_skew,
                           r.H_fit, r.se_H, r.rmse, r.skew_rmse, r.n_quotes, r.n_expiries, r.converged), ','))
    end
    close(out)
    println("Saved → results/identification_study.csv")
end

bias_map()
snapshot_study()
