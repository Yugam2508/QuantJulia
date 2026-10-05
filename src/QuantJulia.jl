module QuantJulia

# erf builds the normal CDF; gamma feeds the fractional Adams weights. Both
# carry ForwardDiff rules, which matters because Duals flow through prices —
# and through the fractional solver, where even α = H + 1/2 can be a Dual.
using SpecialFunctions: erf, gamma

include("blackscholes.jl")
include("heston.jl")
include("fourier.jl")
include("fractional_riccati.jl")
include("rough_heston.jl")
include("lifted_heston.jl")
include("cboe.jl")
include("calibration.jl")
include("diagnostics.jl")
include("greeks.jl")
include("identification.jl")
include("svi.jl")
include("montecarlo.jl")
include("heston_mc.jl")
include("rbergomi.jl")
include("localvol.jl")
include("jumps.jl")
include("cos.jl")
include("variance.jl")
include("exotics.jl")
include("marketdata.jl")

export normal_cdf, normal_pdf, bs_price, bs_vega, bs_delta, implied_vol
export bs_gamma, bs_theta, bs_rho
export HestonParams, feller_ratio, heston_cf, price_from_cf, heston_price, batch_call_prices
export solve_fractional_riccati, frac_integral_end
export nyse_holidays, business_days, year_fraction
export market_atm_skew, model_atm_skew, skew_powerlaw_H, prepare_identification_set
export rough_joint_loss, calibrate_rough_heston_joint
export RoughHestonParams, rough_heston_cf, make_rough_cf
export lifted_kernel, make_lifted_cf
export ad_greeks, heston_greeks, rough_heston_greeks
export SVIParams, svi_total_variance, svi_iv, svi_butterfly_g, svi_butterfly_free, fit_svi
export SSVIParams, ssvi_total_variance, ssvi_slice, ssvi_arbitrage_free, fit_ssvi
export atm_total_variance, calendar_violations, fit_svi_surface
export dupire_local_variance, ssvi_surface, local_vol_mc_prices
export simulate_rough_heston, rough_heston_mc_price
export simulate_heston, heston_mc_price
export RBergomiParams, simulate_rbergomi, rbergomi_mc_prices
export MertonParams, BatesParams, VGParams, merton_cf, bates_cf, vg_cf, merton_price
export calibrate_cf_model
export cf_cumulants, cos_call_prices, cos_price
export variance_swap_strike, replicate_variance, market_variance_term_structure, model_vix
export simulate_gbm_paths, simulate_heston_paths, crr_price, lsm_american_price
export FlatCurve, ZeroCurve, zero_rate, discount, forward_rate, DividendSchedule, MarketData
export pv_dividends, forward, black_price, market_price
export barrier_price, bgk_barrier, barrier_mc_price, geometric_asian_price, asian_mc_price
export read_cboe, prepare_chain, calibrate_heston, heston_loss
export group_quotes, rough_heston_loss, calibrate_rough_heston
export heston_residuals, heston_batch_residuals, heston_batch_loss
export rough_heston_residuals, levenberg_marquardt
export calibration_diagnostics, heston_diagnostics, rough_heston_diagnostics

end # module
