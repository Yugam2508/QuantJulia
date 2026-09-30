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
include("cboe.jl")
include("calibration.jl")
include("greeks.jl")
include("identification.jl")

export normal_cdf, normal_pdf, bs_price, bs_vega, bs_delta, implied_vol
export bs_gamma, bs_theta, bs_rho
export HestonParams, feller_ratio, heston_cf, price_from_cf, heston_price, batch_call_prices
export solve_fractional_riccati, frac_integral_end
export nyse_holidays, business_days, year_fraction
export market_atm_skew, model_atm_skew, skew_powerlaw_H, prepare_identification_set
export rough_joint_loss, calibrate_rough_heston_joint
export RoughHestonParams, rough_heston_cf, make_rough_cf
export ad_greeks, heston_greeks, rough_heston_greeks
export read_cboe, prepare_chain, calibrate_heston, heston_loss
export group_quotes, rough_heston_loss, calibrate_rough_heston
export heston_residuals, heston_batch_residuals, heston_batch_loss
export rough_heston_residuals, levenberg_marquardt

end # module
