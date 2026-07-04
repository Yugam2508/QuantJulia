module QuantJulia

# erf is used to build the standard normal CDF. It is AD-differentiable
# (ForwardDiff has a rule for it via DiffRules), which matters later when we
# push Dual numbers through prices to get Greeks/calibration gradients.
using SpecialFunctions: erf

include("blackscholes.jl")
include("heston.jl")
include("fourier.jl")

export normal_cdf, bs_price, bs_vega, bs_delta, implied_vol
export HestonParams, heston_cf, price_from_cf, heston_price

end # module
