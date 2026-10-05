using QuantJulia
using Test
using ForwardDiff

@testset "QuantJulia" begin
    include("test_blackscholes.jl")
    include("test_heston.jl")
    include("test_fourier.jl")
    include("test_calibration.jl")
    include("test_diagnostics.jl")
    include("test_rough.jl")
    include("test_lifted.jl")
    include("test_greeks.jl")
    include("test_identification.jl")
    include("test_svi.jl")
    include("test_localvol.jl")
    include("test_montecarlo.jl")
    include("test_heston_mc.jl")
    include("test_rbergomi.jl")
    include("test_jumps.jl")
    include("test_cos.jl")
    include("test_variance.jl")
    include("test_quantlib.jl")
    include("test_marketdata.jl")
    include("test_exotics.jl")
end
