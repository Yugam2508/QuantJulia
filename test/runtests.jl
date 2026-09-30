using QuantJulia
using Test
using ForwardDiff

@testset "QuantJulia" begin
    include("test_blackscholes.jl")
    include("test_heston.jl")
    include("test_fourier.jl")
    include("test_calibration.jl")
    include("test_rough.jl")
end
