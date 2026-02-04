using SafeTestsets

@safetestset "MjContactImplicit.jl" begin
    include("cartpole.jl")
end
