using SafeTestsets

@safetestset "ContactContinuationMPC.jl" begin
    include("cartpole.jl")
end
