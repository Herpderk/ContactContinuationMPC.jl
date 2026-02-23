using SafeTestsets

@safetestset "ContactContinuationMPC.jl" begin
    include("ilqr/cartpole.jl")
    include("sqp/cartpole.jl")
end
