"""
"""
mutable struct ProblemParameters{T <: AbstractFloat=Float64}
	sim::Function
	simjacobian_A::Function
	simjacobian_B::Function
	costfunc::TrajectoryCostFunction{T}
	Xref::Vector{Vector{T}}
	Uref::Vector{Vector{T}}
	xic::Vector{T}
	timestep::T
end

function ProblemParameters{T}(
	sim::Function,
	simjacobian_A::Function,
	simjacobian_B::Function,
	stage_costfunc::Function,
	terminal_costfunc::Function,
	Xref::AbstractVector{<:AbstractVector{<:Real}},
	Uref::AbstractVector{<:AbstractVector{<:Real}},
	xic::AbstractVector{<:AbstractVector{<:Real}},
	timestep::Real,
)::ProblemParameters{T} where T
	# Get problem dimensions
	nx = length(Xref[1])
	nu = length(Uref[1])
	N = length(Xref)

	costfunc = TrajectoryCostFunction{T}(stage_costfunc, terminal_costfunc, nx, nu, N)
	Xref_T = Vector{Vector{T}}(Xref)
	Uref_T = Vector{Vector{T}}(Uref)
	xic_T = Vector{T}(xic)
	timestep_T = T(timestep)
	return ProblemParameters{T}(
		sim, simjacobian_A, simjacobian_B, costfunc, Xref_T, Uref_T, xic_T, timestep_T,
	)
end


"""
"""
mutable struct Solution{T <: AbstractFloat=Float64}
	X::Vector{Vector{T}}
	U::Vector{Vector{T}}
	J::T
end

function Solution{T}(nx::Int, nu::Int, N::Int)::Solution{T} where T
	X = [zeros(T, nx) for k ∈ 1:N]
	U = [zeros(T, nu) for k ∈ 1:(N-1)]
	J = T(0.0)
	return Solution{T}(X, U, J)
end


"""
"""
mutable struct SolverCache{T <: AbstractFloat=Float64}
	fwd::ForwardCache{T}
	bwd::BackwardCache{T}
	tmp::TemporaryCache{T}
end

function SolverCache{T}(params::ProblemParameters{T})::SolverCache{T} where T
	nx = length(params.Xref[1])
	nu = length(params.Uref[1])
	N = length(params.Xref)
	fwd = ForwardCache{T}(nx, nu, N)
	bwd = BackwardCache{T}(nx, nu, N)
	tmp = TemporaryCache(nx, nu)
	return SolverCache{T}(fwd, bwd, tmp)
end


"""
"""
mutable struct SolverOptions{T <: AbstractFloat=Float64}
	regularizer::T
	max_step::T
	defect_rate::T
	stat_tol::T
	defect_tol::T
	max_iter::Int
	max_ls_iter::Int
	multishoot::Bool
	verbose::Bool
end

function SolverOptions{T}(;
	regularizer::AbstractFloat = 1e-6,
	max_step::AbstractFloat = 1.0,
	defect_rate::AbstractFloat = 1.0,
	stat_tol::AbstractFloat = 1e-9,
	defect_tol::AbstractFloat = 1e-9,
	max_iter::Int = 100,
	max_ls_iter::Int = 20,
	multishoot::Bool = false,
	verbose::Bool = true,
)::SolverOptions{T} where T
	return SolverOptions{T}(
		T(regularizer),
		T(max_step),
		T(defect_rate),
		T(stat_tol),
		T(defect_tol),
		max_iter,
		max_ls_iter,
		multishoot,
		verbose,
	)
end
