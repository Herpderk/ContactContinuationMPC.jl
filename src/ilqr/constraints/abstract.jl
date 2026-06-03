abstract type AbstractConstraint{T<:AbstractFloat} end
abstract type AbstractEqualityConstraint{T<:AbstractFloat} <:
              AbstractConstraint{T} end
abstract type AbstractInequalityConstraint{T<:AbstractFloat} <:
              AbstractConstraint{T} end

function update_residuals!(
    constr::AbstractConstraint{T}, inputs::Vector{Vector{T}}
) where {T}
    error("Must implement update_residuals! for $(typeof(constr))")
end

function get_stage_jacobian!(
    ∇c::Matrix{T},
    constr::AbstractConstraint{T},
    inputs::Vector{Vector{T}},
    k::Int,
) where {T}
    error("Must implement get_stage_jacobian! for $(typeof(constr))")
end

struct ConstraintSet <: Dict{Symbol,AbstractConstraint}
    dict::Dict{Symbol,AbstractConstraint}
    n_input::Int

    function ConstraintSet(n_input::Integer)
        dict = Dict{Symbol,AbstractConstraint}()
        return new(dict, n_input)
    end
end

Base.getindex(set::ConstraintSet, key::Symbol) = set.dict[key]
function Base.setindex!(
    set::ConstraintSet, value::AbstractConstraint, key::Symbol
)
    (set.dict[key] = value)
end
Base.keys(set::ConstraintSet) = keys(set.dict)
Base.values(set::ConstraintSet) = values(set.dict)
