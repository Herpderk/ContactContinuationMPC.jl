# Shortcut for throwing exceptions
throwdim(msg::AbstractString) = throw(DimensionMismatch(msg))
throwarg(msg::AbstractString) = throw(ArgumentError(msg))
throwdom(msg::AbstractString) = throw(DomainError(msg))
throwdom(val::Any, msg::AbstractString) = throw(DomainError(val, msg))
