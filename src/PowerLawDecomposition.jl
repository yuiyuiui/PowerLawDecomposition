module PowerLawDecomposition

using LinearAlgebra
using Interpolations

export power_solve
export Method, LogLog, Shanks, Wynn, WynnPola, power_solve

abstract type Method end

include("kit.jl")
include("leading.jl")

function power_solve(f::AbstractVector{T}, grid::AbstractVector{T}, norder::Int;
                     method::Method=WynnPola()) where {T<:Real}
    @assert norder == 1 "Only one order is supported for now"
    return power_solve(f, grid, method)
end

end
