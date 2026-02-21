include("leading.jl")

struct IterSeek <: Method
    lm::LeadingMethod
end

function power_solve(f::AbstractVector{T}, grid::AbstractVector{T}, norder::Int,
                     method::IterSeek) where {T<:Real}
end
