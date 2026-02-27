include("leading.jl")

# scale1 is the initial scale of the signal udes for seeking orders
# scale_rate is the rate of the scales of the signal used for seeking orders
struct IterSeek <: Method
    lm_vec::Vector{LeadingMethod}
    scale_rate::Real
    scale1::Real
end


function power_solve(f::AbstractVector{T}, grid::AbstractVector{T}, norder::Int,
                     method::IterSeek) where {T<:Real}
    r = method.scale_rate
    @assert r >= 1 "scale_rate must be >= 1"
    @assert length(method.lm_vec) >= norder "number of leading methods must be >= norder"
    len = round(Int, method.scale1)
    N = length(f)
    fcopy = copy(f)
    order_vec = T[]
    coff_vec = T[]
    note_vec = []
    @show norm(fcopy)
    for i in 1:norder
        @assert len <= N "scale for seeking leading item is too large"
        lm = method.lm_vec[i]
        order, coff, note = leading_solve(view(fcopy, (N - len + 1):N),
                                          view(grid, (N - len + 1):N), lm)
        push!(order_vec, order[1])
        push!(coff_vec, coff[1])
        push!(note_vec, note)
        fcopy .= fcopy .- coff[1] .* grid .^ (-order[1])
        len = round(Int, len * r)
        @show norm(fcopy)
    end
    return order_vec, coff_vec, note_vec
end
