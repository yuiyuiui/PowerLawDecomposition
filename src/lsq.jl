struct LSQ{T<:Real} <: Method
    deg1::T
    d::T
    ndeg::Int
    ntail::Int
end

function power_solve(f::AbstractVector{T}, grid::AbstractVector{T}, norder::Int,
                     method::LSQ{T}) where {T<:Real}
    @assert length(f) == length(grid)
    ntail = method.ntail
    @assert length(f) >= ntail
    @assert ntail >= 1
    deg1 = method.deg1
    d = method.d
    ndeg = method.ndeg
    deg_vec = [deg1 + i * d for i in 0:(ndeg - 1)]
    A = [f[end-ntail+i]^(-deg_vec[j]) for i in 1:ntail, j in 1:ndeg]
    deg_res = A \ f[(end - ntail + 1):end]
    @show deg_res
end
