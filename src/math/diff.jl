const fd_coff_central = [1 // 280, -4 // 105, 1 // 5, -4 // 5, 0, 4 // 5, -1 // 5,
                         4 // 105, -1 // 280]
const fd_coff_forward = [-761 // 280, 8, -14, 56 // 3, -35 // 2, 56 // 5, -14 // 3, 8 // 7,
                         -1 // 8]

function fd8!(u::AbstractVector{<:Number}, v::AbstractVector{<:Number},
                  h::Real)
    n = length(v)
    @assert n >= 12
    @assert length(u) == n
    @inbounds for i in 1:4
        u[i] = dot(fd_coff_forward, view(v, i:(i + 8))) / h
    end
    @inbounds for i in 5:(n - 4)
        u[i] = dot(fd_coff_central, view(v, (i - 4):(i + 4))) / h
    end
    @inbounds for i in (n - 3):n
        u[i] = -dot(fd_coff_forward, view(v, i:-1:(i - 8))) / h
    end
    return u
end

function fd8(v::AbstractVector{<:Number}, h::Real)
    return fd8!(zeros(eltype(v), length(v)), v, h)
end