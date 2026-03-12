abstract type IntegralMethod end

struct Simpson <: IntegralMethod
end

function integrate(f::AbstractVector{<:Real}, h::Real, im::Simpson)
    @assert h > 0 "h must be positive"
    h = eltype(f)(h)
    N = length(f)
    @assert N>1 "number of points must be at least 2"
    if N == 2
        return (f[1] + f[2]) * h / 2
    end
    S = zero(eltype(f))
    if N % 2 == 1
        S += f[1] + f[N]
        for i in 2:2:(N - 1)
            S += 4 * f[i]
        end
        for i in 3:2:(N - 2)
            S += 2 * f[i]
        end
        return S * h / 3
    else
        res = integrate(view(f, 1:(N - 1)), h, im)
        res += (f[N] + f[N-1]) * h / 2
        return res
    end
end
