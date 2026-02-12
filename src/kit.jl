function uniform_check(grid::AbstractVector{<:Real})
    dgrif = diff(grid)
    tol = length(grid) * eps(eltype(grid))^(2 // 3)
    @. dgrif = abs(dgrif - dgrif[1])
    maxerr = maximum(dgrif)
    maxerr > tol &&
        error("grid is not uniform, the error is $maxerr, larger than tol = $tol")
    return true
end

function grid_check(grid::AbstractVector{<:Real})
    @assert grid[1] > 0 "grid must be positive"
    @assert length(grid) > 1 "grid must have at least 2 points"
    uniform_check(grid)
    @assert grid[2] > grid[1] "grid must be increasing"
    return true
end

# get h and point number on (0, L] (N0)
function get_point_num(grid::AbstractVector{<:Real})
    h = grid[2] - grid[1]
    N0 = length(grid) + round(Int, grid[1] / h) - 1
    return h, N0
end

function int_simpson(f::AbstractVector{<:Real}, h::Real)
    @assert h > 0 "h must be positive"
    h = eltype(f)(h)
    N = length(f)
    @assert N>1 "number of points must be at least 2"
    S = 0
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
        res = int_simpson(view(f, 1:(N - 1)), h)
        res += (f[N] + f[N-1]) * h / 2
        return res
    end
end
