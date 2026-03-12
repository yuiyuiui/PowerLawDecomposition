abstract type PolaMethod end

struct LagrangePola <: PolaMethod
    hw::Int
end

function LagrangePola(; hw::Int=10)
    @assert hw >= 3 "hw must be at least 3"
    return LagrangePola(hw)
end

function interpolate(grid::AbstractVector{T}, f::AbstractVector{T}, x::T,
                     pm::LagrangePola) where {T}
    hw = pm.hw
    @assert length(grid) == length(f) "grid and f must have the same length"
    N = length(grid)
    h = grid[2] - grid[1]
    L0 = grid[1]

    @assert x >= L0 - h / 2 && x <= grid[end] + h / 2 "x=$x out of grid range [$(L0), $(grid[end])]"
    @assert 2 * hw + 1 <= N "hw=$hw too large for grid of length $N"

    idx = clamp(round(Int, (x - L0) / h) + 1, 1, N)

    if abs(x - grid[idx]) < h * max(T(1e-14), 128 * eps(T))
        return f[idx]
    end

    rng = _find_support(N, idx, hw)
    return _lagrange_eval(x, view(grid, rng), view(f, rng))
end

function _find_support(N::Int, idx_center::Int, half_width::Int)
    p = 2 * half_width
    i_lo = max(1, idx_center - half_width)
    i_hi = i_lo + p
    if i_hi > N
        i_hi = N
        i_lo = max(1, i_hi - p)
    end
    return i_lo:i_hi
end

function _lagrange_eval(x::T, xsup::AbstractVector{T}, fsup::AbstractVector{T}) where {T}
    n = length(xsup)
    result = zero(T)
    for j in 1:n
        Lj = one(T)
        for k in 1:n
            k == j && continue
            Lj *= (x - xsup[k]) / (xsup[j] - xsup[k])
        end
        result += fsup[j] * Lj
    end
    return result
end

# ================================================================
#  B-spline interpolation wrapper
# ================================================================

struct BsplinePola <: PolaMethod
end

function build_bspline_itp(grid::AbstractVector{T}, f::AbstractVector{T}) where {T}
    itp_base = Interpolations.interpolate(f, BSpline(Cubic(Line(OnGrid()))))
    grid_range = range(grid[1], grid[end]; length=length(grid))
    return Interpolations.scale(itp_base, grid_range)
end

"""
    bspline_interp(grid, f, x)

B-spline interpolation. precision is O(h⁴).
"""
function bspline_interp(grid::AbstractVector{T}, f::AbstractVector{T}, x::T,
                        pm::BsplinePola) where {T}
    itp = build_bspline_itp(grid, f)
    return itp(x)
end
