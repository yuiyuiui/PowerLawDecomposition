# ====================================================
# Wynn method
# ====================================================
"""
    Wynn(; k=2, n=3)

Wynn's epsilon algorithm for extracting the leading power-law exponent and coefficient.

Generalizes the Shanks method by using `n` apparent orders (instead of 3) and applying
Wynn's epsilon acceleration. Requires `n` to be an odd integer >= 3. When `n=3`, this
reduces to the standard Shanks transformation.

The grid must be integer-spaced (uniform with integer ratio `k`).

# Fields
- `k::Int`: integer geometric sampling ratio between consecutive intervals (default: 2).
- `n::Int`: number of apparent orders to compute; must be odd and >= 3 (default: 3).

# Returns
`(order, coefficient)` — the refined exponent `a` and coefficient `c`.
"""
struct Wynn
    k::Int
    n::Int
end

function Wynn(; k::Int=2, n::Int=3)
    return Wynn(k, n)
end

function leading_solve(f::AbstractVector{T}, grid::AbstractVector{T},
                       method::Wynn) where {T<:Real}
    k = method.k
    n = method.n
    # 0. parameter check
    @assert n >= 3 && isodd(n) "Parameter n must be an odd integer >= 3 for Wynn's epsilon algorithm"
    grid_check(grid)
    N = length(f)

    # we need n A, which means we need n+1 S, and then we need n+2 nodes
    # the corresponding exponent span is k^(n+1)
    @assert grid[end] / k^(n - 1) > 1 "grid must be large enough"

    # pre-allocate arrays
    # Svec need n+1 elements
    Svec = zeros(T, n + 1)
    # Avec, Cvec need n elements
    Avec = zeros(T, n)
    Cvec = zeros(T, n)
    # nvec need n+2 node indices
    nvec = zeros(Int, n + 2)

    h, N0 = get_point_num(grid)
    idx_shift = N0 - N

    # check if the grid is large enough to contain the span of k^(n+1)
    @assert floor(Int, grid[end] / k^(n + 1)) >= grid[1] "grid must be large enough for n=$n steps"

    # 1. get sampling indices (Geometric Sampling)
    # nvec[1] is set to the position where the farthest end is pushed back, ensuring that nvec[end] does not exceed the boundary
    nvec[1] = floor(Int, N0 / k^(n + 1))
    for i in 2:(n + 2)
        nvec[i] = nvec[i - 1] * k
    end

    # 2. calculate interval integrals (Interval Integrals)
    # we need n+1 integral values
    for i in 1:(n + 1)
        # the integral interval is from nvec[i] to nvec[i+1]
        idx_start = nvec[i] - idx_shift
        idx_end = nvec[i + 1] - idx_shift
        Svec[i] = int_simpson(view(f, idx_start:idx_end), h)
    end

    # 3. calculate apparent order sequences A and coefficient sequences C
    for i in 1:n
        # Svec[i] corresponds to the interval [nvec[i], nvec[i+1]]
        # Svec[i+1] corresponds to the interval [nvec[i+1], nvec[i+2]]

        # the corresponding physical coordinates (for calculating C)
        # we use the coordinates of the "next" interval Svec[i+1], because it is closer to the far field
        x_low = nvec[i + 1] * h
        x_up = nvec[i + 2] * h

        # calculate local apparent order A_i
        val_ratio = Svec[i + 1] / Svec[i]
        # prevent log(0) or negative number (although it should not happen at the power law tail)
        if val_ratio <= 0
            error("Negative or zero ratio in sequence, cannot take log.")
        end
        Avec[i] = 1 - log(val_ratio) / log(k)

        # inverse to find the local coefficient C_i
        # Formula: c = S * (1-a) / (x_up^(1-a) - x_low^(1-a))
        a_tmp = Avec[i]
        denom = x_up^(1 - a_tmp) - x_low^(1 - a_tmp)
        if abs(denom) < 100 * eps(T)
            Cvec[i] = zero(T) # numerical protection
        else
            Cvec[i] = Svec[i + 1] * (1 - a_tmp) / denom
        end
    end

    # 4. perform Wynn's Epsilon Algorithm
    # accelerate the A and C sequences respectively
    a_refined = wynn_epsilon_core(Avec)
    c_refined = wynn_epsilon_core(Cvec)

    return a_refined, c_refined
end

"""
    wynn_epsilon_core(S)

The core implementation of Wynn's Epsilon algorithm.
Input a sequence S of length n (odd), output the accelerated limit value.
"""
function wynn_epsilon_core(S::AbstractVector{T}) where {T<:Real}
    n = length(S)

    # eps table only needs two columns to iterate, for clarity here we use a one-dimensional array to iterate and update
    # this is a in-place update technique (similar to the one-dimensional array generation of Pascal triangle)
    # e_prev corresponds to epsilon_{k-2}, e_curr corresponds to epsilon_{k}

    # initialization:
    # e_curr (epsilon_0) is the input S
    # e_prev (epsilon_-1) is all 0

    e_curr = copy(S)
    e_prev = zeros(T, n)

    # we need to iterate n-1 times (corresponding to columns 1 to n-1)
    # the final result is in e_curr[1] (shrink to the pyramid tip)
    # but the structure of Wynn's algorithm is:
    # Col 0 (S): n items
    # Col 1 (1/dS): n-1 items
    # Col 2 (Shanks): n-2 items
    # ...
    # Col n-1: 1 item

    # we use `width` to represent the length of the current column
    for col in 1:(n - 1)
        width = n - col

        # temporarily store the next round of e_prev (i.e. the current e_curr)
        # because e_curr[i] update needs to use the old e_curr[i] and e_curr[i+1]
        e_next_prev = copy(e_curr)

        for i in 1:width
            denom = e_curr[i + 1] - e_curr[i]
            if abs(denom) < 100 * eps(T)
                e_curr[i] = e_prev[i + 1] +
                            1 * denom / (eps(T)^2 + denom^2)
            else
                e_curr[i] = e_prev[i + 1] + 1 / denom
            end
        end

        # update e_prev to the e_curr before the start of this round (for the k-2 item in the next round)
        e_prev = e_next_prev
    end

    # the result is in the first element of e_curr
    # note: in Wynn's algorithm, even columns (col=2, 4...) are approximate values, odd columns are auxiliary values.
    # when n is odd, the last column col = n-1 is even (e.g. n=3, col=2), which is exactly what we need.
    return e_curr[1]
end

#
# ====================================================
# Wynn Pola method
# ====================================================
"""
    WynnPola(; k=1.3, n=21, interp_type=BSpline(Cubic(Line(OnGrid()))), points_per_interval=101)

Interpolation-based Wynn's epsilon method for extracting the leading power-law exponent and coefficient.

Unlike `Wynn`, this method supports **non-integer** geometric sampling ratios by first
interpolating the input data with a high-order spline, then evaluating the interpolant
on geometrically-spaced sub-intervals. Each sub-interval is integrated via Simpson's rule
with `points_per_interval` sample points.

# Fields
- `k::Real`: geometric sampling ratio, can be non-integer (default: 1.3, recommended 1.05–1.2).
- `n::Int`: number of apparent orders to compute; must be odd and >= 3 (default: 21).
- `interp_type::InterpolationType`: interpolation scheme for the input data (default: cubic B-spline).
- `points_per_interval::Int`: number of quadrature points per sub-interval (default: 101, should be odd).
- `use_a_final::Bool`: whether to use the final apparent order `a_final` to calculate the coefficient `c` (default: false).
- `nc::Int`: `n` used for solving the coefficient `c` (default: 5).

# Returns
`(order, coefficient)` — the refined exponent `a` and coefficient `c`.
"""
struct WynnPola
    k::Real
    n::Int
    interp_type::Interpolations.InterpolationType
    points_per_interval::Int
    use_a_final::Bool
    nc::Int
end

function WynnPola(; k::Real=1.3, n::Int=21, interp_type=BSpline(Cubic(Line(OnGrid()))),
                  points_per_interval::Int=101, use_a_final::Bool=false, nc::Int=5)
    return WynnPola(k, n, interp_type, points_per_interval, use_a_final, nc)
end

function leading_solve(f::AbstractVector{T}, grid::AbstractVector{T},
                       method::WynnPola) where {T<:Real}
    k = method.k
    n = method.n
    interp_type = method.interp_type
    points_per_interval = method.points_per_interval

    # 0. basic checks
    @assert n >= 3 && isodd(n) "Parameter n must be an odd integer >= 3"
    @assert isodd(method.nc) && (method.nc > 0) "nc must be an odd integer (e.g., 3, 5, 7)"
    method.use_a_final && @assert method.nc < method.n "nc must be less than n"

    grid_check(grid)
    h, _ = get_point_num(grid)
    L_start, L_end = grid[1], grid[end]

    # 1. interpolate the original data to get a continuous function f_itp(x)
    # scale the interpolation object to the physical coordinate grid
    # scale() requires AbstractRange, not Vector
    itp_base = interpolate(f, interp_type)
    grid_range = range(L_start, L_end; length=length(grid))
    f_itp = scale(itp_base, grid_range)

    # 2. generate geometric sampling nodes (from the far end backward)
    nodes = zeros(T, n + 2)
    nodes[end] = L_end
    for i in (n + 1):-1:1
        nodes[i] = nodes[i + 1] / k
    end
    @assert nodes[1] >= L_start "Grid too short for k=$k, n=$n. Needs start <= $(nodes[1])"

    # 3. calculate interval integrals Svec
    # we uniformly sample M points in each interval [nodes[i], nodes[i+1]], then call your int_simpson
    Svec = zeros(T, n + 1)
    M = points_per_interval
    for i in 1:(n + 1)
        x_a, x_b = nodes[i], nodes[i + 1]
        local_h = (x_b - x_a) / (M - 1)

        # generate sampling points in the interval
        local_x = range(x_a, x_b; length=M)
        local_f = [f_itp(tx) for tx in local_x]

        # call your implemented int_simpson function
        Svec[i] = int_simpson(local_f, local_h)
    end

    # 4. calculate apparent order sequences A and coefficient sequences C
    Avec = zeros(T, n)

    for i in 1:n
        ratio = Svec[i + 1] / Svec[i]
        # calculate apparent order
        Avec[i] = 1 - log(ratio) / log(k)
    end

    # 5. perform Wynn acceleration
    # here we call the shared function wynn_epsilon_core discussed earlier
    a_final = wynn_epsilon_core(Avec)

    nc = method.nc
    n1 = method.use_a_final ? nc : n
    Cvec = zeros(T, n1)

    for i in (n - n1 + 1):n
        x_low, x_up = nodes[i + 1], nodes[i + 2]
        a_tmp = method.use_a_final ? a_final : Avec[i]
        denom = abs(a_tmp - 1) < eps(T) * 1000 ? log(x_up / x_low) :
                (x_up^(1 - a_tmp) - x_low^(1 - a_tmp)) / (1 - a_tmp)
        Cvec[i + n1 - n] = Svec[i + 1] / denom
    end
    c_final = wynn_epsilon_core(Cvec)
    return a_final, c_final
end

# ====================================================
# IterSeek method with Wynn
# ====================================================
# scale1 is the initial scale of the signal udes for seeking orders
# scale_rate is the rate of the scales of the signal used for seeking orders
struct IterSeek
    wynn_pola_vec::Vector{WynnPola}
    scale_rate::Real
    scale1::Real
    norder::Int
end

function power_solve(f::AbstractVector{T}, grid::AbstractVector{T},
                     method::IterSeek) where {T<:Real}
    r = method.scale_rate
    order = method.norder
    @assert r >= 1 "scale_rate must be >= 1"
    @assert length(method.wynn_pola_vec) >= order "number of leading methods must be >= norder"
    len = round(Int, method.scale1)
    N = length(f)
    fcopy = copy(f)
    order_vec = T[]
    coff_vec = T[]
    note_vec = []
    @show norm(fcopy)
    for i in 1:order
        @assert len <= N "scale for seeking leading item is too large"
        wynn_pola = method.wynn_pola_vec[i]
        order, coff = leading_solve(view(fcopy, (N - len + 1):N),
                                    view(grid, (N - len + 1):N), wynn_pola)
        push!(order_vec, order)
        push!(coff_vec, coff)
        fcopy .= fcopy .- coff .* grid .^ (-order)
        len = round(Int, len * r)
        @show norm(fcopy)
    end
    return order_vec, coff_vec
end
