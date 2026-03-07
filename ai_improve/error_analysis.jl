using Random, LinearAlgebra
using PowerLawDecomposition
using Interpolations
import PowerLawDecomposition: int_simpson, grid_check

# ===== Common setup (same RNG as user's test) =====
Random.seed!(666)
T = BigFloat
nord = 10
L = 2^14
d = T(0.5)

a_vec = T[]
for i in 1:nord
    if i == 1
        push!(a_vec, (1 + rand(T)) / 2)
    else
        push!(a_vec, a_vec[end] + (1 + rand(T)) * d)
    end
end
c_vec = T[]
for i in 1:nord
    push!(c_vec, 1 + rand(T))
end
_a = T.(a_vec)
_c = T.(c_vec)

# ===== ASP core loop on pre-computed Svec =====
function asp_core(Svec0::Vector{T}, k::T, n0::Int, norder::Int) where {T<:Real}
    Svec = copy(Svec0)
    nS = length(Svec)
    wynn_n = n0
    n = n0
    order_vec = T[]
    Avec = zeros(T, wynn_n)
    ratio_vec = zeros(T, wynn_n)

    for i in 1:norder
        n <= 1 && break
        for j in 1:n
            ratio_vec[end - j + 1] = Svec[end - j + 1] / Svec[end - j]
        end
        sign_flag = 0
        while true
            n < 3 && break
            vr = view(ratio_vec, (wynn_n - n + 1):wynn_n)
            if all(vr .> 0)
                sign_flag = 1; break
            elseif all(vr .< 0)
                sign_flag = -1; break
            else
                n -= 2
            end
        end
        n < 3 && break
        vA = view(Avec, (wynn_n - n + 1):wynn_n)
        vr = view(ratio_vec, (wynn_n - n + 1):wynn_n)
        vA .= 1 .- log.(vr .* sign_flag) ./ log(k)
        a_final = wynn_epsilon_core(vA)
        push!(order_vec, a_final)
        i == norder && break
        λ = k^(1 - a_final)
        nS -= 1
        for j in 1:nS
            Svec[end + 1 - j] = Svec[end + 1 - j] - λ * Svec[end - j]
        end
        !(nS >= n + 1) && (n -= 2)
    end
    return order_vec
end

# Analytical integral: ∫_a^b Σ c_i x^{-α_i} dx
function analytical_S(a::T, b::T, cv, av) where T
    s = zero(T)
    for (c, α) in zip(cv, av)
        s += c * (b^(1 - α) - a^(1 - α)) / (1 - α)
    end
    return s
end

# Direct grid integration of f_data over [a, b], using grid data for interior
# and B-spline interpolation only at 2 boundary points
function direct_grid_integral(f_data, grid, a::T, b::T, f_itp, h::T) where T
    L0 = grid[1]

    # grid indices strictly inside [a, b]
    idx_a = clamp(ceil(Int, (a - L0) / h) + 1, 1, length(grid))
    idx_b = clamp(floor(Int, (b - L0) / h) + 1, 1, length(grid))

    if grid[idx_a] < a - h / 1000
        idx_a += 1
    end
    if grid[idx_b] > b + h / 1000
        idx_b -= 1
    end

    result = zero(T)
    n_pts = idx_b - idx_a + 1

    if n_pts >= 3
        result = int_simpson(view(f_data, idx_a:idx_b), h)
    elseif n_pts == 2
        result = (f_data[idx_a] + f_data[idx_b]) * h / 2
    end

    # left boundary correction: [a, grid[idx_a]]
    gap_l = grid[idx_a] - a
    if gap_l > h * T(1e-6) && idx_a > 1
        f_a = f_itp(a)
        result += gap_l * (f_a + f_data[idx_a]) / 2
    end

    # right boundary correction: [grid[idx_b], b]
    gap_r = b - grid[idx_b]
    if gap_r > h * T(1e-6) && idx_b < length(grid)
        f_b = f_itp(b)
        result += gap_r * (f_data[idx_b] + f_b) / 2
    end

    return result
end

# ============================================================
println("=" ^ 65)
println("  Part 1: Algorithm ceiling — analytical S values")
println("=" ^ 65)

k = T(1.3)
n_wynn = 21
nseek = 10
nS_val = n_wynn + ceil(Int, (nseek - 1) / 2) * 2 + 1
L_end = T(L)

nodes_ana = zeros(T, nS_val + 1)
nodes_ana[end] = L_end
for i in nS_val:-1:1
    nodes_ana[i] = nodes_ana[i + 1] / k
end

Svec_ana = [analytical_S(nodes_ana[i], nodes_ana[i + 1], _c, _a) for i in 1:nS_val]

ov_ana = asp_core(Svec_ana, k, n_wynn, nseek)
println("\nWith EXACT integrals (no interpolation, no quadrature):")
for i in 1:length(ov_ana)
    e = Float64(abs(ov_ana[i] - _a[i]))
    println("  order $i: found=$(Float64(ov_ana[i]))  exact=$(Float64(_a[i]))  err=$e")
end

# ============================================================
println("\n" * "=" ^ 65)
println("  Part 2: Quantify interpolation error in S values")
println("=" ^ 65)

point_density = 10
h_grid = T(1 / point_density)
f_func = x -> sum(_c .* x .^ (-_a))
grid_full = [1 + i * h_grid for i in 0:((L - 1) * point_density)]
f_data_full = f_func.(grid_full)

# build same B-spline as power_solve_asp
N_full = length(f_data_full)
f_view = view(f_data_full, 1:N_full)
grid_view = view(grid_full, 1:N_full)
L_start_g, L_end_g = grid_view[1], grid_view[end]
itp_base = Interpolations.interpolate(f_view, BSpline(Cubic(Line(OnGrid()))))
grid_range = range(L_start_g, L_end_g; length=N_full)
f_itp = Interpolations.scale(itp_base, grid_range)

# use the same geometric nodes as Part 1
nS_cmp = nS_val
nodes_cmp = copy(nodes_ana)

# compute S via B-spline (current approach)
M_pts = 101
Svec_bspline = zeros(T, nS_cmp)
for i in 1:nS_cmp
    xa, xb = nodes_cmp[i], nodes_cmp[i + 1]
    local_h = (xb - xa) / (M_pts - 1)
    local_x = range(xa, xb; length=M_pts)
    local_f = [f_itp(tx) for tx in local_x]
    Svec_bspline[i] = int_simpson(local_f, local_h)
end

# compute S via direct grid integration (improvement)
Svec_direct = zeros(T, nS_cmp)
for i in 1:nS_cmp
    Svec_direct[i] = direct_grid_integral(f_data_full, grid_full,
                                          nodes_cmp[i], nodes_cmp[i + 1],
                                          f_itp, h_grid)
end

println("\nRelative error in S values (point_density=$point_density):")
println("  i    |S_exact|        BSpline_rel_err   DirectGrid_rel_err")
for i in [1, nS_cmp ÷ 4, nS_cmp ÷ 2, 3 * nS_cmp ÷ 4, nS_cmp - 2, nS_cmp - 1, nS_cmp]
    s_ex = abs(Svec_ana[i])
    err_bs = Float64(abs(Svec_bspline[i] - Svec_ana[i]) / s_ex)
    err_dg = Float64(abs(Svec_direct[i] - Svec_ana[i]) / s_ex)
    println("  $i\t$(Float64(s_ex))\t$err_bs\t$err_dg")
end

# ============================================================
println("\n" * "=" ^ 65)
println("  Part 3: ASP accuracy — analytical vs B-spline vs direct grid")
println("=" ^ 65)

ov_bs = asp_core(Svec_bspline, k, n_wynn, nseek)
ov_dg = asp_core(Svec_direct, k, n_wynn, nseek)

n_show = min(length(ov_ana), length(ov_bs), length(ov_dg))
println("\n  order | exact_S_err       | bspline_err       | direct_grid_err")
println("  ------+-------------------+-------------------+-------------------")
for i in 1:n_show
    ea = Float64(abs(ov_ana[i] - _a[i]))
    eb = Float64(abs(ov_bs[i] - _a[i]))
    ed = Float64(abs(ov_dg[i] - _a[i]))
    println("  $i     | $ea | $eb | $ed")
end
if length(ov_ana) > n_show
    println("  (analytical found $(length(ov_ana)) orders, others found $(min(length(ov_bs), length(ov_dg))))")
end

# ============================================================
println("\n" * "=" ^ 65)
println("  Part 4: Effect of point_density (B-spline approach)")
println("=" ^ 65)

for pd in [10, 30]
    println("\n--- point_density = $pd ---")
    h_pd = T(1 / pd)
    grid_pd = [1 + i * h_pd for i in 0:((L - 1) * pd)]
    f_data_pd = f_func.(grid_pd)

    nseek_pd = 7
    asp_pd = ASP(nseek_pd, length(f_data_pd); wynn_pola=WynnPola(; k=1.3, n=21))
    ov_pd = power_solve_asp(f_data_pd, grid_pd, asp_pd)
    for i in 1:length(ov_pd)
        println("  order $i: err=$(Float64(abs(ov_pd[i] - _a[i])))")
    end
end
