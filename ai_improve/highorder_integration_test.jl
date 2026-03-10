using Random, LinearAlgebra, Printf
using PowerLawDecomposition
using Interpolations
import PowerLawDecomposition: int_simpson, grid_check, wynn_epsilon_core

# ================================================================
#  Gauss-Legendre nodes & weights on [-1, 1]
# ================================================================
function gauss_legendre(n::Int, ::Type{T}) where {T<:AbstractFloat}
    nodes = zeros(T, n)
    weights = zeros(T, n)
    pi_T = T(π)
    for i in 1:n
        x = -cos(pi_T * (4i - 1) / (4n + 2))
        for _ in 1:200
            p0, p1 = one(T), x
            for j in 2:n
                p2 = ((2j - 1) * x * p1 - (j - 1) * p0) / T(j)
                p0, p1 = p1, p2
            end
            dp = T(n) * (x * p1 - p0) / (x^2 - 1)
            dx = -p1 / dp
            x += dx
            abs(dx) < 256 * eps(T) && break
        end
        nodes[i] = x
        p0, p1 = one(T), x
        for j in 2:n
            p2 = ((2j - 1) * x * p1 - (j - 1) * p0) / T(j)
            p0, p1 = p1, p2
        end
        dp = T(n) * (x * p1 - p0) / (x^2 - 1)
        weights[i] = 2 / ((1 - x^2) * dp^2)
    end
    return nodes, weights
end

# ================================================================
#  Lagrange interpolation from grid support
# ================================================================
function lagrange_eval(x::T, xsup::AbstractVector{T}, fsup::AbstractVector{T}) where {T}
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

function find_support(grid, idx_center::Int, half_width::Int)
    p = 2 * half_width
    i_lo = max(1, idx_center - half_width)
    i_hi = i_lo + p
    if i_hi > length(grid)
        i_hi = length(grid)
        i_lo = max(1, i_hi - p)
    end
    return i_lo:i_hi
end

# ================================================================
#  Composite GL + local Lagrange integration over [a, b]
#
#  Split [a,b] into chunks of at most max_chunk_h * h width.
#  Each chunk: n_gl GL nodes, each evaluated via degree-(2*hw)
#  Lagrange interpolation from the nearest grid points.
#
#  Error per chunk: O(chunk_width * h^{2*hw+1})
#  Total: O((b-a) * h^{2*hw+1})
# ================================================================
function compute_S_cgl(f_data, grid, a::T, b::T;
                        hw::Int=15, n_gl::Int=20,
                        max_chunk_h::Int=20) where {T}
    h = grid[2] - grid[1]
    L0 = grid[1]
    N = length(grid)
    chunk_max = max_chunk_h * h
    L = b - a
    n_chunks = max(1, ceil(Int, L / chunk_max))
    chunk_w = L / n_chunks

    gl_nodes, gl_weights = gauss_legendre(n_gl, T)

    result = zero(T)
    for c in 1:n_chunks
        ca = a + (c - 1) * chunk_w
        cb = a + c * chunk_w
        mid = (ca + cb) / 2
        hl = (cb - ca) / 2
        s = zero(T)
        for k in eachindex(gl_nodes)
            xk = mid + hl * gl_nodes[k]
            idx = clamp(round(Int, (xk - L0) / h) + 1, 1, N)
            if abs(xk - grid[idx]) < h * T(1e-14)
                s += gl_weights[k] * f_data[idx]
            else
                rng = find_support(grid, idx, hw)
                s += gl_weights[k] * lagrange_eval(xk, view(grid, rng), view(f_data, rng))
            end
        end
        result += hl * s
    end
    return result
end

# ================================================================
#  ASP core on pre-computed Svec
# ================================================================
function asp_core(Svec0::Vector{T}, k::T, n0::Int, norder::Int) where {T<:Real}
    Svec = copy(Svec0)
    nS = length(Svec)
    wynn_n = n0; n = n0
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
            if all(vr .> 0); sign_flag = 1; break
            elseif all(vr .< 0); sign_flag = -1; break
            else n -= 2; end
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

# ================================================================
#  Analytical integral
# ================================================================
function analytical_S(a::T, b::T, cv, av) where T
    s = zero(T)
    for (c, α) in zip(cv, av)
        s += c * (b^(1 - α) - a^(1 - α)) / (1 - α)
    end
    return s
end

# ================================================================
#  Setup
# ================================================================
setprecision(BigFloat, 512)

Random.seed!(666)
T = BigFloat
nord = 10; d = T(0.5); L_val = 2^14

a_vec = T[]
for i in 1:nord
    push!(a_vec, i == 1 ? (1 + rand(T)) / 2 : a_vec[end] + (1 + rand(T)) * d)
end
c_vec = [1 + rand(T) for _ in 1:nord]
_a = T.(a_vec); _c = T.(c_vec)
println("Exponents:    ", round.(Float64.(_a); digits=3))

f_func = x -> sum(_c .* x .^ (-_a))

k_val = T(1.05)
nS_total = 160
nodes = zeros(T, nS_total + 1)
nodes[end] = T(L_val)
for i in nS_total:-1:1; nodes[i] = nodes[i + 1] / k_val; end
println("nodes[1] = ", round(Float64(nodes[1]); digits=2))

Svec_analytical = [analytical_S(nodes[i], nodes[i+1], _c, _a) for i in 1:nS_total]

# ================================================================
#  TEST 1: S value accuracy — BSpline vs Composite GL (pd=10)
# ================================================================
println("\n" * "=" ^ 80)
println("  TEST 1: S value accuracy — BSpline vs Composite GL (pd=10)")
println("=" ^ 80)

pd = 10
h_grid = T(1 / pd)
grid_10 = [1 + i * h_grid for i in 0:((L_val - 1) * pd)]
f_data_10 = f_func.(grid_10)

# B-spline (library code path)
itp_base = Interpolations.interpolate(f_data_10, BSpline(Cubic(Line(OnGrid()))))
gr = range(grid_10[1], grid_10[end]; length=length(grid_10))
fitp = Interpolations.scale(itp_base, gr)

Svec_bs = zeros(T, nS_total)
for i in 1:nS_total
    xa, xb = nodes[i], nodes[i+1]
    lh = (xb - xa) / 100
    lx = range(xa, xb; length=101)
    lf = [fitp(tx) for tx in lx]
    Svec_bs[i] = int_simpson(lf, lh)
end

# Composite GL (proposed)
Svec_cgl = zeros(T, nS_total)
for i in 1:nS_total
    Svec_cgl[i] = compute_S_cgl(f_data_10, grid_10, nodes[i], nodes[i+1];
                                  hw=15, n_gl=20, max_chunk_h=20)
end

println("\n  idx | x_center   | BSpline rel_err | CGL rel_err     | improvement")
println("  " * "-" ^ 80)
for i in [1, 5, 10, 20, 30, 40, 60, 80, 100, 120, 140, nS_total]
    i > nS_total && continue
    s_ex = abs(Svec_analytical[i])
    s_ex < eps(T) && continue
    xc = Float64((nodes[i] + nodes[i+1]) / 2)
    e_bs = Float64(abs(Svec_bs[i] - Svec_analytical[i]) / s_ex)
    e_cgl = Float64(abs(Svec_cgl[i] - Svec_analytical[i]) / s_ex)
    ratio = e_cgl > 0 ? e_bs / e_cgl : Inf
    @printf("  %-4d| %10.1f | %14.2e  | %14.2e  | %.1e×\n", i, xc, e_bs, e_cgl, ratio)
end

worst_bs = maximum(Float64(abs(Svec_bs[i] - Svec_analytical[i]) / abs(Svec_analytical[i]))
                   for i in 1:nS_total if abs(Svec_analytical[i]) > eps(T))
worst_cgl = maximum(Float64(abs(Svec_cgl[i] - Svec_analytical[i]) / abs(Svec_analytical[i]))
                    for i in 1:nS_total if abs(Svec_analytical[i]) > eps(T))
@printf("  Worst S error: BSpline = %.2e, CGL = %.2e\n", worst_bs, worst_cgl)

# ================================================================
#  TEST 2: CGL accuracy vs interpolation order (pd=10)
# ================================================================
println("\n" * "=" ^ 80)
println("  TEST 2: CGL S accuracy vs Lagrange half-width (pd=10)")
println("=" ^ 80)

println("  hw | poly_deg | worst S err    | near-end (i=1) | mid (i=40)     | far (i=160)")
println("  " * "-" ^ 85)
for hw in [3, 5, 8, 10, 15, 20]
    sv = zeros(T, nS_total)
    for i in 1:nS_total
        sv[i] = compute_S_cgl(f_data_10, grid_10, nodes[i], nodes[i+1];
                               hw=hw, n_gl=max(hw + 5, 15), max_chunk_h=2*hw)
    end
    worst = maximum(Float64(abs(sv[i] - Svec_analytical[i]) / abs(Svec_analytical[i]))
                    for i in 1:nS_total if abs(Svec_analytical[i]) > eps(T))
    e1 = Float64(abs(sv[1] - Svec_analytical[1]) / abs(Svec_analytical[1]))
    e40 = Float64(abs(sv[40] - Svec_analytical[40]) / abs(Svec_analytical[40]))
    e160 = Float64(abs(sv[160] - Svec_analytical[160]) / abs(Svec_analytical[160]))
    @printf("  %-3d| %-8d | %13.2e  | %14.2e | %14.2e | %14.2e\n",
            hw, 2*hw, worst, e1, e40, e160)
end

# ================================================================
#  TEST 3: ASP accuracy — Analytical vs BSpline vs CGL
# ================================================================
println("\n" * "=" ^ 80)
println("  TEST 3: ASP accuracy (k=1.05, pd=10)")
println("=" ^ 80)

nseek = 10

for n_wynn in [21, 41, 61, 81, 101]
    nS_need = n_wynn + ceil(Int, (nseek - 1) / 2) * 2 + 1
    nS_need > nS_total && continue

    offset = nS_total - nS_need
    sv_a = Svec_analytical[(offset+1):nS_total]
    sv_b = Svec_bs[(offset+1):nS_total]
    sv_c = Svec_cgl[(offset+1):nS_total]

    ov_a = asp_core(sv_a, k_val, n_wynn, nseek)
    ov_b = asp_core(sv_b, k_val, n_wynn, nseek)
    ov_c = asp_core(sv_c, k_val, n_wynn, nseek)

    println("\n  --- Wynn n=$n_wynn ---")
    println("  order | Analytical     | BSpline        | CGL (hw=15)")
    println("  " * "-" ^ 60)
    n_show = max(length(ov_a), length(ov_b), length(ov_c))
    for i in 1:n_show
        ea = i <= length(ov_a) ? @sprintf("%13.2e", Float64(abs(ov_a[i] - _a[i]))) : "     ---     "
        eb = i <= length(ov_b) ? @sprintf("%13.2e", Float64(abs(ov_b[i] - _a[i]))) : "     ---     "
        ec = i <= length(ov_c) ? @sprintf("%13.2e", Float64(abs(ov_c[i] - _a[i]))) : "     ---     "
        @printf("  %2d    | %s  | %s  | %s\n", i, ea, eb, ec)
    end
    wa = length(ov_a) > 0 ? maximum(Float64(abs(ov_a[i] - _a[i])) for i in 1:length(ov_a)) : Inf
    wb = length(ov_b) > 0 ? maximum(Float64(abs(ov_b[i] - _a[i])) for i in 1:length(ov_b)) : Inf
    wc = length(ov_c) > 0 ? maximum(Float64(abs(ov_c[i] - _a[i])) for i in 1:length(ov_c)) : Inf
    @printf("  worst | %13.2e  | %13.2e  | %13.2e  (orders: %d/%d/%d)\n",
            wa, wb, wc, length(ov_a), length(ov_b), length(ov_c))
end

# ================================================================
#  TEST 4: CGL + higher point_density + optimal Wynn n
# ================================================================
println("\n" * "=" ^ 80)
println("  TEST 4: CGL + varying point_density (k=1.05, n=101)")
println("=" ^ 80)

n_wynn_t4 = 101
nS_need_t4 = n_wynn_t4 + ceil(Int, (nseek - 1) / 2) * 2 + 1
offset_t4 = nS_total - nS_need_t4

# Reference: analytical
ov_ref = asp_core(Svec_analytical[(offset_t4+1):nS_total], k_val, n_wynn_t4, nseek)
println("\n  [Reference] Analytical S:")
for i in 1:length(ov_ref)
    @printf("    order %2d: err = %.2e\n", i, Float64(abs(ov_ref[i] - _a[i])))
end

for pd_t4 in [10, 20, 50]
    h_t4 = T(1 / pd_t4)
    grid_t4 = [1 + i * h_t4 for i in 0:((L_val - 1) * pd_t4)]
    f_data_t4 = f_func.(grid_t4)

    sv_t4 = zeros(T, nS_total)
    for i in 1:nS_total
        sv_t4[i] = compute_S_cgl(f_data_t4, grid_t4, nodes[i], nodes[i+1];
                                  hw=15, n_gl=20, max_chunk_h=20)
    end

    worst_s = maximum(Float64(abs(sv_t4[i] - Svec_analytical[i]) / abs(Svec_analytical[i]))
                      for i in (offset_t4+1):nS_total if abs(Svec_analytical[i]) > eps(T))

    ov_t4 = asp_core(sv_t4[(offset_t4+1):nS_total], k_val, n_wynn_t4, nseek)
    @printf("\n  pd=%d (h=%.4f), worst S rel err in ASP range = %.2e:\n",
            pd_t4, Float64(h_t4), worst_s)
    for i in 1:length(ov_t4)
        @printf("    order %2d: err = %.2e\n", i, Float64(abs(ov_t4[i] - _a[i])))
    end
end

println("\n" * "=" ^ 80)
println("  DONE")
println("=" ^ 80)
