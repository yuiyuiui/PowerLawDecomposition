using Random, LinearAlgebra
using PowerLawDecomposition

# ===== Setup =====
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

# ===== Analytical integral of c*x^{-α} over [a, b] =====
function powerlaw_integral(c::T, α::T, a::T, b::T) where T
    c * (b^(1 - α) - a^(1 - α)) / (1 - α)
end

# ===== Original ASP core: chained subtraction =====
function asp_core_chained(Svec0::Vector{T}, k::T, n0::Int, norder::Int) where {T<:Real}
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

# ===== Improved ASP core: analytical residual subtraction =====
function asp_core_analytical(Svec0::Vector{T}, nodes::Vector{T},
                             k::T, n0::Int, norder::Int) where {T<:Real}
    Svec = copy(Svec0)
    nS = length(Svec)
    wynn_n = n0
    n = n0
    order_vec = T[]
    coeff_vec = T[]
    Avec = zeros(T, wynn_n)
    Cvec = zeros(T, wynn_n)
    ratio_vec = zeros(T, wynn_n)

    for iter in 1:norder
        n <= 1 && break

        # compute ratios from the tail
        for j in 1:n
            ratio_vec[end - j + 1] = Svec[end - j + 1] / Svec[end - j]
        end

        # sign check + n reduction
        sign_flag = 0
        n_local = n
        while true
            n_local < 3 && break
            vr = view(ratio_vec, (wynn_n - n_local + 1):wynn_n)
            if all(vr .> 0)
                sign_flag = 1; break
            elseif all(vr .< 0)
                sign_flag = -1; break
            else
                n_local -= 2
            end
        end
        n_local < 3 && break
        n = n_local

        # Wynn acceleration for order a
        vA = view(Avec, (wynn_n - n + 1):wynn_n)
        vr = view(ratio_vec, (wynn_n - n + 1):wynn_n)
        vA .= 1 .- log.(vr .* sign_flag) ./ log(k)
        a_final = wynn_epsilon_core(vA)

        # Wynn acceleration for coefficient c
        for j in (wynn_n - n + 1):wynn_n
            a_tmp = Avec[j]
            denom = nodes[j + 1]^(1 - a_tmp) - nodes[j]^(1 - a_tmp)
            Cvec[j] = abs(denom) < eps(T) * 100 ? zero(T) :
                      Svec[j] * (1 - a_tmp) / denom
        end
        c_final = wynn_epsilon_core(view(Cvec, (wynn_n - n + 1):wynn_n))

        push!(order_vec, a_final)
        push!(coeff_vec, c_final)
        iter == norder && break

        # Analytical residual subtraction: each S[j] is corrected independently
        for j in 1:nS
            Svec[j] -= powerlaw_integral(c_final, a_final, nodes[j], nodes[j + 1])
        end
        # nS stays the same — no information loss
    end
    return order_vec, coeff_vec
end

# ===== Generate exact S values =====
k = T(1.3)
n_wynn = 21
nseek = 10
nS_val = n_wynn + ceil(Int, (nseek - 1) / 2) * 2 + 1
L_end = T(L)

nodes = zeros(T, nS_val + 1)
nodes[end] = L_end
for i in nS_val:-1:1
    nodes[i] = nodes[i + 1] / k
end

Svec_exact = zeros(T, nS_val)
for i in 1:nS_val
    for (c, α) in zip(_c, _a)
        Svec_exact[i] += powerlaw_integral(c, α, nodes[i], nodes[i + 1])
    end
end

# ===== Test 1: Chained (original) vs Analytical (improved) on EXACT S =====
println("=" ^ 65)
println("  Exact S values: chained vs analytical residual subtraction")
println("=" ^ 65)

ov_chain = asp_core_chained(Svec_exact, k, n_wynn, nseek)
ov_ana, cv_ana = asp_core_analytical(Svec_exact, nodes, k, n_wynn, nseek)

n_max = max(length(ov_chain), length(ov_ana))
println("\n  order | chained_err         | analytical_err      | exact_a")
println("  ------+---------------------+---------------------+--------")
for i in 1:n_max
    ec = i <= length(ov_chain) ? Float64(abs(ov_chain[i] - _a[i])) : NaN
    ea = i <= length(ov_ana) ? Float64(abs(ov_ana[i] - _a[i])) : NaN
    println("  $i     | $ec | $ea | $(Float64(_a[i]))")
end

if length(cv_ana) > 0
    println("\n  Coefficient estimates:")
    for i in 1:length(cv_ana)
        println("  c_$i: found=$(Float64(cv_ana[i]))  exact=$(Float64(_c[i]))  err=$(Float64(abs(cv_ana[i] - _c[i])))")
    end
end

# ===== Test 2: With B-spline interpolated S values =====
println("\n" * "=" ^ 65)
println("  B-spline S values (pd=10): chained vs analytical residual")
println("=" ^ 65)

using Interpolations

point_density = 10
h_grid = T(1 / point_density)
f_func = x -> sum(_c .* x .^ (-_a))
grid_full = [1 + i * h_grid for i in 0:((L - 1) * point_density)]
f_data_full = f_func.(grid_full)

N_full = length(f_data_full)
L_start_g = grid_full[1]
L_end_g = grid_full[end]
itp_base = Interpolations.interpolate(f_data_full, BSpline(Cubic(Line(OnGrid()))))
grid_range = range(L_start_g, L_end_g; length=N_full)
f_itp = Interpolations.scale(itp_base, grid_range)

M_pts = 101
Svec_bspline = zeros(T, nS_val)
for i in 1:nS_val
    xa, xb = nodes[i], nodes[i + 1]
    local_h = (xb - xa) / (M_pts - 1)
    local_x = range(xa, xb; length=M_pts)
    local_f = [f_itp(tx) for tx in local_x]
    Svec_bspline[i] = PowerLawDecomposition.int_simpson(local_f, local_h)
end

ov_bs_chain = asp_core_chained(Svec_bspline, k, n_wynn, nseek)
ov_bs_ana, cv_bs_ana = asp_core_analytical(Svec_bspline, nodes, k, n_wynn, nseek)

n_max2 = max(length(ov_bs_chain), length(ov_bs_ana))
println("\n  order | chained_err         | analytical_err      | exact_a")
println("  ------+---------------------+---------------------+--------")
for i in 1:n_max2
    ec = i <= length(ov_bs_chain) ? Float64(abs(ov_bs_chain[i] - _a[i])) : NaN
    ea = i <= length(ov_bs_ana) ? Float64(abs(ov_bs_ana[i] - _a[i])) : NaN
    println("  $i     | $ec | $ea | $(Float64(_a[i]))")
end
