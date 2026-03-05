using Test, Random, LinearAlgebra
using PowerLawDecomposition
using Interpolations
import PowerLawDecomposition: int_simpson, grid_check

println("="^60)
println("  Test 1: wynn_epsilon_core vs v2 on known sequences")
println("="^60)

# Sequence whose partial sums converge to π²/6 (Basel problem):
# S_n = sum_{k=1}^{n} 1/k^2 → π²/6
for T in [Float64, BigFloat]
    println("\n--- T = $T ---")
    exact = T(π)^2 / 6
    for n in [7, 11, 21]
        S = [sum(one(T) / T(k)^2 for k in 1:i) for i in 1:n]
        r1 = wynn_epsilon_core(S)
        r2 = wynn_epsilon_core_v2(S)
        e1 = abs(r1 - exact)
        e2 = abs(r2 - exact)
        println("  n=$n  v1_err=$e1  v2_err=$e2")
    end
end

# Sequence with known limit: S_n = 1 - (-1/2)^n → 1
for T in [Float64, BigFloat]
    println("\n--- alternating geometric, T = $T ---")
    exact = one(T)
    for n in [7, 11, 21]
        S = [one(T) - (T(-1) / 2)^k for k in 1:n]
        r1 = wynn_epsilon_core(S)
        r2 = wynn_epsilon_core_v2(S)
        e1 = abs(r1 - exact)
        e2 = abs(r2 - exact)
        println("  n=$n  v1_err=$e1  v2_err=$e2")
    end
end

# Slowly converging: ln(2) = 1 - 1/2 + 1/3 - 1/4 + ...
for T in [Float64, BigFloat]
    println("\n--- ln(2) alternating harmonic, T = $T ---")
    exact = log(T(2))
    for n in [7, 11, 21]
        S = [sum(T(-1)^(k + 1) / T(k) for k in 1:i) for i in 1:n]
        r1 = wynn_epsilon_core(S)
        r2 = wynn_epsilon_core_v2(S)
        e1 = abs(r1 - exact)
        e2 = abs(r2 - exact)
        println("  n=$n  v1_err=$e1  v2_err=$e2")
    end
end

println("\n")
println("="^60)
println("  Test 2: power_solve_asp v1 vs v2 core (head-to-head)")
println("="^60)

# A version of power_solve_asp that accepts a core function argument
function power_solve_asp_with(f0::AbstractVector{T}, grid0::AbstractVector{T},
                              asp::ASP, core_fn::Function) where {T<:Real}
    len = round(Int, asp.scale1)
    N = length(f0)
    order_vec = T[]
    norder = asp.norder
    method = asp.wynn_pola
    nS = asp.lenS

    f = view(f0, (N - len + 1):N)
    grid = view(grid0, (N - len + 1):N)

    k = method.k
    n = method.n
    wynn_n = method.n
    interp_type = method.interp_type

    grid_check(grid)
    L_start, L_end = grid[1], grid[end]
    itp_base = Interpolations.interpolate(f, interp_type)
    grid_range = range(L_start, L_end; length=length(grid))
    f_itp = Interpolations.scale(itp_base, grid_range)

    nodes = zeros(T, nS + 1)
    nodes[end] = L_end
    for i in nS:-1:1
        nodes[i] = nodes[i + 1] / k
    end

    Svec = zeros(T, nS)
    M = method.points_per_interval
    for i in 1:nS
        x_a, x_b = nodes[i], nodes[i + 1]
        local_h = (x_b - x_a) / (M - 1)
        local_x = range(x_a, x_b; length=M)
        local_f = [f_itp(tx) for tx in local_x]
        Svec[i] = int_simpson(local_f, local_h)
    end

    Avec = zeros(T, wynn_n)
    ratio_vec = zeros(T, wynn_n)

    for i in 1:norder
        if n <= 1
            break
        end
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

        a_final = core_fn(vA)

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

_a_vec = T.(a_vec)
_c_vec = T.(c_vec)
f_asp = x -> sum(_c_vec .* x .^ (-_a_vec))
h_asp = T(1 / 10)
grid_asp = [1 + i * h_asp for i in 0:((L - 1) * 10)]
f_data_asp = f_asp.(grid_asp)

nseek = 7
asp = ASP(nseek, length(f_data_asp); wynn_pola=WynnPola(; k=1.3, n=21))

println("\n--- v1 (original wynn_epsilon_core) ---")
ov1 = power_solve_asp_with(f_data_asp, grid_asp, asp, wynn_epsilon_core)
for i in 1:length(ov1)
    println("  order $i: err=$(Float64(abs(ov1[i] - _a_vec[i])))")
end

println("\n--- v2 (Gemini's wynn_epsilon_core_v2) ---")
ov2 = power_solve_asp_with(f_data_asp, grid_asp, asp, wynn_epsilon_core_v2)
for i in 1:length(ov2)
    println("  order $i: err=$(Float64(abs(ov2[i] - _a_vec[i])))")
end
