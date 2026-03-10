using Random, LinearAlgebra, Printf
using PowerLawDecomposition
import PowerLawDecomposition: wynn_epsilon_core

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

function analytical_S(a::T, b::T, cv, av) where T
    s = zero(T)
    for (c, α) in zip(cv, av)
        s += c * (b^(1 - α) - a^(1 - α)) / (1 - α)
    end
    return s
end

setprecision(BigFloat, 512)
Random.seed!(666)
T = BigFloat
nord = 10; d = T(0.5)

a_vec = T[]
for i in 1:nord
    push!(a_vec, i == 1 ? (1 + rand(T)) / 2 : a_vec[end] + (1 + rand(T)) * d)
end
c_vec = [1 + rand(T) for _ in 1:nord]
_a = T.(a_vec); _c = T.(c_vec)
println("Exponents: ", round.(Float64.(_a); digits=3))

k = T(1.3)
n_wynn = 21
nseek = 7
nS = n_wynn + ceil(Int, (nseek - 1) / 2) * 2 + 1  # = 28

# ================================================================
#  TEST 1: Analytical S — varying L_end
#  (zero interpolation error, zero quadrature error)
# ================================================================
println("\n" * "=" ^ 75)
println("  TEST 1: Analytical S + ASP — varying L_end (k=1.3, n=21)")
println("=" ^ 75)

for L_end_val in [2^10, 2^12, 2^14, 2^16, 2^18, 2^20]
    L_end = T(L_end_val)
    nodes = zeros(T, nS + 1)
    nodes[end] = L_end
    for i in nS:-1:1; nodes[i] = nodes[i + 1] / k; end

    Svec = [analytical_S(nodes[i], nodes[i+1], _c, _a) for i in 1:nS]
    ov = asp_core(Svec, k, n_wynn, nseek)

    @printf("\n  L=%d, nodes[1]=%.1f, nodes[end]=%.0f:\n", L_end_val, Float64(nodes[1]), Float64(L_end))
    for i in 1:length(ov)
        e = Float64(abs(ov[i] - _a[i]))
        @printf("    order %d: err = %.2e\n", i, e)
    end
end

# ================================================================
#  TEST 2: Diagnostic — chain subtraction noise analysis
#  Show the noise-to-signal ratio after each chain subtraction
# ================================================================
println("\n" * "=" ^ 75)
println("  TEST 2: Chain subtraction — S value magnitude after each step")
println("=" ^ 75)

for L_end_val in [2^14, 2^16]
    L_end = T(L_end_val)
    nodes = zeros(T, nS + 1)
    nodes[end] = L_end
    for i in nS:-1:1; nodes[i] = nodes[i + 1] / k; end

    Svec = [analytical_S(nodes[i], nodes[i+1], _c, _a) for i in 1:nS]
    Svec_copy = copy(Svec)

    @printf("\n  L=%d:\n", L_end_val)
    @printf("  step | a_found        | err(a)       | |S_last|       | |S_first|     | |S_last/S_first|\n")
    @printf("  " * "-" ^ 90 * "\n")

    cur_nS = nS
    for step in 1:nseek
        # compute ratios and Wynn
        nn = n_wynn
        wn = n_wynn
        ratio_vec = zeros(T, wn)
        Avec = zeros(T, wn)
        for j in 1:nn
            ratio_vec[end-j+1] = Svec_copy[cur_nS-j+1] / Svec_copy[cur_nS-j]
        end
        sign_flag = 0
        while true
            nn < 3 && break
            vr = view(ratio_vec, (wn-nn+1):wn)
            if all(vr .> 0); sign_flag = 1; break
            elseif all(vr .< 0); sign_flag = -1; break
            else nn -= 2; end
        end
        nn < 3 && break
        vA = view(Avec, (wn-nn+1):wn)
        vr = view(ratio_vec, (wn-nn+1):wn)
        vA .= 1 .- log.(vr .* sign_flag) ./ log(k)
        a_est = wynn_epsilon_core(vA)

        err = Float64(abs(a_est - _a[step]))
        s_last = Float64(abs(Svec_copy[cur_nS]))
        s_first = Float64(abs(Svec_copy[cur_nS - nn + 1]))
        ratio = s_last / s_first

        @printf("  %d    | %.10f  | %.2e   | %.2e     | %.2e    | %.2e\n",
                step, Float64(a_est), err, s_last, s_first, ratio)

        # chain subtraction
        λ = k^(1 - a_est)
        cur_nS -= 1
        for j in 1:cur_nS
            Svec_copy[cur_nS+1-j] = Svec_copy[cur_nS+1-j] - λ * Svec_copy[cur_nS-j]
        end
    end
end

# ================================================================
#  TEST 3: What is the actual contribution of each term at the far end?
# ================================================================
println("\n" * "=" ^ 75)
println("  TEST 3: Relative weight of each power-law term at geometric nodes")
println("=" ^ 75)

for L_end_val in [2^14, 2^16]
    L_end = T(L_end_val)
    nodes = zeros(T, nS + 1)
    nodes[end] = L_end
    for i in nS:-1:1; nodes[i] = nodes[i + 1] / k; end

    # S values from each individual term at the LAST interval
    x_a, x_b = nodes[nS], nodes[nS+1]
    S_total = zero(T)
    @printf("\n  L=%d, last interval [%.0f, %.0f]:\n", L_end_val, Float64(x_a), Float64(x_b))
    for (idx, (c, a)) in enumerate(zip(_c, _a))
        Si = c * (x_b^(1-a) - x_a^(1-a)) / (1-a)
        @printf("    term %2d (a=%.3f): S_i = %.4e, fraction = %.4e\n",
                idx, Float64(a), Float64(Si), Float64(Si / analytical_S(x_a, x_b, _c, _a)))
    end
end

println("\n" * "=" ^ 75)
println("  DONE")
println("=" ^ 75)
