using Random, LinearAlgebra
using PowerLawDecomposition

Random.seed!(666)
nord = 10
d = 0.5
a_f64 = Float64[]
for i in 1:nord
    if i == 1
        push!(a_f64, (1 + rand()) / 2)
    else
        push!(a_f64, a_f64[end] + (1 + rand()) * d)
    end
end
c_f64 = Float64[]
for i in 1:nord
    push!(c_f64, 1 + rand())
end
println("Exponents: ", round.(a_f64; digits=3))

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

function test_params(k_val, n_wynn, nseek; L_end_val=2^14)
    T = BigFloat
    _a = T.(a_f64)
    _c = T.(c_f64)
    k = T(k_val)
    L_end = T(L_end_val)

    # max nS given grid range [1, L_end]
    nS_max = floor(Int, log(L_end) / log(k))
    nS_need = n_wynn + ceil(Int, (nseek - 1) / 2) * 2 + 1
    nS_val = min(nS_max, nS_need)

    if nS_val < n_wynn + 1
        return nothing
    end

    nodes = zeros(T, nS_val + 1)
    nodes[end] = L_end
    for i in nS_val:-1:1
        nodes[i] = nodes[i + 1] / k
    end

    if nodes[1] < 1
        return nothing
    end

    Svec = zeros(T, nS_val)
    for i in 1:nS_val
        for (c, α) in zip(_c, _a)
            Svec[i] += c * (nodes[i + 1]^(1 - α) - nodes[i]^(1 - α)) / (1 - α)
        end
    end

    ov = asp_core(Svec, k, n_wynn, nseek)
    return [Float64(abs(ov[i] - _a[i])) for i in 1:length(ov)]
end

setprecision(BigFloat, 512)
nseek = 10

println("\n" * "=" ^ 70)
println("  Parameter sweep: (k, n) on analytical S values")
println("=" ^ 70)

configs = [
    (1.3,  21,  "baseline"),
    (1.3,  35,  "larger n"),
    (1.2,  21,  "smaller k"),
    (1.2,  41,  "k=1.2 n=41"),
    (1.2,  51,  "k=1.2 n=51"),
    (1.1,  21,  "k=1.1 n=21"),
    (1.1,  51,  "k=1.1 n=51"),
    (1.1,  71,  "k=1.1 n=71"),
    (1.1,  91,  "k=1.1 n=91"),
    (1.05, 51,  "k=1.05 n=51"),
    (1.05, 101, "k=1.05 n=101"),
    (1.05, 151, "k=1.05 n=151"),
]

results = []
for (k_val, n_wynn, label) in configs
    errs = test_params(k_val, n_wynn, nseek)
    if errs === nothing
        println("\n[$label] k=$k_val, n=$n_wynn — grid too short, skipped")
        continue
    end
    push!(results, (label, errs))
    println("\n[$label] k=$k_val, n=$n_wynn — found $(length(errs)) orders:")
    for (i, e) in enumerate(errs)
        tag = e < 1e-10 ? " ★" : e < 1e-5 ? " ✓" : e < 0.01 ? " ~" : " ✗"
        println("  order $i: err=$e$tag")
    end
end
