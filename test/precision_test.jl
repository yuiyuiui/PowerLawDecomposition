using Random, LinearAlgebra
using PowerLawDecomposition

# Generate parameters in Float64 (fixed across all precision levels)
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

println("True exponents: ", round.(a_f64; digits=4))
println("True coefficients: ", round.(c_f64; digits=4))

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

function run_at_precision(prec, nseek)
    setprecision(BigFloat, prec)
    T = BigFloat
    _a = T.(a_f64)
    _c = T.(c_f64)
    k = T(1.3)
    n_wynn = 21
    L_end = T(2^14)
    nS_val = n_wynn + ceil(Int, (nseek - 1) / 2) * 2 + 1

    nodes = zeros(T, nS_val + 1)
    nodes[end] = L_end
    for i in nS_val:-1:1
        nodes[i] = nodes[i + 1] / k
    end

    Svec = zeros(T, nS_val)
    for i in 1:nS_val
        for (c, α) in zip(_c, _a)
            Svec[i] += c * (nodes[i + 1]^(1 - α) - nodes[i]^(1 - α)) / (1 - α)
        end
    end

    ov = asp_core(Svec, k, n_wynn, nseek)
    errs = [abs(ov[i] - _a[i]) for i in 1:length(ov)]
    return errs
end

println("\n" * "=" ^ 70)
println("  Effect of BigFloat precision on ASP accuracy (exact integrals)")
println("=" ^ 70)

nseek = 10
results = Dict{Int,Vector}()

for prec in [256, 512, 1024, 2048]
    errs = run_at_precision(prec, nseek)
    results[prec] = errs
    digits = round(Int, prec * log10(2))
    println("\nprec=$prec bits (~$digits decimal digits), found $(length(errs)) orders:")
    for (i, e) in enumerate(errs)
        println("  order $i: err=$(Float64(e))")
    end
end

println("\n" * "=" ^ 70)
println("  Summary table")
println("=" ^ 70)

# Find max orders across all precisions
max_ord = maximum(length(v) for v in values(results))
precs = sort(collect(keys(results)))

print("\n  order")
for p in precs
    d = round(Int, p * log10(2))
    print(" | $(d)d")
end
println()
print("  -----")
for _ in precs
    print("-+-----------")
end
println()

for i in 1:max_ord
    print("  $i    ")
    for p in precs
        errs = results[p]
        if i <= length(errs)
            e = Float64(errs[i])
            if e < 1e-50
                # print in scientific notation with BigFloat
                setprecision(BigFloat, p)
                print(" | $(Float64(errs[i]))")
            else
                print(" | $e")
            end
        else
            print(" | ---")
        end
    end
    println()
end
