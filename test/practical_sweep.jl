using Test, Random, LinearAlgebra
using PowerLawDecomposition

Random.seed!(666)
T = BigFloat
nord = 10
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

function run_test(pd, k_val, n_val, nseek; lenS=nothing)
    f = x -> sum(_c_vec .* x .^ (-_a_vec))
    h = T(1 / pd)
    grid = [1 + i * h for i in 0:((2^14 - 1) * pd)]
    f_data = f.(grid)

    kw = if lenS !== nothing
        (; wynn_pola=WynnPola(; k=k_val, n=n_val), lenS=lenS)
    else
        (; wynn_pola=WynnPola(; k=k_val, n=n_val))
    end
    asp = ASP(nseek, length(f_data); kw...)
    order_vec = power_solve_asp(f_data, grid, asp)
    errs = [Float64(abs(order_vec[i] - _a_vec[i])) for i in 1:length(order_vec)]
    return errs
end

nseek = 7

configs = [
    # (pd, k, n, label, lenS_override)
    (10, big"1.3",  21, "baseline: pd=10, k=1.3, n=21", nothing),
    (10, big"1.1",  21, "pd=10, k=1.1, n=21", nothing),
    (10, big"1.1",  41, "pd=10, k=1.1, n=41", nothing),
    (10, big"1.1",  51, "pd=10, k=1.1, n=51", 70),
    (10, big"1.05", 41, "pd=10, k=1.05, n=41", 60),
    (10, big"1.05", 51, "pd=10, k=1.05, n=51", 70),
    (10, big"1.05", 71, "pd=10, k=1.05, n=71", 90),
    (10, big"1.05",101, "pd=10, k=1.05, n=101",120),
    (10, big"1.05",151, "pd=10, k=1.05, n=151",nothing),
    (30, big"1.3",  21, "pd=30, k=1.3, n=21", nothing),
    (30, big"1.05", 51, "pd=30, k=1.05, n=51", 70),
    (30, big"1.05",101, "pd=30, k=1.05, n=101",120),
]

for (pd, k_val, n_val, label, lenS_ov) in configs
    print("\n[$label]  ")
    errs = run_test(pd, k_val, n_val, nseek; lenS=lenS_ov)
    for (i, e) in enumerate(errs)
        tag = e < 1e-10 ? "★" : e < 1e-5 ? "✓" : e < 0.01 ? "~" : "✗"
        print("o$i=$(round(e; sigdigits=2))$tag ")
    end
    println()
end
