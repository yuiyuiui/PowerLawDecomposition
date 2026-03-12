"""
Part 6: Pre-filtering & Delta-Operator Matrix Pencil

Compare 6 methods:
  1. matrix_pencil (standard, L=3N)
  2. matrix_pencil_delayed (stride-k Hankel)
  3. matrix_pencil_prefilter (moving average + subsample)
  4. matrix_pencil_delta (delta operator, L=M/2)
  5. matrix_pencil_delta (delta operator, L=3N)
  6. matrix_pencil_delta_delayed (delta + stride-k combined)
"""

include("hankel.jl")
using Printf, Statistics

function make_signal(::Type{T}, N, L0, L_end, h) where {T}
    a_true = T.(collect(1:N) .* T(1 // 2))
    c_true = ones(T, N)
    M = round(Int, Float64(L_end - L0) / Float64(h)) + 1
    grid = [T(L0) + T(h) * i for i in 0:(M - 1)]
    f = [sum(c_true .* exp.(-a_true .* x)) for x in grid]
    return grid, f, a_true, c_true
end

function max_a_err(res, a_true)
    N = length(a_true)
    nr = length(res.a)
    if nr == N
        return maximum(abs.(Float64.(res.a) .- Float64.(a_true)))
    end
    errs = [minimum(abs.(Float64(res.a[k]) .- Float64.(a_true))) for k in 1:nr]
    return isempty(errs) ? Inf : maximum(errs)
end

function safe_run(fn)
    try
        return fn()
    catch
        return nothing
    end
end

function recommend_L(M, N)
    return clamp(min(3 * N, div(M, 3)), N, M - N - 1)
end

function run_all_methods(grid::AbstractVector{T}, f::AbstractVector{T}, N, a_true) where {T}
    M = length(grid)
    L3n = recommend_L(M, N)

    results = Float64[]

    push!(results, let r = safe_run(() -> matrix_pencil(grid, f, N; pencil_L=L3n))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> matrix_pencil_delayed(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> matrix_pencil_prefilter(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> matrix_pencil_delta(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> matrix_pencil_delta(grid, f, N; pencil_L=L3n))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> matrix_pencil_delta_delayed(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)

    return results
end

function print_header()
    @printf("  %-8s | %5s | %-11s | %-11s | %-11s | %-11s | %-11s | %-11s\n",
            "param", "M", "std(L=3N)", "delayed", "prefilter", "delta(M/2)", "delta(3N)",
            "delta+delay")
    return println("  " * "-" ^ 106)
end

function print_row(label, M, errs)
    @printf("  %-8s | %5d | %.2e   | %.2e   | %.2e   | %.2e   | %.2e   | %.2e\n",
            label, M, errs...)
end

# ================================================================
#  Experiment 1: Fixed domain [1,5], varying h — Float64
# ================================================================

function exp1()
    println("\n" * "=" ^ 110)
    println("  EXP 1: domain=[1,5], varying h — Float64, N=10")
    println("=" ^ 110)

    N = 10
    print_header()

    for h_val in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        grid, f, a_true, _ = make_signal(Float64, N, 1.0, 5.0, h_val)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all_methods(grid, f, N, a_true)
        print_row(string(h_val), M, errs)
    end
end

# ================================================================
#  Experiment 2: Fixed domain [1,5], varying h — BigFloat
#  (limit M ≤ 201 to avoid SVD timeout)
# ================================================================

function exp2()
    println("\n" * "=" ^ 110)
    println("  EXP 2: domain=[1,5], varying h — BigFloat(256-bit), N=10")
    println("=" ^ 110)

    setprecision(BigFloat, 256)
    N = 10
    print_header()

    for h_val in [big"0.5", big"0.2", big"0.1", big"0.05", big"0.02"]
        grid, f, a_true, _ = make_signal(BigFloat, N, big"1.0", big"5.0", h_val)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all_methods(grid, f, N, a_true)
        print_row(string(Float64(h_val)), M, errs)
    end
end

# ================================================================
#  Experiment 3: Varying domain [1, L_end], h=0.1 — Float64
# ================================================================

function exp3()
    println("\n" * "=" ^ 110)
    println("  EXP 3: varying domain [1,L], h=0.1 — Float64, N=10")
    println("=" ^ 110)

    N = 10
    print_header()

    for L_end in [4, 6, 8, 10, 14, 20]
        grid, f, a_true, _ = make_signal(Float64, N, 1.0, Float64(L_end), 0.1)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all_methods(grid, f, N, a_true)
        print_row("L=$L_end", M, errs)
    end
end

# ================================================================
#  Experiment 4: Per-component accuracy
# ================================================================

function exp4()
    println("\n" * "=" ^ 110)
    println("  EXP 4: Per-component accuracy — domain=[1,5], h=0.01, Float64, N=10")
    println("=" ^ 110)

    N = 10
    grid, f, a_true, _ = make_signal(Float64, N, 1.0, 5.0, 0.01)
    M = length(grid)
    L3n = recommend_L(M, N)

    methods = [("std(L=3N)", () -> matrix_pencil(grid, f, N; pencil_L=L3n)),
               ("delayed", () -> matrix_pencil_delayed(grid, f, N)),
               ("prefilter", () -> matrix_pencil_prefilter(grid, f, N)),
               ("delta(L=M/2)", () -> matrix_pencil_delta(grid, f, N)),
               ("delta(L=3N)", () -> matrix_pencil_delta(grid, f, N; pencil_L=L3n)),
               ("delta+delay", () -> matrix_pencil_delta_delayed(grid, f, N))]

    for (name, fn) in methods
        res = safe_run(fn)
        if res === nothing
            println("\n  $name: FAILED")
            continue
        end
        println("\n  $name:")
        for k in 1:min(N, length(res.a))
            err = abs(Float64(res.a[k]) - Float64(a_true[k]))
            @printf("    a[%2d] = %12.8f  (true=%4.1f, err=%.2e)\n",
                    k, Float64(res.a[k]), Float64(a_true[k]), err)
        end
        @printf("    RMSE = %.3e\n", Float64(rmse(res, grid, f)))
    end
end

# ================================================================
#  Experiment 5: Extreme dense grid — domain=[0,20], N=5
# ================================================================

function exp5()
    println("\n" * "=" ^ 110)
    println("  EXP 5: Extreme dense — domain=[0,20], N=5, Float64")
    println("=" ^ 110)

    N = 5
    print_header()

    for h_val in [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
        grid, f, a_true, _ = make_signal(Float64, N, 0.0, 20.0, h_val)
        M = length(grid)
        M < 2N + 1 && continue
        M > 200_001 && continue
        errs = run_all_methods(grid, f, N, a_true)
        print_row(string(h_val), M, errs)
    end
end

# ================================================================
#  Experiment 6: BigFloat per-component — domain=[1,5], h=0.1
# ================================================================

function exp6()
    println("\n" * "=" ^ 110)
    println("  EXP 6: Per-component — domain=[1,5], h=0.1, BigFloat(256), N=10")
    println("=" ^ 110)

    setprecision(BigFloat, 256)
    N = 10
    grid, f, a_true, _ = make_signal(BigFloat, N, big"1.0", big"5.0", big"0.1")
    M = length(grid)
    L3n = recommend_L(M, N)

    methods = [("std(L=3N)", () -> matrix_pencil(grid, f, N; pencil_L=L3n)),
               ("delayed", () -> matrix_pencil_delayed(grid, f, N)),
               ("prefilter", () -> matrix_pencil_prefilter(grid, f, N)),
               ("delta(L=M/2)", () -> matrix_pencil_delta(grid, f, N)),
               ("delta(L=3N)", () -> matrix_pencil_delta(grid, f, N; pencil_L=L3n)),
               ("delta+delay", () -> matrix_pencil_delta_delayed(grid, f, N))]

    for (name, fn) in methods
        res = safe_run(fn)
        if res === nothing
            println("\n  $name: FAILED")
            continue
        end
        println("\n  $name:")
        for k in 1:min(N, length(res.a))
            err = abs(Float64(res.a[k]) - Float64(a_true[k]))
            @printf("    a[%2d] = %12.8f  (true=%4.1f, err=%.2e)\n",
                    k, Float64(res.a[k]), Float64(a_true[k]), err)
        end
        @printf("    RMSE = %.3e\n", Float64(rmse(res, grid, f)))
    end
end

# ================================================================
println("=" ^ 110)
println("  Part 6: Pre-filtering & Delta-Operator Matrix Pencil")
println("=" ^ 110)

exp1()
exp2()
exp3()
exp4()
exp5()
exp6()

println("\n" * "=" ^ 110)
println("  ALL DONE")
println("=" ^ 110)
