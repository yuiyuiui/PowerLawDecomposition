"""
Part 7: Improving Delta+Delay for universal superiority

Compare 10 methods (6 from Part 6 + 4 new):
  1. std(L=3N)       — standard Matrix Pencil, L=3N
  2. delayed          — stride-k delayed Hankel
  3. prefilter        — moving average + subsample (shift operator)
  4. delta(M/2)       — delta operator, L=M/2
  5. delta+delay      — delta + stride-k (Part 6)
  6. bilinear         — bilinear/Cayley transform + delayed [NEW]
  7. prefilt+delta    — prefilter then delta operator [NEW]
  8. delta_auto       — CV over (stride, L) for delta [NEW]
  9. universal        — CV over ALL method families [NEW]
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
    catch e
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
    push!(results, let r = safe_run(() -> matrix_pencil_delta_delayed(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> matrix_pencil_bilinear(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> matrix_pencil_prefilter_bilinear(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> matrix_pencil_universal(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)

    return results
end

const METHOD_NAMES = ["std(L=3N)", "delayed", "prefilter", "delta(M/2)",
                      "delt+dly", "bilinear", "pf+bilin", "universl"]

function print_header()
    @printf("  %-8s | %5s", "param", "M")
    for name in METHOD_NAMES
        @printf(" | %-11s", name)
    end
    println()
    return println("  " * "-" ^ (20 + 14 * length(METHOD_NAMES)))
end

function print_row(label, M, errs)
    @printf("  %-8s | %5d", label, M)
    for e in errs
        @printf(" | %.2e   ", e)
    end
    return println()
end

function highlight_best(errs)
    valid = [i for i in eachindex(errs) if !isnan(errs[i]) && !isinf(errs[i])]
    isempty(valid) && return 0
    return valid[argmin(errs[valid])]
end

# ================================================================
#  EXP 1: Fixed domain [1,5], varying h — Float64, N=10
# ================================================================
function exp1()
    println("\n" * "=" ^ 150)
    println("  EXP 1: domain=[1,5], varying h — Float64, N=10")
    println("=" ^ 150)

    N = 10
    print_header()

    for h_val in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        grid, f, a_true, _ = make_signal(Float64, N, 1.0, 5.0, h_val)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all_methods(grid, f, N, a_true)
        best = highlight_best(errs)
        print_row(string(h_val), M, errs)
        best > 0 && @printf("    ↑ best: %s (%.2e)\n", METHOD_NAMES[best], errs[best])
    end
end

# ================================================================
#  EXP 2: Varying domain [1, L_end], h=0.1 — Float64, N=10
# ================================================================
function exp2()
    println("\n" * "=" ^ 150)
    println("  EXP 2: varying domain [1,L], h=0.1 — Float64, N=10")
    println("=" ^ 150)

    N = 10
    print_header()

    for L_end in [4, 6, 8, 10, 14, 20]
        grid, f, a_true, _ = make_signal(Float64, N, 1.0, Float64(L_end), 0.1)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all_methods(grid, f, N, a_true)
        best = highlight_best(errs)
        print_row("L=$L_end", M, errs)
        best > 0 && @printf("    ↑ best: %s (%.2e)\n", METHOD_NAMES[best], errs[best])
    end
end

# ================================================================
#  EXP 3: Per-component accuracy — [1,5], h=0.01, N=10
# ================================================================
function exp3()
    println("\n" * "=" ^ 150)
    println("  EXP 3: Per-component accuracy — domain=[1,5], h=0.01, Float64, N=10")
    println("=" ^ 150)

    N = 10
    grid, f, a_true, _ = make_signal(Float64, N, 1.0, 5.0, 0.01)
    M = length(grid)
    L3n = recommend_L(M, N)

    methods = [("std(L=3N)", () -> matrix_pencil(grid, f, N; pencil_L=L3n)),
               ("delayed", () -> matrix_pencil_delayed(grid, f, N)),
               ("prefilter", () -> matrix_pencil_prefilter(grid, f, N)),
               ("delta(M/2)", () -> matrix_pencil_delta(grid, f, N)),
               ("delta+delay", () -> matrix_pencil_delta_delayed(grid, f, N)),
               ("bilinear", () -> matrix_pencil_bilinear(grid, f, N)),
               ("pf+bilinear", () -> matrix_pencil_prefilter_bilinear(grid, f, N)),
               ("universal", () -> matrix_pencil_universal(grid, f, N))]

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
        @printf("    max|err| = %.3e,  RMSE = %.3e\n",
                max_a_err(res, a_true), Float64(rmse(res, grid, f)))
    end
end

# ================================================================
#  EXP 4: Extreme dense grid — domain=[0,20], N=5
# ================================================================
function exp4()
    println("\n" * "=" ^ 150)
    println("  EXP 4: Extreme dense — domain=[0,20], N=5, Float64")
    println("=" ^ 150)

    N = 5
    print_header()

    for h_val in [1.0, 0.1, 0.01, 0.005]
        grid, f, a_true, _ = make_signal(Float64, N, 0.0, 20.0, h_val)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all_methods(grid, f, N, a_true)
        best = highlight_best(errs)
        print_row(string(h_val), M, errs)
        best > 0 && @printf("    ↑ best: %s (%.2e)\n", METHOD_NAMES[best], errs[best])
    end
end

# ================================================================
#  EXP 5: Per-component at sparse grid — [1,5], h=0.1, N=10
# ================================================================
function exp5()
    println("\n" * "=" ^ 150)
    println("  EXP 5: Per-component — domain=[1,5], h=0.1, Float64, N=10 (sparse grid)")
    println("=" ^ 150)

    N = 10
    grid, f, a_true, _ = make_signal(Float64, N, 1.0, 5.0, 0.1)
    M = length(grid)

    methods = [("std(L=3N)", () -> matrix_pencil(grid, f, N; pencil_L=recommend_L(M, N))),
               ("delta(M/2)", () -> matrix_pencil_delta(grid, f, N)),
               ("delta+delay", () -> matrix_pencil_delta_delayed(grid, f, N)),
               ("bilinear", () -> matrix_pencil_bilinear(grid, f, N)),
               ("pf+bilinear", () -> matrix_pencil_prefilter_bilinear(grid, f, N)),
               ("universal", () -> matrix_pencil_universal(grid, f, N))]

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
        @printf("    max|err| = %.3e,  RMSE = %.3e\n",
                max_a_err(res, a_true), Float64(rmse(res, grid, f)))
    end
end

# ================================================================
#  EXP 6: Varying domain [1,L], h=0.05 — Float64, N=10
# ================================================================
function exp6()
    println("\n" * "=" ^ 150)
    println("  EXP 6: varying domain [1,L], h=0.05 — Float64, N=10")
    println("=" ^ 150)

    N = 10
    print_header()

    for L_end in [4, 6, 8, 10, 14, 20]
        grid, f, a_true, _ = make_signal(Float64, N, 1.0, Float64(L_end), 0.05)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all_methods(grid, f, N, a_true)
        best = highlight_best(errs)
        print_row("L=$L_end", M, errs)
        best > 0 && @printf("    ↑ best: %s (%.2e)\n", METHOD_NAMES[best], errs[best])
    end
end

# ================================================================
#  EXP 7: Fewer components N=5 on [1,5]
# ================================================================
function exp7()
    println("\n" * "=" ^ 150)
    println("  EXP 7: domain=[1,5], N=5, varying h — Float64")
    println("=" ^ 150)

    N = 5
    print_header()

    for h_val in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        grid, f, a_true, _ = make_signal(Float64, N, 1.0, 5.0, h_val)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all_methods(grid, f, N, a_true)
        best = highlight_best(errs)
        print_row(string(h_val), M, errs)
        best > 0 && @printf("    ↑ best: %s (%.2e)\n", METHOD_NAMES[best], errs[best])
    end
end

# ================================================================
#  EXP 8: Per-component accuracy — [1,5], h=0.2, N=10 (very sparse)
# ================================================================
function exp8()
    println("\n" * "=" ^ 150)
    println("  EXP 8: Per-component — domain=[1,5], h=0.2, Float64, N=10 (very sparse)")
    println("=" ^ 150)

    N = 10
    grid, f, a_true, _ = make_signal(Float64, N, 1.0, 5.0, 0.2)
    M = length(grid)

    methods = [("std(L=3N)", () -> matrix_pencil(grid, f, N; pencil_L=recommend_L(M, N))),
               ("delta+delay", () -> matrix_pencil_delta_delayed(grid, f, N)),
               ("bilinear", () -> matrix_pencil_bilinear(grid, f, N)),
               ("pf+bilinear", () -> matrix_pencil_prefilter_bilinear(grid, f, N)),
               ("universal", () -> matrix_pencil_universal(grid, f, N))]

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
        @printf("    max|err| = %.3e,  RMSE = %.3e\n",
                max_a_err(res, a_true), Float64(rmse(res, grid, f)))
    end
end

# ================================================================
println("=" ^ 150)
println("  Part 7: Bilinear Transform + Universal Adaptive Selection")
println("=" ^ 150)

exp1()
exp2()
exp3()
exp4()
exp5()
exp6()
exp7()
exp8()

println("\n" * "=" ^ 150)
println("  ALL DONE")
println("=" ^ 150)
