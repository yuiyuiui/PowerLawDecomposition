"""
Part 8: Wynn Epsilon for exponential sum decomposition — comparison with Matrix Pencil

The Wynn epsilon algorithm accelerates convergence of sequences. For
f(x) = Σ cₙ exp(-aₙ x), the ratio r_k = f(x_{k+1})/f(x_k) converges
geometrically to z₁ = exp(-a₁ h) (the slowest-decaying component).
Wynn epsilon can dramatically speed up this convergence.

Strategy: iterative peeling
  1. Form ratio sequence r_k from signal tail (where convergence is best)
  2. Apply Wynn ε to get z₁ → extract a₁
  3. Estimate c₁ via least-squares; subtract c₁ exp(-a₁ x)
  4. Repeat on residual for a₂, a₃, ...
"""

include("hankel.jl")
include("../src/math/sequence.jl")
using Printf, LinearAlgebra

# ════════════════════════════════════════════════════════════
#  Wynn-based exponential decomposition
# ════════════════════════════════════════════════════════════

"""
    wynn_peel(grid, f, N) -> ExponentialSum

Extract N exponential components by iterated Wynn-epsilon peeling.
Extracts slowest-decaying component first (from the signal tail),
subtracts it, and repeats.
"""
function wynn_peel(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                   n_wynn::Int=0) where {T<:Real}
    M = length(f)
    h = grid[2] - grid[1]

    remaining = copy(f)
    a_vals = T[]

    for iter in 1:N
        npts = n_wynn > 0 ? n_wynn : min(2 * (N - iter + 1) + 5, M - 1)
        npts = min(npts, M - 1)

        ratios = remaining[2:end] ./ remaining[1:(end - 1)]
        valid = findall(isfinite.(ratios) .& (abs.(ratios) .> eps(T)))
        isempty(valid) && break

        start_idx = max(valid[end] - npts + 1, valid[1])
        sub = ratios[start_idx:valid[end]]
        length(sub) < 3 && break
        if length(sub) % 2 == 0
            sub = sub[2:end]
        end

        z_est = wynn_epsilon_core(sub)

        if !isfinite(z_est) || abs(z_est) < eps(T) || abs(z_est) > one(T) + T(0.1)
            z_est = T(real(sub[end]))
        end

        a_est = -log(abs(z_est)) / h
        a_est < zero(T) && (a_est = zero(T))
        push!(a_vals, a_est)

        basis = T[exp(-a_est * x) for x in grid]
        c_est = dot(remaining, basis) / dot(basis, basis)
        remaining = remaining - c_est * basis
    end

    length(a_vals) < N && append!(a_vals, fill(a_vals[end], N - length(a_vals)))

    a_vec = T.(a_vals)
    c_vec = _solve_amplitudes(grid, f, a_vec)
    return _pack(c_vec, a_vec)
end

"""
    wynn_peel_both_ends(grid, f, N) -> ExponentialSum

Extract components from both ends of the signal:
  - Tail → slowest decay (a₁, a₂, ...)
  - Head → fastest decay (aₙ, aₙ₋₁, ...)
Then merge and refine via least-squares.
"""
function wynn_peel_both_ends(grid::AbstractVector{T}, f::AbstractVector{T},
                             N::Int) where {T<:Real}
    M = length(f)
    h = grid[2] - grid[1]

    n_slow = div(N + 1, 2)
    n_fast = N - n_slow

    a_slow = T[]
    remaining = copy(f)

    for iter in 1:n_slow
        npts = min(2 * (N - iter + 1) + 5, M - 1)
        ratios = remaining[2:end] ./ remaining[1:(end - 1)]
        valid = findall(isfinite.(ratios) .& (abs.(ratios) .> eps(T)))
        isempty(valid) && break

        start_idx = max(valid[end] - npts + 1, valid[1])
        sub = ratios[start_idx:valid[end]]
        length(sub) < 3 && break
        if length(sub) % 2 == 0
            ;
            sub = sub[2:end];
        end

        z_est = wynn_epsilon_core(sub)
        if !isfinite(z_est) || abs(z_est) < eps(T) || abs(z_est) > one(T) + T(0.1)
            z_est = T(real(sub[end]))
        end
        a_est = max(zero(T), -log(abs(z_est)) / h)
        push!(a_slow, a_est)

        basis = T[exp(-a_est * x) for x in grid]
        c_est = dot(remaining, basis) / dot(basis, basis)
        remaining = remaining - c_est * basis
    end

    a_fast = T[]
    remaining2 = copy(f)
    for a in a_slow
        basis = T[exp(-a * x) for x in grid]
        c = dot(remaining2, basis) / dot(basis, basis)
        remaining2 = remaining2 - c * basis
    end

    for iter in 1:n_fast
        npts = min(2 * (n_fast - iter + 1) + 5, M - 1)
        ratios = remaining2[2:end] ./ remaining2[1:(end - 1)]
        valid = findall(isfinite.(ratios) .& (abs.(ratios) .> eps(T)))
        isempty(valid) && break

        sub = ratios[valid[1]:min(valid[1] + npts - 1, valid[end])]
        length(sub) < 3 && break
        if length(sub) % 2 == 0
            ;
            sub = sub[2:end];
        end

        z_est = wynn_epsilon_core(sub)
        if !isfinite(z_est) || abs(z_est) < eps(T)
            z_est = T(real(sub[1]))
        end
        a_est = max(zero(T), -log(abs(z_est)) / h)
        push!(a_fast, a_est)

        basis = T[exp(-a_est * x) for x in grid]
        c_est = dot(remaining2, basis) / dot(basis, basis)
        remaining2 = remaining2 - c_est * basis
    end

    all_a = vcat(a_slow, a_fast)
    if length(all_a) < N
        append!(all_a, fill(all_a[end], N - length(all_a)))
    end
    a_vec = T.(all_a)
    c_vec = _solve_amplitudes(grid, f, a_vec)
    return _pack(c_vec, a_vec)
end

"""
    wynn_ratio_grid(grid, f, N) -> ExponentialSum

Use Wynn epsilon on ratio sequences at multiple grid offsets,
then cluster the results to identify N distinct decay rates.
"""
function wynn_ratio_grid(grid::AbstractVector{T}, f::AbstractVector{T},
                         N::Int) where {T<:Real}
    M = length(f)
    h = grid[2] - grid[1]

    ratios = f[2:end] ./ f[1:(end - 1)]
    valid = findall(isfinite.(ratios) .& (abs.(ratios) .> eps(T)) .& (ratios .> zero(T)))
    length(valid) < 3 && return wynn_peel(grid, f, N)

    a_estimates = T[]
    for window_size in [2N + 1, 2N + 3, 2N + 5, min(4N + 1, length(valid))]
        window_size > length(valid) && continue
        ws = window_size % 2 == 0 ? window_size - 1 : window_size
        ws < 3 && continue

        for start in [valid[1], max(valid[1], valid[end] - ws + 1),
                      max(valid[1], div(valid[1] + valid[end] - ws + 1, 2))]
            start < valid[1] && continue
            stop = min(start + ws - 1, valid[end])
            sub = ratios[start:stop]
            if length(sub) % 2 == 0
                ;
                sub = sub[1:(end - 1)];
            end
            length(sub) < 3 && continue

            z_est = wynn_epsilon_core(sub)
            if isfinite(z_est) && abs(z_est) > eps(T) && z_est > zero(T) &&
               z_est <= one(T) + T(0.01)
                a_est = -log(z_est) / h
                if a_est > zero(T)
                    push!(a_estimates, a_est)
                end
            end
        end
    end

    isempty(a_estimates) && return wynn_peel(grid, f, N)

    sort!(a_estimates)
    clusters = T[]
    tol = T(0.1)
    i = 1
    while i <= length(a_estimates) && length(clusters) < N
        cluster = [a_estimates[i]]
        j = i + 1
        while j <= length(a_estimates) && abs(a_estimates[j] - a_estimates[i]) < tol
            push!(cluster, a_estimates[j])
            j += 1
        end
        push!(clusters, mean(cluster))
        i = j
    end

    if length(clusters) < N
        return wynn_peel(grid, f, N)
    end

    a_vec = T.(clusters[1:N])
    c_vec = _solve_amplitudes(grid, f, a_vec)
    return _pack(c_vec, a_vec)
end

function mean(v)
    return sum(v) / length(v)
end

# ════════════════════════════════════════════════════════════
#  Helpers (reused from Part 7)
# ════════════════════════════════════════════════════════════

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
        ; return fn();
    catch
        ; return nothing;
    end
end

function recommend_L(M, N)
    return clamp(min(3 * N, div(M, 3)), N, M - N - 1)
end

# ════════════════════════════════════════════════════════════
#  Experiments
# ════════════════════════════════════════════════════════════

const METHOD_NAMES = ["std(L=3N)", "delayed", "prefilter", "delt+dly",
                      "bilinear", "pf+bilin",
                      "wynn_peel", "wynn_2end", "wynn_grid"]

function run_all(grid::AbstractVector{T}, f::AbstractVector{T}, N, a_true) where {T}
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
    push!(results, let r = safe_run(() -> matrix_pencil_delta_delayed(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> matrix_pencil_bilinear(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> matrix_pencil_prefilter_bilinear(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> wynn_peel(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> wynn_peel_both_ends(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)
    push!(results, let r = safe_run(() -> wynn_ratio_grid(grid, f, N))
              r !== nothing ? max_a_err(r, a_true) : NaN
          end)

    return results
end

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
    N = 10;
    print_header()
    for h_val in [0.2, 0.1, 0.05, 0.02, 0.01]
        grid, f, a_true, _ = make_signal(Float64, N, 1.0, 5.0, h_val)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all(grid, f, N, a_true)
        best = highlight_best(errs)
        print_row(string(h_val), M, errs)
        best > 0 && @printf("    ↑ best: %s (%.2e)\n", METHOD_NAMES[best], errs[best])
    end
end

# ================================================================
#  EXP 2: Varying domain [1,L], h=0.1 — Float64, N=10
# ================================================================
function exp2()
    println("\n" * "=" ^ 150)
    println("  EXP 2: varying domain [1,L], h=0.1 — Float64, N=10")
    println("=" ^ 150)
    N = 10;
    print_header()
    for L_end in [4, 6, 8, 10, 14, 20]
        grid, f, a_true, _ = make_signal(Float64, N, 1.0, Float64(L_end), 0.1)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all(grid, f, N, a_true)
        best = highlight_best(errs)
        print_row("L=$L_end", M, errs)
        best > 0 && @printf("    ↑ best: %s (%.2e)\n", METHOD_NAMES[best], errs[best])
    end
end

# ================================================================
#  EXP 3: Per-component accuracy — [1,5], h=0.1, N=10
# ================================================================
function exp3()
    println("\n" * "=" ^ 150)
    println("  EXP 3: Per-component — domain=[1,5], h=0.1, Float64, N=10")
    println("=" ^ 150)
    N = 10
    grid, f, a_true, _ = make_signal(Float64, N, 1.0, 5.0, 0.1)
    M = length(grid)
    L3n = recommend_L(M, N)

    methods = [("std(L=3N)", () -> matrix_pencil(grid, f, N; pencil_L=L3n)),
               ("delt+dly", () -> matrix_pencil_delta_delayed(grid, f, N)),
               ("bilinear", () -> matrix_pencil_bilinear(grid, f, N)),
               ("wynn_peel", () -> wynn_peel(grid, f, N)),
               ("wynn_2end", () -> wynn_peel_both_ends(grid, f, N)),
               ("wynn_grid", () -> wynn_ratio_grid(grid, f, N))]

    for (name, fn) in methods
        res = safe_run(fn)
        if res === nothing
            println("\n  $name: FAILED");
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
#  EXP 4: Easy case — [0,20], N=5
# ================================================================
function exp4()
    println("\n" * "=" ^ 150)
    println("  EXP 4: Easy — domain=[0,20], N=5, Float64")
    println("=" ^ 150)
    N = 5;
    print_header()
    for h_val in [1.0, 0.1, 0.01]
        grid, f, a_true, _ = make_signal(Float64, N, 0.0, 20.0, h_val)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all(grid, f, N, a_true)
        best = highlight_best(errs)
        print_row(string(h_val), M, errs)
        best > 0 && @printf("    ↑ best: %s (%.2e)\n", METHOD_NAMES[best], errs[best])
    end
end

# ================================================================
#  EXP 5: Per-component on easy case — [0,20], h=0.1, N=5
# ================================================================
function exp5()
    println("\n" * "=" ^ 150)
    println("  EXP 5: Per-component — domain=[0,20], h=0.1, Float64, N=5")
    println("=" ^ 150)
    N = 5
    grid, f, a_true, _ = make_signal(Float64, N, 0.0, 20.0, 0.1)
    M = length(grid)

    methods = [("std(L=3N)", () -> matrix_pencil(grid, f, N; pencil_L=recommend_L(M, N))),
               ("delayed", () -> matrix_pencil_delayed(grid, f, N)),
               ("bilinear", () -> matrix_pencil_bilinear(grid, f, N)),
               ("wynn_peel", () -> wynn_peel(grid, f, N)),
               ("wynn_2end", () -> wynn_peel_both_ends(grid, f, N)),
               ("wynn_grid", () -> wynn_ratio_grid(grid, f, N))]

    for (name, fn) in methods
        res = safe_run(fn)
        if res === nothing
            println("\n  $name: FAILED");
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
#  EXP 6: N=5 on [1,5]
# ================================================================
function exp6()
    println("\n" * "=" ^ 150)
    println("  EXP 6: domain=[1,5], N=5, varying h — Float64")
    println("=" ^ 150)
    N = 5;
    print_header()
    for h_val in [0.2, 0.1, 0.05, 0.02, 0.01]
        grid, f, a_true, _ = make_signal(Float64, N, 1.0, 5.0, h_val)
        M = length(grid)
        M < 2N + 1 && continue
        errs = run_all(grid, f, N, a_true)
        best = highlight_best(errs)
        print_row(string(h_val), M, errs)
        best > 0 && @printf("    ↑ best: %s (%.2e)\n", METHOD_NAMES[best], errs[best])
    end
end

# ================================================================
#  EXP 7: BigFloat validation — [0,20], h=0.1, N=5
# ================================================================
function exp7()
    println("\n" * "=" ^ 150)
    println("  EXP 7: BigFloat(256) — domain=[0,20], h=0.1, N=5")
    println("=" ^ 150)
    setprecision(BigFloat, 256)
    N = 5
    grid, f, a_true, _ = make_signal(BigFloat, N, big"0.0", big"20.0", big"0.1")
    M = length(grid)

    methods = [("std(L=3N)", () -> matrix_pencil(grid, f, N; pencil_L=recommend_L(M, N))),
               ("wynn_peel", () -> wynn_peel(grid, f, N)),
               ("wynn_2end", () -> wynn_peel_both_ends(grid, f, N))]

    for (name, fn) in methods
        res = safe_run(fn)
        if res === nothing
            println("\n  $name: FAILED");
            continue
        end
        println("\n  $name:")
        for k in 1:min(N, length(res.a))
            err = abs(Float64(res.a[k]) - Float64(a_true[k]))
            @printf("    a[%2d]  err=%.2e\n", k, err)
        end
        @printf("    max|err| = %.3e,  RMSE = %.3e\n",
                max_a_err(res, a_true), Float64(rmse(res, grid, f)))
    end
end

# ================================================================
println("=" ^ 150)
println("  Part 8: Wynn Epsilon vs Matrix Pencil")
println("=" ^ 150)

exp1()
exp2()
exp3()
exp4()
exp5()
exp6()
exp7()

println("\n" * "=" ^ 150)
println("  ALL DONE")
println("=" ^ 150)
