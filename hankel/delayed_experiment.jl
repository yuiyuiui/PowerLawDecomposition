"""
Part 4: Delayed Hankel Matrix Pencil 实验

对比方法：
  1. matrix_pencil (default L=M/2)          — Part 1 基线
  2. matrix_pencil (L=3N)                    — Part 3 理论 L
  3. matrix_pencil_subsample (target_M=8N)   — Part 3 降采样
  4. matrix_pencil_delayed (auto stride)     — Part 4 延迟 Hankel

测试信号: f(x) = Σ exp(-n·0.5·x), n=1..N
"""

include("hankel.jl")
using Printf, Statistics, Dates

# ================================================================
#  辅助函数
# ================================================================

function make_test_signal(::Type{T}, N, L0, L_end, h) where {T}
    a_true = T.(collect(1:N) .* 0.5)
    c_true = ones(T, N)
    M = round(Int, Float64(L_end - L0) / Float64(h)) + 1
    grid = [T(L0) + T(h) * i for i in 0:(M-1)]
    f = [sum(c_true .* exp.(-a_true .* x)) for x in grid]
    return grid, f, a_true, c_true
end

function max_a_error(res, a_true)
    N = length(a_true)
    n_res = length(res.a)
    if n_res == N
        return maximum(abs.(Float64.(res.a) .- Float64.(a_true)))
    else
        errs = Float64[]
        for k in 1:n_res
            best = minimum(abs.(Float64(res.a[k]) .- Float64.(a_true)))
            push!(errs, best)
        end
        return isempty(errs) ? Inf : maximum(errs)
    end
end

function recommend_L_theory(M, N)
    L = min(3 * N, div(M, 3))
    L = clamp(L, N, M - N - 1)
    return L
end

function matrix_pencil_subsample(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                                 target_M::Int=0) where {T}
    M = length(grid)
    if target_M <= 0
        target_M = max(8 * N, 2 * N + 1)
    end
    target_M = min(target_M, M)
    if M <= target_M
        L = recommend_L_theory(M, N)
        return matrix_pencil(grid, f, N; pencil_L=L)
    end
    stride = M ÷ target_M
    stride = max(stride, 1)
    indices = 1:stride:M
    grid_sub = grid[indices]
    f_sub = f[indices]
    actual_M = length(indices)
    L = recommend_L_theory(actual_M, N)
    res_sub = matrix_pencil(grid_sub, f_sub, N; pencil_L=L)
    c_refined = _solve_amplitudes(grid, f, res_sub.a)
    return _pack(c_refined, res_sub.a)
end

function safe_run(func)
    try
        return func()
    catch e
        return nothing
    end
end

# ================================================================
#  实验 1: 核心对比 — 固定域，变化 h
# ================================================================

function experiment_core_comparison()
    println("\n" * "=" ^ 90)
    println("  实验 1: 核心对比 — 固定域，变化 h，四种方法精度对比")
    println("=" ^ 90)

    N = 10

    for (L0, L_end, label) in [(1.0, 5.0, "[1,5]"), (0.0, 20.0, "[0,20]")]
        println("\n  域 $label, N=$N, f(x)=Σexp(-0.5n·x)")
        println("  " * "-" ^ 88)
        @printf("  %-6s | %4s | %-14s | %-14s | %-14s | %-14s\n",
                "h", "M", "default(L=M/2)", "theory(L=3N)", "subsample(8N)", "delayed(auto)")
        println("  " * "-" ^ 88)

        for h_val in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
            grid, f, a_true, _ = make_test_signal(Float64, N, L0, L_end, h_val)
            M = length(grid)
            M < 2 * N + 1 && continue

            L_half = div(M - 1, 2)
            L_3n = recommend_L_theory(M, N)

            err_default = let r = safe_run(() -> matrix_pencil(grid, f, N; pencil_L=L_half))
                r !== nothing ? max_a_error(r, a_true) : NaN
            end

            err_3n = let r = safe_run(() -> matrix_pencil(grid, f, N; pencil_L=L_3n))
                r !== nothing ? max_a_error(r, a_true) : NaN
            end

            err_sub = let r = safe_run(() -> matrix_pencil_subsample(grid, f, N))
                r !== nothing ? max_a_error(r, a_true) : NaN
            end

            err_delayed = let r = safe_run(() -> matrix_pencil_delayed(grid, f, N))
                r !== nothing ? max_a_error(r, a_true) : NaN
            end

            @printf("  %-6.3f | %4d | %.3e      | %.3e      | %.3e      | %.3e\n",
                    h_val, M, err_default, err_3n, err_sub, err_delayed)
        end
    end
end

# ================================================================
#  实验 2: Delayed Hankel 的 stride 扫描
# ================================================================

function experiment_stride_scan()
    println("\n" * "=" ^ 90)
    println("  实验 2: Delayed Hankel — stride 扫描 (L=3N 固定)")
    println("=" ^ 90)

    N = 10

    for (L0, L_end, h_val, label) in [
        (1.0, 5.0, 0.01, "[1,5] h=0.01 M=401"),
        (1.0, 5.0, 0.005, "[1,5] h=0.005 M=801"),
        (0.0, 20.0, 0.01, "[0,20] h=0.01 M=2001"),
        (0.0, 20.0, 0.1, "[0,20] h=0.1 M=201"),
    ]
        grid, f, a_true, _ = make_test_signal(Float64, N, L0, L_end, h_val)
        M = length(grid)
        h = Float64(grid[2] - grid[1])
        L = min(3 * N, div(M, 3))

        println("\n  $label (M=$M)")
        println("  stride | h_eff   | nrow  | max|Δa|         | 备注")
        println("  " * "-" ^ 70)

        for k in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]
            nrow = M - k * L
            nrow < N + 1 && continue
            h_eff = k * h

            res = safe_run(() -> matrix_pencil_delayed(grid, f, N; stride=k, pencil_L=L))
            err = res !== nothing ? max_a_error(res, a_true) : NaN

            note = ""
            if k == 1; note = "(标准 Hankel)"; end

            res_auto = safe_run(() -> matrix_pencil_delayed(grid, f, N))
            auto_stride = max(1, round(Int, (Float64(grid[end]) - Float64(grid[1])) / (5 * N) / h))
            if k == auto_stride; note *= " ← auto"; end

            @printf("  %4d   | %-7.4f | %5d | %.3e       | %s\n",
                    k, h_eff, nrow, err, note)
        end

        res_auto = safe_run(() -> matrix_pencil_delayed(grid, f, N))
        err_auto = res_auto !== nothing ? max_a_error(res_auto, a_true) : NaN
        auto_stride = max(1, round(Int, (Float64(grid[end]) - Float64(grid[1])) / (5 * N) / h))
        @printf("  auto(%d) — max|Δa| = %.3e\n", auto_stride, err_auto)

        res_sub = safe_run(() -> matrix_pencil_subsample(grid, f, N))
        err_sub = res_sub !== nothing ? max_a_error(res_sub, a_true) : NaN
        @printf("  subsample(Part3) — max|Δa| = %.3e\n", err_sub)
    end
end

# ================================================================
#  实验 3: 行数效应 — delayed vs subsample 的本质差异
# ================================================================

function experiment_row_comparison()
    println("\n" * "=" ^ 90)
    println("  实验 3: 行数效应 — 相同 h_eff 下 delayed vs subsample")
    println("=" ^ 90)

    N = 10
    L = 3 * N  # = 30

    for (L0, L_end, h_val, label) in [
        (1.0, 5.0, 0.005, "[1,5] h=0.005"),
        (0.0, 20.0, 0.01, "[0,20] h=0.01"),
    ]
        grid, f, a_true, _ = make_test_signal(Float64, N, L0, L_end, h_val)
        M = length(grid)
        h = Float64(grid[2] - grid[1])

        println("\n  $label, M=$M, L=$L")
        println("  stride | h_eff  | delayed_rows | subsamp_rows | err_delayed  | err_subsamp  | 行数比")
        println("  " * "-" ^ 90)

        for k in [2, 5, 10, 20, 50]
            h_eff = k * h
            nrow_delayed = M - k * L
            nrow_delayed < N + 1 && continue

            M_sub = length(1:k:M)
            nrow_sub = M_sub - L
            nrow_sub < N + 1 && continue

            res_del = safe_run(() -> matrix_pencil_delayed(grid, f, N; stride=k, pencil_L=L))
            err_del = res_del !== nothing ? max_a_error(res_del, a_true) : NaN

            indices = 1:k:M
            grid_sub = grid[indices]
            f_sub = f[indices]
            res_sub = safe_run(() -> matrix_pencil(grid_sub, f_sub, N; pencil_L=L))
            err_sub = res_sub !== nothing ? max_a_error(res_sub, a_true) : NaN

            ratio = nrow_delayed / nrow_sub

            @printf("  %4d   | %-6.3f | %5d        | %5d        | %.3e    | %.3e    | %.1fx\n",
                    k, h_eff, nrow_delayed, nrow_sub, err_del, err_sub, ratio)
        end
    end
end

# ================================================================
#  实验 4: 更多 N 值的鲁棒性测试
# ================================================================

function experiment_robustness()
    println("\n" * "=" ^ 90)
    println("  实验 4: 不同 N 值的鲁棒性测试")
    println("=" ^ 90)

    for (N, L0, L_end, h_val) in [
        (3, 1.0, 5.0, 0.01),
        (5, 1.0, 5.0, 0.01),
        (10, 1.0, 5.0, 0.01),
        (3, 0.0, 10.0, 0.01),
        (5, 0.0, 10.0, 0.01),
        (10, 0.0, 10.0, 0.01),
    ]
        grid, f, a_true, _ = make_test_signal(Float64, N, L0, L_end, h_val)
        M = length(grid)

        err_default = let r = safe_run(() -> matrix_pencil(grid, f, N))
            r !== nothing ? max_a_error(r, a_true) : NaN
        end
        err_sub = let r = safe_run(() -> matrix_pencil_subsample(grid, f, N))
            r !== nothing ? max_a_error(r, a_true) : NaN
        end
        err_del = let r = safe_run(() -> matrix_pencil_delayed(grid, f, N))
            r !== nothing ? max_a_error(r, a_true) : NaN
        end

        @printf("  N=%2d [%.0f,%.0f] h=%.3f M=%4d  | default: %.3e | subsample: %.3e | delayed: %.3e\n",
                N, L0, L_end, h_val, M, err_default, err_sub, err_del)
    end
end

# ================================================================
#  实验 5: delayed + 振幅精修 (用全部网格重新求 c)
# ================================================================

function experiment_delayed_with_refinement()
    println("\n" * "=" ^ 90)
    println("  实验 5: Delayed Hankel + 振幅精修")
    println("=" ^ 90)

    N = 10

    for (L0, L_end, h_val, label) in [
        (1.0, 5.0, 0.01, "[1,5] h=0.01"),
        (1.0, 5.0, 0.005, "[1,5] h=0.005"),
        (0.0, 20.0, 0.01, "[0,20] h=0.01"),
    ]
        grid, f, a_true, _ = make_test_signal(Float64, N, L0, L_end, h_val)
        M = length(grid)

        res_del = safe_run(() -> matrix_pencil_delayed(grid, f, N))
        if res_del !== nothing
            err_a = max_a_error(res_del, a_true)
            r_fit = rmse(res_del, grid, f)

            c_refined = _solve_amplitudes(grid, f, res_del.a)
            res_refined = _pack(c_refined, res_del.a)
            r_refined = rmse(res_refined, grid, f)

            @printf("  %-20s M=%4d | max|Δa|=%.3e | RMSE(原始)=%.3e | RMSE(精修)=%.3e\n",
                    label, M, err_a, Float64(r_fit), Float64(r_refined))
        end
    end
end

# ================================================================
#  实验 6: 最佳 stride 的自动选择策略对比
# ================================================================

function experiment_auto_stride()
    println("\n" * "=" ^ 90)
    println("  实验 6: stride 自适应策略")
    println("=" ^ 90)
    println("  策略A: h_target = domain/(5N), stride = h_target/h  (Gemini推荐)")
    println("  策略B: h_target = domain/(10N)")
    println("  策略C: stride = M/(5N)")
    println()

    N = 10

    @printf("  %-20s | %4s | %5s | %5s | %5s | %-12s | %-12s | %-12s | %-12s\n",
            "场景", "M", "k_A", "k_B", "k_C", "err_A", "err_B", "err_C", "err_穷举最优")
    println("  " * "-" ^ 120)

    for (L0, L_end, h_val) in [
        (1.0, 5.0, 0.1),
        (1.0, 5.0, 0.05),
        (1.0, 5.0, 0.01),
        (1.0, 5.0, 0.005),
        (0.0, 20.0, 0.1),
        (0.0, 20.0, 0.05),
        (0.0, 20.0, 0.01),
        (0.0, 20.0, 0.005),
    ]
        grid, f, a_true, _ = make_test_signal(Float64, N, L0, L_end, h_val)
        M = length(grid)
        h = Float64(grid[2] - grid[1])
        domain_len = Float64(L_end - L0)
        L = min(3 * N, div(M, 3))

        k_A = max(1, round(Int, domain_len / (5 * N) / h))
        k_B = max(1, round(Int, domain_len / (10 * N) / h))
        k_C = max(1, M ÷ (5 * N))

        for k_ref in [k_A, k_B, k_C]
            while M - k_ref * L < N + 1 && k_ref > 1
                k_ref -= 1
            end
        end

        err_A = let r = safe_run(() -> matrix_pencil_delayed(grid, f, N; stride=k_A, pencil_L=L))
            r !== nothing ? max_a_error(r, a_true) : NaN
        end
        err_B = let r = safe_run(() -> matrix_pencil_delayed(grid, f, N; stride=k_B, pencil_L=L))
            r !== nothing ? max_a_error(r, a_true) : NaN
        end
        err_C = let r = safe_run(() -> matrix_pencil_delayed(grid, f, N; stride=k_C, pencil_L=L))
            r !== nothing ? max_a_error(r, a_true) : NaN
        end

        best_err = Inf
        best_k = 1
        for k in 1:max(1, M ÷ (L + 1))
            nrow = M - k * L
            nrow < N + 1 && break
            r = safe_run(() -> matrix_pencil_delayed(grid, f, N; stride=k, pencil_L=L))
            r === nothing && continue
            e = max_a_error(r, a_true)
            if e < best_err
                best_err = e
                best_k = k
            end
        end

        label = @sprintf("[%.0f,%.0f] h=%g", L0, L_end, h_val)
        @printf("  %-20s | %4d | %5d | %5d | %5d | %.3e    | %.3e    | %.3e    | %.3e (k=%d)\n",
                label, M, k_A, k_B, k_C, err_A, err_B, err_C, best_err, best_k)
    end
end

# ================================================================
#  运行全部
# ================================================================

println("Part 4: Delayed Hankel Matrix Pencil 对比实验")
println("="^90)
println("时间: ", Dates.now())
println()

experiment_core_comparison()
experiment_stride_scan()
experiment_row_comparison()
experiment_robustness()
experiment_delayed_with_refinement()
experiment_auto_stride()

println("\n" * "=" ^ 90)
println("  全部实验完成")
println("=" ^ 90)
