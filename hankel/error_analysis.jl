"""
Matrix Pencil 方法误差来源分析

三个潜在误差源：
  1. 输入信号的机器精度限制（Float64 噪声地板）
  2. 算法本身的设计（矩阵铅笔法是否最优）
  3. SVD 求解中的 Float64 精度不足

通过对照实验定量分离各误差源的贡献。
"""

include("hankel.jl")
using GenericLinearAlgebra
using LinearAlgebra
using Printf, Dates

# ================================================================
#  实验 1: SVD 奇异值谱 — Float64 vs BigFloat
# ================================================================

function experiment_svd_spectrum()
    println("\n" * "=" ^ 80)
    println("  实验 1: SVD 奇异值谱分析")
    println("=" ^ 80)

    N = 10
    h_val = 1 // 10
    L0, L = 1, 5

    for T in [Float64, BigFloat]
        if T == BigFloat
            ;
            setprecision(BigFloat, 256);
        end

        d = T(1 // 2)
        a_vec = collect(1:N) .* d
        func(x) = sum(exp.(-a_vec .* x))

        grid = [T(L0) + T(h_val) * i for i in 0:((L - L0) * 10)]
        f = func.(grid)
        M = length(grid)
        Lp = div(M - 1, 2)

        nrow = M - Lp
        Y = zeros(T, nrow, Lp + 1)
        for i in 1:nrow, j in 1:(Lp + 1)
            Y[i, j] = f[i + j - 1]
        end
        Y0 = Y[:, 1:Lp]
        F = svd(Y0)

        println("\n--- $T (M=$M, L=$Lp, Y0: $(nrow)×$(Lp)) ---")
        println("  奇异值谱:")
        for k in 1:min(N + 2, length(F.S))
            ratio = k > 1 ? F.S[k-1] / F.S[k] : NaN
            @printf("    S[%2d] = %.6e    S[k-1]/S[k] = %.1e\n", k, Float64(F.S[k]), ratio)
        end
        @printf("  条件数 S[1]/S[%d] = %.2e\n", N, Float64(F.S[1] / F.S[N]))
    end
end

# ================================================================
#  实验 2: 误差源隔离 — 三组对照
# ================================================================

function experiment_error_isolation()
    println("\n" * "=" ^ 80)
    println("  实验 2: 误差源隔离")
    println("=" ^ 80)

    setprecision(BigFloat, 512)

    N = 10
    d = BigFloat(1 // 2)
    a_true = collect(1:N) .* d
    c_true = ones(BigFloat, N)

    h_val = BigFloat(1 // 10)
    L0 = BigFloat(1)
    grid_big = [L0 + h_val * i for i in 0:40]
    f_big = [sum(c_true .* exp.(-a_true .* x)) for x in grid_big]

    grid_f64 = Float64.(grid_big)
    f_f64 = Float64.(f_big)

    println("\n--- 场景 A: BigFloat 数据 + BigFloat 计算 (纯算法测试) ---")
    res_a = matrix_pencil(grid_big, f_big, N)
    err_a = maximum(abs.(Float64.(res_a.a .- a_true)))
    @printf("  max|Δa| = %.3e\n", err_a)
    println("  恢复的 a 值:")
    for k in 1:N
        @printf("    a[%2d] = true=%.1f  recovered=%.1f  error=%.2e\n",
                k, Float64(a_true[k]), Float64(res_a.a[k]),
                Float64(abs(res_a.a[k] - a_true[k])))
    end

    println("\n--- 场景 B: Float64 数据 → BigFloat 计算 (隔离输入噪声) ---")
    grid_b = BigFloat.(grid_f64)
    f_b = BigFloat.(f_f64)
    res_b = matrix_pencil(grid_b, f_b, N)
    err_b = maximum(abs.(Float64.(res_b.a .- a_true)))
    @printf("  max|Δa| = %.3e\n", err_b)
    println("  恢复的 a 值:")
    for k in 1:N
        @printf("    a[%2d] = true=%.1f  recovered=%.6f  error=%.2e\n",
                k, Float64(a_true[k]), Float64(res_b.a[k]),
                Float64(abs(res_b.a[k] - a_true[k])))
    end

    println("\n--- 场景 C: Float64 数据 + Float64 计算 (原始问题) ---")
    res_c = matrix_pencil(grid_f64, f_f64, N)
    err_c = maximum(abs.(Float64.(res_c.a .- a_true)))
    @printf("  max|Δa| = %.3e\n", err_c)
    println("  恢复的 a 值:")
    for k in 1:N
        @printf("    a[%2d] = true=%.1f  recovered=%.6f  error=%.2e\n",
                k, Float64(a_true[k]), Float64(res_c.a[k]),
                Float64(abs(res_c.a[k] - a_true[k])))
    end

    println("\n--- 归因分析 ---")
    @printf("  场景 A (纯算法):     max|Δa| = %.3e\n", err_a)
    @printf("  场景 B (输入噪声):   max|Δa| = %.3e\n", err_b)
    @printf("  场景 C (全Float64):  max|Δa| = %.3e\n", err_c)
    @printf("\n  Source 1 (输入精度):  B/A = %.1e × (B 相对 A 的劣化)\n", err_b / max(err_a, 1e-300))
    @printf("  Source 3 (SVD精度):   C/B = %.2f × (C 相对 B 的额外劣化)\n",
            err_c / max(err_b, 1e-300))
    println("\n  结论:")
    if err_b / max(err_a, 1e-300) > 1e6 && err_c / max(err_b, 1e-300) < 10
        println("  → Source 1 (输入精度) 是绝对主因。")
        println("    Float64 的机器精度 (~1e-16) 淹没了慢衰减分量的信息。")
        println("    SVD 的计算精度 (Source 3) 几乎不影响结果。")
    elseif err_c / max(err_b, 1e-300) > 100
        println("  → Source 3 (SVD精度) 是显著贡献因子。")
    else
        println("  → 需要进一步分析。")
    end

    return (err_a, err_b, err_c)
end

# ================================================================
#  实验 3: 网格参数敏感性
# ================================================================

function experiment_grid_sensitivity()
    println("\n" * "=" ^ 80)
    println("  实验 3: 网格参数对恢复精度的影响")
    println("=" ^ 80)

    setprecision(BigFloat, 256)

    N = 10
    d = BigFloat(1 // 2)
    a_true = collect(1:N) .* d

    # 3a: 变化域起点 L0 (固定 h=0.1, 域长度=4)
    println("\n--- 3a: 域起点 L0 的影响 (h=0.1, 域长度=4, Float64) ---")
    println("  L0    | M  | cond(Y0) | max|Δa| F64 | max|Δa| Big | S[N]/eps")
    println("  " * "-" ^ 75)

    for L0_val in [0, 0.5, 1, 2]
        h_val = 0.1
        M = round(Int, 4.0 / h_val) + 1
        grid_f64 = Float64[L0_val + h_val * i for i in 0:(M - 1)]
        f_f64 = Float64[sum(exp.(-Float64.(a_true) .* x)) for x in grid_f64]

        grid_big = BigFloat.(grid_f64)
        f_big_exact = [sum(exp.(-a_true .* x)) for x in grid_big]

        res_f64 = try
            matrix_pencil(grid_f64, f_f64, N)
        catch e
            nothing
        end

        Lp = div(M - 1, 2)
        nrow = M - Lp
        Y0 = zeros(Float64, nrow, Lp)
        for i in 1:nrow, j in 1:Lp
            Y0[i, j] = f_f64[i + j - 1]
        end
        F = svd(Y0)
        cond_val = F.S[1] / F.S[min(N, length(F.S))]
        sN_over_eps = F.S[min(N, length(F.S))] / eps(Float64)

        err_f64 = res_f64 !== nothing ? maximum(abs.(res_f64.a .- Float64.(a_true))) : NaN

        res_big = try
            matrix_pencil(BigFloat.(grid_f64), BigFloat.(f_f64), N)
        catch e
            nothing
        end
        err_big = res_big !== nothing ? maximum(abs.(Float64.(res_big.a .- a_true))) : NaN

        @printf("  %-5.1f | %2d | %.2e | %.3e     | %.3e     | %.1f\n",
                L0_val, M, cond_val, err_f64, err_big, sN_over_eps)
    end

    # 3b: 变化步长 h (固定 L0=1, 域 [1,5])
    println("\n--- 3b: 步长 h 的影响 (L0=1, 域=[1,5], Float64) ---")
    println("  h     | M   | cond(Y0) | max|Δa| F64 | S[N] | S[N]/eps")
    println("  " * "-" ^ 70)

    for h_val in [0.2, 0.1, 0.05, 0.02, 0.01]
        L0_val = 1.0
        M = round(Int, 4.0 / h_val) + 1
        grid_f64 = Float64[L0_val + h_val * i for i in 0:(M - 1)]
        f_f64 = Float64[sum(exp.(-Float64.(a_true) .* x)) for x in grid_f64]

        res = try
            matrix_pencil(grid_f64, f_f64, N)
        catch e
            nothing
        end

        Lp = div(M - 1, 2)
        nrow = M - Lp
        Y0_mat = zeros(Float64, nrow, Lp)
        for i in 1:nrow, j in 1:Lp
            Y0_mat[i, j] = f_f64[i + j - 1]
        end
        F = svd(Y0_mat)
        cond_val = F.S[1] / F.S[min(N, length(F.S))]
        sN = F.S[min(N, length(F.S))]
        sN_over_eps = sN / eps(Float64)

        err = res !== nothing ? maximum(abs.(res.a .- Float64.(a_true))) : NaN

        @printf("  %-5.3f | %3d | %.2e | %.3e     | %.2e | %.1f\n",
                h_val, M, cond_val, err, sN, sN_over_eps)
    end

    # 3c: 变化数据量 M (固定 L0=1, 域端点变化)
    println("\n--- 3c: 数据量 M 的影响 (L0=1, h=0.1, Float64) ---")
    println("  M   | 域     | cond(Y0) | max|Δa| F64 | S[N]     | S[N]/eps")
    println("  " * "-" ^ 70)

    for L_end in [3, 5, 10, 20]
        h_val = 0.1
        M = round(Int, (L_end - 1.0) / h_val) + 1
        grid_f64 = Float64[1.0 + h_val * i for i in 0:(M - 1)]
        f_f64 = Float64[sum(exp.(-Float64.(a_true) .* x)) for x in grid_f64]

        res = try
            matrix_pencil(grid_f64, f_f64, N)
        catch e
            nothing
        end

        Lp = div(M - 1, 2)
        nrow = M - Lp
        Y0_mat = zeros(Float64, nrow, Lp)
        for i in 1:nrow, j in 1:Lp
            Y0_mat[i, j] = f_f64[i + j - 1]
        end
        F = svd(Y0_mat)
        cond_val = F.S[1] / F.S[min(N, length(F.S))]
        sN = F.S[min(N, length(F.S))]
        sN_over_eps = sN / eps(Float64)

        err = res !== nothing ? maximum(abs.(res.a .- Float64.(a_true))) : NaN

        @printf("  %3d | [1,%-2d] | %.2e | %.3e     | %.2e | %.1f\n",
                M, L_end, cond_val, err, sN, sN_over_eps)
    end
end

# ================================================================
#  实验 4: 最有效的改进策略
# ================================================================

function experiment_improvements()
    println("\n" * "=" ^ 80)
    println("  实验 4: 改进策略效果对比")
    println("=" ^ 80)

    setprecision(BigFloat, 256)
    N = 10
    d = BigFloat(1 // 2)
    a_true = collect(1:N) .* d

    # 基准: 原始设置 L0=1, h=0.1, 域=[1,5], Float64
    h_val = 0.1;
    L0 = 1.0;
    L_end = 5.0
    M_base = round(Int, (L_end - L0) / h_val) + 1
    grid_base = Float64[L0 + h_val * i for i in 0:(M_base - 1)]
    f_base = Float64[sum(exp.(-Float64.(a_true) .* x)) for x in grid_base]

    println("\n--- 基准 (L0=1, h=0.1, [1,5], M=41, Float64) ---")
    res_base = matrix_pencil(grid_base, f_base, N)
    err_base = maximum(abs.(res_base.a .- Float64.(a_true)))
    @printf("  max|Δa| = %.3e\n", err_base)

    # 改进 1: 从 x=0 开始
    println("\n--- 改进 1: 域起点 L0=0 (h=0.1, [0,4], M=41, Float64) ---")
    grid_1 = Float64[0.0 + h_val * i for i in 0:40]
    f_1 = Float64[sum(exp.(-Float64.(a_true) .* x)) for x in grid_1]
    res_1 = matrix_pencil(grid_1, f_1, N)
    err_1 = maximum(abs.(res_1.a .- Float64.(a_true)))
    @printf("  max|Δa| = %.3e  (改善 %.0f×)\n", err_1, err_base / err_1)

    # 改进 2: 更密的网格
    println("\n--- 改进 2: 更密网格 (L0=0, h=0.01, [0,4], M=401, Float64) ---")
    h2 = 0.01;
    M2 = round(Int, 4.0 / h2) + 1
    grid_2 = Float64[0.0 + h2 * i for i in 0:(M2 - 1)]
    f_2 = Float64[sum(exp.(-Float64.(a_true) .* x)) for x in grid_2]
    res_2 = matrix_pencil(grid_2, f_2, N)
    err_2 = maximum(abs.(res_2.a .- Float64.(a_true)))
    @printf("  max|Δa| = %.3e  (改善 %.0f×)\n", err_2, err_base / err_2)

    # 改进 3: 请求更少分量（只要 7 个）
    println("\n--- 改进 3: 只请求 7 个分量 (原始网格, Float64) ---")
    res_3 = matrix_pencil(grid_base, f_base, 7)
    err_3_a = Float64[]
    println("  恢复的 a 值 vs 真实值:")
    for k in 1:7
        best_match = argmin(abs.(Float64.(a_true) .- res_3.a[k]))
        push!(err_3_a, abs(res_3.a[k] - Float64(a_true[best_match])))
        @printf("    a[%d] = %.6f  (最近真值=%.1f, error=%.2e)\n",
                k, res_3.a[k], Float64(a_true[best_match]), err_3_a[end])
    end
    @printf("  max|Δa|(前7个匹配) = %.3e\n", maximum(err_3_a))

    # 改进 4: svd_tol 自适应截断
    println("\n--- 改进 4: SVD 自适应截断 svd_tol=eps*S[1] ---")
    # 先获取 S[1] 估计
    Lp = div(M_base - 1, 2)
    nrow = M_base - Lp
    Y0_mat = zeros(Float64, nrow, Lp)
    for i in 1:nrow, j in 1:Lp
        Y0_mat[i, j] = f_base[i + j - 1]
    end
    F = svd(Y0_mat)
    tol_adaptive = eps(Float64) * F.S[1]
    svd_rank = count(F.S .> tol_adaptive)
    @printf("  S[1] = %.3e, tol = %.3e, 自适应秩 = %d\n",
            F.S[1], tol_adaptive, svd_rank)
    res_4 = matrix_pencil(grid_base, f_base, N; svd_tol=tol_adaptive)
    println("  恢复的 $(length(res_4.a)) 个分量:")
    for k in 1:length(res_4.a)
        best_match = argmin(abs.(Float64.(a_true) .- res_4.a[k]))
        err_k = abs(res_4.a[k] - Float64(a_true[best_match]))
        @printf("    a[%d] = %.6f  (最近真值=%.1f, error=%.2e)\n",
                k, res_4.a[k], Float64(a_true[best_match]), err_k)
    end

    # 改进 5: 组合方案 — 从 x=0 开始 + 密网格
    println("\n--- 改进 5: 组合 (L0=0, h=0.02, [0,8], M=401, Float64) ---")
    h5 = 0.02;
    M5 = round(Int, 8.0 / h5) + 1
    grid_5 = Float64[0.0 + h5 * i for i in 0:(M5 - 1)]
    f_5 = Float64[sum(exp.(-Float64.(a_true) .* x)) for x in grid_5]
    res_5 = matrix_pencil(grid_5, f_5, N)
    err_5 = maximum(abs.(res_5.a .- Float64.(a_true)))
    @printf("  max|Δa| = %.3e  (改善 %.0f×)\n", err_5, err_base / err_5)
    println("  恢复的 a 值:")
    for k in 1:N
        @printf("    a[%2d] = true=%.1f  recovered=%.10f  error=%.2e\n",
                k, Float64(a_true[k]), res_5.a[k],
                abs(res_5.a[k] - Float64(a_true[k])))
    end

    return err_base
end

# ================================================================
#  实验 5: 信息论视角 — 每个分量贡献多少信息
# ================================================================

function experiment_information_content()
    println("\n" * "=" ^ 80)
    println("  实验 5: 各分量对信号的信息贡献分析")
    println("=" ^ 80)

    N = 10
    a_true = Float64.(collect(1:N) .* 0.5)

    println("\n  各分量在域 [1,5] 上的能量贡献:")
    println("  n | a_n | max|c_n*exp(-a_n*x)| on [1,5] | min on [1,5] | 动态范围")
    println("  " * "-" ^ 75)

    for n in 1:N
        val_at_1 = exp(-a_true[n] * 1.0)
        val_at_5 = exp(-a_true[n] * 5.0)
        dyn_range = val_at_1 / max(val_at_5, 1e-300)
        @printf("  %2d | %.1f | %.3e                | %.3e    | %.1e\n",
                n, a_true[n], val_at_1, val_at_5, dyn_range)
    end

    println("\n  信号总值中各分量的相对贡献 (x=1处):")
    total_at_1 = sum(exp.(-a_true .* 1.0))
    for n in 1:N
        frac = exp(-a_true[n] * 1.0) / total_at_1
        @printf("  分量 %2d (a=%.1f): %.6f  (占比 %.2f%%)\n",
                n, a_true[n], exp(-a_true[n] * 1.0), frac * 100)
    end

    println("\n  Float64 可分辨性分析:")
    println("  如果分量贡献 < eps(Float64) × total ≈ $(total_at_1 * eps(Float64))")
    for n in 1:N
        contrib = exp(-a_true[n] * 1.0)
        resolvable = contrib > total_at_1 * eps(Float64)
        status = resolvable ? "可分辨" : "淹没在噪声中"
        @printf("  分量 %2d: %.3e  %s  (S/N ≈ %.1e)\n",
                n, contrib, status, contrib / (total_at_1 * eps(Float64)))
    end
end

# ================================================================
#  生成报告
# ================================================================

function generate_error_report(output_path::String)
    io = IOBuffer()

    println(io, "# Matrix Pencil 方法误差来源分析报告")
    println(io, "")
    println(io, "生成时间: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "")

    report_str = String(take!(io))
    open(output_path, "w") do f
        return write(f, report_str)
    end
end

# ================================================================
#  运行全部实验
# ================================================================

experiment_svd_spectrum()
experiment_error_isolation()
experiment_grid_sensitivity()
experiment_improvements()
experiment_information_content()

# ================================================================
#  实验 6: Pencil 参数 L 的优化
# ================================================================

function experiment_pencil_L()
    println("\n" * "=" ^ 80)
    println("  实验 6: Pencil 参数 L 的影响 (M >> 2N+1 时)")
    println("=" ^ 80)

    N = 10
    a_true = Float64.(collect(1:N) .* 0.5)

    println("\n--- 6a: 域 [0,4], h=0.01, M=401, 变化 L ---")
    h_val = 0.01;
    M = 401
    grid = Float64[0.0 + h_val * i for i in 0:(M - 1)]
    f = Float64[sum(exp.(-a_true .* x)) for x in grid]

    println("  L     | 矩阵尺寸      | max|Δa|     | 前3个分量误差")
    println("  " * "-" ^ 70)

    for L_val in [10, 15, 20, 30, 50, 100, 200]
        if L_val >= M - N
            ;
            continue;
        end
        res = try
            matrix_pencil(grid, f, N; pencil_L=L_val)
        catch e
            nothing
        end
        if res === nothing
            @printf("  %-5d | fail\n", L_val)
            continue
        end
        nrow = M - L_val
        err = maximum(abs.(res.a .- a_true))
        e1 = abs(res.a[1] - a_true[1])
        e2 = abs(res.a[2] - a_true[2])
        e3 = abs(res.a[3] - a_true[3])
        @printf("  %-5d | %3d × %-3d      | %.3e     | %.1e, %.1e, %.1e\n",
                L_val, nrow, L_val, err, e1, e2, e3)
    end

    println("\n--- 6b: 域 [1,5], h=0.1, M=41 (原始问题), 变化 L ---")
    grid2 = Float64[1.0 + 0.1 * i for i in 0:40]
    f2 = Float64[sum(exp.(-a_true .* x)) for x in grid2]
    println("  L     | 矩阵尺寸      | max|Δa|     | 前3个分量误差")
    println("  " * "-" ^ 70)

    for L_val in [10, 12, 15, 20]
        res = try
            matrix_pencil(grid2, f2, N; pencil_L=L_val)
        catch e
            nothing
        end
        res === nothing && continue
        nrow = 41 - L_val
        err = maximum(abs.(res.a .- a_true))
        e1 = abs(res.a[1] - a_true[1])
        e2 = abs(res.a[2] - a_true[2])
        e3 = abs(res.a[3] - a_true[3])
        @printf("  %-5d | %3d × %-3d      | %.3e     | %.1e, %.1e, %.1e\n",
                L_val, nrow, L_val, err, e1, e2, e3)
    end

    println("\n--- 6c: 最优组合：域 [0,20], h=0.1, M=201, L=20~30 ---")
    grid3 = Float64[0.0 + 0.1 * i for i in 0:200]
    f3 = Float64[sum(exp.(-a_true .* x)) for x in grid3]
    println("  L     | 矩阵尺寸      | max|Δa|     | 前5个分量误差")
    println("  " * "-" ^ 80)

    for L_val in [15, 20, 25, 30, 40, 50, 100]
        res = try
            matrix_pencil(grid3, f3, N; pencil_L=L_val)
        catch e
            nothing
        end
        res === nothing && continue
        nrow = 201 - L_val
        err = maximum(abs.(res.a .- a_true))
        errs = [abs(res.a[k] - a_true[k]) for k in 1:min(5, length(res.a))]
        errs_str = join([@sprintf("%.1e", e) for e in errs], ", ")
        @printf("  %-5d | %3d × %-3d      | %.3e     | %s\n",
                L_val, nrow, L_val, err, errs_str)
    end
end

experiment_pencil_L()

println("\n\n" * "=" ^ 80)
println("  全部实验完成")
println("=" ^ 80)
