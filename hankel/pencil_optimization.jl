"""
Part 3: Pencil 参数 L 和步长 h 的最优选择

内容：
  1. recommend_L_theory / recommend_L_crossval — 最优 pencil_L
  2. matrix_pencil_auto — 自动选择 L 的改进版 matrix_pencil
  3. recommend_h_theory / recommend_h_crossval — 最优 h
  4. matrix_pencil_subsample — 对过密网格降采样的改进算法
  5. 数值实验与对比
"""

include("hankel.jl")
using GenericLinearAlgebra
using LinearAlgebra
using Printf, Dates, Statistics

# ================================================================
#  1. pencil_L 的理论推荐
# ================================================================

"""
    recommend_L_theory(M, N)

基于理论分析推荐 pencil 参数 L。

Hankel 矩阵 Y₀ 的尺寸为 (M-L)×L。
- 信号奇异值 ~ √(M-L) (更多行 → 更好的最小二乘平均)
- Hankel 噪声谱范数 ~ ε·√(min(M-L, L)) (Hankel 结构噪声)
- SNR ∝ √(M-L) / √L = √((M-L)/L)

SNR 随 L 减小而增大，但 L ≥ N 是必须的。
同时 L 过小会导致 Vandermonde 矩阵条件数增大。

推荐：L = clamp(min(3N, M÷3), N, M-N-1)
- 当 M ≈ 2N+1（数据刚好够）: L ≈ N
- 当 M >> 2N+1（大量过采样）: L ≈ 3N（经验最优区间 2N~5N）
"""
function recommend_L_theory(M::Int, N::Int)
    L_raw = min(3 * N, div(M, 3))
    return clamp(L_raw, N, M - N - 1)
end

# ================================================================
#  2. pencil_L 的交叉验证
# ================================================================

"""
    recommend_L_crossval(grid, f, N; L_candidates, verbose)

通过交叉验证选择最优 pencil_L。

方法：将数据分为前 2/3（训练）和后 1/3（验证）。
对每个候选 L，用训练集运行 matrix_pencil，在验证集上计算 RMSE。
选 RMSE 最小的 L。
"""
function recommend_L_crossval(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                              L_candidates::Union{Nothing,Vector{Int}}=nothing,
                              verbose::Bool=false) where {T}
    M = length(grid)

    if L_candidates === nothing
        L_min = N
        L_max = min(10 * N, M - N - 1)
        step = max(1, (L_max - L_min) ÷ 15)
        L_candidates = collect(L_min:step:L_max)
        if L_candidates[end] != L_max
            push!(L_candidates, L_max)
        end
    end

    split_idx = round(Int, 2 * M / 3)
    split_idx = max(split_idx, 2 * N + 2)
    split_idx = min(split_idx, M - 10)

    grid_train = grid[1:split_idx]
    f_train = f[1:split_idx]
    grid_val = grid[(split_idx + 1):end]
    f_val = f[(split_idx + 1):end]

    best_L = L_candidates[1]
    best_rmse = T(Inf)

    for Lc in L_candidates
        Lc >= split_idx - N && continue
        Lc < N && continue

        res = try
            matrix_pencil(grid_train, f_train, N; pencil_L=Lc)
        catch
            nothing
        end
        res === nothing && continue

        pred = eval_expsum(res, grid_val)
        r = sqrt(sum((pred .- f_val) .^ 2) / length(f_val))

        if verbose
            @printf("    L=%3d: RMSE_val = %.3e\n", Lc, Float64(r))
        end

        if r < best_rmse
            best_rmse = r
            best_L = Lc
        end
    end

    return (best_L, Float64(best_rmse))
end

"""
    recommend_L_stability(grid, f, N; L_candidates, verbose)

通过参数稳定性选择最优 pencil_L。

方法：对每个候选 L，同时也运行 L±1 和 L±2。
比较恢复参数 a 的一致性（标准差）。选标准差最小的 L。
"""
function recommend_L_stability(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                               L_candidates::Union{Nothing,Vector{Int}}=nothing,
                               verbose::Bool=false) where {T}
    M = length(grid)

    if L_candidates === nothing
        L_min = N
        L_max = min(10 * N, M - N - 1)
        step = max(1, (L_max - L_min) ÷ 15)
        L_candidates = collect(L_min:step:L_max)
    end

    best_L = L_candidates[1]
    best_variation = Inf

    for Lc in L_candidates
        nearby = [Lc - 2, Lc - 1, Lc, Lc + 1, Lc + 2]
        filter!(l -> N <= l <= M - N - 1, nearby)
        length(nearby) < 3 && continue

        a_sets = Vector{Float64}[]
        for Ln in nearby
            res = try
                matrix_pencil(grid, f, N; pencil_L=Ln)
            catch
                nothing
            end
            res === nothing && continue
            length(res.a) == N || continue
            push!(a_sets, Float64.(res.a))
        end

        length(a_sets) < 3 && continue

        variation = 0.0
        for k in 1:N
            vals = [s[k] for s in a_sets]
            variation += std(vals)
        end
        variation /= N

        if verbose
            @printf("    L=%3d: avg_std(a) = %.3e\n", Lc, variation)
        end

        if variation < best_variation
            best_variation = variation
            best_L = Lc
        end
    end

    return (best_L, best_variation)
end

# ================================================================
#  3. 改进算法：自动选 L + 对过密网格降采样
# ================================================================

"""
    matrix_pencil_auto(grid, f, N; method=:theory)

自动选择 pencil_L 的改进版 matrix_pencil。

method 选项：
  :theory    — 使用理论公式 L = min(3N, M÷3)
  :crossval  — 使用交叉验证
  :stability — 使用参数稳定性
"""
function matrix_pencil_auto(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                            method::Symbol=:theory, svd_tol::Real=0) where {T}
    M = length(grid)

    if method == :theory
        L = recommend_L_theory(M, N)
    elseif method == :crossval
        L, _ = recommend_L_crossval(grid, f, N)
    elseif method == :stability
        L, _ = recommend_L_stability(grid, f, N)
    else
        error("Unknown method: $method")
    end

    return matrix_pencil(grid, f, N; pencil_L=L, svd_tol=svd_tol)
end

"""
    matrix_pencil_subsample(grid, f, N; target_M)

对过密网格降采样后运行 matrix_pencil。

当 M >> 10N 时，直接使用全部数据通常不如降采样到 target_M ≈ 6N~10N
然后用 L ≈ 3N 运行。原因：
- Hankel 噪声谱范数 ~ ε·√(min(M-L, L))，对固定 L 随 M 缓慢增长
- 但降采样（跳点）不增加额外噪声，只是减少冗余
- 同时减小 Hankel 矩阵尺寸，加速计算

如果 target_M == 0，自动选择 target_M = max(8N, 2N+1)。
"""
function matrix_pencil_subsample(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                                 target_M::Int=0, svd_tol::Real=0) where {T}
    M = length(grid)
    if target_M <= 0
        target_M = max(8 * N, 2 * N + 1)
    end
    target_M = min(target_M, M)

    if M <= target_M
        L = recommend_L_theory(M, N)
        return matrix_pencil(grid, f, N; pencil_L=L, svd_tol=svd_tol)
    end

    stride = M ÷ target_M
    stride = max(stride, 1)
    indices = 1:stride:M
    actual_M = length(indices)

    grid_sub = grid[indices]
    f_sub = f[indices]

    L = recommend_L_theory(actual_M, N)
    res_sub = matrix_pencil(grid_sub, f_sub, N; pencil_L=L, svd_tol=svd_tol)

    c_refined = _solve_amplitudes(grid, f, res_sub.a)
    return _pack(c_refined, res_sub.a)
end

# ================================================================
#  4. 最优 h 的理论推荐与交叉验证
# ================================================================

"""
    recommend_h_theory(L0, L_end, N, T)

给定固定域 [L0, L_end]，推荐最优步长 h。

理论分析：设 M = (L_end - L0)/h + 1
- 太少的点（M 小）→ 矩阵秩不够，方法误差大
- 太多的点（M 大）→ Hankel 噪声累积，噪声放大
- 最优 M ≈ 6N~10N，对应 h_opt = (L_end - L0) / (M_opt - 1)

返回 (h_recommended, M_recommended)。
"""
function recommend_h_theory(L0::Real, L_end::Real, N::Int, (::Type{T})=Float64) where {T}
    domain_len = Float64(L_end - L0)
    M_opt = clamp(8 * N, 2 * N + 1, round(Int, domain_len / 0.001))
    h_opt = domain_len / (M_opt - 1)
    return (h_opt, M_opt)
end

"""
    recommend_h_crossval(grid_fine, f_fine, N; h_candidates, verbose)

通过交叉验证选择最优步长 h。

方法：从最密的网格 (grid_fine, f_fine) 开始，按不同的降采样率 k
得到有效 h' = k·h_fine。对每个 h' 运行 matrix_pencil_auto，
用参数稳定性（相邻 k 值的参数一致性）评估。
"""
function recommend_h_crossval(grid_fine::AbstractVector{T}, f_fine::AbstractVector{T},
                              N::Int; verbose::Bool=false) where {T}
    M_fine = length(grid_fine)
    h_fine = Float64(grid_fine[2] - grid_fine[1])
    M_min = 2 * N + 1

    strides = Int[]
    k = 1
    while true
        M_sub = length(1:k:M_fine)
        M_sub < M_min && break
        push!(strides, k)
        k += 1
    end
    length(strides) < 3 && return (h_fine, M_fine)

    results = Dict{Int,Vector{Float64}}()
    rmses = Dict{Int,Float64}()

    for k in strides
        indices = 1:k:M_fine
        grid_sub = grid_fine[indices]
        f_sub = f_fine[indices]
        M_sub = length(grid_sub)
        L = recommend_L_theory(M_sub, N)

        res = try
            matrix_pencil(grid_sub, f_sub, N; pencil_L=L)
        catch
            nothing
        end
        res === nothing && continue
        length(res.a) == N || continue

        results[k] = Float64.(res.a)
        rmses[k] = Float64(rmse(res, grid_sub, f_sub))
    end

    valid_ks = sort(collect(keys(results)))
    length(valid_ks) < 3 && return (h_fine, M_fine)

    stability = Dict{Int,Float64}()
    for i in 2:(length(valid_ks) - 1)
        k = valid_ks[i]
        k_prev = valid_ks[i - 1]
        k_next = valid_ks[i + 1]
        haskey(results, k_prev) && haskey(results, k_next) || continue

        var = 0.0
        for j in 1:N
            a_prev = results[k_prev][j]
            a_curr = results[k][j]
            a_next = results[k_next][j]
            var += abs(a_curr - (a_prev + a_next) / 2)
        end
        stability[k] = var / N

        if verbose
            h_eff = h_fine * k
            M_eff = length(1:k:M_fine)
            @printf("    stride=%2d (h=%.4f, M=%3d): stability=%.3e, RMSE=%.3e\n",
                    k, h_eff, M_eff, var / N, rmses[k])
        end
    end

    isempty(stability) && return (h_fine, M_fine)

    best_k = first(sort(collect(stability); by=kv -> kv.second)).first
    h_best = h_fine * best_k
    M_best = length(1:best_k:M_fine)
    return (h_best, M_best)
end

# ================================================================
#  5. 数值实验
# ================================================================

function make_test_signal(::Type{T}, N, L0, L_end, h) where {T}
    a_true = T.(collect(1:N) .* 0.5)
    c_true = ones(T, N)
    M = round(Int, Float64(L_end - L0) / Float64(h)) + 1
    grid = [T(L0) + T(h) * i for i in 0:(M - 1)]
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
        return maximum(errs)
    end
end

# ── 实验 A: pencil_L 选择方法对比 ─────────────────────────

function experiment_L_selection()
    println("\n" * "=" ^ 80)
    println("  实验 A: pencil_L 选择方法对比")
    println("=" ^ 80)

    N = 10
    configs = [(L0=1.0, L_end=5.0, h=0.1, label="[1,5] h=0.1 M=41"),
               (L0=1.0, L_end=5.0, h=0.05, label="[1,5] h=0.05 M=81"),
               (L0=0.0, L_end=4.0, h=0.1, label="[0,4] h=0.1 M=41"),
               (L0=1.0, L_end=20.0, h=0.1, label="[1,20] h=0.1 M=191"),
               (L0=0.0, L_end=20.0, h=0.1, label="[0,20] h=0.1 M=201")]

    for cfg in configs
        grid, f, a_true, _ = make_test_signal(Float64, N, cfg.L0, cfg.L_end, cfg.h)
        M = length(grid)

        L_default = div(M - 1, 2)
        L_th = recommend_L_theory(M, N)
        L_cv, _ = recommend_L_crossval(grid, f, N)
        L_st, _ = recommend_L_stability(grid, f, N)

        println("\n--- $(cfg.label) (M=$M) ---")
        @printf("  理论推荐 L=%d, 交叉验证 L=%d, 稳定性 L=%d, 默认 L=%d\n",
                L_th, L_cv, L_st, L_default)

        for (tag, Lv) in [("default(M/2)", L_default), ("theory", L_th),
                          ("crossval", L_cv), ("stability", L_st)]
            res = try
                matrix_pencil(grid, f, N; pencil_L=Lv)
            catch
                nothing
            end
            if res !== nothing
                err = max_a_error(res, a_true)
                @printf("  %-15s L=%-3d max|Δa|=%.3e\n", tag, Lv, err)
            else
                @printf("  %-15s L=%-3d FAILED\n", tag, Lv)
            end
        end

        # 穷举找真正最优 L
        best_L = N
        best_err = Inf
        for Lc in N:(M - N - 1)
            res = try
                matrix_pencil(grid, f, N; pencil_L=Lc)
            catch
                nothing
            end
            res === nothing && continue
            err = max_a_error(res, a_true)
            if err < best_err
                best_err = err
                best_L = Lc
            end
        end
        @printf("  %-15s L=%-3d max|Δa|=%.3e (穷举)\n", "TRUE OPTIMAL", best_L, best_err)
    end
end

# ── 实验 B: 更密 h 是否还会劣化 ─────────────────────────

function experiment_h_with_fixed_L()
    println("\n" * "=" ^ 80)
    println("  实验 B: 固定域 [1,5], 固定 L=3N, 变化 h")
    println("=" ^ 80)

    N = 10

    println("\n  h      | M    | L  | max|Δa| (L=3N) | max|Δa| (L=M/2) | max|Δa| (subsample)")
    println("  " * "-" ^ 85)

    for h_val in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        grid, f, a_true, _ = make_test_signal(Float64, N, 1.0, 5.0, h_val)
        M = length(grid)
        M < 2 * N + 1 && continue

        L_3n = recommend_L_theory(M, N)
        L_half = div(M - 1, 2)

        res_3n = try
            matrix_pencil(grid, f, N; pencil_L=L_3n)
        catch
            ; nothing;
        end
        err_3n = res_3n !== nothing ? max_a_error(res_3n, a_true) : NaN

        res_half = try
            matrix_pencil(grid, f, N; pencil_L=L_half)
        catch
            ; nothing;
        end
        err_half = res_half !== nothing ? max_a_error(res_half, a_true) : NaN

        res_sub = try
            matrix_pencil_subsample(grid, f, N)
        catch
            ; nothing;
        end
        err_sub = res_sub !== nothing ? max_a_error(res_sub, a_true) : NaN

        @printf("  %-6.3f | %4d | %2d | %.3e        | %.3e        | %.3e\n",
                h_val, M, L_3n, err_3n, err_half, err_sub)
    end

    println("\n--- 同上，域 [0,20] ---")
    println("\n  h      | M    | L  | max|Δa| (L=3N) | max|Δa| (L=M/2) | max|Δa| (subsample)")
    println("  " * "-" ^ 85)

    for h_val in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        grid, f, a_true, _ = make_test_signal(Float64, N, 0.0, 20.0, h_val)
        M = length(grid)
        M < 2 * N + 1 && continue

        L_3n = recommend_L_theory(M, N)
        L_half = div(M - 1, 2)

        res_3n = try
            matrix_pencil(grid, f, N; pencil_L=L_3n)
        catch
            ; nothing;
        end
        err_3n = res_3n !== nothing ? max_a_error(res_3n, a_true) : NaN

        res_half = try
            matrix_pencil(grid, f, N; pencil_L=L_half)
        catch
            ; nothing;
        end
        err_half = res_half !== nothing ? max_a_error(res_half, a_true) : NaN

        res_sub = try
            matrix_pencil_subsample(grid, f, N)
        catch
            ; nothing;
        end
        err_sub = res_sub !== nothing ? max_a_error(res_sub, a_true) : NaN

        @printf("  %-6.3f | %4d | %2d | %.3e        | %.3e        | %.3e\n",
                h_val, M, L_3n, err_3n, err_half, err_sub)
    end
end

# ── 实验 C: 最优 h 选择 ─────────────────────────────────

function experiment_h_selection()
    println("\n" * "=" ^ 80)
    println("  实验 C: 最优 h 选择方法对比")
    println("=" ^ 80)

    N = 10

    for (L0, L_end, label) in [(1.0, 5.0, "[1,5]"), (0.0, 20.0, "[0,20]")]
        println("\n--- 域 $(label) ---")

        h_th, M_th = recommend_h_theory(L0, L_end, N, Float64)
        @printf("  理论推荐: h=%.4f, M=%d\n", h_th, M_th)

        h_fine = 0.005
        grid_fine, f_fine, a_true, _ = make_test_signal(Float64, N, L0, L_end, h_fine)
        M_fine = length(grid_fine)
        println("  最密网格: h=$h_fine, M=$M_fine")

        println("\n  交叉验证详情:")
        h_cv, M_cv = recommend_h_crossval(grid_fine, f_fine, N; verbose=true)
        @printf("  交叉验证推荐: h=%.4f, M=%d\n", h_cv, M_cv)

        println("\n  各 h 对应的真实误差:")
        println("  stride | h_eff  | M_eff | L   | max|Δa|")
        println("  " * "-" ^ 55)

        for k in [1, 2, 3, 5, 10, 20, 50, 100]
            indices = 1:k:M_fine
            M_eff = length(indices)
            M_eff < 2 * N + 1 && continue

            grid_sub = grid_fine[indices]
            f_sub = f_fine[indices]
            h_eff = Float64(grid_sub[2] - grid_sub[1])
            L = recommend_L_theory(M_eff, N)

            res = try
                matrix_pencil(grid_sub, f_sub, N; pencil_L=L)
            catch
                ; nothing;
            end
            err = res !== nothing ? max_a_error(res, a_true) : NaN

            marker = ""
            if abs(h_eff - h_th) < 0.001
                ;
                marker = " ← 理论推荐";
            end
            if abs(h_eff - h_cv) < 0.001
                ;
                marker = " ← 交叉验证推荐";
            end

            @printf("  %4d   | %-6.3f | %4d  | %3d | %.3e%s\n",
                    k, h_eff, M_eff, L, err, marker)
        end
    end
end

# ── 实验 D: matrix_pencil_auto 综合对比 ─────────────────

function experiment_auto_comparison()
    println("\n" * "=" ^ 80)
    println("  实验 D: matrix_pencil_auto 综合对比")
    println("=" ^ 80)

    N = 10
    configs = [(L0=1.0, L_end=5.0, h=0.1, label="[1,5] h=0.1"),
               (L0=1.0, L_end=5.0, h=0.02, label="[1,5] h=0.02"),
               (L0=0.0, L_end=20.0, h=0.1, label="[0,20] h=0.1"),
               (L0=0.0, L_end=20.0, h=0.02, label="[0,20] h=0.02")]

    println("\n  场景               | 方法         | max|Δa|     | 恢复 N")
    println("  " * "-" ^ 65)

    for cfg in configs
        grid, f, a_true, _ = make_test_signal(Float64, N, cfg.L0, cfg.L_end, cfg.h)
        M = length(grid)

        methods = [("default", () -> matrix_pencil(grid, f, N)),
                   ("auto_theory", () -> matrix_pencil_auto(grid, f, N; method=:theory)),
                   ("auto_crossval",
                    () -> matrix_pencil_auto(grid, f, N; method=:crossval)),
                   ("subsample", () -> matrix_pencil_subsample(grid, f, N))]

        for (mname, mfunc) in methods
            res = try
                mfunc()
            catch
                ; nothing;
            end

            if res !== nothing
                err = max_a_error(res, a_true)
                @printf("  %-20s | %-12s | %.3e     | %d\n",
                        cfg.label, mname, err, length(res.a))
            else
                @printf("  %-20s | %-12s | FAILED\n", cfg.label, mname)
            end
        end
        println("  " * "-" ^ 65)
    end
end

# ================================================================
#  运行全部实验
# ================================================================

experiment_L_selection()
experiment_h_with_fixed_L()
experiment_h_selection()
experiment_auto_comparison()

println("\n" * "=" ^ 80)
println("  全部实验完成")
println("=" ^ 80)
