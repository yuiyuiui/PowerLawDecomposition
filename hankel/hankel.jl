using LinearAlgebra

"""
    ExponentialSum{T}

Represents f(x) = Σᵢ c[i] * exp(-a[i] * x).
Fields are sorted by ascending decay rate `a`.
"""
struct ExponentialSum{T<:Real}
    c::Vector{T}
    a::Vector{T}
end

function eval_expsum(es::ExponentialSum{T}, x::T) where {T}
    s = zero(T)
    @inbounds for i in eachindex(es.c)
        s += es.c[i] * exp(-es.a[i] * x)
    end
    return s
end

function eval_expsum(es::ExponentialSum{T}, grid::AbstractVector{T}) where {T}
    return T[eval_expsum(es, x) for x in grid]
end

function rmse(es::ExponentialSum{T}, grid::AbstractVector{T},
              f::AbstractVector{T}) where {T}
    err = zero(T)
    @inbounds for k in eachindex(f)
        r = eval_expsum(es, grid[k]) - f[k]
        err += r * r
    end
    return sqrt(err / length(f))
end

function _solve_amplitudes(grid::AbstractVector{T}, f::AbstractVector{T},
                           a::Vector{T}) where {T}
    M, N = length(f), length(a)
    V = zeros(T, M, N)
    @inbounds for k in 1:M, n in 1:N
        V[k, n] = exp(-a[n] * grid[k])
    end
    return V \ f
end

function _pack(c::Vector{T}, a::Vector{T}) where {T}
    perm = sortperm(a)
    return ExponentialSum(c[perm], a[perm])
end

# ════════════════════════════════════════════════════════════
#  Prony's method (least-squares variant)
# ════════════════════════════════════════════════════════════

"""
    prony(grid, f, N) -> ExponentialSum

Decompose `f` sampled on a uniform `grid` into `N` real exponential terms
`f(x) ≈ Σ c_n exp(-a_n x)` via the classical Prony method.

Requires `length(f) ≥ 2N`.
"""
function prony(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int) where {T<:Real}
    M = length(f)
    @assert M >= 2N "Need M ≥ 2N data points (got $M for N=$N)"
    @assert length(grid) == M
    h = grid[2] - grid[1]

    # Linear prediction: build over-determined Toeplitz system H q = b
    nrows = M - N
    H = zeros(T, nrows, N)
    b = zeros(T, nrows)
    @inbounds for i in 1:nrows
        for j in 1:N
            H[i, j] = f[i + N - j]
        end
        b[i] = f[i + N]
    end
    p = -(H \ b)   # AR coefficients: z^N + p₁z^{N-1} + … + pₙ = 0

    # Roots via companion matrix
    C = zeros(T, N, N)
    @inbounds for i in 2:N
        ;
        C[i, i - 1] = one(T);
    end
    @inbounds for i in 1:N
        ;
        C[i, N] = -p[N + 1 - i];
    end

    z = eigvals(C)
    a_vals = T[-log(abs(T(real(zk)))) / h for zk in z]
    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Matrix Pencil Method  (Hua–Sarkar, 1990)
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil(grid, f, N) -> ExponentialSum

Decompose `f` sampled on a uniform `grid` into `N` real exponential terms
`f(x) ≈ Σ c_n exp(-a_n x)` via the Matrix Pencil Method.

Requires `length(f) ≥ 2N + 1`.
"""
function matrix_pencil(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                       svd_tol::Real=0, pencil_L::Int=0) where {T<:Real}
    M = length(f)
    @assert M >= 2N + 1 "Need M ≥ 2N+1 data points (got $M for N=$N)"
    @assert length(grid) == M
    h = grid[2] - grid[1]
    L = pencil_L > 0 ? min(pencil_L, M - N - 1) : div(M - 1, 2)

    # Hankel matrix
    nrow = M - L
    Y = zeros(T, nrow, L + 1)
    @inbounds for i in 1:nrow, j in 1:(L + 1)
        Y[i, j] = f[i + j - 1]
    end

    Y0 = Y[:, 1:L]
    Y1 = Y[:, 2:(L + 1)]

    # SVD, truncate to rank N
    F = svd(Y0)
    svd_n = count(F.S .> svd_tol)
    (svd_n < N) && @warn "SVD truncated to rank $svd_n, which is less than $N"
    N = min(svd_n, N)

    U_N = F.U[:, 1:N]
    S_inv = Diagonal(one(T) ./ F.S[1:N])
    V_N = F.V[:, 1:N]

    Z = S_inv * (U_N' * Y1 * V_N)
    z = eigvals(Z)
    a_vals = T[-log(abs(T(real(zk)))) / h for zk in z]
    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Delayed Hankel Matrix Pencil (stride-k construction)
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil_delayed(grid, f, N; stride, pencil_L, svd_tol) -> ExponentialSum

Matrix Pencil with delayed (strided) Hankel matrix construction.

Standard Hankel uses column step = 1 sample (= h).
Delayed Hankel uses column step = `stride` samples (= stride·h),
while rows still advance by 1 sample. This decouples:
  - Column spacing (large → good pole separation, h_eff = stride·h)
  - Row density (dense → many rows for SVD noise averaging)

If `stride == 0`, automatically chooses stride so that h_eff ≈ domain_length / (5N).
"""
function matrix_pencil_delayed(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                               stride::Int=0, pencil_L::Int=0,
                               svd_tol::Real=0) where {T<:Real}
    M = length(f)
    @assert length(grid) == M
    h = grid[2] - grid[1]
    domain_len = grid[end] - grid[1]

    L = pencil_L > 0 ? pencil_L : min(3 * N, div(M, 3))
    L = clamp(L, N, M - 1)

    if stride <= 0
        h_target = domain_len / (5 * N)
        stride = max(1, round(Int, Float64(h_target / h)))
    end

    nrow = M - stride * L
    while nrow < N + 1 && stride > 1
        stride -= 1
        nrow = M - stride * L
    end
    while nrow < N + 1 && L > N
        L -= 1
        nrow = M - stride * L
    end
    @assert nrow >= N + 1 "Cannot build delayed Hankel: M=$M, stride=$stride, L=$L → nrow=$nrow < $(N+1)"

    h_eff = stride * h

    Y0 = zeros(T, nrow, L)
    Y1 = zeros(T, nrow, L)
    @inbounds for i in 1:nrow, j in 1:L
        Y0[i, j] = f[i + stride * (j - 1)]
        Y1[i, j] = f[i + stride * j]
    end

    F = svd(Y0)
    svd_n = count(F.S .> svd_tol)
    (svd_n < N) && @warn "SVD truncated to rank $svd_n, which is less than $N"
    Neff = min(svd_n, N)

    U_N = F.U[:, 1:Neff]
    S_inv = Diagonal(one(T) ./ F.S[1:Neff])
    V_N = F.V[:, 1:Neff]

    Z = S_inv * (U_N' * Y1 * V_N)
    z = eigvals(Z)
    a_vals = T[-log(abs(T(real(zk)))) / h_eff for zk in z]
    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Pre-filtering + Subsampling Matrix Pencil
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil_prefilter(grid, f, N; stride, pencil_L, svd_tol) -> ExponentialSum

Moving-average pre-filter followed by subsampling, then standard Matrix Pencil.

For f(x) = Σ cₙ exp(-aₙ x), the k-point moving average preserves all
decay rates aₙ (only amplitudes change by a known factor). Averaging
reduces noise by ~√k while subsampling restores good pole separation.

Amplitudes are re-estimated from the original (unfiltered) data via
least-squares after extracting the decay rates.
"""
function matrix_pencil_prefilter(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                                 stride::Int=0, pencil_L::Int=0,
                                 svd_tol::Real=0) where {T<:Real}
    M = length(f)
    @assert length(grid) == M
    h = grid[2] - grid[1]
    domain_len = grid[end] - grid[1]

    if stride <= 0
        h_target = domain_len / (5 * N)
        stride = max(1, round(Int, Float64(h_target / h)))
    end

    k = stride

    if k <= 1
        L = pencil_L > 0 ? pencil_L : div(M - 1, 2)
        return matrix_pencil(grid, f, N; pencil_L=L, svd_tol=svd_tol)
    end

    M_filt = M - k + 1
    g = zeros(T, M_filt)
    @inbounds for i in 1:M_filt
        s = zero(T)
        for m in 0:(k - 1)
            s += f[i + m]
        end
        g[i] = s / k
    end

    sub_idx = 1:k:M_filt
    g_sub = g[sub_idx]
    grid_sub = grid[sub_idx]
    M_sub = length(g_sub)

    if M_sub < 2N + 1
        L = pencil_L > 0 ? pencil_L : div(M - 1, 2)
        return matrix_pencil(grid, f, N; pencil_L=L, svd_tol=svd_tol)
    end

    L = pencil_L > 0 ? pencil_L : min(3 * N, div(M_sub - 1, 2))
    L = clamp(L, N, M_sub - N - 1)
    h_sub = grid_sub[2] - grid_sub[1]

    nrow = M_sub - L
    Y = zeros(T, nrow, L + 1)
    @inbounds for i in 1:nrow, j in 1:(L + 1)
        Y[i, j] = g_sub[i + j - 1]
    end

    Y0 = Y[:, 1:L]
    Y1 = Y[:, 2:(L + 1)]

    F = svd(Y0)
    svd_n = count(F.S .> svd_tol)
    (svd_n < N) && @warn "Pre-filter SVD truncated to rank $svd_n < $N"
    Neff = min(svd_n, N)

    U_N = F.U[:, 1:Neff]
    S_inv = Diagonal(one(T) ./ F.S[1:Neff])
    V_N = F.V[:, 1:Neff]

    Z = S_inv * (U_N' * Y1 * V_N)
    z = eigvals(Z)
    a_vals = T[-log(abs(T(real(zk)))) / h_sub for zk in z]

    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Delta-Operator Matrix Pencil
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil_delta(grid, f, N; pencil_L, svd_tol) -> ExponentialSum

Matrix Pencil using the delta operator δ = (z - 1)/h instead of the
shift operator z.

Standard MP solves for z = e^{-ah} ≈ 1 when h is small (poles cluster
near 1, causing ill-conditioning). The delta operator maps poles to
λ = (e^{-ah} - 1)/h → -a as h → 0, giving well-separated eigenvalues
regardless of h.

Implementation:
  1. Build standard Hankel Y₀, Y₁
  2. Compute Yδ = (Y₁ - Y₀) / h  (amplifies the signal differences)
  3. SVD on Y₀, project Yδ onto signal subspace
  4. Eigenvalues λ of projected matrix satisfy a = -log(1 + λh)/h
"""
function matrix_pencil_delta(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                             pencil_L::Int=0, svd_tol::Real=0) where {T<:Real}
    M = length(f)
    @assert M >= 2N + 1 "Need M ≥ 2N+1 data points (got $M for N=$N)"
    @assert length(grid) == M
    h = grid[2] - grid[1]
    L = pencil_L > 0 ? min(pencil_L, M - N - 1) : div(M - 1, 2)

    nrow = M - L
    Y = zeros(T, nrow, L + 1)
    @inbounds for i in 1:nrow, j in 1:(L + 1)
        Y[i, j] = f[i + j - 1]
    end

    Y0 = Y[:, 1:L]
    Y1 = Y[:, 2:(L + 1)]
    Y_delta = (Y1 - Y0) / h

    F = svd(Y0)
    svd_n = count(F.S .> svd_tol)
    (svd_n < N) && @warn "Delta SVD truncated to rank $svd_n < $N"
    Neff = min(svd_n, N)

    U_N = F.U[:, 1:Neff]
    S_inv = Diagonal(one(T) ./ F.S[1:Neff])
    V_N = F.V[:, 1:Neff]

    Z_delta = S_inv * (U_N' * Y_delta * V_N)
    lam = eigvals(Z_delta)

    a_vals = Vector{T}(undef, Neff)
    for i in 1:Neff
        li = T(real(lam[i]))
        arg = one(T) + li * h
        if arg > zero(T)
            a_vals[i] = -log(arg) / h
        else
            a_vals[i] = -li
        end
    end

    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Delta-Operator + Delayed Hankel (combined)
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil_delta_delayed(grid, f, N; stride, pencil_L, svd_tol) -> ExponentialSum

Combines the delta operator with delayed (strided) Hankel construction.
Uses stride-k columns for pole separation AND the delta transform
for additional numerical stability.
"""
function matrix_pencil_delta_delayed(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                                     stride::Int=0, pencil_L::Int=0,
                                     svd_tol::Real=0) where {T<:Real}
    M = length(f)
    @assert length(grid) == M
    h = grid[2] - grid[1]
    domain_len = grid[end] - grid[1]

    L = pencil_L > 0 ? pencil_L : min(3 * N, div(M, 3))
    L = clamp(L, N, M - 1)

    if stride <= 0
        h_target = domain_len / (5 * N)
        stride = max(1, round(Int, Float64(h_target / h)))
    end

    nrow = M - stride * L
    while nrow < N + 1 && stride > 1
        stride -= 1
        nrow = M - stride * L
    end
    while nrow < N + 1 && L > N
        L -= 1
        nrow = M - stride * L
    end
    @assert nrow >= N + 1 "Cannot build delta-delayed Hankel: M=$M, stride=$stride, L=$L → nrow=$nrow < $(N+1)"

    h_eff = stride * h

    Y0 = zeros(T, nrow, L)
    Y1 = zeros(T, nrow, L)
    @inbounds for i in 1:nrow, j in 1:L
        Y0[i, j] = f[i + stride * (j - 1)]
        Y1[i, j] = f[i + stride * j]
    end
    Y_delta = (Y1 - Y0) / h_eff

    F = svd(Y0)
    svd_n = count(F.S .> svd_tol)
    (svd_n < N) && @warn "Delta-delayed SVD truncated to rank $svd_n < $N"
    Neff = min(svd_n, N)

    U_N = F.U[:, 1:Neff]
    S_inv = Diagonal(one(T) ./ F.S[1:Neff])
    V_N = F.V[:, 1:Neff]

    Z_delta = S_inv * (U_N' * Y_delta * V_N)
    lam = eigvals(Z_delta)

    a_vals = Vector{T}(undef, Neff)
    for i in 1:Neff
        li = T(real(lam[i]))
        arg = one(T) + li * h_eff
        if arg > zero(T)
            a_vals[i] = -log(arg) / h_eff
        else
            a_vals[i] = -li
        end
    end

    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Improved Delta+Delayed: balanced Hankel matrix
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil_delta_balanced(grid, f, N; stride, pencil_L, svd_tol) -> ExponentialSum

Delta operator + delayed Hankel with **balanced** matrix sizing.

Key difference from `matrix_pencil_delta_delayed`: defaults L ≈ M/(stride+1)
so that nrow ≈ L (square-ish Hankel). The delta operator benefits greatly from
a balanced matrix because the 1/h amplification of noise in Yδ = (Y₁−Y₀)/h
requires both sufficient columns (for eigenvalue resolution) and sufficient rows
(for SVD noise averaging).
"""
function matrix_pencil_delta_balanced(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                                      stride::Int=0, pencil_L::Int=0,
                                      svd_tol::Real=0) where {T<:Real}
    M = length(f)
    @assert length(grid) == M
    h = grid[2] - grid[1]
    domain_len = grid[end] - grid[1]

    if stride <= 0
        h_target = domain_len / (5 * N)
        stride = max(1, round(Int, Float64(h_target / h)))
    end

    if pencil_L > 0
        L = pencil_L
    else
        L = max(N, div(M, stride + 1))
    end

    nrow = M - stride * L
    while nrow < N + 1 && stride > 1
        stride -= 1
        nrow = M - stride * L
    end
    while nrow < N + 1 && L > N
        L -= 1
        nrow = M - stride * L
    end
    @assert nrow >= N + 1 "Cannot build balanced delta Hankel: M=$M, stride=$stride, L=$L → nrow=$nrow < $(N+1)"

    h_eff = stride * h

    Y0 = zeros(T, nrow, L)
    Y1 = zeros(T, nrow, L)
    @inbounds for i in 1:nrow, j in 1:L
        Y0[i, j] = f[i + stride * (j - 1)]
        Y1[i, j] = f[i + stride * j]
    end
    Y_delta = (Y1 - Y0) / h_eff

    F = svd(Y0)
    svd_n = count(F.S .> svd_tol)
    (svd_n < N) && @warn "Delta-balanced SVD truncated to rank $svd_n < $N"
    Neff = min(svd_n, N)

    U_N = F.U[:, 1:Neff]
    S_inv = Diagonal(one(T) ./ F.S[1:Neff])
    V_N = F.V[:, 1:Neff]

    Z_delta = S_inv * (U_N' * Y_delta * V_N)
    lam = eigvals(Z_delta)

    a_vals = Vector{T}(undef, Neff)
    for i in 1:Neff
        li = T(real(lam[i]))
        arg = one(T) + li * h_eff
        if arg > zero(T)
            a_vals[i] = -log(arg) / h_eff
        else
            a_vals[i] = -li
        end
    end

    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Pre-filter + Delta Operator
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil_prefilter_delta(grid, f, N; stride, pencil_L, svd_tol) -> ExponentialSum

Moving-average pre-filter followed by subsampling, then **delta operator** MP.

Combines two orthogonal improvements:
  - Pre-filter reduces quantization/machine noise by ~√k
  - Delta operator maps eigenvalues to λ ≈ -a (avoids z→1 clustering)

When stride=1 (sparse grid), falls back to delta with L=M/2.
"""
function matrix_pencil_prefilter_delta(grid::AbstractVector{T}, f::AbstractVector{T},
                                       N::Int;
                                       stride::Int=0, pencil_L::Int=0,
                                       svd_tol::Real=0) where {T<:Real}
    M = length(f)
    @assert length(grid) == M
    h = grid[2] - grid[1]
    domain_len = grid[end] - grid[1]

    if stride <= 0
        h_target = domain_len / (5 * N)
        stride = max(1, round(Int, Float64(h_target / h)))
    end

    k = stride

    if k <= 1
        L = pencil_L > 0 ? pencil_L : div(M - 1, 2)
        return matrix_pencil_delta(grid, f, N; pencil_L=L, svd_tol=svd_tol)
    end

    M_filt = M - k + 1
    g = zeros(T, M_filt)
    inv_k = one(T) / T(k)
    @inbounds for i in 1:M_filt
        s = zero(T)
        for m in 0:(k - 1)
            s += f[i + m]
        end
        g[i] = s * inv_k
    end

    sub_idx = 1:k:M_filt
    g_sub = g[sub_idx]
    grid_sub = grid[sub_idx]
    M_sub = length(g_sub)

    if M_sub < 2N + 1
        L = pencil_L > 0 ? pencil_L : div(M - 1, 2)
        return matrix_pencil_delta(grid, f, N; pencil_L=L, svd_tol=svd_tol)
    end

    L = pencil_L > 0 ? pencil_L : div(M_sub - 1, 2)
    L = clamp(L, N, M_sub - N - 1)
    h_sub = grid_sub[2] - grid_sub[1]

    nrow = M_sub - L
    Y = zeros(T, nrow, L + 1)
    @inbounds for i in 1:nrow, j in 1:(L + 1)
        Y[i, j] = g_sub[i + j - 1]
    end

    Y0 = Y[:, 1:L]
    Y1 = Y[:, 2:(L + 1)]
    Y_delta = (Y1 - Y0) / h_sub

    F = svd(Y0)
    svd_n = count(F.S .> svd_tol)
    (svd_n < N) && @warn "Prefilter-delta SVD truncated to rank $svd_n < $N"
    Neff = min(svd_n, N)

    U_N = F.U[:, 1:Neff]
    S_inv = Diagonal(one(T) ./ F.S[1:Neff])
    V_N = F.V[:, 1:Neff]

    Z_delta = S_inv * (U_N' * Y_delta * V_N)
    lam = eigvals(Z_delta)

    a_vals = Vector{T}(undef, Neff)
    for i in 1:Neff
        li = T(real(lam[i]))
        arg = one(T) + li * h_sub
        if arg > zero(T)
            a_vals[i] = -log(arg) / h_sub
        else
            a_vals[i] = -li
        end
    end

    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Adaptive Delta: cross-validation over (stride, L)
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil_delta_auto(grid, f, N; svd_tol) -> ExponentialSum

Automatically selects the best (stride, L) combination for delta operator MP
via internal RMSE evaluation on the input data.

Tries several candidate configurations spanning:
  - stride ∈ {1, auto_stride/2, auto_stride, auto_stride*2}
  - L ∈ {3N, M/(s+1), M/2/s} for each stride s

Picks the configuration with smallest reconstruction RMSE.
"""
function matrix_pencil_delta_auto(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                                  svd_tol::Real=0) where {T<:Real}
    M = length(f)
    h = grid[2] - grid[1]
    domain_len = grid[end] - grid[1]

    h_target = domain_len / (5 * N)
    auto_stride = max(1, round(Int, Float64(h_target / h)))

    stride_candidates = sort(unique(clamp.([1, max(1, div(auto_stride, 2)), auto_stride,
                                            min(div(M, 2N + 1), auto_stride * 2)],
                                           1, max(1, div(M, 2N + 1)))))

    best_result = nothing
    best_rmse = T(Inf)

    for s in stride_candidates
        L_balanced = max(N, div(M, s + 1))
        L_3n = clamp(min(3 * N, div(M, 3)), N, M - 1)
        L_half = max(N, div(M, 2 * s))

        L_candidates = sort(unique(clamp.([L_3n, L_balanced, L_half], N, M - 1)))

        for L in L_candidates
            nrow = M - s * L
            nrow < N + 1 && continue

            res = _try_delta_delayed(grid, f, N, s, L, h, svd_tol)
            res === nothing && continue

            r = rmse(res, grid, f)
            if r < best_rmse
                best_rmse = r
                best_result = res
            end
        end
    end

    if best_result === nothing
        return matrix_pencil_delta_balanced(grid, f, N; svd_tol=svd_tol)
    end
    return best_result
end

function _try_delta_delayed(grid::AbstractVector{T}, f::AbstractVector{T},
                            N::Int, stride::Int, L::Int, h::T,
                            svd_tol::Real) where {T<:Real}
    M = length(f)
    nrow = M - stride * L
    nrow < N + 1 && return nothing
    h_eff = stride * h

    Y0 = zeros(T, nrow, L)
    Y1 = zeros(T, nrow, L)
    @inbounds for i in 1:nrow, j in 1:L
        Y0[i, j] = f[i + stride * (j - 1)]
        Y1[i, j] = f[i + stride * j]
    end
    Y_delta = (Y1 - Y0) / h_eff

    F = try
        ; svd(Y0);
    catch
        ; return nothing;
    end
    svd_n = count(F.S .> svd_tol)
    svd_n < 1 && return nothing
    Neff = min(svd_n, N)

    U_N = F.U[:, 1:Neff]
    S_inv = Diagonal(one(T) ./ F.S[1:Neff])
    V_N = F.V[:, 1:Neff]

    Z_delta = S_inv * (U_N' * Y_delta * V_N)
    lam = try
        ; eigvals(Z_delta);
    catch
        ; return nothing;
    end

    a_vals = Vector{T}(undef, Neff)
    for i in 1:Neff
        li = T(real(lam[i]))
        arg = one(T) + li * h_eff
        if arg > zero(T)
            a_vals[i] = -log(arg) / h_eff
        else
            a_vals[i] = -li
        end
    end

    any(a_vals .< zero(T)) && return nothing

    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Bilinear (Cayley/Tustin) Transform Matrix Pencil
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil_bilinear(grid, f, N; stride, pencil_L, svd_tol) -> ExponentialSum

Matrix Pencil using the bilinear (Cayley/Tustin) transform instead of
the first-order delta operator.

Standard MP: z = e^{-ah} (clusters near 1 when h→0)
First-order delta: λ = (z-1)/h ≈ -a + a²h/2 + ⋯ (O(h) linearization error)
**Bilinear**: γ = 2(z-1)/(h(z+1)) → recovers a = 2/h·atanh(-γh/2) **exactly**

The bilinear transform maps the z-plane to the γ-plane via the Möbius
transform, preserving the analytic structure. Unlike first-order delta,
it has zero systematic bias for any h — the only error is numerical.

Implementation via generalized eigenvalue problem:
  1. Build Y₀, Y₁ (with optional stride-k for dense grids)
  2. SVD of Y₀ → U, S, V; project to signal subspace
  3. Compute Z_num = Σ⁻¹ Uᵀ(Y₁-Y₀)V and Z_den = Σ⁻¹ Uᵀ(Y₁+Y₀)V
  4. Generalized eigenvalues μ of (Z_num, Z_den)
  5. Recover a = -2/h_eff · atanh(μ)  (exact for real exponentials)
"""
function matrix_pencil_bilinear(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                                stride::Int=0, pencil_L::Int=0,
                                svd_tol::Real=0) where {T<:Real}
    M = length(f)
    @assert length(grid) == M
    h = grid[2] - grid[1]
    domain_len = grid[end] - grid[1]

    if stride <= 0
        h_target = domain_len / (5 * N)
        stride = max(1, round(Int, Float64(h_target / h)))
    end

    if pencil_L > 0
        L = pencil_L
    else
        L = max(N, div(M, stride + 1))
    end

    nrow = M - stride * L
    while nrow < N + 1 && stride > 1
        stride -= 1
        nrow = M - stride * L
    end
    while nrow < N + 1 && L > N
        L -= 1
        nrow = M - stride * L
    end
    @assert nrow >= N + 1 "Cannot build bilinear Hankel: M=$M, stride=$stride, L=$L → nrow=$nrow < $(N+1)"

    h_eff = stride * h

    Y0 = zeros(T, nrow, L)
    Y1 = zeros(T, nrow, L)
    @inbounds for i in 1:nrow, j in 1:L
        Y0[i, j] = f[i + stride * (j - 1)]
        Y1[i, j] = f[i + stride * j]
    end

    F = svd(Y0)
    svd_n = count(F.S .> svd_tol)
    (svd_n < N) && @warn "Bilinear SVD truncated to rank $svd_n < $N"
    Neff = min(svd_n, N)

    U_N = F.U[:, 1:Neff]
    S_inv = Diagonal(one(T) ./ F.S[1:Neff])
    V_N = F.V[:, 1:Neff]

    Y_diff = Y1 - Y0
    Y_sum = Y1 + Y0
    Z_num = S_inv * (U_N' * Y_diff * V_N)
    Z_den = S_inv * (U_N' * Y_sum * V_N)

    # Standard eigenvalue of inv(Z_den)*Z_num is more stable than
    # generalized eigvals(Z_num, Z_den) for small matrices
    Z_bi = Z_den \ Z_num
    mu = eigvals(Z_bi)

    a_vals = Vector{T}(undef, Neff)
    for i in 1:Neff
        mi = T(real(mu[i]))
        abs_mi = abs(mi)
        if abs_mi < one(T)
            a_vals[i] = -T(2) / h_eff * atanh(mi)
        else
            a_vals[i] = -log(abs((one(T) + mi) / (one(T) - mi))) / h_eff
        end
    end

    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Pre-filter + Bilinear Transform
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil_prefilter_bilinear(grid, f, N; stride, pencil_L, svd_tol) -> ExponentialSum

Moving-average pre-filter + subsampling + bilinear transform.

Combines three orthogonal improvements:
  - Pre-filter reduces noise by ~√k
  - Subsampling restores pole separation  
  - Bilinear transform provides exact eigenvalue-to-a mapping (zero bias)

Falls back to plain bilinear with L=M/2 when stride=1 (sparse grid).
"""
function matrix_pencil_prefilter_bilinear(grid::AbstractVector{T}, f::AbstractVector{T},
                                          N::Int;
                                          stride::Int=0, pencil_L::Int=0,
                                          svd_tol::Real=0) where {T<:Real}
    M = length(f)
    @assert length(grid) == M
    h = grid[2] - grid[1]
    domain_len = grid[end] - grid[1]

    if stride <= 0
        h_target = domain_len / (5 * N)
        stride = max(1, round(Int, Float64(h_target / h)))
    end

    k = stride

    if k <= 1
        L = pencil_L > 0 ? pencil_L : div(M - 1, 2)
        return matrix_pencil_bilinear(grid, f, N; stride=1, pencil_L=L, svd_tol=svd_tol)
    end

    M_filt = M - k + 1
    g = zeros(T, M_filt)
    inv_k = one(T) / T(k)
    @inbounds for i in 1:M_filt
        s = zero(T)
        for m in 0:(k - 1)
            s += f[i + m]
        end
        g[i] = s * inv_k
    end

    sub_idx = 1:k:M_filt
    g_sub = g[sub_idx]
    grid_sub = grid[sub_idx]
    M_sub = length(g_sub)

    if M_sub < 2N + 1
        L = pencil_L > 0 ? pencil_L : div(M - 1, 2)
        return matrix_pencil_bilinear(grid, f, N; stride=1, pencil_L=L, svd_tol=svd_tol)
    end

    L = pencil_L > 0 ? pencil_L : div(M_sub - 1, 2)
    L = clamp(L, N, M_sub - N - 1)
    h_sub = grid_sub[2] - grid_sub[1]

    nrow = M_sub - L
    Y = zeros(T, nrow, L + 1)
    @inbounds for i in 1:nrow, j in 1:(L + 1)
        Y[i, j] = g_sub[i + j - 1]
    end

    Y0 = Y[:, 1:L]
    Y1 = Y[:, 2:(L + 1)]
    Y_diff = Y1 - Y0
    Y_sum = Y1 + Y0

    F = svd(Y0)
    svd_n = count(F.S .> svd_tol)
    (svd_n < N) && @warn "Prefilter-bilinear SVD truncated to rank $svd_n < $N"
    Neff = min(svd_n, N)

    U_N = F.U[:, 1:Neff]
    S_inv = Diagonal(one(T) ./ F.S[1:Neff])
    V_N = F.V[:, 1:Neff]

    Z_num = S_inv * (U_N' * Y_diff * V_N)
    Z_den = S_inv * (U_N' * Y_sum * V_N)
    Z_bi = Z_den \ Z_num
    mu = eigvals(Z_bi)

    a_vals = Vector{T}(undef, Neff)
    for i in 1:Neff
        mi = T(real(mu[i]))
        abs_mi = abs(mi)
        if abs_mi < one(T)
            a_vals[i] = -T(2) / h_sub * atanh(mi)
        else
            a_vals[i] = -log(abs((one(T) + mi) / (one(T) - mi))) / h_sub
        end
    end

    c_vals = _solve_amplitudes(grid, f, a_vals)
    return _pack(c_vals, a_vals)
end

# ════════════════════════════════════════════════════════════
#  Universal Adaptive: tries bilinear, delta, and standard MP
# ════════════════════════════════════════════════════════════

"""
    matrix_pencil_universal(grid, f, N; svd_tol) -> ExponentialSum

Automatically selects the best method and parameters via RMSE evaluation.

Tries candidates from multiple method families:
  - Standard MP (good for sparse grids)
  - Delta + delayed (good for dense grids)
  - Bilinear + delayed (good across all h, exact eigenvalue mapping)
  - Prefilter variants (good for medium grids)

Scoring: RMSE on input data, with penalty for near-duplicate poles
(which indicate numerical artifacts rather than genuine decomposition).
"""
function matrix_pencil_universal(grid::AbstractVector{T}, f::AbstractVector{T}, N::Int;
                                 svd_tol::Real=0) where {T<:Real}
    M = length(f)
    h = grid[2] - grid[1]
    domain_len = grid[end] - grid[1]

    best_result = nothing
    best_score = T(Inf)

    function score_result(res)
        r = rmse(res, grid, f)
        penalty = one(T)
        sorted_a = sort(Float64.(res.a))
        for i in 2:length(sorted_a)
            gap = sorted_a[i] - sorted_a[i - 1]
            if gap < 0.01
                penalty *= T(100)
            elseif gap < 0.1
                penalty *= T(10)
            end
        end
        return r * penalty
    end

    function try_and_update(fn)
        res = try
            ; fn();
        catch
            ; nothing;
        end
        res === nothing && return
        length(res.a) < N && return
        any(res.a .< zero(T)) && return
        s = score_result(res)
        if s < best_score
            best_score = s
            best_result = res
        end
    end

    h_target = domain_len / (5 * N)
    auto_stride = max(1, round(Int, Float64(h_target / h)))
    stride_set = sort(unique(clamp.([1, max(1, div(auto_stride, 2)), auto_stride,
                                     min(div(M, 2N + 1), max(1, auto_stride * 2))],
                                    1, max(1, div(M, 2N + 1)))))

    for s in stride_set
        L_balanced = max(N, div(M, s + 1))
        L_half_s = max(N, div(M, 2 * s))
        L_3n = clamp(min(3 * N, div(M, 3)), N, M - 1)
        L_set = sort(unique(clamp.([L_3n, L_balanced, L_half_s], N, M - 1)))

        for L in L_set
            nrow = M - s * L
            nrow < N + 1 && continue
            try_and_update(() -> matrix_pencil_bilinear(grid, f, N; stride=s, pencil_L=L))
            try_and_update(() -> _try_delta_delayed(grid, f, N, s, L, h, svd_tol))
        end
    end

    L_std_half = max(N, div(M - 1, 2))
    L_std_3n = clamp(min(3 * N, div(M, 3)), N, M - N - 1)
    for L in sort(unique([L_std_half, L_std_3n]))
        try_and_update(() -> matrix_pencil(grid, f, N; pencil_L=L))
    end

    try_and_update(() -> matrix_pencil_prefilter(grid, f, N))
    try_and_update(() -> matrix_pencil_prefilter_bilinear(grid, f, N))

    if best_result === nothing
        return matrix_pencil(grid, f, N)
    end
    return best_result
end
