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
    @inbounds for i in 2:N; C[i, i - 1] = one(T); end
    @inbounds for i in 1:N; C[i, N] = -p[N + 1 - i]; end

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
                       svd_tol::Real = 0, pencil_L::Int = 0) where {T<:Real}
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
