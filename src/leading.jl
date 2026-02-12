function loglog_fit(f::AbstractVector{T}, grid::AbstractVector{T}; ntail::Int=10,
                    sign_rate::Real=0.9) where {T<:Real}
    N = length(f)

    @assert N == length(grid)
    grid_check(grid)
    @assert round(Int, (grid[end] - grid[1]) / (grid[2] - grid[1])) >= ntail "grid must be large enough"

    n = ntail
    sign_flag = 0
    while n >= 1
        fview = view(f, (n + 1):N)
        if isempty(fview)
            n = floor(Int, n * sign_rate)
            n < 1 && error("data too short for sign detection in loglog demode")
            continue
        end
        if all(abs.(fview) .< 1000 * eps(T))
            println("all elements are close to zero, stopping loglog demode")
            return 0, 0, n
        end
        if all(fview .> 0)
            sign_flag = 1
            break
        elseif all(fview .< 0)
            sign_flag = -1
            break
        else
            n = floor(Int, n * sign_rate)
            if n < 1
                error("alternating positive and negative, stopping loglog demode")
            end
        end
    end

    e = ones(T, N)
    logf = log.(f * sign_flag)
    logx = log.(grid)
    # k, b = [dot(logx, logx) dot(logx, e); dot(e, logx) dot(e, e)] \ [dot(logx, logf); dot(e, logf)]
    A = hcat(logx, e)
    k, b = A \ logf
    order = -k
    coff = exp(b) * sign_flag
    return order, coff, n
end

function shanks_int_fit(f::AbstractVector{T}, grid::AbstractVector{T};
                        k::Int=2) where {T<:Real}
    grid_check(grid)
    N = length(f)
    @assert grid[end] / k^2 > 1 "grid must be large enough"

    Svec = zeros(T, 4)
    Avec = zeros(T, 3)
    Cvec = zeros(T, 3) # for storing local estimates of c
    nvec = zeros(Int, 5)

    h, N0 = get_point_num(grid)
    idx_shift = N0 - N
    @assert floor(Int, grid[end] / k^4) >= grid[1] "grid must be large enough"

    # 1. get sampling indices
    nvec[1] = floor(Int, N0 / k^4)
    for i in 2:5
        nvec[i] = nvec[i - 1] * k
    end

    # 2. calculate interval integrals
    for i in 1:4
        Svec[i] = int_simpson(view(f, (nvec[i] - idx_shift):(nvec[i + 1] - idx_shift)), h)
    end

    # 3. calculate apparent order sequence A and coefficient sequence C
    for i in 1:3
        # actual physical coordinates
        x_low = nvec[i + 1] * h
        x_up = nvec[i + 2] * h

        # calculate apparent order A_i (based on S_i and S_{i+1})
        # 注意：Avec[i] 实际上反映的是从 x_i 到 x_{i+2} 范围内的平均特性
        Avec[i] = 1 - log(Svec[i + 1] / Svec[i]) / log(k)

        # use the just calculated A_i, inverse to find the c_i of the interval
        # formula: c = S * (1-a) / (x_up^(1-a) - x_low^(1-a))
        # we use Svec[i+1] (the integral at the higher physical position) to calculate, because it is closer to the limit
        a_tmp = Avec[i]
        Cvec[i] = Svec[i + 1] * (1 - a_tmp) / (x_up^(1 - a_tmp) - x_low^(1 - a_tmp))
    end

    # 4. perform Shanks transformation on a
    da2 = Avec[3] - Avec[2]
    da1 = Avec[2] - Avec[1]
    a_refined = Avec[3] - (da2^2) / (da2 - da1)

    # 5. perform Shanks transformation on c
    dc2 = Cvec[3] - Cvec[2]
    dc1 = Cvec[2] - Cvec[1]
    c_refined = Cvec[3] - (dc2^2) / (dc2 - dc1)

    return a_refined, c_refined
end
