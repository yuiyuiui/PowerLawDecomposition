using Test, Random, LinearAlgebra
using PowerLawDecomposition

@testset "power_solve wynn pola" begin
    Random.seed!(666)
    T = Float64
    nord = 10
    d = 0.5
    a_vec = T[]
    for i in 1:nord
        if i == 1
            push!(a_vec, (1 + rand()) / 2)
        else
            push!(a_vec, a_vec[end] + (1 + rand()) * d)
        end
    end
    c_vec = T[]
    for i in 1:nord
        push!(c_vec, 1 + rand())
    end

    L0 = 1
    L = 2^14
    point_density = 10

    _a_vec = T.(a_vec)
    _c_vec = T.(c_vec)
    f = x -> sum(_c_vec .* x .^ (-_a_vec))
    h = T(1 / point_density)
    grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
    f_data = f.(grid)

    f_data = BigFloat.(f_data)
    grid = BigFloat.(grid)
    nseek = 4
    is_use_a_final = false
    n1 = 21
    n2 = 21
    n3 = 21
    n4 = 21
    lm1 = WynnPola(; k=big"1.3", n=n1, use_a_final=is_use_a_final)
    lm2 = WynnPola(; k=big"1.3", n=n2, use_a_final=is_use_a_final)
    lm3 = WynnPola(; k=big"1.3", n=n3, use_a_final=is_use_a_final)
    lm4 = WynnPola(; k=big"1.3", n=n4, use_a_final=is_use_a_final)
    lm_vec = [lm1, lm2, lm3, lm4]
    iter_seek = IterSeek(lm_vec, 1, length(f_data), nseek)
    order_vec = power_solve(f_data, grid, iter_seek)
    @show n1, n2, n3
    @show norm.(order_vec .- _a_vec[1:nseek])
end

@testset "power_solve wynn" begin
    Random.seed!(666)
    T = Float64
    nord = 10
    d = 0.5
    a_vec = T[]
    for i in 1:nord
        if i == 1
            push!(a_vec, (1 + rand()) / 2)
        else
            push!(a_vec, a_vec[end] + (1 + rand()) * d)
        end
    end
    c_vec = T[]
    for i in 1:nord
        push!(c_vec, 1 + rand())
    end

    L0 = 1
    L = 2^14
    point_density = 10

    _a_vec = T.(a_vec)
    _c_vec = T.(c_vec)
    f = x -> sum(_c_vec .* x .^ (-_a_vec))
    h = T(1 / point_density)
    grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
    f_data = f.(grid)

    nseek = 3
    is_use_a_final = false
    n1 = 7
    n2 = 7
    n3 = 7
    lm1 = Wynn(; k=2, n=n1)
    lm2 = Wynn(; k=2, n=n2)
    lm3 = Wynn(; k=2, n=n3)
    lm_vec = [lm1, lm2, lm3]
    iter_seek = IterSeek(lm_vec, 1, length(f_data), nseek)
    order_vec = power_solve(f_data, grid, iter_seek)
    @show norm.(order_vec .- _a_vec[1:nseek])
end
