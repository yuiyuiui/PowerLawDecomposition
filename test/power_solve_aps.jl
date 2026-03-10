using Test, Random, LinearAlgebra
using PowerLawDecomposition

@testset "power_solve_asp wynn pola" begin
    Random.seed!(666)
    T = BigFloat
    nord = 10
    d = T(0.5)
    a_vec = T[]
    for i in 1:nord
        if i == 1
            push!(a_vec, (1 + rand(T)) / 2)
        else
            push!(a_vec, a_vec[end] + (1 + rand(T)) * d)
        end
    end
    c_vec = T[]
    for i in 1:nord
        push!(c_vec, 1 + rand(T))
    end

    L0 = 1
    L = 2^16
    @show L
    point_density = 10

    _a_vec = T.(a_vec)
    _c_vec = T.(c_vec)
    f = x -> sum(_c_vec .* x .^ (-_a_vec))
    h = T(1 / point_density)
    grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
    f_data = f.(grid)

    #f_data = BigFloat.(f_data);
    #grid = BigFloat.(grid);

    nseek = 3
    asp = ASP(nseek, length(f_data); wynn_pola=WynnPola(; k=big"1.3", n=21))
    order_vec = power_solve_asp(f_data, grid, asp)
    res = norm.(order_vec .- _a_vec[1:nseek])
    for i in 1:nseek
        @show res[i]
    end
end
