@testset "wynn.jl" begin
    Random.seed!(123456789)
    a1 = 0.55 + 0.01 * rand()
    a2 = 1.22 + 0.01 * rand()
    a3 = 1.83 + 0.01 * rand()
    #=
    0.5503026711692189
    1.2208577781394374
    1.8311472770231354
    =#
    c1, c2, c3 = rand(3) .+ 1
    #=
    1.1576090409417932
    1.8329810111138598
    1.7569774924284722
    =#
    L0 = 1
    L = 2^14
    point_density = 100

    for T in [Float32, Float64]
        @show T
        _a1, _a2, _a3 = T.([a1, a2, a3])
        _c1, _c2, _c3 = T.([c1, c2, c3])
        f = x -> _c1 * x^(-_a1) + _c2 * x^(-_a2) + _c3 * x^(-_a3)
        h = T(1 / point_density)
        grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
        f_data = f.(grid)

        order_wynnpola, coff_wynnpola = leading_solve(f_data, grid,
                                                      WynnPola(; k=1.3, n=21))
        order_wynn, coff_wynn = leading_solve(f_data, grid,
                                              Wynn(; k=2, n=7))

        @show abs(order_wynnpola - _a1)
        @show abs(coff_wynnpola - _c1)
        @show abs(order_wynn - _a1)
        @show abs(coff_wynn - _c1)

        if T == Float64
            @test abs(order_wynnpola - _a1) < 2e-11
            @test abs(coff_wynnpola - _c1) < 2e-6
            @test abs(order_wynn - _a1) < 8e-8
            @test abs(coff_wynn - _c1) < 2e-4
        end
    end

    T = BigFloat
    @show T
    _a1, _a2, _a3 = T.([a1, a2, a3])
    _c1, _c2, _c3 = T.([c1, c2, c3])
    f = x -> _c1 * x^(-_a1) + _c2 * x^(-_a2) + _c3 * x^(-_a3)
    h = T(1 / point_density)
    grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
    f_data = f.(grid)

    order, coff = leading_solve(f_data, grid, WynnPola(; k=1.2, n=41))

    @show abs(order - _a1)
    @show abs(coff - _c1)

    if T == BigFloat
        @test abs(order - _a1) < 4e-16
        @test abs(coff - _c1) < 4e-10
    end
end

@testset "wynn_pola with use_a_final=true" begin
    Random.seed!(123456789)
    a1 = 0.55 + 0.01 * rand()
    a2 = 1.22 + 0.01 * rand()
    a3 = 1.83 + 0.01 * rand()
    #=
    0.5503026711692189
    1.2208577781394374
    1.8311472770231354
    =#
    c1, c2, c3 = rand(3) .+ 1
    #=
    1.1576090409417932
    1.8329810111138598
    1.7569774924284722
    =#
    L0 = 1
    L = 2^14
    point_density = 10
    for T in [Float64, BigFloat]
        @show T
        _a1, _a2, _a3 = T.([a1, a2, a3])
        _c1, _c2, _c3 = T.([c1, c2, c3])
        f = x -> _c1 * x^(-_a1) + _c2 * x^(-_a2) + _c3 * x^(-_a3)
        h = T(1 / point_density)
        grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
        f_data = f.(grid)

        order_false, coff_false = leading_solve(f_data, grid,
                                                WynnPola(; k=1.3, n=21,
                                                         use_a_final=false))

        order_true, coff_true = leading_solve(f_data, grid,
                                              WynnPola(; k=1.3, n=21,
                                                       use_a_final=true))

        @show abs(order_false - _a1)
        @show abs(coff_false - _c1)
        @show abs(order_true - _a1)
        @show abs(coff_true - _c1)

        if T == Float64
            @test abs(order_false - _a1) < 3e-11
            @test abs(coff_false - _c1) < 5e-5
            @test abs(order_true - _a1) < 3e-11
            @test abs(coff_true - _c1) < 4e-10
        elseif T == BigFloat
            @test abs(order_false - _a1) < 2e-12
            @test abs(coff_false - _c1) < 2e-8
            @test abs(order_true - _a1) < 2e-12
            @test abs(coff_true - _c1) < 3e-11
        end
    end
end
