@testset "leading.jl" begin
    Random.seed!(123456789)
    a1 = 0.55
    a2 = 1.22
    a3 = 1.83
    c1, c2, c3 = rand(3) .+ 1
    wynn_n = 3
    wynn_polarate = 2
    #=
    1.0302671169218902
    1.0857778139437397
    1.114727702313528
    =#
    for T in [Float32, Float64]
        @show T
        @show wynn_n
        _a1, _a2, _a3 = T.([a1, a2, a3])
        _c1, _c2, _c3 = T.([c1, c2, c3])
        f = x -> _c1 * x^(-_a1) + _c2 * x^(-_a2) + _c3 * x^(-_a3)
        L0 = 1
        L = 2^14
        point_density = 10
        h = T(1 / point_density)
        grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
        f_data = f.(grid)

        order_loglog_arr, coff_loglog_arr, _ = power_solve(f_data, grid, 1; method=LogLog())
        order_shanks_arr, coff_shanks_arr, _ = power_solve(f_data, grid, 1;
                                                           method=Shanks(k=wynn_polarate))
        order_wynn_arr, coff_wynn_arr, _ = power_solve(f_data, grid, 1;
                                                       method=Wynn(k=wynn_polarate,
                                                                   n=wynn_n))
        order_wynn_pola_arr, coff_wynn_pola_arr, _ = power_solve(f_data, grid, 1;
                                                                 method=WynnPola(k=wynn_polarate,
                                                                                 n=wynn_n))

        @show abs(order_loglog_arr[1] - _a1)
        @show abs(coff_loglog_arr[1] - _c1)
        @show abs(order_shanks_arr[1] - _a1)
        @show abs(coff_shanks_arr[1] - _c1)
        @show abs(order_wynn_arr[1] - _a1)
        @show abs(coff_wynn_arr[1] - _c1)
        @show abs(order_wynn_pola_arr[1] - _a1)
        @show abs(coff_wynn_pola_arr[1] - _c1)

        @test order_loglog_arr isa Array{T,1}
        @test length(order_loglog_arr) == 1
        @test coff_loglog_arr isa Array{T,1}
        @test length(coff_loglog_arr) == 1
        @test order_shanks_arr isa Array{T,1}
        @test length(order_shanks_arr) == 1
        @test coff_shanks_arr isa Array{T,1}
        @test length(coff_shanks_arr) == 1
        @test order_wynn_arr isa Array{T,1}
        @test length(order_wynn_arr) == 1
        @test coff_wynn_arr isa Array{T,1}
        @test length(coff_wynn_arr) == 1
        @test order_wynn_pola_arr isa Array{T,1}
        @test length(order_wynn_pola_arr) == 1
        @test coff_wynn_pola_arr isa Array{T,1}
        @test length(coff_wynn_pola_arr) == 1
        if T == Float64
            @test abs(order_loglog_arr[1] - _a1) < 0.008
            @test abs(coff_loglog_arr[1] - _c1) < 0.08
            @test abs(order_shanks_arr[1] - _a1) < 1.7e-5
            @test abs(coff_shanks_arr[1] - _c1) < 3e-4
            @test abs(order_wynn_arr[1] - _a1) < 1.7e-5
            @test abs(coff_wynn_arr[1] - _c1) < 3e-4
            @test abs(order_wynn_pola_arr[1] - _a1) < 3.7e-5
            @test abs(coff_wynn_pola_arr[1] - _c1) < 3e-4
        end
    end
end

@testset "wynn_pola.jl" begin
    Random.seed!(123456789)
    a1 = 0.55
    a2 = 1.22
    a3 = 1.83
    c1, c2, c3 = rand(3) .+ 1
    wynn_n = 21
    wynn_polarate = 1.3
    #=
    1.0302671169218902
    1.0857778139437397
    1.114727702313528
    =#
    for T in [Float32, Float64, BigFloat]
        @show T
        @show wynn_n
        _a1, _a2, _a3 = T.([a1, a2, a3])
        _c1, _c2, _c3 = T.([c1, c2, c3])
        f = x -> _c1 * x^(-_a1) + _c2 * x^(-_a2) + _c3 * x^(-_a3)
        L0 = 1
        L = 2^14
        point_density = 10
        h = T(1 / point_density)
        grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
        f_data = f.(grid)

        order_arr, coff_arr, _ = power_solve(f_data, grid, 1;
                                             method=WynnPola(k=wynn_polarate, n=wynn_n))
        order = order_arr[1]
        coff = coff_arr[1]

        @show abs(order - _a1)
        @show abs(coff - _c1)

        if T == Float64
            @test abs(order - _a1) < 2e-11
            @test abs(coff - _c1) < 1.1e-7
        elseif T == BigFloat
            @test abs(order - _a1) < 6e-13
            @test abs(coff - _c1) < 1.6e-8
        end
    end
end
