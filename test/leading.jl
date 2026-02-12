@testset "leading.jl" begin
    Random.seed!(123456789)
    a1 = 0.55
    a2 = 1.22
    a3 = 1.83
    c1, c2, c3 = rand(3) .+ 1
    #=
    1.0302671169218902
    1.0857778139437397
    1.114727702313528
    =#
    for T in [Float32, Float64, BigFloat]
        @show T
        _a1, _a2, _a3 = T.([a1, a2, a3])
        _c1, _c2, _c3 = T.([c1, c2, c3])
        f = x -> _c1 * x^(-_a1) + _c2 * x^(-_a2) + _c3 * x^(-_a3) * sin(x^2)
        L0 = 1
        L = 2^14
        point_density = 10
        h = T(1 / point_density)
        grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
        f_data = f.(grid)

        order_loglog, coff_loglog, _ = loglog_fit(f_data, grid)
        order_shanks, coff_shanks = shanks_int_fit(f_data, grid)

        @show abs(order_loglog - _a1)
        @show abs(coff_loglog - _c1)
        @show abs(order_shanks - _a1)
        @show abs(coff_shanks - _c1)

        @test order_loglog isa T
        @test coff_loglog isa T
        @test order_shanks isa T
        @test coff_shanks isa T
        if T == Float64
            @test abs(order_loglog - _a1) < 0.0072
            @test abs(coff_loglog - _c1) < 0.071
            @test abs(order_shanks - _a1) < 1.4e-5
            @test abs(coff_shanks - _c1) < 7e-4
        end
    end
end
