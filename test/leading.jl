@testset "leading.jl" begin
    T = Float64
    a1 = T(0.55)
    a2 = T(1.22)
    a3 = T(1.83)
    c1, c2, c3 = rand(T, 3) .+ 1
    #=
     1.0302671169218902
    1.0857778139437397
    1.114727702313528

    =#
    f = x -> c1*x^(-a1) + c2*x^(-a2) + c3 * x^(-a3) * sin(x^2)
    L0 = 1
    L = 2^14
    point_density = 10
    h = T(1/point_density)
    grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
    f_data = f.(grid)

    order_loglog, coff_loglog, _ = loglog_fit(f_data, grid)
    order_shanks, coff_shanks = shanks_int_fit(f_data, grid)

    @show abs(order_loglog - a1)
    @show abs(coff_loglog - c1)
    @show abs(order_shanks - a1)
    @show abs(coff_shanks - c1)

    @test abs(order_loglog - a1) < 0.0072
    @test abs(coff_loglog - c1) < 0.071
    @test abs(order_shanks - a1) < 1.4e-5
    @test abs(coff_shanks - c1) < 7e-4
end
