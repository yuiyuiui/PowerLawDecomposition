@testset "LagrangePola interpolation" begin
    @testset "correctness — exact for polynomials" begin
        grid = collect(range(0.0, 2.0; length=101))
        f = @. 3grid^5 - 2grid^3 + grid - 7
        pm = LagrangePola(hw=10)

        for x in [0.01, 0.5, 1.0, 1.5, 1.99]
            exact = 3x^5 - 2x^3 + x - 7
            @test isapprox(interpolate(grid, f, x, pm), exact; atol=1e-10)
        end
    end

    @testset "correctness — sin function" begin
        grid = collect(range(0.0, 2π; length=201))
        f = sin.(grid)
        pm = LagrangePola(hw=10)

        for x in [0.1, 1.0, π, 5.0, 6.0]
            @test isapprox(interpolate(grid, f, x, pm), sin(x); atol=1e-12)
        end
    end

    @testset "type stability" begin
        grid64 = collect(range(0.0, 1.0; length=50))
        f64 = sin.(grid64)
        pm = LagrangePola(hw=5)
        @test @inferred(interpolate(grid64, f64, 0.5, pm)) isa Float64

        grid32 = Float32.(grid64)
        f32 = Float32.(f64)
        @test @inferred(interpolate(grid32, f32, 0.5f0, pm)) isa Float32
    end

    @testset "boundary — x at grid endpoints" begin
        grid = collect(range(-1.0, 1.0; length=51))
        f = grid .^ 2
        pm = LagrangePola(hw=5)

        @test isapprox(interpolate(grid, f, grid[1], pm), f[1]; atol=1e-14)
        @test isapprox(interpolate(grid, f, grid[end], pm), f[end]; atol=1e-14)
    end

    @testset "boundary — x at exact grid points" begin
        grid = collect(range(0.0, 1.0; length=101))
        f = exp.(grid)
        pm = LagrangePola(hw=10)

        for idx in [1, 25, 50, 75, 101]
            @test interpolate(grid, f, grid[idx], pm) == f[idx]
        end
    end

    @testset "boundary — minimum hw" begin
        pm = LagrangePola(hw=3)
        grid = collect(range(0.0, 1.0; length=21))
        f = sin.(grid)
        @test isfinite(interpolate(grid, f, 0.5, pm))
    end

    @testset "edge — hw assertion" begin
        @test_throws AssertionError LagrangePola(hw=2)
    end

    @testset "edge — x out of range" begin
        grid = collect(range(0.0, 1.0; length=51))
        f = sin.(grid)
        pm = LagrangePola(hw=5)
        @test_throws AssertionError interpolate(grid, f, -1.0, pm)
        @test_throws AssertionError interpolate(grid, f, 2.0, pm)
    end

    @testset "edge — grid too small for hw" begin
        grid = collect(range(0.0, 1.0; length=5))
        f = sin.(grid)
        pm = LagrangePola(hw=10)
        @test_throws AssertionError interpolate(grid, f, 0.5, pm)
    end

    @testset "precision — convergence with increasing hw" begin
        grid = collect(range(0.0, 2π; length=201))
        f = sin.(grid)
        x = 1.234

        errs = Float64[]
        for hw in [3, 5, 8, 10]
            pm = LagrangePola(hw=hw)
            push!(errs, abs(interpolate(grid, f, x, pm) - sin(x)))
        end
        for i in 2:length(errs)
            @test errs[i] <= errs[i - 1] + eps()
        end
    end

    @testset "precision — Float32" begin
        grid = Float32.(collect(range(0.0f0, 1.0f0; length=101)))
        f = sin.(grid)
        pm = LagrangePola(hw=5)
        x = 0.5f0
        @test isapprox(interpolate(grid, f, x, pm), sin(x); atol=1e-5)
    end
end

@testset "BsplinePola interpolation" begin
    @testset "correctness — smooth function" begin
        grid = collect(range(0.0, 2π; length=201))
        f = sin.(grid)
        pm = BsplinePola()

        for x in [0.5, 1.0, π, 5.0]
            @test isapprox(interpolate(grid, f, x, pm), sin(x); atol=1e-6)
        end
    end

    @testset "type stability" begin
        grid = collect(range(0.0, 1.0; length=50))
        f = sin.(grid)
        pm = BsplinePola()
        @test @inferred(interpolate(grid, f, 0.5, pm)) isa Float64
    end
end
