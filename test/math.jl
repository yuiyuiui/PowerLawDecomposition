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
            @test isapprox(bspline_interp(grid, f, x, pm), sin(x); atol=1e-6)
        end
    end

    @testset "type stability" begin
        grid = collect(range(0.0, 1.0; length=50))
        f = sin.(grid)
        pm = BsplinePola()
        @test @inferred(bspline_interp(grid, f, 0.5, pm)) isa Float64
    end
end

@testset "Simpson integration" begin
    im = Simpson()

    @testset "correctness — constant function" begin
        f = fill(3.0, 11)
        h = 0.1
        @test isapprox(integrate(f, h, im), 3.0; atol=1e-14)
    end

    @testset "correctness — linear" begin
        grid = range(0.0, 2.0; length=101)
        f = collect(2.0 .* grid .+ 1.0)
        h = step(grid)
        @test isapprox(integrate(f, h, im), 6.0; atol=1e-12)
    end

    @testset "correctness — quadratic (exact)" begin
        grid = range(0.0, 1.0; length=51)
        f = collect(grid .^ 2)
        h = step(grid)
        @test isapprox(integrate(f, h, im), 1 / 3; atol=1e-12)
    end

    @testset "correctness — cubic (exact)" begin
        grid = range(0.0, 1.0; length=51)
        f = collect(grid .^ 3)
        h = step(grid)
        @test isapprox(integrate(f, h, im), 0.25; atol=1e-12)
    end

    @testset "correctness — even N (trapezoidal fallback)" begin
        grid = range(0.0, 1.0; length=50)
        f = collect(grid .^ 2)
        h = step(grid)
        @test isapprox(integrate(f, h, im), 1 / 3; atol=1e-4)
    end

    @testset "correctness — sin integral" begin
        grid = range(0.0, π; length=1001)
        f = sin.(collect(grid))
        h = step(grid)
        @test isapprox(integrate(f, h, im), 2.0; atol=1e-10)
    end

    @testset "type stability" begin
        f64 = collect(range(0.0, 1.0; length=11))
        @test @inferred(integrate(f64, 0.1, im)) isa Float64

        f32 = Float32.(f64)
        @test @inferred(integrate(f32, 0.1f0, im)) isa Float32
    end

    @testset "boundary — N=2" begin
        f = [1.0, 2.0]
        @test isapprox(integrate(f, 1.0, im), 1.5; atol=1e-14)
    end

    @testset "boundary — N=3" begin
        f = [0.0, 1.0, 0.0]
        h = 1.0
        @test isapprox(integrate(f, h, im), 4 / 3; atol=1e-14)
    end

    @testset "edge — h must be positive" begin
        @test_throws AssertionError integrate([1.0, 2.0], -1.0, im)
        @test_throws AssertionError integrate([1.0, 2.0], 0.0, im)
    end

    @testset "edge — N=1 rejected" begin
        @test_throws AssertionError integrate([1.0], 1.0, im)
    end

    @testset "precision — O(h⁴) convergence" begin
        exact = 2.0
        errs = Float64[]
        ns = [51, 101, 201, 401]
        for n in ns
            grid = range(0.0, π; length=n)
            f = sin.(collect(grid))
            push!(errs, abs(integrate(f, step(grid), im) - exact))
        end
        for i in 2:length(errs)
            ratio = errs[i - 1] / max(errs[i], 1e-16)
            @test ratio > 10
        end
    end
end

@testset "fd8 finite differences" begin
    @testset "correctness — polynomial degree ≤ 8 (central)" begin
        N = 50
        h = 0.1
        grid = [h * (i - 1) for i in 1:N]
        f = grid .^ 7
        df_exact = 7 .* grid .^ 6

        df = fd8(f, h)
        for i in 5:(N - 4)
            @test isapprox(df[i], df_exact[i]; rtol=1e-6)
        end
    end

    @testset "correctness — sin function" begin
        N = 100
        h = 0.05
        grid = [h * (i - 1) for i in 1:N]
        f = sin.(grid)
        df_exact = cos.(grid)

        df = fd8(f, h)
        for i in 5:(N - 4)
            @test isapprox(df[i], df_exact[i]; atol=1e-10)
        end
    end

    @testset "correctness — constant (derivative = 0)" begin
        f = fill(42.0, 20)
        df = fd8(f, 1.0)
        @test all(x -> abs(x) < 1e-12, df)
    end

    @testset "correctness — linear" begin
        N = 20
        h = 0.5
        f = [3.0 * i * h + 1.0 for i in 0:(N - 1)]
        df = fd8(f, h)
        @test all(x -> isapprox(x, 3.0; atol=1e-10), df)
    end

    @testset "type stability" begin
        f64 = sin.(collect(range(0.0, 1.0; length=20)))
        @test @inferred(fd8(f64, 0.05)) isa Vector{Float64}

        u64 = zeros(20)
        @test @inferred(fd8!(u64, f64, 0.05)) isa Vector{Float64}
    end

    @testset "boundary — minimum length 12" begin
        f = sin.(collect(range(0.0, 1.0; length=12)))
        h = 1.0 / 11
        df = fd8(f, h)
        @test length(df) == 12
        @test all(isfinite, df)
    end

    @testset "boundary — length < 12 rejected" begin
        @test_throws AssertionError fd8(ones(11), 1.0)
    end

    @testset "edge — fd8! in-place" begin
        N = 20
        h = 0.1
        f = sin.(collect(range(0.0, (N - 1) * h; length=N)))
        u = zeros(N)
        fd8!(u, f, h)
        @test u == fd8(f, h)
    end

    @testset "precision — 8th-order convergence" begin
        function central_error(N)
            h = 2.0 / (N - 1)
            grid = collect(range(0.0, 2.0; length=N))
            f = exp.(grid)
            df = fd8(f, h)
            idx = round(Int, 1.0 / h) + 1
            return (abs(df[idx] - exp(1.0)), h)
        end

        # Two coarse grids where truncation error >> roundoff
        err1, h1 = central_error(21)   # h=0.1,  h^8=1e-8
        err2, h2 = central_error(41)   # h=0.05, h^8≈4e-11

        order = log(err1 / err2) / log(h1 / h2)
        @test order > 7.0  # expect ~8
    end
end
