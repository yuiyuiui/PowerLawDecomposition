@testset "kit.jl" begin
    @testset "uniform_check" begin
        # uniform grid should pass
        @test PowerLawDecomposition.uniform_check([1.0, 2.0, 3.0, 4.0]) == true
        @test PowerLawDecomposition.uniform_check([0.5, 1.0, 1.5]) == true

        # non-uniform grid should throw error
        @test_throws ErrorException PowerLawDecomposition.uniform_check([1.0, 2.0, 4.0])
        @test_throws ErrorException PowerLawDecomposition.uniform_check([1.0, 3.0, 4.0,
                                                                         5.0])
    end

    @testset "grid_check" begin
        # valid grid should pass
        @test PowerLawDecomposition.grid_check([1.0, 2.0, 3.0, 4.0]) == true
        @test PowerLawDecomposition.grid_check([0.5, 1.0, 1.5, 2.0]) == true

        # grid must be positive
        @test_throws AssertionError PowerLawDecomposition.grid_check([-1.0, 0.0, 1.0])
        @test_throws AssertionError PowerLawDecomposition.grid_check([0.0, 1.0, 2.0])

        # grid must have at least 2 points
        @test_throws AssertionError PowerLawDecomposition.grid_check([1.0])

        # grid must be increasing
        @test_throws AssertionError PowerLawDecomposition.grid_check([3.0, 2.0, 1.0])

        # grid must be uniform
        @test_throws ErrorException PowerLawDecomposition.grid_check([1.0, 2.0, 5.0])
    end

    @testset "get_point_num" begin
        # grid = [1.0, 2.0, 3.0, 4.0], h=1.0, grid[1]/h=1, N0 = 4+1-1 = 4
        h, N0 = PowerLawDecomposition.get_point_num([1.0, 2.0, 3.0, 4.0])
        @test h ≈ 1.0
        @test N0 == 4

        # grid = [0.5, 1.0, 1.5, 2.0], h=0.5, grid[1]/h=1, N0 = 4+1-1 = 4
        h, N0 = PowerLawDecomposition.get_point_num([0.5, 1.0, 1.5, 2.0])
        @test h ≈ 0.5
        @test N0 == 4

        # grid = [2.0, 3.0, 4.0], h=1.0, grid[1]/h=2, N0 = 3+2-1 = 4
        h, N0 = PowerLawDecomposition.get_point_num([2.0, 3.0, 4.0])
        @test h ≈ 1.0
        @test N0 == 4

        # grid = [0.1, 0.2, 0.3], h=0.1, grid[1]/h=1, N0 = 3+1-1 = 3
        h, N0 = PowerLawDecomposition.get_point_num([0.1, 0.2, 0.3])
        @test h ≈ 0.1
        @test N0 == 3
    end

    @testset "int_simpson" begin
        # integral of f(x)=1 on [0,1] with N=5 points (h=0.25)
        # result should be 1.0
        h = 0.25
        f = ones(5)
        @test PowerLawDecomposition.int_simpson(f, h) ≈ 1.0

        # integral of f(x)=x on [0,1] with N=5 points
        # exact value = 0.5
        x = range(0, 1, length=5)
        f = collect(x)
        @test PowerLawDecomposition.int_simpson(f, step(x)) ≈ 0.5

        # integral of f(x)=x^2 on [0,1] with N=5 points
        # exact value = 1/3, Simpson is exact for quadratic polynomials
        f = collect(x) .^ 2
        @test PowerLawDecomposition.int_simpson(f, step(x)) ≈ 1 / 3

        # integral of f(x)=x^3 on [0,1] with N=5 points
        # Simpson is exact for cubic polynomials, exact value = 1/4
        f = collect(x) .^ 3
        @test PowerLawDecomposition.int_simpson(f, step(x)) ≈ 1 / 4

        # even number of points (N=4)
        x_even = range(0, 1, length=4)
        f_const = ones(4)
        @test PowerLawDecomposition.int_simpson(f_const, step(x_even)) ≈ 1.0

        f_lin = collect(x_even)
        @test PowerLawDecomposition.int_simpson(f_lin, step(x_even)) ≈ 0.5

        # at least 2 points
        @test_throws AssertionError PowerLawDecomposition.int_simpson([1.0], 0.1)

        # h must be positive
        @test_throws AssertionError PowerLawDecomposition.int_simpson([1.0, 2.0], -0.1)
    end
end
