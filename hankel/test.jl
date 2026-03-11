using Test
using LinearAlgebra
using GenericLinearAlgebra
using Printf

include(joinpath(@__DIR__, "hankel.jl"))

# ── helpers ──────────────────────────────────────────────────

function make_test_data(::Type{T}, c_true, a_true, grid_spec) where {T}
    grid = T.(collect(grid_spec))
    f = zeros(T, length(grid))
    for i in eachindex(c_true)
        ci, ai = T(c_true[i]), T(a_true[i])
        @inbounds for k in eachindex(grid)
            f[k] += ci * exp(-ai * grid[k])
        end
    end
    return grid, f
end

function max_param_error(es::ExponentialSum{T}, c_true, a_true) where {T}
    ea = maximum(abs(es.a[i] - T(a_true[i])) for i in eachindex(a_true))
    ec = maximum(abs(es.c[i] - T(c_true[i])) for i in eachindex(c_true))
    return (a_err=ea, c_err=ec)
end

# ── Prony tests ──────────────────────────────────────────────

@testset "Prony method" begin
    @testset "single exponential" begin
        grid, f = make_test_data(Float64, [5.0], [2.0], range(0, 3; length=30))
        es = prony(grid, f, 1)
        @test isapprox(es.a[1], 2.0; rtol=1e-10)
        @test isapprox(es.c[1], 5.0; rtol=1e-10)
        @test rmse(es, grid, f) < 1e-12
    end

    @testset "two exponentials" begin
        c, a = [3.0, 2.0], [1.0, 3.0]
        grid, f = make_test_data(Float64, c, a, range(0, 5; length=50))
        es = prony(grid, f, 2)
        for i in 1:2
            @test isapprox(es.a[i], a[i]; rtol=1e-8)
            @test isapprox(es.c[i], c[i]; rtol=1e-8)
        end
        @test rmse(es, grid, f) < 1e-10
    end

    @testset "three exponentials" begin
        c, a = [1.0, 2.0, 3.0], [0.5, 1.5, 4.0]
        grid, f = make_test_data(Float64, c, a, range(0, 4; length=80))
        es = prony(grid, f, 3)
        for i in 1:3
            @test isapprox(es.a[i], a[i]; rtol=1e-6)
            @test isapprox(es.c[i], c[i]; rtol=1e-6)
        end
    end

    @testset "type stability" begin
        grid, f = make_test_data(Float64, [3.0, 2.0], [1.0, 3.0], range(0, 3; length=30))
        @test @inferred(prony(grid, f, 2)) isa ExponentialSum{Float64}
    end

    @testset "assertion — insufficient data" begin
        grid, f = make_test_data(Float64, [1.0], [1.0], range(0, 1; length=3))
        @test_throws AssertionError prony(grid, f, 3)
    end
end

# ── Matrix Pencil tests ──────────────────────────────────────

@testset "Matrix Pencil method" begin
    @testset "single exponential" begin
        grid, f = make_test_data(Float64, [5.0], [2.0], range(0, 3; length=31))
        es = matrix_pencil(grid, f, 1)
        @test isapprox(es.a[1], 2.0; rtol=1e-10)
        @test isapprox(es.c[1], 5.0; rtol=1e-10)
        @test rmse(es, grid, f) < 1e-12
    end

    @testset "two exponentials" begin
        c, a = [3.0, 2.0], [1.0, 3.0]
        grid, f = make_test_data(Float64, c, a, range(0, 5; length=51))
        es = matrix_pencil(grid, f, 2)
        for i in 1:2
            @test isapprox(es.a[i], a[i]; rtol=1e-8)
            @test isapprox(es.c[i], c[i]; rtol=1e-8)
        end
        @test rmse(es, grid, f) < 1e-10
    end

    @testset "three exponentials" begin
        c, a = [1.0, 2.0, 3.0], [0.5, 1.5, 4.0]
        grid, f = make_test_data(Float64, c, a, range(0, 4; length=81))
        es = matrix_pencil(grid, f, 3)
        for i in 1:3
            @test isapprox(es.a[i], a[i]; rtol=1e-6)
            @test isapprox(es.c[i], c[i]; rtol=1e-6)
        end
    end

    @testset "type stability" begin
        grid, f = make_test_data(Float64, [3.0, 2.0], [1.0, 3.0], range(0, 3; length=31))
        @test @inferred(matrix_pencil(grid, f, 2)) isa ExponentialSum{Float64}
    end

    @testset "assertion — insufficient data" begin
        grid, f = make_test_data(Float64, [1.0], [1.0], range(0, 1; length=4))
        @test_throws AssertionError matrix_pencil(grid, f, 3)
    end
end

# ── Multi-type precision ─────────────────────────────────────

@testset "Multi-type precision" begin
    c_true, a_true = [3.0, 2.0], [1.0, 3.0]

    for (T, tol_rmse) in [(Float32, 1e-3), (Float64, 1e-10), (BigFloat, 1e-15)]
        @testset "$(T) — Prony" begin
            grid, f = make_test_data(T, c_true, a_true, range(0, 5; length=50))
            es = prony(grid, f, 2)
            @test eltype(es.c) == T
            @test eltype(es.a) == T
            @test rmse(es, grid, f) < T(tol_rmse)
        end

        @testset "$(T) — Matrix Pencil" begin
            grid, f = make_test_data(T, c_true, a_true, range(0, 5; length=51))
            es = matrix_pencil(grid, f, 2)
            @test eltype(es.c) == T
            @test eltype(es.a) == T
            @test rmse(es, grid, f) < T(tol_rmse)
        end
    end
end

# ── Boundary / edge cases ────────────────────────────────────

@testset "Boundary cases" begin
    @testset "minimum data — Prony (M=2N)" begin
        grid, f = make_test_data(Float64, [1.0, 2.0], [1.0, 3.0], range(0, 2; length=4))
        es = prony(grid, f, 2)
        @test isfinite(rmse(es, grid, f))
    end

    @testset "minimum data — Matrix Pencil (M=2N+1)" begin
        grid, f = make_test_data(Float64, [1.0, 2.0], [1.0, 3.0], range(0, 2; length=5))
        es = matrix_pencil(grid, f, 2)
        @test isfinite(rmse(es, grid, f))
    end

    @testset "wide decay-rate spread" begin
        c, a = [1.0, 1.0], [0.01, 100.0]
        grid, f = make_test_data(Float64, c, a, range(0, 1; length=200))
        es_p = prony(grid, f, 2)
        es_m = matrix_pencil(grid, f, 2)
        @test rmse(es_p, grid, f) < 1e-3
        @test rmse(es_m, grid, f) < 1e-3
    end
end

# ── Method comparison (output for report) ─────────────────────

println("\n" * "="^70)
println("  PRECISION COMPARISON REPORT")
println("="^70)

test_cases = [
    ("2-exp simple",  [3.0, 2.0],         [1.0, 3.0],         50, 51),
    ("3-exp mixed",   [1.0, 2.0, 3.0],    [0.5, 1.5, 4.0],    80, 81),
    ("2-exp close",   [1.0, 1.5],         [1.0, 1.2],          100, 101),
    ("2-exp spread",  [1.0, 1.0],         [0.1, 10.0],         100, 101),
]

for (T, label) in [(Float32, "Float32"), (Float64, "Float64"), (BigFloat, "BigFloat")]
    println("\n── $label ─────────────────────────────────")
    @printf("  %-16s  %12s  %12s  %12s  %12s\n",
            "Case", "Prony RMSE", "MPM RMSE", "max|Δa|_P", "max|Δa|_M")
    println("  " * "-"^66)
    for (name, c_true, a_true, mp, mm) in test_cases
        N = length(c_true)

        grid_p, f_p = make_test_data(T, c_true, a_true, range(0, 5; length=mp))
        es_p = prony(grid_p, f_p, N)
        ep = max_param_error(es_p, c_true, a_true)
        rp = rmse(es_p, grid_p, f_p)

        grid_m, f_m = make_test_data(T, c_true, a_true, range(0, 5; length=mm))
        es_m = matrix_pencil(grid_m, f_m, N)
        em = max_param_error(es_m, c_true, a_true)
        rm = rmse(es_m, grid_m, f_m)

        @printf("  %-16s  %12.3e  %12.3e  %12.3e  %12.3e\n",
                name, Float64(rp), Float64(rm), Float64(ep.a_err), Float64(em.a_err))
    end
end
println()
