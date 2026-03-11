include("hankel.jl")
using GenericLinearAlgebra

for T in [Float64, BigFloat]
    for type_pro in [false, true]
        n = 10

        d = T(1 // 2)

        a_vec = collect(1:n) .* d

        func(x) = sum(exp.(-a_vec .* x))

        h = T(1 // 10)

        L0 = 1
        L = 5

        grid = [T(L0 + h * i) for i in 0:((L - L0) * 10)]
        f = func.(grid)

        S = promote_type(T, BigFloat)

        if type_pro && T === Float64
            grid = BigFloat.(grid)
            f = BigFloat.(f)
        end

        res1 = matrix_pencil(grid, f, n)
        res2 = matrix_pencil(grid, f, n; svd_tol=1e-12)
        @show T
        @show res1.a
        @show res2.a
    end
end

# 1. machine noise in input signal
# 2. The processing of the alg itself
# 3. ability to solve the generalized eignevalues problem

T = Float64
n = 5

d = T(1 // 2)

lnγ = 1//2

a_vec = collect(0:n-1) .* d .+ 1//2
a_vec .= a_vec .* lnγ

func(x) = sum(exp.(-a_vec .* x))

h = T(1)

L0 = 1
L = round(Int, 10 / lnγ)

grid = [T(L0 + h * i) for i in 0:((L - L0))]
f = func.(grid)



res1 = matrix_pencil(grid, f, n)
res2 = matrix_pencil(grid, f, n; svd_tol=eps())
@show T
@show res1.a
@show res2.a