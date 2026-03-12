include("hankel.jl")
using GenericLinearAlgebra

for L in 4:2:20
    for T in [Float64, BigFloat]
        n = 10

        d = T(1 // 2)

        a_vec = collect(1:n) .* d

        func(x) = sum(exp.(-a_vec .* x))

        h = T(1 // 10)

        L0 = 1

        grid = [T(L0 + h * i) for i in 0:((L - L0) * 10)]
        f = func.(grid)

        res = matrix_pencil(grid, f, n)
        @show T, L
        @show res.a
    end
end

# 1. machine noise in input signal
# 2. The processing of the alg itself
# 3. ability to solve the generalized eignevalues problem

L = 10
n = 10
T = Float64

d = T(1 // 2)

a_vec = collect(1:n) .* d

func(x) = sum(exp.(-a_vec .* x))

h = T(1 // 10)

L0 = 1

grid = [T(L0 + h * i) for i in 0:((L - L0) * 10)]
f = func.(grid)

res1 = matrix_pencil(grid, f, n)
res1.a
