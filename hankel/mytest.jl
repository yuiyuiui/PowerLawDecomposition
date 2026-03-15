include("hankel.jl")
using GenericLinearAlgebra

L = 20
n = 10
T = Float64

d = T(1//2)

a_vec = collect(1:n) .* d

func(x) = sum(exp.(-a_vec .* x))

point_density = 10;

h = T(1 // point_density);

L0 = 1;

grid = [T(L0 + h * i) for i in 0:((L - L0) * point_density)];
f = func.(grid);

println("normal")
@show matrix_pencil(grid, f, n).a

println("delayed")
@show matrix_pencil_delayed(grid, f, n).a

println("prefilter")
@show matrix_pencil_prefilter(grid, f, n).a

println("delta")
@show matrix_pencil_delta(grid, f, n).a

println("delta delayed")
@show matrix_pencil_delta_delayed(grid, f, n).a

println("delta balanced") # better
@show matrix_pencil_delta_balanced(grid, f, n).a

println("prefilter delta")
@show matrix_pencil_prefilter_delta(grid, f, n).a

println("delta auto")
@show matrix_pencil_delta_auto(grid, f, n).a

println("bilinear")
@show matrix_pencil_bilinear(grid, f, n).a

println("prefilter bilinear")
@show matrix_pencil_prefilter_bilinear(grid, f, n).a

println("universal")
@show matrix_pencil_universal(grid, f, n).a

println("delayed 2")
@show matrix_pencil_delayed_2(grid, f, n; stride=20).a
