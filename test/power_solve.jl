Random.seed!(123456789)
a1 = 0.55 + 0.01 * rand()
a2 = 1.22 + 0.01 * rand()
a3 = 1.83 + 0.01 * rand()
#=
0.5503026711692189
1.2208577781394374
1.8311472770231354
=#
c1, c2, c3 = rand(3) .+ 1
#=
1.1576090409417932
1.8329810111138598
1.7569774924284722
=#
L0 = 1
L = 2^14
point_density = 10

T = Float64
_a1, _a2, _a3 = T.([a1, a2, a3])
_c1, _c2, _c3 = T.([c1, c2, c3])
f = x -> _c1 * x^(-_a1) + _c2 * x^(-_a2) + _c3 * x^(-_a3)
h = T(1 / point_density)
grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
f_data = f.(grid)

lm1 = WynnPola(; k=1.3, n=21, use_a_final=true)
lm2 = WynnPola(; k=1.3, n=19, use_a_final=true)
lm3 = WynnPola(; k=1.3, n=17, use_a_final=true)
lm_vec = [lm1, lm2, lm3]
iter_seek = IterSeek(lm_vec, 1, length(f_data))
order_vec, coff_vec, note_vec = power_solve(f_data, grid, 3, iter_seek)
@show order_vec
@show a1, a2, a3
@show coff_vec
@show c1, c2, c3