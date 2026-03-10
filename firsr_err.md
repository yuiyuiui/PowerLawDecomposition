为了研究其他误差，我们先保证输入的数据本身是“可分辨”的，也就是说用F128来生成数据

using Test, Random, LinearAlgebra
using PowerLawDecomposition

@testset "power_solve_asp wynn pola" begin
    Random.seed!(666)
    T = BigFloat
    nord = 10
    d = T(0.5)
    a_vec = T[]
    for i in 1:nord
        if i == 1
            push!(a_vec, (1 + rand(T)) / 2)
        else
            push!(a_vec, a_vec[end] + (1 + rand(T)) * d)
        end
    end
    c_vec = T[]
    for i in 1:nord
        push!(c_vec, 1 + rand(T))
    end

    L0 = 1
    L = 2^16
    @show L
    point_density = 10

    _a_vec = T.(a_vec)
    _c_vec = T.(c_vec)
    f = x -> sum(_c_vec .* x .^ (-_a_vec))
    h = T(1 / point_density)
    grid = [L0 + i * h for i in 0:((L - L0) * point_density)]
    f_data = f.(grid)

    #f_data = BigFloat.(f_data);
    #grid = BigFloat.(grid);

    nseek = 7
    asp = ASP(nseek, length(f_data); wynn_pola=WynnPola(; k=big"1.3", n=21))
    order_vec = power_solve_asp(f_data, grid, asp)
    res = norm.(order_vec .- _a_vec[1:nseek])
    for i in 1:nseek
        @show res[i]
    end
end


L = 65536
(i, n) = (1, 21)
(i, n) = (2, 21)
(i, n) = (3, 21)
(i, n) = (4, 21)
(i, n) = (5, 21)
(i, n) = (6, 15)
(i, n) = (7, 11)
res[i] = 2.309623123116215386007424848862732525979531847739573893597158489367420839362445e-15
res[i] = 1.238363965930164837244807619079343347815010293089671946129594371656864141092027e-08
res[i] = 6.651785096523025668707297122391323979155668172883968862359436103012022214917469e-08
res[i] = 1.688564925263971128928906341753880848235278782864861138746054197427489818402325e-05
res[i] = 0.1007647187111903499131117536474634778259138160527739216949398832665176848935325
res[i] = 2.648514048341159697193135431638549124535986874081676824620552273628077567443187
res[i] = 1.318766879741667458472066730064680521178556119441275199820107241219186677837381
Test Summary:             | Total   Time
power_solve_asp wynn pola |     0  27.8s
Test.DefaultTestSet("power_solve_asp wynn pola", Any[], 0, false, false, true, 1.772734620106077e9, 1.772734647909592e9, false, "/Users/syyui/projects/PowerLawDecomposition/test/power_solve_aps.jl")

julia> 


L = 16384
(i, n) = (1, 21)
(i, n) = (2, 21)
(i, n) = (3, 21)
(i, n) = (4, 21)
(i, n) = (5, 21)
(i, n) = (6, 21)
(i, n) = (7, 9)
res[i] = 7.994822907779804015853546810659832642751515324850766712656351554250104886514676e-14
res[i] = 1.370403894927420178632873575170279957100765660387057754476934425485777355722606e-08
res[i] = 1.046016008902922434140800859788285560053683702788916011315473705299826498623687e-07
res[i] = 2.239387110467756641761723657256909040131166547031050944445001447161954893225506e-05
res[i] = 0.0004312970657528816024183217260717006316520215042639062262987144010249507329174934
res[i] = 0.05939494631414219711394768424507359270495046859636857245363064968703187859324728
res[i] = 3.171945401979192031490410398813495620163833738088875884722549054691077360171947
Test Summary:             | Total  Time
power_solve_asp wynn pola |     0  6.5s
Test.DefaultTestSet("power_solve_asp wynn pola", Any[], 0, false, false, true, 1.772734711461011e9, 1.772734717922258e9, false, "/Users/syyui/projects/PowerLawDecomposition/test/power_solve_aps.jl")

julia> 


这时候应该不存在输入数据的order不可分辨的问题吧，但为什么L变长之后提取oredr的精度还是下降了？？？

请注意，我的问题的关键在于，我并不是要通过增长L提高精度，而是现在的你的理论似乎完全无法解释，为什么对BigFloat，提高L误差反而变大？