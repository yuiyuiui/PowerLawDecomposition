module PowerLawDecomposition

using LinearAlgebra
using Interpolations

export power_solve, power_solve_asp, leading_solve
export IterSeek, ASP, WynnPola, Wynn
export wynn_epsilon_core, wynn_epsilon_core_v2

include("kit.jl")
include("iterseek.jl")

end
