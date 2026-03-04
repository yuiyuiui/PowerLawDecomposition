module PowerLawDecomposition

using LinearAlgebra
using Interpolations

export power_solve, power_solve_asp, leading_solve
export IterSeek, ASP, WynnPola, Wynn

include("kit.jl")
include("iterseek.jl")

end
