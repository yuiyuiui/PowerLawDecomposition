module PowerLawDecomposition

using LinearAlgebra
using Interpolations

export power_solve, leading_solve
export IterSeek, WynnPola, Wynn

include("kit.jl")
include("iterseek.jl")

end
