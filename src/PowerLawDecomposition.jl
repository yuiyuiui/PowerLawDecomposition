module PowerLawDecomposition

using LinearAlgebra
using Interpolations

include("pola.jl")

export PolaMethod, LagrangePola, BsplinePola, interpolate
export matrix_pencil, solve_hankel

end
