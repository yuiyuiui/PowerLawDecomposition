module PowerLawDecomposition

using LinearAlgebra
using Interpolations

include("math/pola.jl")
include("math/integral.jl")
include("math/diff.jl")

export PolaMethod, LagrangePola, BsplinePola, interpolate, bspline_interp, build_bspline_itp
export IntegralMethod, Simpson, integrate
export fd8, fd8!

end
