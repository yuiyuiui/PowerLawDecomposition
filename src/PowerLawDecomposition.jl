module PowerLawDecomposition

using LinearAlgebra
using Interpolations

export Method, power_solve
export IterSeek, LeadingMethod, leading_solve, LogLog, Shanks, Wynn, WynnPola
export MellinTrans

abstract type Method end

include("kit.jl")
include("iterseek/iterseek.jl")
include("mellintrans.jl")

end
