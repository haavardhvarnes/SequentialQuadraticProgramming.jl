module SequentialQuadraticProgramming

using ForwardDiff
using FiniteDiff
using COSMO
using LinearAlgebra
using SparseArrays

include("types.jl")
include("problem.jl")
include("derivatives.jl")
include("merit.jl")
include("line_search.jl")
include("hessian_update.jl")
include("qp_subproblem.jl")
include("solver.jl")

export sqp_solve, SQPOptions, SQPResult, NLPProblem, COSMOQPSolver

end # module
