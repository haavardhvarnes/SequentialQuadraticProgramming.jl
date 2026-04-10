module SequentialQuadraticProgramming

import DifferentiationInterface as DI
import ADTypes
using ForwardDiff
using FiniteDiff
using COSMO
using LinearAlgebra
using SparseArrays
using Random

include("types.jl")
include("problem.jl")
include("derivatives.jl")
include("diagnostics.jl")
include("merit.jl")
include("line_search.jl")
include("hessian_update.jl")
include("qp_subproblem.jl")
include("trust_region.jl")
include("solver.jl")

export sqp_solve, SQPOptions, SQPResult, NLPProblem, COSMOQPSolver
export ProblemDiagnostics, diagnose_problem

# Placeholder for MOI extension — set by SequentialQuadraticProgrammingMOIExt.__init__
global Optimizer::Any = nothing
export Optimizer

# Placeholder for Clarabel extension — set by SequentialQuadraticProgrammingClarabelExt.__init__
global ClarabelQPSolver::Any = nothing
export ClarabelQPSolver

# Placeholder for HiGHS extension — set by SequentialQuadraticProgrammingHiGHSExt.__init__
global HiGHSQPSolver::Any = nothing
export HiGHSQPSolver

end # module
