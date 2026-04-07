module SequentialQuadraticProgrammingHiGHSExt

import SequentialQuadraticProgramming as SQP
import HiGHS
using SparseArrays
using LinearAlgebra

include("highs_qp.jl")

function __init__()
    setglobal!(SQP, :HiGHSQPSolver, HiGHSQPSolver)
    return
end

end # module
