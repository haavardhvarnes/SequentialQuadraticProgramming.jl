module SequentialQuadraticProgrammingClarabelExt

import SequentialQuadraticProgramming as SQP
import Clarabel
using SparseArrays
using LinearAlgebra

include("clarabel_qp.jl")

function __init__()
    setglobal!(SQP, :ClarabelQPSolver, ClarabelQPSolver)
    return
end

end # module
