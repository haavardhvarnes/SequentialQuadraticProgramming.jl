module SequentialQuadraticProgrammingMOIExt

import SequentialQuadraticProgramming
import MathOptInterface as MOI

include("MOI_wrapper.jl")

function __init__()
    setglobal!(SequentialQuadraticProgramming, :Optimizer, Optimizer)
    return
end

end # module
