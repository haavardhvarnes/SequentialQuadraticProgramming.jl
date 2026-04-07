# SequentialQuadraticProgramming.jl

A Julia package for nonlinear constrained optimization using Sequential Quadratic Programming (SQP).

## Algorithm Flow

```
        Problem: min f(x)  s.t. g(x) <= 0, h(x) = 0
                            |
                    [ Initialize x0, H = I ]
                            |
              +-------------+--------------+
              |                            |
    [ Solve QP subproblem ]     [ Compute derivatives ]
    [  min d'Hd + df'd    ]     [  df, dg, dh via AD  ]
    [  s.t. dg*d <= -g    ]     [  or MOI evaluator   ]
    [       dh*d  = -h    ]
              |
    +----[ Globalization ]----+
    |                         |
  Line Search          Trust Region
  (Schittkowski)       (||d|| <= Delta)
    |                         |
    +----[ Update x ]--------+
              |
    [ Update Hessian ]
    [ Analytical -> Robust BFGS -> L-BFGS -> I ]
              |
    [ Check convergence ]
    [ |p| < xtol && cv < tol ]
```

## Features

| Feature | Options |
|:--------|:--------|
| **QP Solvers** | COSMO (default), Clarabel, HiGHS |
| **AD Backends** | ForwardDiff (default), FiniteDiff, Enzyme, ... via DifferentiationInterface.jl |
| **Globalization** | Line search (Schittkowski), Trust region |
| **Hessian** | Analytical, Robust BFGS (Yang 2022), L-BFGS fallback |
| **JuMP Support** | Full MathOptInterface wrapper with exact evaluator derivatives |

## Quick Example

```julia
using SequentialQuadraticProgramming

f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
g(x) = zeros(0)    # no inequality constraints
h(x) = zeros(0)    # no equality constraints

result = sqp_solve(f, g, h, [-1.9, 2.0])
result.x          # [1.0, 1.0]
result.objective  # ~0.0
result.converged  # true
```

## Contents

```@contents
Pages = [
    "getting_started.md",
    "tutorials/basic_constrained.md",
    "tutorials/jump_integration.md",
    "api/solver.md",
    "design/algorithm.md",
]
Depth = 1
```
