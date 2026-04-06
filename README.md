# SequentialQuadraticProgramming.jl

A Julia implementation of Sequential Quadratic Programming (SQP) for nonlinear constrained optimization.

## Installation

```julia
using Pkg
pkg"registry add https://github.com/haavardhvarnes/JuliaRegistry"
Pkg.add("SequentialQuadraticProgramming")
```

## Usage

```julia
using SequentialQuadraticProgramming

# min f(x) s.t. g(x) <= 0, h(x) = 0, lb <= x <= ub
f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
g(x) = [-prod(x) + 25.0]
h(x) = [x' * x - 40.0]

x0 = [1.0, 5.0, 5.0, 1.0]
lb = ones(4)
ub = 5.0 * ones(4)

result = sqp_solve(f, g, h, x0, lb, ub)
result.x          # optimal solution
result.objective  # objective value
result.converged  # true/false
```

## Algorithm

Schittkowski-style SQP with:
- Augmented Lagrangian merit function with adaptive penalty parameters
- COSMO conic solver for QP subproblems (with Schittkowski delta-slack for infeasibility)
- Robust BFGS Hessian updates (Yang et al. 2022)
- Quadratic interpolation line search with Armijo condition
- ForwardDiff automatic differentiation with FiniteDiff fallback

## References

- Schittkowski, K. "On the convergence of a sequential quadratic programming method" (NM_SQP2)
- Yang et al. "A robust BFGS algorithm for unconstrained nonlinear optimization problems" (2022)
