# Basic Constrained Optimization

## Rosenbrock with Variable Bounds

The Rosenbrock function with a lower bound on `x[1]`:

```julia
using SequentialQuadraticProgramming

f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
g(x) = zeros(0)
h(x) = zeros(0)
lb = [1.25, -Inf]
ub = [Inf, Inf]
x0 = [-1.9, 2.0]

result = sqp_solve(f, g, h, x0, lb, ub)
# x[1] will be at the bound: result.x[1] ~ 1.25
```

## HS071: Mixed Constraints

This classic benchmark combines inequality constraints, equality constraints, and variable bounds:

```julia
using SequentialQuadraticProgramming, LinearAlgebra

f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
g(x) = [-prod(x) + 25.0]   # x1*x2*x3*x4 >= 25
h(x) = [x' * x - 40.0]     # sum(xi^2) = 40

result = sqp_solve(f, g, h, [1.0, 5.0, 5.0, 1.0], ones(4), 5.0 * ones(4);
    options = SQPOptions(max_iterations = 2000, verbose = true))

println("Optimal: x = ", round.(result.x, digits = 3))
println("Objective: ", round(result.objective, digits = 4))
println("Iterations: ", result.iterations)
println("Status: ", result.status)
```

## Understanding SQPResult

The solver returns an [`SQPResult`](@ref) with:

| Field | Description |
|:------|:------------|
| `x` | Optimal solution vector |
| `objective` | Objective value ``f(x^*)`` |
| `iterations` | Number of SQP iterations |
| `converged` | `true` if all convergence criteria met |
| `constraint_violation` | ``\|[g(x)^+; |h(x)|]\|_\infty`` |
| `status` | `:converged`, `:max_iterations`, `:qp_failed`, `:line_search_failed`, `:trust_region_failed` |

## Configuring the Solver

All options are set via [`SQPOptions`](@ref):

```julia
opts = SQPOptions(
    max_iterations = 500,       # maximum SQP iterations
    xtol = 1e-8,               # step size tolerance
    ftol = 1e-8,               # objective change tolerance
    constraint_tol = 1e-8,     # constraint violation tolerance
    verbose = true,            # print iteration table
)
```

The `verbose = true` option prints an iteration table:

```
Iter    objective    norm_dx     step      c_viol
1       16.0625      1.3438     1.0       1.38672   BFGS
2       16.9565      0.0357     1.0       0.09843   BFGS
3       17.0143      0.0011     1.0       0.00114   BFGS
...
```
