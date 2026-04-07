# Getting Started

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/haavardhvarnes/SequentialQuadraticProgramming.jl")
```

Or via the private registry (if set up):
```julia
pkg"registry add https://github.com/haavardhvarnes/JuliaRegistry"
Pkg.add("SequentialQuadraticProgramming")
```

## Your First Problem

Solve the Rosenbrock function (unconstrained):

```julia
using SequentialQuadraticProgramming

f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
g(x) = zeros(0)  # no inequality constraints
h(x) = zeros(0)  # no equality constraints

result = sqp_solve(f, g, h, [-1.9, 2.0])
```

## Inspecting the Result

```julia
result.x                   # optimal solution vector
result.objective           # objective value at optimum
result.converged           # true if converged
result.status              # :converged, :max_iterations, :qp_failed, ...
result.iterations          # number of SQP iterations
result.constraint_violation  # max constraint violation (Inf-norm)
```

## Adding Constraints

The SQP solver handles three types of constraints:

- **Inequality**: `g(x) <= 0`
- **Equality**: `h(x) = 0`
- **Variable bounds**: `lb <= x <= ub`

### Example: HS071 (Hock-Schittkowski Problem 71)

```julia
using SequentialQuadraticProgramming, LinearAlgebra

# min x1*x4*(x1+x2+x3) + x3
# s.t. x1*x2*x3*x4 >= 25  (inequality)
#      x1^2+x2^2+x3^2+x4^2 = 40  (equality)
#      1 <= xi <= 5  (bounds)

f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
g(x) = [-prod(x) + 25.0]       # g(x) <= 0 form: 25 - prod(x) <= 0
h(x) = [x' * x - 40.0]         # h(x) = 0 form
x0 = [1.0, 5.0, 5.0, 1.0]
lb = ones(4)
ub = 5.0 * ones(4)

result = sqp_solve(f, g, h, x0, lb, ub)
# result.objective ~ 17.014
```

## Solver Options

```julia
options = SQPOptions(
    max_iterations = 500,
    xtol = 1e-8,
    ftol = 1e-8,
    verbose = true,
)
result = sqp_solve(f, g, h, x0; options = options)
```

See the [Solver Interface](@ref) API page for all options.

## Next Steps

- [Basic Constrained Optimization](@ref) tutorial for more examples
- [JuMP Integration](@ref) to use with JuMP modelling
- [Solver Interface](@ref) for full API details
