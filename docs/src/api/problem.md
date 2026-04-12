# Problem Definition

## NLPProblem

```@docs
NLPProblem
```

## Usage

The simplest way to solve a problem is to pass functions directly to [`sqp_solve`](@ref):

```julia
result = sqp_solve(f, g, h, x0, lb, ub)
```

This constructs an `NLPProblem` internally. You can also construct one explicitly for use with [`diagnose_problem`](@ref):

```julia
problem = NLPProblem(f, g, h, x0, lb, ub)
diag = diagnose_problem(problem)
result = sqp_solve(problem)
```

### Constraint Convention

- **Inequality constraints**: ``g(x) \leq 0`` — return a vector; each entry must be ``\leq 0`` at the optimum
- **Equality constraints**: ``h(x) = 0`` — return a vector; each entry must be ``= 0`` at the optimum
- **No constraints**: return `zeros(0)` (empty vector)
