# JuMP Integration

SequentialQuadraticProgramming.jl provides a MathOptInterface (MOI) wrapper, enabling use with [JuMP](https://jump.dev/).

## Basic Usage

```julia
using SequentialQuadraticProgramming, JuMP

model = Model(SequentialQuadraticProgramming.Optimizer)
set_silent(model)

@variable(model, 1 <= x[i = 1:4] <= 5, start = [1, 5, 5, 1][i])

# Register a nonlinear operator for the objective
@operator(model, obj_fn, 4,
    (x1, x2, x3, x4) -> x1 * x4 * (x1 + x2 + x3) + x3)

@objective(model, Min, obj_fn(x[1], x[2], x[3], x[4]))
@constraint(model, x[1] * x[2] * x[3] * x[4] >= 25)
@constraint(model, x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 == 40)

optimize!(model)

println("Status: ", termination_status(model))
println("Objective: ", objective_value(model))
println("Solution: ", value.(x))
```

## Setting Solver Options

Use `set_optimizer_attribute` to configure SQP options:

```julia
model = Model(SequentialQuadraticProgramming.Optimizer)
set_optimizer_attribute(model, "max_iterations", 2000)
set_optimizer_attribute(model, "xtol", 1e-8)
set_optimizer_attribute(model, "ftol", 1e-8)
set_optimizer_attribute(model, "globalization", :filter_line_search)
set_optimizer_attribute(model, "hessian_strategy", :analytical)
set_silent(model)
```

Any field of [`SQPOptions`](@ref) can be set as a string attribute.

## Exact Evaluator Derivatives

When using JuMP, the solver automatically uses JuMP's built-in reverse-mode AD for exact derivatives:

- **Gradient** of the objective (`:Grad`)
- **Jacobian** of constraints (`:Jac`)
- **Hessian** of the Lagrangian (`:Hess`)

This bypasses ForwardDiff/FiniteDiff entirely and provides machine-precision derivatives at no extra cost. This is handled automatically by the MOI wrapper.

## Maximization

The wrapper supports `Max` sense:

```julia
model = Model(SequentialQuadraticProgramming.Optimizer)
set_silent(model)

@variable(model, 0 <= x[1:2] <= 10, start = 1)
@operator(model, neg_quad, 2, (x1, x2) -> -(x1 - 3)^2 - (x2 - 4)^2)
@objective(model, Max, neg_quad(x[1], x[2]))

optimize!(model)
# x ~ [3.0, 4.0]
```

## Engineering Design Example

A pressure vessel design problem with nonlinear constraints, combining material cost, welding cost, and volume constraints:

```julia
using SequentialQuadraticProgramming, JuMP

model = Model(SequentialQuadraticProgramming.Optimizer)
set_silent(model)

# Design variables: shell thickness, head thickness, radius, length
@variable(model, 0.5 <= Ts <= 10.0, start = 1.0)
@variable(model, 0.5 <= Th <= 10.0, start = 1.0)
@variable(model, 10.0 <= R <= 200.0, start = 50.0)
@variable(model, 10.0 <= L <= 200.0, start = 50.0)

# Objective: material + welding cost
@operator(model, cost, 4, (ts, th, r, l) ->
    0.6224 * ts * r * l +
    1.7781 * th * r^2 +
    3.1661 * ts^2 * l +
    19.84 * ts^2 * r)

@objective(model, Min, cost(Ts, Th, R, L))

# Stress constraints
@constraint(model, -Ts + 0.0193 * R <= 0)
@constraint(model, -Th + 0.00954 * R <= 0)

# Volume constraint: enclosed volume >= 1296000
@constraint(model, pi * R^2 * L + (4/3) * pi * R^3 >= 1296000)

optimize!(model)

println("Status: ", termination_status(model))
println("Cost:   ", round(objective_value(model), digits = 2))
println("Ts = ", round(value(Ts), digits = 3),
        ", Th = ", round(value(Th), digits = 3),
        ", R = ", round(value(R), digits = 3),
        ", L = ", round(value(L), digits = 3))
```

## Functional API vs JuMP

The same problem can be solved both ways:

**Functional API** (ForwardDiff, faster for small problems):
```julia
f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
g(x) = [-prod(x) + 25.0]
h(x) = [x' * x - 40.0]
result = sqp_solve(f, g, h, [1.0, 5.0, 5.0, 1.0], ones(4), 5.0 * ones(4))
```

**JuMP** (MOI evaluator derivatives, better for larger/complex models):
```julia
model = Model(SequentialQuadraticProgramming.Optimizer)
# ... build model with @variable, @operator, @objective, @constraint ...
optimize!(model)
```

**When to use which:**

| Scenario | Recommended |
|:---------|:------------|
| Small problem (< 20 vars), simple constraints | Functional API |
| Complex model structure (indexed sets, conditional constraints) | JuMP |
| Need sparse Jacobian/Hessian | JuMP (automatic) |
| Comparing with other solvers (Ipopt, KNITRO) | JuMP (swap optimizer) |
| Custom AD backend (ForwardDiff, FiniteDiff) | Functional API |

## Termination Status Mapping

| SQP Status | MOI Status |
|:-----------|:-----------|
| `:converged` | `LOCALLY_SOLVED` |
| `:max_iterations` | `ITERATION_LIMIT` |
| `:qp_failed` | `NUMERICAL_ERROR` |
| `:line_search_failed` | `NUMERICAL_ERROR` |
| `:trust_region_failed` | `NUMERICAL_ERROR` |
