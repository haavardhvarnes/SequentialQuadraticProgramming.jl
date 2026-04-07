# JuMP Integration

SequentialQuadraticProgramming.jl provides a MathOptInterface (MOI) wrapper, enabling use with JuMP.

## Basic Usage

```julia
using SequentialQuadraticProgramming, JuMP

model = Model(SequentialQuadraticProgramming.Optimizer)
set_silent(model)

@variable(model, 1 <= x[i = 1:4] <= 5, start = [1, 5, 5, 1][i])
@NLobjective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
@NLconstraint(model, x[1] * x[2] * x[3] * x[4] >= 25)
@NLconstraint(model, x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 == 40)

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
set_silent(model)
```

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
@NLobjective(model, Max, -(x[1] - 3)^2 - (x[2] - 4)^2)

optimize!(model)
# x ~ [3.0, 4.0]
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
# ... build model with @variable, @NLobjective, @NLconstraint ...
optimize!(model)
```
