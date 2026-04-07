# AD Backends

The solver uses [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) for automatic differentiation, supporting any AD backend from the Julia ecosystem.

## Default Behavior

By default, the solver auto-selects the AD backend:

1. Try **ForwardDiff** (fast, exact, supports most Julia functions)
2. Fall back to **FiniteDiff** if ForwardDiff fails (e.g., FFI calls, MOI evaluator closures)

```julia
result = sqp_solve(f, g, h, x0)  # auto-selects ForwardDiff
```

## Explicit Backend Selection

Pass `ad_backend` to choose a specific backend:

```julia
using SequentialQuadraticProgramming
import ADTypes: AutoForwardDiff, AutoFiniteDiff

# Explicit ForwardDiff
result = sqp_solve(f, g, h, x0; ad_backend = AutoForwardDiff())

# Explicit FiniteDiff (useful when ForwardDiff fails)
result = sqp_solve(f, g, h, x0; ad_backend = AutoFiniteDiff())
```

## Using Enzyme

If [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is loaded:

```julia
using Enzyme
import ADTypes: AutoEnzyme

result = sqp_solve(f, g, h, x0; ad_backend = AutoEnzyme())
```

## Using ReverseDiff

For problems with many variables and few outputs (large Jacobians):

```julia
using ReverseDiff
import ADTypes: AutoReverseDiff

result = sqp_solve(f, g, h, x0; ad_backend = AutoReverseDiff())
```

## How It Works

The solver uses DifferentiationInterface's `prepare_*` pattern for zero-overhead repeated evaluation:

```julia
# Internally, the solver does:
prep = DI.prepare_gradient(f, backend, x0)       # one-time setup
grad = DI.gradient(f, prep, backend, x)           # reused every iteration
```

The `prepare_*` step caches tape compilations, configuration objects, and sparsity patterns, so subsequent calls are as fast as calling the backend directly.

## When to Use What

| Backend | Best for | Notes |
|:--------|:---------|:------|
| `AutoForwardDiff()` | Small-medium problems, general use | Default, exact, fast |
| `AutoFiniteDiff()` | Functions that don't support AD | Approximate, slower |
| `AutoEnzyme()` | Performance-critical, GPU | Requires Enzyme.jl |
| `AutoReverseDiff()` | Many variables, few outputs | Reverse-mode AD |
| JuMP/MOI evaluator | JuMP models | Automatic, exact, uses JuMP's own AD |
