# Nonlinear Constraints

## Constraint Conventions

The SQP solver expects constraints in standard form:

- **Inequality**: `g(x) <= 0` (return a vector, each element ``\leq 0``)
- **Equality**: `h(x) = 0` (return a vector, each element ``= 0``)

Empty constraints are specified as `g(x) = zeros(0)` or `h(x) = zeros(0)`.

## Converting Common Constraints

| Original | SQP form |
|:---------|:---------|
| ``c(x) \geq b`` | `g(x) = [b - c(x)]` |
| ``c(x) \leq b`` | `g(x) = [c(x) - b]` |
| ``a \leq c(x) \leq b`` | `g(x) = [a - c(x); c(x) - b]` |
| ``c(x) = b`` | `h(x) = [c(x) - b]` |

## Example: Multiple Nonlinear Constraints

HS108 has 14 inequality constraints and 9 variables:

```julia
using SequentialQuadraticProgramming

x0 = ones(9)
f(x) = -0.5(x[1]x[4] - x[2]x[3] + x[3]x[9] - x[5]x[9] + x[5]x[8] - x[6]x[7])

g(x) = [
    -(1 - x[3]^2 - x[4]^2),
    -(1 - x[5]^2 - x[6]^2),
    -(1 - x[9]^2),
    -(1 - x[1]^2 - (x[2] - x[9])^2),
    -(1 - (x[1] - x[5])^2 - (x[2] - x[6])^2),
    -(1 - (x[1] - x[7])^2 - (x[2] - x[8])^2),
    -(1 - (x[3] - x[5])^2 - (x[4] - x[6])^2),
    -(1 - (x[3] - x[7])^2 - (x[4] - x[8])^2),
    -(1 - x[7]^2 - (x[8] - x[9])^2),
    -(x[1]x[4] - x[2]x[3]),
    -x[3]x[9],
    x[5]x[9],
    -(x[5]x[8] - x[6]x[7]),
    -x[9],
]
h(x) = zeros(0)

result = sqp_solve(f, g, h, x0, zeros(9), 80000.0 * ones(9);
    options = SQPOptions(max_iterations = 2000))
# result.objective ~ -0.866
```

## How Bounds Are Handled

When you pass `lb` and `ub`, the solver internally converts them to inequality constraints:

```
g_wrapped(x) = [g(x); lb - x; x - ub]
```

This means variable bounds `lb[i] <= x[i] <= ub[i]` become two inequality constraints per variable. The conversion is automatic and transparent.

## Equality Constraints with Large Problems

HS119 has 16 variables and 8 equality constraints:

```julia
h(x) = [
    0.22x[1] + 0.2x[2] + 0.19x[3] + 0.25x[4] + 0.15x[5] +
    0.11x[6] + 0.12x[7] + 0.13x[8] + x[9] - 2.5,
    -1.46x[1] - 1.3x[3] + 1.82x[4] - 1.15x[5] + 0.8x[7] + x[10] - 1.1,
    # ... more rows ...
]
```

The solver handles mixed inequality + equality constraints seamlessly.
