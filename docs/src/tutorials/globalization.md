# Globalization Strategies

The SQP solver supports three globalization strategies: **line search** (default), **filter line search**, and **trust region**.

## Line Search (Default)

The default strategy uses Schittkowski's quadratic interpolation with the Armijo sufficient decrease condition:

```julia
result = sqp_solve(f, g, h, x0)
# equivalent to:
result = sqp_solve(f, g, h, x0; options = SQPOptions(globalization = :line_search))
```

**How it works:**
1. Solve QP subproblem to get search direction `dx`
2. Find step size ``\alpha \in (0, 1]`` such that the merit function decreases sufficiently
3. Update: ``x \leftarrow x + \alpha \cdot dx``

**Line search parameters:**
- `line_search_mu` (default `1e-4`): Armijo condition parameter
- `line_search_beta` (default `0.5`): Backtracking contraction factor
- `phi0_lookback` (default `5`): Non-monotone lookback window for merit history

## Trust Region

The trust region strategy constrains the step to ``\|d\|_\infty \leq \Delta`` and adjusts the radius based on actual vs predicted reduction:

```julia
result = sqp_solve(f, g, h, x0;
    options = SQPOptions(
        globalization = :trust_region,
        trust_region_init = 1.0,     # initial radius
        trust_region_max = 1e4,      # maximum radius
        trust_region_eta = 0.1,      # acceptance threshold
    ))
```

**How it works:**
1. Solve QP subproblem with ``\|d\|_\infty \leq \Delta``
2. Compute reduction ratio ``\rho = \text{actual} / \text{predicted}``
3. If ``\rho > \eta``: accept step, possibly expand ``\Delta``
4. If ``\rho \leq \eta``: reject step, shrink ``\Delta``

**Radius updates:**
- ``\rho > 0.75`` and step at boundary: ``\Delta \leftarrow \min(2\Delta, \Delta_\text{max})``
- ``\rho < 0.25``: ``\Delta \leftarrow \Delta / 4``

## Filter Line Search (Wächter-Biegler)

The filter line search replaces the merit function with a **bi-objective filter** that tracks pairs ``(f, \theta)`` where ``\theta`` is the constraint violation. A trial step is accepted if it either:

- **f-step**: Reduces the objective sufficiently (switching condition met, Armijo on ``f``)
- **h-step**: Is not dominated by any entry in the filter (improves ``f`` or ``\theta`` or both)

This avoids the penalty parameter tuning inherent in merit-function approaches.

```julia
result = sqp_solve(f, g, h, x0;
    options = SQPOptions(
        globalization = :filter_line_search,
        filter_max_size = 50,         # maximum filter entries
        filter_alpha_min = 1e-6,      # minimum step size before fallback
    ))
```

**When the filter accepts a step, the result tracks which type:**
- `result.n_filter_f_steps` — steps accepted by objective descent
- `result.n_filter_h_steps` — steps accepted by filter dominance
- `result.n_filter_fallbacks` — iterations where the filter fell through to the Schittkowski merit line search

**Best for:** Problems where the augmented Lagrangian merit function oscillates, e.g., HS092 (352 iters with `:line_search` → ~58 iters with `:filter_line_search`).

## When to Use Which

| Scenario | Recommended |
|:---------|:------------|
| General-purpose | Line search (default) |
| Merit function oscillating (large penalty swings) | Filter line search |
| Ill-conditioned Hessian | Trust region |
| Fast convergence expected | Line search |
| Robustness over speed | Trust region |
| Near-quadratic problems | Line search |

## Comparison Example

```julia
using SequentialQuadraticProgramming, LinearAlgebra

f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
g(x) = [-prod(x) + 25.0]
h(x) = [x' * x - 40.0]
x0 = [1.0, 5.0, 5.0, 1.0]
lb = ones(4); ub = 5.0 * ones(4)

r_ls = sqp_solve(f, g, h, x0, lb, ub;
    options = SQPOptions(globalization = :line_search))
r_tr = sqp_solve(f, g, h, x0, lb, ub;
    options = SQPOptions(globalization = :trust_region, max_iterations = 500))

println("Line search: $(r_ls.iterations) iters, obj = $(round(r_ls.objective, digits = 4))")
println("Trust region: $(r_tr.iterations) iters, obj = $(round(r_tr.objective, digits = 4))")
```
