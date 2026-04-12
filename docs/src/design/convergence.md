# Convergence & Tuning

## Status Codes

The solver returns a `status` symbol in [`SQPResult`](@ref):

| Status | Meaning | Action |
|:-------|:--------|:-------|
| `:converged` | All convergence criteria satisfied | Problem solved |
| `:max_iterations` | Iteration limit reached | Increase `max_iterations` or loosen tolerances |
| `:qp_failed` | QP subproblem failed (infeasible or numerical) | Check problem feasibility, try different QP solver |
| `:line_search_failed` | Line search could not find sufficient decrease | Check problem scaling, try trust region |
| `:trust_region_failed` | Trust region radius shrunk below tolerance | Problem may be ill-conditioned |

## Convergence Criteria

The solver declares convergence when **both** conditions hold:

1. **Optimality**: ``\|p\|_\infty \leq \texttt{xtol}`` or ``|f_k - f_{k-1}| < \texttt{ftol}``
2. **Feasibility**: ``\|[\max(g(x), 0);\, |h(x)|]\|_\infty < \min(\texttt{xtol}, \texttt{constraint\_tol})``

## Tuning Guidelines

### Problem Not Converging

1. **Increase iterations**: `SQPOptions(max_iterations = 2000)`
2. **Loosen tolerances**: `SQPOptions(xtol = 1e-4, ftol = 1e-4)`
3. **Try trust region**: `SQPOptions(globalization = :trust_region)`
4. **Better starting point**: Closer to feasible region
5. **Enable verbose**: `SQPOptions(verbose = true)` to diagnose

### Slow Convergence

- Watch the `norm_dx` column in verbose output. If it's very small but `c_viol` is large, the penalty parameters may need time to adjust.
- If `dphi0 > 0` appears frequently, the merit function gradient is positive — the solver is adjusting penalties.
- BFGS Hessian updates may take several iterations to build a good approximation.

### QP Failures

- Try a different QP solver: `qp_solver = HiGHSQPSolver()` or `ClarabelQPSolver()`
- Check if the problem is feasible at the starting point
- The solver automatically falls back to the delta-slack QP formulation on QP failure

### Numerical Issues

- Scale your problem: keep objective and constraint values in similar ranges
- Avoid very large bounds (use `Inf` instead of `1e20`)
- Check that constraint functions return consistent-length vectors

### Filter Line Search Not Converging

- Check `n_filter_fallbacks` — frequent fallbacks mean the filter is too conservative. Try increasing `filter_max_size`.
- If `n_filter_h_steps` dominates, the solver is mostly improving feasibility. The problem may need a feasible starting point.
- The filter seeds with ``(\!-\infty, \theta_\max)`` to bound constraint violation. If ``\theta_0`` is already near zero, increase `filter_switching_delta` to bias toward f-steps.

### Numerical Safeguard Tuning

The default safeguards work well for most problems. When tuning:

| Option | Default | Effect of increasing | Effect of decreasing |
|:-------|:--------|:---------------------|:---------------------|
| `step_clamp_factor` | 100 | Allows larger steps (less conservative) | Clamps more aggressively |
| `bfgs_skip_alpha` | 1e-3 | Skips BFGS on more iterations | Only skips on near-zero steps |
| `lm_grow` | 4 | LM damping ramps up faster | Slower ramp-up |
| `lm_shrink` | 0.25 | LM damping decays faster on good steps | Slower decay |
| `lm_max` | 1e4 | Higher maximum damping | Limits regularization |

### Hessian Strategy Selection

- `:bfgs` (default): Safe for all problems. May stall on bilinear/highly nonlinear objectives (HS108).
- `:analytical`: Computes true ``\nabla^2 L`` with eigenvalue correction. Faster convergence on small non-convex problems, but ``O(n^3)`` per iteration and may find wrong stationary point on some problems (HS092).
- `:auto`: Picks `:analytical` for ``n \leq 50``, `:bfgs` otherwise.

See [Hessian Strategies](@ref) tutorial for detailed guidance.
