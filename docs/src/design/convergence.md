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
