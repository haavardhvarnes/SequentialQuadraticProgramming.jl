# Hessian Strategies

The Hessian approximation ``H_k \approx \nabla^2 L(x_k)`` drives the QP search direction at each iteration. Choosing the right strategy can dramatically affect convergence.

## Available Strategies

Set via `SQPOptions(hessian_strategy = ...)`:

### `:bfgs` (Default)

Builds ``H`` from scratch using the robust BFGS update (Yang et al. 2022). Each iteration:

1. Try computing the analytical Hessian ``\nabla^2 L(x)`` via AD
2. If positive definite, use it directly
3. If not PD, fall through to BFGS update using the curvature pair ``(s, y)``
4. L-BFGS history (10 pairs) is stored; if ``H`` ever loses positive definiteness, it is reconstructed from history
5. If history is insufficient, reset ``H = I``

Best for: general-purpose use, problems where the Lagrangian Hessian is frequently indefinite (e.g., HS092).

```julia
result = sqp_solve(f, g, h, x0;
    options = SQPOptions(hessian_strategy = :bfgs))
```

### `:analytical`

Uses the true Hessian ``\nabla^2 L(x)`` at every iteration, with eigenvalue correction when it is not positive definite:

1. Compute ``\nabla^2 L(x)`` via AD
2. If PD, use directly
3. If not PD, apply eigenvalue correction: decompose ``H = V \Lambda V^T``, replace ``\lambda_i \to \max(|\lambda_i|, \epsilon)``

This preserves the true curvature *directions* while ensuring positive definiteness. The modified-Newton approach (`:abs` method) reflects negative eigenvalues rather than clipping them, so the QP makes progress along directions of negative curvature.

Best for: problems with bilinear or highly nonlinear objectives where BFGS builds a poor approximation (e.g., HS108).

```julia
result = sqp_solve(f, g, h, x0;
    options = SQPOptions(hessian_strategy = :analytical))
```

### `:auto`

Selects `:analytical` when ``n \leq`` `auto_hessian_max_n` (default 50), `:bfgs` otherwise. The rationale: eigenvalue decomposition is ``O(n^3)`` per iteration, which is affordable for small problems but prohibitive for large ones.

```julia
result = sqp_solve(f, g, h, x0;
    options = SQPOptions(hessian_strategy = :auto, auto_hessian_max_n = 30))
```

## Comparison Example

HS108 (9 variables, bilinear objective, 14 quadratic constraints) illustrates the difference:

```julia
using SequentialQuadraticProgramming

x0 = ones(9)
f(x) = -0.5(x[1]x[4] - x[2]x[3] + x[3]x[9] - x[5]x[9] + x[5]x[8] - x[6]x[7])
g(x) = [-(1 - x[3]^2 - x[4]^2);
        -(1 - x[5]^2 - x[6]^2);
        -(1 - x[9]^2);
        -(1 - x[1]^2 - (x[2] - x[9])^2);
        -(1 - (x[1] - x[5])^2 - (x[2] - x[6])^2);
        -(1 - (x[1] - x[7])^2 - (x[2] - x[8])^2);
        -(1 - (x[3] - x[5])^2 - (x[4] - x[6])^2);
        -(1 - (x[3] - x[7])^2 - (x[4] - x[8])^2);
        -(1 - x[7]^2 - (x[8] - x[9])^2);
        -(x[1]x[4] - x[2]x[3]); -x[3]x[9]; x[5]x[9];
        -(x[5]x[8] - x[6]x[7]); -x[9]]
h(x) = zeros(0)
lb = zeros(9); ub = 80000.0 * ones(9)

r_bfgs = sqp_solve(f, g, h, x0, lb, ub;
    options = SQPOptions(max_iterations = 500, hessian_strategy = :bfgs))
r_anal = sqp_solve(f, g, h, x0, lb, ub;
    options = SQPOptions(max_iterations = 500, hessian_strategy = :analytical))

println("BFGS:       converged=$(r_bfgs.converged), obj=$(round(r_bfgs.objective, digits=4)), iters=$(r_bfgs.iterations)")
println("Analytical: converged=$(r_anal.converged), obj=$(round(r_anal.objective, digits=4)), iters=$(r_anal.iterations)")
# BFGS:       converged=false, obj=-0.3399, iters=500
# Analytical: converged=true,  obj=-0.866,  iters=226
```

## When to Use Which

| Problem characteristics | Strategy | Why |
|:----------------------|:---------|:----|
| General-purpose, unknown structure | `:bfgs` | Robust, no eigenvalue decomposition cost |
| Bilinear/quadratic objective (HS108-class) | `:analytical` | BFGS builds poor curvature model for bilinear terms |
| Small problem (n < 50) where speed matters | `:analytical` | True curvature gives faster convergence |
| Large problem (n > 100) | `:bfgs` | Eigenvalue decomposition is ``O(n^3)`` |
| Highly nonlinear constraints, PD objective | `:bfgs` | Analytical Hessian is often indefinite, causing frequent corrections |

## Eigenvalue Correction Floor

When using `:analytical`, the correction floor controls the minimum eigenvalue:

```julia
options = SQPOptions(
    hessian_strategy = :analytical,
    hessian_correction_floor = 1e-6,  # default 1e-8
)
```

A larger floor makes the corrected Hessian more like a steepest-descent step; a smaller floor preserves more of the true curvature structure.

## Mid-Solve Fallback

If the `:analytical` strategy drives Levenberg-Marquardt damping to its maximum for 10 consecutive iterations (indicating the analytical direction isn't helping), the solver automatically switches to `:bfgs` for the remainder of the solve, resets ``H = I``, and clears L-BFGS history.
