# Problem Diagnostics & Strategy Selection

For hard nonlinear problems, choosing the right solver options can mean the difference between convergence in 50 iterations and failure at 2000. This tutorial shows how to use the built-in diagnostics to guide that choice.

## Running Diagnostics

```julia
using SequentialQuadraticProgramming, LinearAlgebra

f(x) = sum(x .^ 2)
g(x) = [some_nonlinear_constraint(x)]
h(x) = zeros(0)
x0 = randn(6)

problem = NLPProblem(f, g, h, x0)
diag = diagnose_problem(problem)
```

The [`ProblemDiagnostics`](@ref) result contains:

| Field | What it tells you |
|:------|:-----------------|
| `constraint_nonlinearity` | How nonlinear the constraints are (0 = linear, > 0.3 = highly nonlinear) |
| `jacobian_condition` | Condition number of the constraint Jacobian at ``x_0`` |
| `hessian_posdef` | Whether the Lagrangian Hessian is positive definite at ``x_0`` |
| `initial_feasibility` | Constraint violation at the starting point |
| `recommended_strategy` | Suggested globalization: `:line_search`, `:line_search_soc`, or `:trust_region` |
| `warnings` | Human-readable notes about potential issues |

## Automatic Diagnostics

Enable diagnostics at solve time with `diagnose = true`:

```julia
result = sqp_solve(f, g, h, x0;
    options = SQPOptions(diagnose = true, verbose = true))

# Access diagnostics from result
result.diagnostics.recommended_strategy
result.diagnostics.warnings
```

## Decision Guide

### Globalization Strategy

| Diagnostic signal | Recommended option |
|:-----------------|:-------------------|
| Low nonlinearity, PD Hessian | `globalization = :line_search` (default) |
| High nonlinearity (> 0.3) | `globalization = :filter_line_search` |
| High nonlinearity + SOC recommended | `use_soc = true` (with `:line_search`) |
| Ill-conditioned Jacobian | `globalization = :trust_region` |
| Merit function oscillating (visible in verbose trace) | `globalization = :filter_line_search` |

### Hessian Strategy

| Problem type | Recommended option |
|:------------|:-------------------|
| General-purpose | `hessian_strategy = :bfgs` (default) |
| Bilinear/highly nonlinear objective (HS108-class) | `hessian_strategy = :analytical` |
| Large problem (n > 50) | `hessian_strategy = :bfgs` (analytical Hessian is dense) |
| Unsure | `hessian_strategy = :auto` (uses `:analytical` when n <= 50) |

### Numerical Safeguards

Safeguards are enabled by default (`numerical_safeguards = true`). They protect against:

- **Step clamping**: Prevents pathologically large QP steps (``\|dx\|_\infty > 100 \cdot (1 + \|x\|_\infty)``)
- **BFGS skip**: Skips Hessian update when the line search collapsed to a tiny step (``\alpha < 10^{-3}``)
- **Levenberg-Marquardt damping**: Adds ``\lambda I`` to ``H`` when the solver detects ill-conditioning

Disable for well-conditioned problems where you want maximum speed:
```julia
options = SQPOptions(numerical_safeguards = false)
```

## Worked Example: HS092

HS092 is a 6-variable problem with a single highly nonlinear constraint. The default line search takes 352 iterations; the filter line search converges in ~58.

```julia
using SequentialQuadraticProgramming

# ... (define f, g, h, x0 for HS092) ...

# Step 1: Diagnose
problem = NLPProblem(f, g, h, x0)
diag = diagnose_problem(problem)
println("Nonlinearity: ", diag.constraint_nonlinearity)
println("Recommended:  ", diag.recommended_strategy)

# Step 2: Solve with recommended strategy
result_default = sqp_solve(f, g, h, x0;
    options = SQPOptions(max_iterations = 500))
println("Default:  $(result_default.iterations) iters, obj=$(round(result_default.objective, digits=4))")

result_filter = sqp_solve(f, g, h, x0;
    options = SQPOptions(max_iterations = 500,
                          globalization = :filter_line_search))
println("Filter:   $(result_filter.iterations) iters, obj=$(round(result_filter.objective, digits=4))")
println("f-steps:  $(result_filter.n_filter_f_steps)")
println("h-steps:  $(result_filter.n_filter_h_steps)")
```

## Second-Order Correction

For problems exhibiting the **Maratos effect** (the merit function increases at the full Newton step despite making progress toward the solution), enable SOC:

```julia
result = sqp_solve(f, g, h, x0;
    options = SQPOptions(use_soc = true, soc_max_tries = 3))
```

SOC computes a correction step ``d_c`` that drives the linearized constraint residual toward zero at the trial point. This is most effective when:

- The solver takes very small steps (``\alpha \ll 1``) near the solution
- Verbose output shows `dphi0 > 0` frequently
- The problem has tight nonlinear equality constraints

!!! note
    SOC is currently supported only with `COSMOQPSolver` (the default).
