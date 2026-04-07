# SequentialQuadraticProgramming.jl

## Purpose

Schittkowski-style SQP solver for nonlinear constrained optimization:
`min f(x) s.t. g(x) <= 0, h(x) = 0, lb <= x <= ub`

Ported from the original script-based implementation at `/Users/havard/Documents/devroot/jl/SQP/src/csqp.jl`.

## Architecture

```
src/
  SequentialQuadraticProgramming.jl  # Module: imports + includes + exports
  types.jl          # SQPOptions, SQPResult, SQPWorkspace, AbstractQPSolver
  problem.jl        # NLPProblem struct (bounds -> inequality conversion)
  derivatives.jl    # ForwardDiff gradient/jacobian/hessian with FiniteDiff fallback
  qp_subproblem.jl  # COSMOQPSolver: solve_qp() and solve_qp_with_slack()
  merit.jl          # augmented_lagrangian() merit function
  line_search.jl    # schittkowski_line_search() with quadratic interpolation + Armijo
  hessian_update.jl # robust_bfgs_update!() (Yang 2022), bfgs_update!(), ensure_positive_definite!()
  solver.jl         # sqp_solve() main loop + convenience wrappers
ext/
  SequentialQuadraticProgrammingMOIExt/  # MathOptInterface wrapper (Phase 2)
```

## Algorithm Flow (solver.jl)

1. Build derivative functions (ForwardDiff, FiniteDiff fallback)
2. Initialize Hessian (try analytical, fall back to identity)
3. For each iteration:
   a. Ensure H positive definite
   b. Update rho_k (Schittkowski eq 10 from NM_SQP2.pdf)
   c. Solve QP subproblem -> dx, multipliers, delta
   d. Update penalty parameters r (eq 14)
   e. Line search (quadratic interpolation, Armijo condition)
   f. Update x, check convergence
   g. Update Hessian (analytical or robust BFGS)

## Design Decisions

- **COSMO as default QP solver**: Matches the original working code. COSMO.Box constraints.
- **ForwardDiff.derivative replaces Zygote**: Merit function derivative w.r.t. alpha is scalar-to-scalar. Removes Zygote+ReverseDiff hard dependencies.
- **Generic types**: `T <: AbstractFloat` throughout, not hardcoded `Float64`.
- **Immutable options/result, mutable workspace**: SQPOptions and SQPResult are immutable. SQPWorkspace is mutable for in-place updates.
- **Bounds as inequalities**: Variable bounds converted to `g(x) <= 0` form in NLPProblem constructor.

## Conventions

Follow the SciML Style Guide:
- `lower_snake_case` functions, `CamelCase` types
- 4-space indentation
- Generic code (support multiple numeric types)
- Prefer immutable structs
- `!` suffix for in-place functions

## Key References

- Schittkowski NM_SQP2.pdf: Core SQP algorithm, penalty update (eq 14), line search
- Yang et al. 2022: Robust BFGS update (https://arxiv.org/pdf/1212.5929.pdf)
- NLPQL article 1985: QP subproblem formulation (eq 9)

## Registry

Private registry at `https://github.com/haavardhvarnes/JuliaRegistry`. Use `LocalRegistry.jl` to register new versions.

## MOI Extension (Phase 2)

Located in `ext/SequentialQuadraticProgrammingMOIExt/`. Activated when `MathOptInterface` (or `JuMP`) is loaded.

- `Optimizer <: MOI.AbstractOptimizer` — JuMP-compatible solver
- Supports: NLPBlock, variable bounds, linear LE/GE/EQ constraints, MAX_SENSE
- NLP evaluator callbacks use `Float64.(x)` to prevent ForwardDiff Dual propagation (MOI evaluators don't support AD). Derivatives computed via FiniteDiff fallback.
- Status mapping: `:converged` → `LOCALLY_SOLVED`, `:max_iterations` → `ITERATION_LIMIT`
- Usage: `Model(SequentialQuadraticProgramming.Optimizer)`

## Solver accepts optional derivative functions (Phase 3)

`sqp_solve` accepts keyword arguments `grad_f`, `jac_g`, `jac_h`, `hess_lag` to provide pre-computed derivatives. When provided, ForwardDiff/FiniteDiff are bypassed entirely. The MOI wrapper uses this to pass exact derivatives from the JuMP evaluator (reverse-mode AD).

## L-BFGS fallback (Phase 3)

When the analytical Hessian is not positive definite, the solver tries L-BFGS reconstruction from stored (s, y) history pairs before falling back to identity. This preserves curvature information across iterations.

## DifferentiationInterface.jl (Phase 4)

`src/derivatives.jl` uses DifferentiationInterface.jl (DI) with `prepare_*` caching for zero-overhead repeated differentiation. Users can pass any ADTypes.jl backend via `ad_backend` kwarg:

```julia
using ADTypes
result = sqp_solve(f, g, h, x0; ad_backend=AutoForwardDiff())  # explicit
result = sqp_solve(f, g, h, x0; ad_backend=AutoFiniteDiff())   # finite diff
result = sqp_solve(f, g, h, x0)                                # auto-select
```

Default: auto-selects `AutoForwardDiff()`, falls back to `AutoFiniteDiff()` if ForwardDiff fails (e.g., functions with FFI calls).

## Phase Roadmap

- **v0.1.0**: Core solver, functional API, COSMO QP, ForwardDiff, test suite
- **v0.2.0**: MathOptInterface extension for JuMP integration
- **v0.3.0**: Exact MOI evaluator derivatives, L-BFGS fallback, optional derivative kwargs
- **v0.4.0** (current): DifferentiationInterface.jl pluggable AD backends
- **v0.5.0**: Clarabel extension (alternative QP solver via package extension)
- **v0.6.0** (current): Trust region globalization variant

## Trust Region Globalization (Phase 6)

Alternative to line search. Enabled via `SQPOptions(globalization=:trust_region)`.

Constrains the QP step to `||d||∞ ≤ Δ` and uses reduction ratio to accept/reject steps
and adjust the radius. More conservative than line search but can be more robust for
ill-conditioned problems. Line search remains the default.

```julia
result = sqp_solve(f, g, h, x0; options=SQPOptions(
    globalization=:trust_region,
    trust_region_init=1.0,    # initial Δ
    trust_region_max=1e4,     # max Δ
    trust_region_eta=0.1,     # acceptance threshold
))
```

## Clarabel Extension (Phase 5)

Package extension in `ext/SequentialQuadraticProgrammingClarabelExt/`. Activated by `using Clarabel`.
Provides `ClarabelQPSolver <: AbstractQPSolver` as alternative to the default `COSMOQPSolver`.
Converts COSMO Box constraints to Clarabel's conic form (NonnegativeCone).

Note: COSMO is faster on typical SQP QP subproblems (Box constraints are native). Clarabel is
useful when COSMO is unavailable or for problems that benefit from Clarabel's interior point method.
