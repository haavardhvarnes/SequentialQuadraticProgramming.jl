# QP Subsolvers

The SQP algorithm solves a quadratic programming (QP) subproblem at each iteration. Three QP solvers are available:

| Solver | Package | Type | Best for |
|:-------|:--------|:-----|:---------|
| `COSMOQPSolver()` | COSMO.jl (built-in) | Interior point | Default, small-medium QPs |
| `ClarabelQPSolver()` | Clarabel.jl (extension) | Interior point (conic) | Alternative solver |
| `HiGHSQPSolver()` | HiGHS.jl (extension) | Simplex / IPM | Large sparse QPs |

## Abstract Type

```@docs
SequentialQuadraticProgramming.AbstractQPSolver
```

## COSMO (Default)

```@docs
COSMOQPSolver
```

## QP Methods

```@docs
SequentialQuadraticProgramming.solve_qp
SequentialQuadraticProgramming.solve_qp_with_slack
SequentialQuadraticProgramming.solve_qp_trust_region
```
