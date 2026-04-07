# Hessian Updates

The solver maintains an approximation ``H`` of the Hessian of the Lagrangian. The update strategy is:

1. Try analytical Hessian (ForwardDiff or external) - use if positive definite
2. L-BFGS reconstruction from stored `(s, y)` history pairs
3. Robust BFGS update (Yang et al. 2022)
4. Fall back to identity matrix

## Robust BFGS

```@docs
SequentialQuadraticProgramming.robust_bfgs_update!
```

## Classic BFGS

```@docs
SequentialQuadraticProgramming.bfgs_update!
```

## L-BFGS

```@docs
SequentialQuadraticProgramming.lbfgs_hessian
```

## Utilities

```@docs
SequentialQuadraticProgramming.ensure_positive_definite!
SequentialQuadraticProgramming.update_hessian!
```
