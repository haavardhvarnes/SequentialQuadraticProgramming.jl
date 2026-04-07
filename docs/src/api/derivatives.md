# Derivatives

Derivative functions are constructed via [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) with `prepare_*` caching for zero-overhead repeated evaluation.

## Gradient

```@docs
SequentialQuadraticProgramming.make_gradient
```

## Jacobian

```@docs
SequentialQuadraticProgramming.make_jacobian
```

## Hessian

```@docs
SequentialQuadraticProgramming.make_hessian
```
