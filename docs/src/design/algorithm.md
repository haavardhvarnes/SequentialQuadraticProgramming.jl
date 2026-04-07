# Algorithm Overview

## Problem Formulation

The solver addresses the general nonlinear programming problem:

```math
\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad g(x) \leq 0, \quad h(x) = 0, \quad l \leq x \leq u
```

Variable bounds are internally converted to inequality constraints: ``g_\text{wrapped}(x) = [g(x);\, l - x;\, x - u]``.

## SQP Iteration

At each iteration ``k``, the algorithm:

### 1. QP Subproblem

Solve a quadratic approximation of the original problem:

```math
\min_d \frac{1}{2} d^T H_k d + \nabla f(x_k)^T d \quad \text{s.t.} \quad \nabla g(x_k) d \leq -g(x_k), \quad \nabla h(x_k) d = -h(x_k)
```

where ``H_k`` approximates the Hessian of the Lagrangian. The QP is solved using COSMO (default), Clarabel, or HiGHS.

An optional **Schittkowski delta slack variable** ``\delta`` relaxes the constraints for infeasibility detection:

```math
\min_{d,\delta} \frac{1}{2} d^T H_k d + \nabla f(x_k)^T d + \frac{1}{2} \rho \delta^2 \quad \text{s.t.} \quad \nabla g(x_k) d + g(x_k)\delta \leq -g(x_k), \quad 0 \leq \delta \leq 1
```

### 2. Multiplier Update

Lagrange multipliers ``\sigma`` (inequality) and ``\lambda`` (equality) are extracted from the QP dual solution.

### 3. Penalty Parameter Update

Adaptive penalty parameters ``r`` are updated using Schittkowski's formula (NM\_SQP2, eq. 14):

```math
r_j \leftarrow \max\left(\pi_j r_j, \; \frac{2(m+p)(u_j - v_j)}{(1-\delta)(d^T H d)}\right)
```

where ``\pi_j = \min(1, k / (\log r_j + 1))``.

### 4. Globalization

**Line search** (default): Find ``\alpha \in (0, 1]`` satisfying the Armijo condition on the augmented Lagrangian merit function using quadratic interpolation.

**Trust region** (optional): Constrain ``\|d\|_\infty \leq \Delta`` and accept/reject based on actual vs predicted merit reduction.

### 5. Hessian Update

The Hessian approximation ``H_k`` is updated using a cascade strategy:

1. **Analytical Hessian**: Compute ``\nabla^2 L(x_k)`` via AD. Use if positive definite.
2. **L-BFGS reconstruction**: If analytical Hessian is not PD and history is available, reconstruct from stored ``(s, y)`` pairs.
3. **Robust BFGS**: Apply the Yang et al. (2022) update, which guarantees positive definiteness through a modified curvature condition.
4. **Identity fallback**: If all else fails, reset ``H = I``.

## Augmented Lagrangian Merit Function

The merit function combines objective, constraint penalty, and multiplier estimates:

```math
\psi(x, v, r) = f(x) + \sum_{j} \begin{cases} v_j g_j(x) + \frac{1}{2} r_j g_j(x)^2 & \text{if } g_j \geq v_j/r_j \\ \frac{1}{2} v_j^2 / r_j & \text{otherwise} \end{cases} + \sum_{j} \left(\lambda_j h_j(x) + \frac{1}{2} r_j h_j(x)^2\right)
```

## Convergence Criteria

The solver converges when:

```math
(\|p\|_\infty \leq \epsilon_x \;\text{or}\; |f_k - f_{k-1}| < \epsilon_f) \quad \text{and} \quad \|[\max(g(x), 0);\, |h(x)|]\|_\infty < \min(\epsilon_x, \epsilon_c)
```

## References

- Schittkowski, K. "On the convergence of a sequential quadratic programming method with an augmented Lagrangian line search function" (NM\_SQP2)
- Yang, Y. et al. "A robust BFGS algorithm for unconstrained nonlinear optimization problems" (2022)
- Schittkowski, K. "NLPQL: A FORTRAN subroutine for solving constrained nonlinear programming problems" (1985)
