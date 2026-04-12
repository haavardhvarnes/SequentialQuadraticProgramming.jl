# Algorithm Overview

## Problem Formulation

The solver addresses the general nonlinear programming problem:

```math
\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad g(x) \leq 0, \quad h(x) = 0, \quad l \leq x \leq u
```

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

### 4. Numerical Safeguards

When enabled (default), three safeguards protect against pathological BFGS behavior:

- **Step clamping**: If ``\|d\|_\infty > C \cdot (1 + \|x\|_\infty)`` (default ``C = 100``), rescale the step to prevent the line search from sampling extreme points.
- **BFGS skip**: If the line search collapsed to ``\alpha < 10^{-3}`` or the step was clamped, skip the BFGS update to avoid polluting ``H`` with a bad curvature sample.
- **Levenberg-Marquardt damping**: Add ``\lambda I`` to ``H`` before the QP solve. ``\lambda`` grows on bad iterations (clamped/skipped) and shrinks on good ones (``\alpha \geq 0.9``), bounding the step in directions where ``H`` has small eigenvalues.

### 5. Globalization

Three strategies are available:

#### Line Search (default)

Find ``\alpha \in (0, 1]`` satisfying the Armijo condition on the augmented Lagrangian merit function using quadratic interpolation (Schittkowski NM\_SQP2).

#### Filter Line Search (Wächter-Biegler 2006)

Replace the merit function with a bi-objective filter tracking ``(f, \theta)`` pairs, where ``\theta = \|[\max(g,0); |h|]\|_1``. A trial step is accepted if:

- **f-step** (switching condition met): ``\theta_k`` is small relative to the predicted descent, and the Armijo condition holds on ``f`` alone.
- **h-step**: The trial ``(f_\text{trial}, \theta_\text{trial})`` is not dominated by any filter entry.

After an h-step, the current ``(f_k, \theta_k)`` is added to the filter (with margin). The filter has FIFO eviction when it exceeds `filter_max_size`. If the filter rejects all backtracked ``\alpha`` values, the solver falls back to the Schittkowski merit line search.

#### Trust Region

Constrain the step to ``\|d\|_\infty \leq \Delta`` and accept/reject based on actual vs predicted merit reduction. The radius ``\Delta`` grows on good steps and shrinks on rejected ones.

### 6. Second-Order Correction

When the line search detects a non-descent direction (``\nabla \phi(0) \geq 0``), the **Maratos effect** may be the cause. SOC (Nocedal-Wright Algorithm 18.3) computes a correction step ``d_c`` by solving a second QP that drives the linearized constraint residual at the trial point toward zero:

```math
\min_{d_c} \frac{1}{2} d_c^T H d_c + \nabla f(x_k)^T d_c \quad \text{s.t.} \quad J(x_k) d_c \leq -c(x_k + d)
```

The corrected direction ``d + d_c`` often produces a descent step where ``d`` alone fails.

### 7. Hessian Update

The Hessian approximation ``H_k`` is updated using a cascade strategy:

1. **Analytical Hessian**: Compute ``\nabla^2 L(x_k)`` via AD. Use if positive definite.
2. **Eigenvalue correction** (`:analytical` strategy): If not PD, decompose ``H = V\Lambda V^T`` and replace ``\lambda_i \to \max(|\lambda_i|, \epsilon)``. This modified-Newton approach preserves curvature directions while ensuring positive definiteness.
3. **Robust BFGS** (`:bfgs` strategy): Apply the Yang et al. (2022) update with modified curvature condition, guaranteeing positive definiteness.
4. **L-BFGS reconstruction**: If ``H`` loses PD and history is available, reconstruct from stored ``(s, y)`` pairs.
5. **Identity fallback**: If all else fails, reset ``H = I``.

A **mid-solve fallback** detects when the analytical strategy drives LM damping to its maximum for 10 consecutive iterations and switches to BFGS.

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
- Wächter, A. and Biegler, L. T. "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming" (2006)
- Nocedal, J. and Wright, S. "Numerical Optimization", 2nd ed., Algorithm 18.3 (Second-Order Correction)
