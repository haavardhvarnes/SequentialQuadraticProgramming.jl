"""
    COSMOQPSolver <: AbstractQPSolver

QP subproblem solver using COSMO (conic solver).
"""
struct COSMOQPSolver <: AbstractQPSolver
    max_iter::Int
    eps_abs::Float64
    eps_rel::Float64
end

COSMOQPSolver(; max_iter::Int = 2500, eps_abs::Float64 = 1e-11, eps_rel::Float64 = 1e-11) =
    COSMOQPSolver(max_iter, eps_abs, eps_rel)

"""
    solve_qp(solver, H, g, h, df, dg, dh, x)

Solve the QP subproblem without the delta slack variable.

    min  0.5 * d' H d + df(x)' d
    s.t. dg(x) d <= -g(x)     (inequality)
         dh(x) d  = -h(x)     (equality)

Returns `(dx, multipliers, delta, A)`.
"""
function solve_qp(
    solver::COSMOQPSolver,
    H::AbstractMatrix{T}, g, h, df, dg, dh,
    x::AbstractVector{T},
) where {T <: AbstractFloat}
    n = length(x)

    df_ = df(x)
    dg_ = dg(x)
    dh_ = dh(x)
    neq = size(dh_, 1)
    nlt = size(dg_, 1)

    lb = if neq > 0 || nlt > 0
        T[fill(T(-Inf), nlt); -h(x)]
    else
        T[]
    end
    ub = if neq > 0 || nlt > 0
        T[-g(x); -h(x)]
    else
        T[]
    end
    A = if neq > 0 || nlt > 0
        [dg_; dh_]
    else
        zeros(T, 0, n)
    end

    m = size(A, 1)

    con = COSMO.Constraint(sparse(A), zeros(T, length(lb)), COSMO.Box(lb, ub))
    settings = COSMO.Settings(max_iter = solver.max_iter, verbose = false,
                              eps_abs = solver.eps_abs, eps_rel = solver.eps_rel)
    model = COSMO.Model()
    COSMO.assemble!(model, sparse(H), df_, con, settings = settings)
    result = COSMO.optimize!(model)

    return result.x[1:n], result.y[1:m], zero(T), A
end

"""
    solve_qp_with_slack(solver, H, g, h, df, dg, dh, x, multiplier, rho)

Solve the QP subproblem with Schittkowski delta slack variable for infeasibility.

    min  0.5 * [d; delta]' Hqp [d; delta] + [df(x); 0]' [d; delta]
    s.t. dg(x) d + g(x) * delta <= -g(x)   (inequality, relaxed)
         dh(x) d                 = -h(x)    (equality)
         0 <= delta <= 1 - eps

Returns `(dx, multipliers, delta, A)`.
"""
function solve_qp_with_slack(
    solver::COSMOQPSolver,
    H::AbstractMatrix{T}, g, h, df, dg, dh,
    x::AbstractVector{T}, multiplier::AbstractVector{T}, rho::T = T(1.1),
) where {T <: AbstractFloat}
    n = length(x)

    df_ = df(x)
    dg_ = dg(x)
    dh_ = dh(x)
    neq = size(dh_, 1)
    nlt = size(dg_, 1)

    lb = if neq > 0 || nlt > 0
        T[fill(T(-Inf), nlt); -h(x)]
    else
        T[]
    end
    ub = if neq > 0 || nlt > 0
        T[-g(x); -h(x)]
    else
        T[]
    end
    A = if neq > 0 || nlt > 0
        [dg_; dh_]
    else
        zeros(T, 0, n)
    end

    m = size(A, 1)

    # Extend H for delta variable
    Hqp = [H zeros(T, n); zeros(T, n)' rho]

    if !isposdef(sparse(Hqp))
        Hqp = T(100) * ones(T, n + 1, n + 1) .* I(n + 1)
        Hqp[n + 1, n + 1] = rho
    end

    # Extend A for delta variable
    uqp_col = copy(ub)
    for i in 1:nlt
        uqp_col[i] = !isinf(ub[i]) && (multiplier[i] > zero(T) || T(1e-6) < ub[i]) ? ub[i] : zero(T)
    end
    Aqp = [A uqp_col; zeros(T, n)' one(T)]

    # Extended bounds
    lqp = T[lb; zero(T)]
    uqp = T[ub; one(T) - T(1e-5)]
    cqp = T[df_; zero(T)]

    con = COSMO.Constraint(sparse(Aqp), zeros(T, length(lqp)), COSMO.Box(lqp, uqp))
    settings = COSMO.Settings(max_iter = solver.max_iter, verbose = false,
                              eps_abs = solver.eps_abs, eps_rel = solver.eps_rel)
    model = COSMO.Model()
    COSMO.assemble!(model, sparse(Hqp), cqp, con, settings = settings)
    result = COSMO.optimize!(model)

    return result.x[1:n], result.y[1:m], result.x[n + 1], A
end

"""
    solve_qp_correction(solver::COSMOQPSolver, H, c_trial, J, df_x, n_ineq)

Solve the Second-Order Correction (SOC) subproblem at a failed trial point:

    min  0.5 d_c' H d_c + df_x' d_c
    s.t. dg(x) * d_c ≤ -g(x_trial)   (inequality rows — first n_ineq rows of J)
         dh(x) * d_c  = -h(x_trial)  (equality rows — remaining rows of J)

Here `J = [dg(x); dh(x)]` is the primary-QP constraint Jacobian evaluated
at the current iterate `x` (not at the trial point), and
`c_trial = [g(x_trial); h(x_trial)]` is the true constraint residual
evaluated at the trial point `x_trial = x + dx`.

The first `n_ineq` rows of `c_trial`/`J` correspond to inequality
constraints; the remaining `n_eq = size(J,1) - n_ineq` rows correspond to
equality constraints. Inequality rows use a one-sided bound so that
currently-satisfied constraints stay feasible (they aren't forced back to
their boundary), while violated ones are driven to zero. Equality rows use
a two-sided bound.

The correction `d_c` drives the linearized constraint residual to zero at
the trial point, producing a corrected search direction `dx + d_c` that
typically recovers descent when the primary SQP step triggers the Maratos
effect. Reference: Nocedal & Wright, *Numerical Optimization* 2nd ed.,
Algorithm 18.3.

Returns the correction vector `d_c::Vector{T}`, or `nothing` if the
correction QP is infeasible or the solver fails.
"""
function solve_qp_correction(
    solver::COSMOQPSolver,
    H::AbstractMatrix{T},
    c_trial::AbstractVector{T},
    J::AbstractMatrix{T},
    df_x::AbstractVector{T},
    n_ineq::Int,
) where {T <: AbstractFloat}
    n = size(H, 1)
    m = size(J, 1)
    m == 0 && return zeros(T, n)

    # COSMO.Constraint(A, b, set) encodes A*x + b ∈ set.
    # With b = zeros we need J*d_c ∈ Box(lb, ub) where:
    #   inequality rows (i ∈ 1:n_ineq): lb = -Inf, ub = -c_trial[i]
    #     meaning the linearized constraint g(x) + dg(x)·d_c ≤ g(x_trial)
    #     but since g(x_trial) may be positive (violated) or negative (ok),
    #     the -c_trial[i] value drives the linearised model toward zero.
    #   equality rows: lb = ub = -c_trial[i]
    lb = Vector{T}(undef, m)
    ub = Vector{T}(undef, m)
    for i in 1:m
        if i <= n_ineq
            lb[i] = T(-Inf)
            ub[i] = -c_trial[i]
        else
            lb[i] = -c_trial[i]
            ub[i] = -c_trial[i]
        end
    end

    con = COSMO.Constraint(sparse(J), zeros(T, m), COSMO.Box(lb, ub))
    settings = COSMO.Settings(max_iter = solver.max_iter, verbose = false,
                              eps_abs = solver.eps_abs, eps_rel = solver.eps_rel)
    model = COSMO.Model()

    # Regularize the Hessian mildly to ensure the correction QP is well-posed
    # (the primary Hessian may have indefinite directions along the active set).
    H_reg = Matrix{T}(H)
    if !isposdef(H_reg)
        H_reg = H_reg + T(1e-6) * I
    end

    try
        COSMO.assemble!(model, sparse(H_reg), df_x, con, settings = settings)
        result = COSMO.optimize!(model)
        if result.status != :Solved && result.status != :Solved_inaccurate
            return nothing
        end
        d_c = result.x[1:n]
        any(isnan, d_c) && return nothing
        return d_c
    catch
        return nothing
    end
end
