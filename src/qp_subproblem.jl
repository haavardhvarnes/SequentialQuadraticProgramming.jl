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
