"""
    ClarabelQPSolver <: AbstractQPSolver

QP subproblem solver using Clarabel (conic solver).
Alternative to the default COSMOQPSolver.

Usage:
    using SequentialQuadraticProgramming, Clarabel
    result = sqp_solve(f, g, h, x0; qp_solver=ClarabelQPSolver())
"""
struct ClarabelQPSolver <: SQP.AbstractQPSolver
    verbose::Bool
end

ClarabelQPSolver(; verbose::Bool = false) = ClarabelQPSolver(verbose)

"""
    solve_qp(solver::ClarabelQPSolver, H, g, h, df, dg, dh, x)

Solve the QP subproblem using Clarabel's conic solver.

Clarabel form: min 0.5 x'Px + q'x  s.t.  Ax + s = b, s ∈ K
We convert box constraints [lb, ub] to nonnegative cone form:
  [-A; A] d + s = [-lb; ub],  s ≥ 0
"""
function SQP.solve_qp(
    solver::ClarabelQPSolver,
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

    if m == 0
        # No constraints — solve unconstrained QP
        dx = -H \ df_
        return dx, T[], zero(T), A
    end

    # Convert to Clarabel conic form: [-A; A] d + s = [-lb; ub], s ≥ 0
    Ac = [-A; A]
    b = [-lb; ub]
    m_conic = size(Ac, 1)

    cones = [Clarabel.NonnegativeConeT(m_conic)]
    settings = Clarabel.Settings(; verbose = solver.verbose)

    clarabel_solver = Clarabel.Solver(sparse(T(2) * H), df_, sparse(Ac), b, cones, settings)
    Clarabel.solve!(clarabel_solver)
    result = clarabel_solver.solution

    # Extract multipliers: max of upper and lower dual variables
    mhalf = m
    multipliers = max.(result.z[1:mhalf], result.z[(mhalf + 1):(2 * mhalf)])
    multipliers = max.(multipliers, zero(T))

    return result.x[1:n], multipliers, zero(T), A
end

"""
    solve_qp_with_slack(solver::ClarabelQPSolver, H, g, h, df, dg, dh, x, multiplier, rho)

Solve the QP subproblem with Schittkowski delta slack variable using Clarabel.
"""
function SQP.solve_qp_with_slack(
    solver::ClarabelQPSolver,
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

    m_ext = size(Aqp, 1)

    # Convert to conic form
    Ac = [-Aqp; Aqp]
    b = [-lqp; uqp]
    m_conic = size(Ac, 1)

    cones = [Clarabel.NonnegativeConeT(m_conic)]
    settings = Clarabel.Settings(; verbose = solver.verbose)

    clarabel_solver = Clarabel.Solver(T(2) * sparse(Hqp), cqp, sparse(Ac), b, cones, settings)
    Clarabel.solve!(clarabel_solver)
    result = clarabel_solver.solution

    # Extract multipliers
    multipliers = max.(result.z[1:m], result.z[(m_ext + 1):(m_ext + m)])
    multipliers = max.(multipliers, zero(T))

    return result.x[1:n], multipliers, result.x[n + 1], A
end
