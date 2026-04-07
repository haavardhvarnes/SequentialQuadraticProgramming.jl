"""
    HiGHSQPSolver <: AbstractQPSolver

QP subproblem solver using HiGHS (simplex/interior point solver).
Uses the HiGHS C API directly for minimal overhead.

HiGHS natively supports row-based box constraints `lb <= Ax <= ub`,
which maps directly to the SQP QP subproblem form.

Usage:
    using SequentialQuadraticProgramming, HiGHS
    result = sqp_solve(f, g, h, x0; qp_solver=HiGHSQPSolver())
"""
struct HiGHSQPSolver <: SQP.AbstractQPSolver
    verbose::Bool
end

HiGHSQPSolver(; verbose::Bool = false) = HiGHSQPSolver(verbose)

# HiGHS constants
const _HighsInt = Cint
const _kColwise = _HighsInt(1)
const _kTriangular = _HighsInt(1)
const _kMinimize = _HighsInt(1)

"""
Build sparse lower-triangular CSC representation of H for HiGHS.
HiGHS expects lower triangular in CSC format (= upper triangular in CSR).
"""
function _build_hessian_csc(H::AbstractMatrix{T}, n::Int) where {T}
    Q = sparse(tril(H))
    colptr = Vector{_HighsInt}(Q.colptr .- _HighsInt(1))
    rowidx = Vector{_HighsInt}(Q.rowval .- _HighsInt(1))
    vals = Vector{Cdouble}(Q.nzval)
    return colptr, rowidx, vals, _HighsInt(length(vals))
end

"""
Solve a QP using HiGHS C API:
    min  0.5 d'Hd + c'd
    s.t. lb_row <= A d <= ub_row
         lb_col <= d <= ub_col
"""
function _solve_qp_highs(
    H::AbstractMatrix, c::Vector{Float64},
    A::AbstractMatrix, lb_row::Vector{Float64}, ub_row::Vector{Float64},
    lb_col::Vector{Float64}, ub_col::Vector{Float64};
    verbose::Bool = false,
)
    n = length(c)
    m = length(lb_row)

    highs = HiGHS.Highs_create()
    try
        if !verbose
            HiGHS.Highs_setBoolOptionValue(highs, "output_flag", _HighsInt(0))
        end

        # Build sparse A in CSC format
        As = sparse(A)
        a_colptr = Vector{_HighsInt}(As.colptr .- _HighsInt(1))
        a_rowidx = Vector{_HighsInt}(As.rowval .- _HighsInt(1))
        a_vals = Vector{Cdouble}(As.nzval)
        a_nnz = _HighsInt(length(a_vals))

        # Build sparse Hessian (lower triangular CSC)
        q_colptr, q_rowidx, q_vals, q_nnz = _build_hessian_csc(H, n)

        # Pass full model
        ret = HiGHS.Highs_passModel(
            highs,
            _HighsInt(n),           # num_col
            _HighsInt(m),           # num_row
            a_nnz,                  # num_nz (constraint matrix)
            q_nnz,                  # q_num_nz (Hessian)
            _kColwise,              # a_format
            _kTriangular,           # q_format
            _kMinimize,             # sense
            Cdouble(0.0),           # offset
            Vector{Cdouble}(c),     # col_cost
            Vector{Cdouble}(lb_col),# col_lower
            Vector{Cdouble}(ub_col),# col_upper
            Vector{Cdouble}(lb_row),# row_lower
            Vector{Cdouble}(ub_row),# row_upper
            a_colptr,               # a_start
            a_rowidx,               # a_index
            a_vals,                 # a_value
            q_colptr,               # q_start
            q_rowidx,               # q_index
            q_vals,                 # q_value
            C_NULL,                 # integrality (NULL = continuous)
        )

        # Solve
        HiGHS.Highs_run(highs)

        # Extract solution
        col_value = zeros(Cdouble, n)
        col_dual = zeros(Cdouble, n)
        row_value = zeros(Cdouble, m)
        row_dual = zeros(Cdouble, m)
        HiGHS.Highs_getSolution(highs, col_value, col_dual, row_value, row_dual)

        return Vector{Float64}(col_value), Vector{Float64}(row_dual)
    finally
        HiGHS.Highs_destroy(highs)
    end
end

function SQP.solve_qp(
    solver::HiGHSQPSolver,
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
        Float64[fill(-Inf, nlt); -h(x)]
    else
        Float64[]
    end
    ub = if neq > 0 || nlt > 0
        Float64[-g(x); -h(x)]
    else
        Float64[]
    end
    A = if neq > 0 || nlt > 0
        Float64[dg_; dh_]
    else
        zeros(Float64, 0, n)
    end

    m = size(A, 1)

    if m == 0
        dx = -H \ df_
        return Vector{T}(dx), T[], zero(T), A
    end

    col_lb = fill(-1e20, n)  # HiGHS uses finite bounds, not -Inf
    col_ub = fill(1e20, n)

    dx, row_dual = _solve_qp_highs(
        Float64.(H), Float64.(df_), Float64.(A),
        Float64.(lb), Float64.(ub), col_lb, col_ub;
        verbose = solver.verbose)

    return Vector{T}(dx), Vector{T}(row_dual[1:m]), zero(T), A
end

function SQP.solve_qp_with_slack(
    solver::HiGHSQPSolver,
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
        Float64[fill(-Inf, nlt); -h(x)]
    else
        Float64[]
    end
    ub = if neq > 0 || nlt > 0
        Float64[-g(x); -h(x)]
    else
        Float64[]
    end
    A = if neq > 0 || nlt > 0
        Float64[dg_; dh_]
    else
        zeros(Float64, 0, n)
    end

    m = size(A, 1)

    # Extend H for delta variable
    Hqp = Float64[H zeros(T, n); zeros(T, n)' rho]
    if !isposdef(sparse(Hqp))
        Hqp = Float64(100) * ones(Float64, n + 1, n + 1) .* I(n + 1)
        Hqp[n + 1, n + 1] = Float64(rho)
    end

    # Extend A for delta variable
    uqp_col = copy(ub)
    for i in 1:nlt
        uqp_col[i] = !isinf(ub[i]) && (multiplier[i] > zero(T) || T(1e-6) < ub[i]) ? ub[i] : zero(Float64)
    end
    Aqp = Float64[A uqp_col; zeros(Float64, n)' 1.0]

    lqp = Float64[lb; 0.0]
    uqp = Float64[ub; 1.0 - 1e-5]
    cqp = Float64[df_; 0.0]

    m_ext = size(Aqp, 1)

    col_lb = [fill(-1e20, n); 0.0]      # delta >= 0
    col_ub = [fill(1e20, n); 1.0 - 1e-5] # delta <= 1-eps

    dx, row_dual = _solve_qp_highs(
        Hqp, cqp, Aqp, lqp, uqp, col_lb, col_ub;
        verbose = solver.verbose)

    return Vector{T}(dx[1:n]), Vector{T}(row_dual[1:m]), T(dx[n + 1]), A
end
