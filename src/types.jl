"""
    SQPOptions{T}

Options for the SQP solver.
"""
struct SQPOptions{T <: AbstractFloat}
    max_iterations::Int
    xtol::T
    ftol::T
    constraint_tol::T
    verbose::Bool
    line_search_mu::T
    line_search_beta::T
    phi0_lookback::Int
    qp_max_iter::Int
end

function SQPOptions{T}(;
    max_iterations::Int = 200,
    xtol::T = T(1e-6),
    ftol::T = T(1e-6),
    constraint_tol::T = T(1e-6),
    verbose::Bool = false,
    line_search_mu::T = T(1e-4),
    line_search_beta::T = T(0.5),
    phi0_lookback::Int = 5,
    qp_max_iter::Int = 2500,
) where {T <: AbstractFloat}
    SQPOptions{T}(max_iterations, xtol, ftol, constraint_tol, verbose,
                  line_search_mu, line_search_beta, phi0_lookback, qp_max_iter)
end

SQPOptions(; kwargs...) = SQPOptions{Float64}(; kwargs...)

"""
    SQPResult{T, V}

Result returned by the SQP solver.
"""
struct SQPResult{T <: AbstractFloat, V <: AbstractVector{T}}
    x::V
    objective::T
    iterations::Int
    converged::Bool
    constraint_violation::T
    status::Symbol
end

"""
    SQPWorkspace{T, M, V}

Pre-allocated working memory for the SQP solver. Not user-facing.
"""
mutable struct SQPWorkspace{T <: AbstractFloat, M <: AbstractMatrix{T}, V <: AbstractVector{T}}
    x::V
    dx::V
    p::V
    H::M
    sigma::V
    lambda::V
    pensig::V
    penlam::V
    r::V
    u::V
    phi_history::Vector{T}
    k_reset::Int
    f_last::T
    # L-BFGS history
    s_history::Vector{V}
    y_history::Vector{V}
    lbfgs_memory::Int
end

function SQPWorkspace(x0::AbstractVector{T}, n_ineq::Int, n_eq::Int;
                      lbfgs_memory::Int = 10) where {T <: AbstractFloat}
    n = length(x0)
    SQPWorkspace{T, Matrix{T}, Vector{T}}(
        copy(x0),
        zeros(T, n),
        zeros(T, n),
        Matrix{T}(I, n, n),
        zeros(T, n_ineq),
        zeros(T, n_eq),
        zeros(T, n_ineq),
        zeros(T, n_eq),
        ones(T, n_eq + n_ineq),
        ones(T, n_ineq + n_eq),
        T[],
        0,
        zero(T),
        Vector{T}[],
        Vector{T}[],
        lbfgs_memory,
    )
end

"""
    AbstractQPSolver

Abstract type for QP subproblem solvers.
"""
abstract type AbstractQPSolver end
