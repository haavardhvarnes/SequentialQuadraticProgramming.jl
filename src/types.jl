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
    # Trust region options
    globalization::Symbol       # :line_search or :trust_region
    trust_region_init::T        # initial trust region radius
    trust_region_max::T         # maximum radius
    trust_region_eta::T         # step acceptance threshold
    # Phase 8: diagnostics and SOC
    diagnose::Bool              # run `diagnose_problem` at solver start
    use_soc::Bool               # enable Second-Order Correction on uphill line search
    soc_max_tries::Int          # max SOC attempts per iteration
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
    globalization::Symbol = :line_search,
    trust_region_init::T = T(1.0),
    trust_region_max::T = T(1e4),
    trust_region_eta::T = T(0.1),
    diagnose::Bool = false,
    use_soc::Bool = false,
    soc_max_tries::Int = 1,
) where {T <: AbstractFloat}
    SQPOptions{T}(max_iterations, xtol, ftol, constraint_tol, verbose,
                  line_search_mu, line_search_beta, phi0_lookback, qp_max_iter,
                  globalization, trust_region_init, trust_region_max, trust_region_eta,
                  diagnose, use_soc, soc_max_tries)
end

SQPOptions(; kwargs...) = SQPOptions{Float64}(; kwargs...)

"""
    ProblemDiagnostics{T}

Characterization of a nonlinear programming problem at the initial point `x0`.
Returned by [`diagnose_problem`](@ref). High `constraint_nonlinearity` indicates
the QP linearization is a poor model of the true constraints (Maratos regime).
In that case, enable `use_soc = true` in `SQPOptions` for Second-Order Correction.

# Fields
- `constraint_nonlinearity::T` — average ratio
  `‖c(x0+δ) - c(x0) - J(x0)·δ‖ / ‖J(x0)·δ‖` over random perturbations.
  Values > 2 indicate strong nonlinearity.
- `jacobian_condition::T` — condition number of `[Jg(x0); Jh(x0)]` via SVD.
- `hessian_posdef::Bool` — whether `∇²f(x0)` is positive definite.
- `initial_feasibility::T` — `‖max(g(x0), 0); h(x0)‖_∞`.
- `n_variables::Int`, `n_ineq::Int`, `n_eq::Int` — problem dimensions.
- `recommended_strategy::Symbol` — one of `:line_search`, `:line_search_soc`,
  `:trust_region`.
- `warnings::Vector{String}` — human-readable concerns.
"""
struct ProblemDiagnostics{T <: AbstractFloat}
    constraint_nonlinearity::T
    jacobian_condition::T
    hessian_posdef::Bool
    initial_feasibility::T
    n_variables::Int
    n_ineq::Int
    n_eq::Int
    recommended_strategy::Symbol
    warnings::Vector{String}
end

"""
    SQPResult{T, V}

Result returned by the SQP solver.

# Fields
- `x::V` — optimal solution
- `objective::T` — objective value at `x`
- `iterations::Int` — number of SQP iterations performed
- `converged::Bool` — whether convergence criteria were met
- `constraint_violation::T` — `‖max(g(x), 0); h(x)‖_∞` at termination
- `status::Symbol` — `:converged`, `:max_iterations`, `:qp_failed`,
  `:line_search_failed`, `:trust_region_failed`, etc.
- `diagnostics::Union{Nothing, ProblemDiagnostics{T}}` — populated when
  `SQPOptions(diagnose=true)` was used, otherwise `nothing`.
- `n_soc_steps::Int` — number of Second-Order Correction steps taken
  (zero unless `use_soc=true`).
"""
struct SQPResult{T <: AbstractFloat, V <: AbstractVector{T}}
    x::V
    objective::T
    iterations::Int
    converged::Bool
    constraint_violation::T
    status::Symbol
    diagnostics::Union{Nothing, ProblemDiagnostics{T}}
    n_soc_steps::Int
end

# Backward-compatible constructor — old call sites without diagnostics/n_soc_steps
function SQPResult(x::V, objective::T, iterations::Int, converged::Bool,
                   constraint_violation::T, status::Symbol) where {
                   T <: AbstractFloat, V <: AbstractVector{T}}
    return SQPResult{T, V}(x, objective, iterations, converged,
                           constraint_violation, status, nothing, 0)
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
