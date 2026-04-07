"""
    solve_qp_trust_region(solver, H, g, h, df, dg, dh, x, delta_radius)

Solve the QP subproblem with trust region constraint `||d||∞ <= Δ`.

    min  0.5 d'Hd + df(x)'d
    s.t. dg(x) d <= -g(x)
         dh(x) d  = -h(x)
         -Δ <= d_i <= Δ      (trust region)

Returns `(dx, multipliers, delta_slack, A)`.
"""
function solve_qp_trust_region(
    solver::COSMOQPSolver,
    H::AbstractMatrix{T}, g, h, df, dg, dh,
    x::AbstractVector{T}, delta_radius::T,
) where {T <: AbstractFloat}
    n = length(x)

    df_ = df(x)
    dg_ = dg(x)
    dh_ = dh(x)
    neq = size(dh_, 1)
    nlt = size(dg_, 1)

    # Original constraint bounds
    lb_con = if neq > 0 || nlt > 0
        T[fill(T(-Inf), nlt); -h(x)]
    else
        T[]
    end
    ub_con = if neq > 0 || nlt > 0
        T[-g(x); -h(x)]
    else
        T[]
    end
    A_con = if neq > 0 || nlt > 0
        [dg_; dh_]
    else
        zeros(T, 0, n)
    end

    # Add trust region bounds: -Δ <= d_i <= Δ via identity rows
    A_tr = Matrix{T}(I, n, n)
    lb_tr = fill(-delta_radius, n)
    ub_tr = fill(delta_radius, n)

    # Stack constraints
    A_full = [A_con; A_tr]
    lb_full = [lb_con; lb_tr]
    ub_full = [ub_con; ub_tr]

    m_con = size(A_con, 1)
    m_full = size(A_full, 1)

    con = COSMO.Constraint(sparse(A_full), zeros(T, m_full), COSMO.Box(lb_full, ub_full))
    settings = COSMO.Settings(max_iter = solver.max_iter, verbose = false,
                              eps_abs = solver.eps_abs, eps_rel = solver.eps_rel)
    model = COSMO.Model()
    COSMO.assemble!(model, sparse(H), df_, con, settings = settings)
    result = COSMO.optimize!(model)

    # Return only the constraint multipliers (not the trust region ones)
    return result.x[1:n], result.y[1:m_con], zero(T), A_con
end

"""
    compute_reduction_ratio(merit_current, merit_trial, predicted_reduction)

Compute the trust region reduction ratio ρ = actual / predicted.
"""
function compute_reduction_ratio(
    merit_current::T, merit_trial::T, predicted_reduction::T,
) where {T <: AbstractFloat}
    actual = merit_current - merit_trial
    if abs(predicted_reduction) < eps(T)
        return actual >= zero(T) ? one(T) : zero(T)
    end
    return actual / predicted_reduction
end

"""
    update_trust_radius(delta, rho, dx_norm, delta_max)

Update the trust region radius based on the reduction ratio.

- ρ > 0.75 and step at boundary: expand
- ρ < 0.25: shrink
- otherwise: keep
"""
function update_trust_radius(
    delta::T, rho::T, dx_norm::T, delta_max::T,
) where {T <: AbstractFloat}
    if rho > T(0.75) && dx_norm >= T(0.95) * delta
        return min(T(2) * delta, delta_max)
    elseif rho < T(0.25)
        return delta / T(4)
    else
        return delta
    end
end
