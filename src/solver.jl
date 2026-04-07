"""
    sqp_solve(problem::NLPProblem; options, qp_solver, grad_f, jac_g, jac_h, hess_lag)

Solve a nonlinear programming problem using Sequential Quadratic Programming.

Optional derivative functions can be provided to bypass automatic differentiation:
- `grad_f`: `x -> gradient_vector` for the objective
- `jac_g`: `x -> jacobian_matrix` for inequality constraints
- `jac_h`: `x -> jacobian_matrix` for equality constraints
- `hess_lag`: `(x, sigma, lambda) -> hessian_matrix` for the Lagrangian

When called from the MOI wrapper, these are built from the evaluator's exact AD.

Returns an `SQPResult`.
"""
function sqp_solve(
    problem::NLPProblem{T};
    options::SQPOptions{T} = SQPOptions{T}(),
    qp_solver::AbstractQPSolver = COSMOQPSolver(),
    ad_backend = nothing,
    grad_f = nothing,
    jac_g = nothing,
    jac_h = nothing,
    hess_lag = nothing,
) where {T <: AbstractFloat}
    (; f, g, h, x0, n, n_ineq, n_eq) = problem

    if options.verbose
        @info "SQP solver" iterations=options.max_iterations xtol=options.xtol ftol=options.ftol
    end

    # Build derivative functions (use provided, or DI with specified/auto backend)
    df = grad_f !== nothing ? grad_f : make_gradient(f, x0; backend = ad_backend)
    dg = jac_g !== nothing ? jac_g : make_jacobian(g, x0; backend = ad_backend)
    dh = jac_h !== nothing ? jac_h : make_jacobian(h, x0; backend = ad_backend)

    # Build Lagrangian and its derivatives
    ws = SQPWorkspace(x0, n_ineq, n_eq)

    has_external_hessian = hess_lag !== nothing

    # Only build Lagrangian derivative functions if no external Hessian provided
    dl = nothing
    d2l = nothing
    if !has_external_hessian
        lagrangian(x) = if n_eq > 0 && n_ineq > 0
            f(x) + dot(ws.lambda, h(x)) + dot(ws.sigma, g(x))
        elseif n_eq > 0
            f(x) + dot(ws.lambda, h(x))
        elseif n_ineq > 0
            f(x) + dot(ws.sigma, g(x))
        else
            f(x)
        end
        dl = make_gradient(lagrangian, x0; backend = ad_backend)
        d2l = make_hessian(lagrangian, x0; backend = ad_backend)
    end

    # Initialize Hessian
    ws.f_last = f(x0)
    if has_external_hessian
        Had = try
            hess_lag(x0, ws.sigma, ws.lambda)
        catch
            Matrix{T}(I, n, n)
        end
    else
        Had = try
            d2l(x0)
        catch
            Matrix{T}(I, n, n)
        end
    end

    if isposdef(Had)
        ws.H .= Matrix{T}(Had)
        if options.verbose
            @info "Hessian is positive definite"
        end
    else
        ws.H .= Matrix{T}(I, n, n)
    end

    # Merit function (captures mutable penalty params via workspace)
    merit_phi(alpha) = begin
        x_trial = ws.x .+ alpha .* ws.dx
        pensig_trial = ws.pensig .+ alpha .* (ws.sigma .- ws.pensig)
        penlam_trial = ws.penlam .+ alpha .* (ws.lambda .- ws.penlam)
        augmented_lagrangian(f(x_trial), g(x_trial), h(x_trial),
                            pensig_trial, penlam_trial, ws.r)
    end

    dphi(alpha) = ForwardDiff.derivative(merit_phi, alpha)

    push!(ws.phi_history, merit_phi(zero(T)))

    constraint_violation_last = norm([max.(g(ws.x), zero(T)); abs.(h(ws.x))], Inf)

    rho_k = one(T)
    rho_0 = T(1.001)
    rho_s = T(1.2)
    delta = zero(T)
    A = zeros(T, 0, 0)
    trust_radius = options.trust_region_init
    use_trust_region = options.globalization == :trust_region

    if options.verbose
        header = use_trust_region ? "Iter\t\tobjective\tnorm_dx\t\tΔ\t\tc_viol" :
                                    "Iter\t\tobjective\tnorm_dx\t\tstep\t\tc_viol"
        println(header)
    end

    for i in 1:options.max_iterations
        # Ensure H is positive definite — try L-BFGS reconstruction if available
        if !isposdef(ws.H)
            if length(ws.s_history) >= 2
                ws.H .= lbfgs_hessian(ws.s_history, ws.y_history, n)
            else
                ws.H .= Matrix{T}(I, n, n)
            end
        end

        # Update rho_k (Schittkowski eq 10) — only for line search
        if !use_trust_region && i > 1 && (n_eq > 0 || n_ineq > 0)
            dxHdx = dot(ws.dx, ws.H * ws.dx)
            if abs(dxHdx) > eps(T) && abs(1 - delta) > eps(T)
                rho_k = max(rho_0, rho_s * (dot(ws.dx, A' * ws.u))^2 / ((1 - delta)^2 * dxHdx))
            end
        end

        # Solve QP subproblem
        H_dense = Matrix{Float64}(ws.H)
        qp_success = true

        if use_trust_region
            try
                ws.dx, ws.u, delta, A = solve_qp_trust_region(
                    qp_solver, H_dense, g, h, df, dg, dh, ws.x, trust_radius)
            catch e
                @warn "Trust region QP failed" exception=e
                qp_success = false
            end
        else
            try
                ws.dx, ws.u, delta, A = solve_qp(qp_solver, H_dense, g, h, df, dg, dh, ws.x)
            catch
                try
                    ws.dx, ws.u, delta, A = solve_qp_with_slack(
                        qp_solver, H_dense, g, h, df, dg, dh, ws.x, ws.u, rho_k)
                catch e
                    @warn "QP subproblem failed" exception=e
                    qp_success = false
                end
            end
        end

        if !qp_success || any(isnan, ws.dx)
            @warn "QP direction step failed"
            return SQPResult(ws.x, ws.f_last, i, false, constraint_violation_last, :qp_failed)
        end

        # Update multipliers
        ws.sigma .= n_ineq > 0 ? ws.u[1:n_ineq] : T[]
        ws.lambda .= n_eq > 0 ? ws.u[(n_ineq + 1):(n_ineq + n_eq)] : T[]

        # Update penalty parameters (Schittkowski eq 14)
        v_i = [ws.pensig; ws.penlam]
        parsigma = i == 1 ? ones(T, n_eq + n_ineq) : min.(one(T), T(i) ./ (log.(ws.r) .+ one(T)))
        denom = (1 - delta) * dot(ws.dx, ws.H * ws.dx)
        if abs(denom) > eps(T)
            ws.r .= Vector{T}(max.(parsigma .* ws.r,
                              T(2) * (n_ineq + n_eq) .* (ws.u .- v_i) / denom))
        end

        # === Globalization: line search or trust region ===
        alpha = one(T)
        ls_comment = ""
        step_accepted = true

        if use_trust_region
            # Trust region: accept step if merit function decreases
            merit_current = merit_phi(zero(T))
            merit_trial = merit_phi(one(T))  # full step along dx
            actual_reduction = merit_current - merit_trial
            dx_norm = norm(ws.dx, Inf)

            # Predicted reduction from QP model: -(df'dx + 0.5 dx'H dx)
            predicted_reduction = -(dot(df(ws.x), ws.dx) + T(0.5) * dot(ws.dx, ws.H * ws.dx))
            predicted_reduction = max(predicted_reduction, eps(T))  # safeguard

            rho_tr = actual_reduction / predicted_reduction

            # Update trust radius
            trust_radius = update_trust_radius(trust_radius, rho_tr, dx_norm, options.trust_region_max)

            if actual_reduction > zero(T) || rho_tr > options.trust_region_eta
                # Accept step
                ws.p = ws.dx
                ws.pensig .= ws.sigma
                ws.penlam .= ws.lambda
                ls_comment = string("ρ=", round(rho_tr, digits = 3), " Δ=", round(trust_radius, digits = 3))
            else
                # Reject step — shrink radius, don't update x
                step_accepted = false
                ls_comment = string("rejected Δ=", round(trust_radius, digits = 3))
                if trust_radius < options.xtol
                    if options.verbose
                        @info "Trust region radius too small"
                    end
                    return SQPResult(ws.x, ws.f_last, i, false, constraint_violation_last, :trust_region_failed)
                end
            end
        else
            # Line search
            lookback = options.phi0_lookback
            start_idx = max(1, length(ws.phi_history) - min(i, lookback))
            phi0max = maximum(ws.phi_history[start_idx:end])

            alpha, phi_val, ls_comment = schittkowski_line_search(
                merit_phi, dphi, phi0max, one(T);
                mu = options.line_search_mu, beta = options.line_search_beta)

            ws.pensig .+= alpha .* (ws.sigma .- ws.pensig)
            ws.penlam .+= alpha .* (ws.lambda .- ws.penlam)
            push!(ws.phi_history, min(phi_val, phi0max))

            if isnan(alpha) || isinf(alpha)
                @warn "Line search failed" alpha
                return SQPResult(ws.x, ws.f_last, i, false, constraint_violation_last, :line_search_failed)
            end

            ws.p = alpha * ws.dx
        end

        if step_accepted
            # Update x
            ws.x .+= ws.p
            f_new = f(ws.x)
            constraint_violation = norm([max.(g(ws.x), zero(T)); abs.(h(ws.x))], Inf)

            # Check convergence
            if (norm(ws.p, Inf) <= options.xtol || abs(f_new - ws.f_last) < options.ftol) &&
               constraint_violation < min(options.xtol, options.constraint_tol)
                if options.verbose
                    step_info = use_trust_region ? round(trust_radius, digits = 5) : round(alpha, digits = 5)
                    println(i, "\t\t", round(f_new, digits = 4), "\t\t",
                            round(dot(ws.dx, ws.dx), digits = 4), "\t\t",
                            step_info, "\t\t",
                            round(constraint_violation, digits = 5), "\t", ls_comment)
                    @info "SQP converged"
                end
                return SQPResult(ws.x, f_new, i, true, constraint_violation, :converged)
            end

            # Hessian update — store (s, y) for L-BFGS history
            use_bfgs = false

            if has_external_hessian
                Had_new = try
                    hess_lag(ws.x, ws.sigma, ws.lambda)
                catch
                    use_bfgs = true
                    nothing
                end
            else
                Had_new = try
                    d2l(ws.x)
                catch
                    use_bfgs = true
                    nothing
                end
            end

            if !use_bfgs && isposdef(Had_new)
                ws.H .= Matrix{T}(Had_new)
                ws.k_reset = i
            else
                use_bfgs = true
                if has_external_hessian
                    grad_new = df(ws.x)
                    grad_old = df(ws.x .- ws.p)
                    q = grad_new .- grad_old
                else
                    q = dl(ws.x) .- dl(ws.x .- ws.p)
                end
                ws.H, ws.k_reset = update_hessian!(ws.H, ws.p, q, i, ws.k_reset)
            end

            # Store L-BFGS history
            push!(ws.s_history, copy(ws.p))
            if has_external_hessian
                push!(ws.y_history, df(ws.x) .- df(ws.x .- ws.p))
            else
                push!(ws.y_history, dl(ws.x) .- dl(ws.x .- ws.p))
            end
            while length(ws.s_history) > ws.lbfgs_memory
                popfirst!(ws.s_history)
                popfirst!(ws.y_history)
            end

            constraint_violation_last = constraint_violation
            ws.f_last = f_new

            if options.verbose
                step_info = use_trust_region ? round(trust_radius, digits = 5) : round(alpha, digits = 5)
                println(i, "\t\t", round(f_new, digits = 4), "\t\t",
                        round(dot(ws.dx, ws.dx), digits = 4), "\t\t",
                        step_info, "\t\t",
                        round(constraint_violation, digits = 5),
                        use_bfgs ? "\tBFGS" : "", "\t", ls_comment)
            end
        else
            # Step rejected (trust region) — only print
            if options.verbose
                println(i, "\t\t", round(ws.f_last, digits = 4), "\t\t",
                        round(dot(ws.dx, ws.dx), digits = 4), "\t\t",
                        round(trust_radius, digits = 5), "\t\t",
                        round(constraint_violation_last, digits = 5), "\t", ls_comment)
            end
        end
    end

    if options.verbose
        @info "No convergence" iterations=options.max_iterations
    end

    return SQPResult(ws.x, ws.f_last, options.max_iterations, false,
                     constraint_violation_last, :max_iterations)
end

# Convenience wrappers

"""
    sqp_solve(f, g, h, x0; kwargs...)

Solve min f(x) s.t. g(x) <= 0, h(x) = 0.
"""
function sqp_solve(f, g, h, x0::AbstractVector{T}; kwargs...) where {T <: AbstractFloat}
    problem = NLPProblem(f, g, h, x0)
    sqp_solve(problem; kwargs...)
end

"""
    sqp_solve(f, g, h, x0, lb, ub; kwargs...)

Solve min f(x) s.t. g(x) <= 0, h(x) = 0, lb <= x <= ub.
"""
function sqp_solve(f, g, h, x0::AbstractVector{T}, lb::AbstractVector{T},
                   ub::AbstractVector{T}; kwargs...) where {T <: AbstractFloat}
    problem = NLPProblem(f, g, h, x0, lb, ub)
    sqp_solve(problem; kwargs...)
end
