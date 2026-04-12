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

    # Problem diagnostics (optional)
    diagnostics = nothing
    if options.diagnose
        diagnostics = diagnose_problem(problem; ad_backend = ad_backend)
        if options.verbose
            @info "Problem diagnostics" nonlinearity=diagnostics.constraint_nonlinearity cond=diagnostics.jacobian_condition hessian_posdef=diagnostics.hessian_posdef feasibility=diagnostics.initial_feasibility recommended=diagnostics.recommended_strategy
            for w in diagnostics.warnings
                @warn w
            end
        end
    end

    # SOC is implemented for COSMOQPSolver only in v0.8.1
    if options.use_soc && !(qp_solver isa COSMOQPSolver)
        @warn "Second-Order Correction (use_soc=true) is only supported with COSMOQPSolver in v0.8.x — falling back to standard line search"
    end

    # Build derivative functions (use provided, or DI with specified/auto backend)
    df = grad_f !== nothing ? grad_f : make_gradient(f, x0; backend = ad_backend)
    dg = jac_g !== nothing ? jac_g : make_jacobian(g, x0; backend = ad_backend)
    dh = jac_h !== nothing ? jac_h : make_jacobian(h, x0; backend = ad_backend)

    # Build Lagrangian and its derivatives
    ws = SQPWorkspace(x0, n_ineq, n_eq)
    n_soc_steps = 0
    # Phase 8.2 — numerical safeguard counters
    n_steps_clamped = 0
    n_bfgs_skipped = 0
    lm_lambda = zero(T)
    lm_good_streak = 0
    lm_ever_activated = false
    # Phase 9.0 — analytical Hessian counter
    n_hessian_corrections = 0
    # Mid-solve fallback tracking: if we're in analytical mode but LM damping
    # saturates the maximum for several consecutive iterations, the analytical
    # path isn't helping and we switch to BFGS for the rest of the solve.
    lm_saturated_streak = 0
    # Phase 9.1 — filter line search state
    use_filter = options.globalization == :filter_line_search
    filter = Filter{T}(; max_size = options.filter_max_size)
    n_filter_f_steps = 0
    n_filter_h_steps = 0
    n_filter_fallbacks = 0

    has_external_hessian = hess_lag !== nothing

    # Phase 9.0 — decide whether to use analytical Hessian this run.
    # :analytical forces it on, :bfgs forces it off, :auto picks based on size.
    use_analytical_hessian =
        !has_external_hessian && (
            options.hessian_strategy == :analytical ||
            (options.hessian_strategy == :auto && n <= options.auto_hessian_max_n)
        )

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
    elseif use_analytical_hessian || has_external_hessian
        # Phase 9.0 — keep the true curvature directions via eigenvalue
        # correction; BFGS would discard them entirely.
        ws.H .= Matrix{T}(Had)
        _, corrected, λmin = modify_eigenvalues!(ws.H;
                                floor = options.hessian_correction_floor)
        if corrected
            n_hessian_corrections += 1
            if options.verbose
                @info "Initial Hessian eigenvalue-corrected" λmin
            end
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

    # Phase 9.1 — seed the filter with an upper bound on constraint violation.
    # The Wächter-Biegler convention starts the filter with an entry
    # (−∞, θ_max) that enforces θ ≤ θ_max for all accepted iterates. This
    # gives the filter *something to bound* even on nearly-feasible starts,
    # which otherwise leave the filter empty and let the f-step branch
    # accept any descent direction blindly.
    if use_filter
        theta_0 = theta_constraint_violation(g(ws.x), h(ws.x))
        theta_max = max(T(1e4), T(10) * theta_0 + one(T))
        # Store (-Inf, theta_max) so any trial with theta > theta_max is dominated
        push!(filter.entries, (T(-Inf), theta_max))
    end

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

        # Phase 8.2 Part C — Levenberg-Marquardt regularization of H.
        # When `lm_lambda > 0`, add `lm_lambda·I` to H before the QP solve.
        # This bounds the step in directions where H has a small eigenvalue,
        # preventing the QP from returning garbage steps on near-singular H.
        if options.numerical_safeguards && lm_lambda > eps(T)
            for jj in 1:n
                ws.H[jj, jj] += lm_lambda
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
            return SQPResult{T, typeof(ws.x)}(ws.x, ws.f_last, i, false,
                                              constraint_violation_last, :qp_failed,
                                              diagnostics, n_soc_steps,
                                              n_steps_clamped, n_bfgs_skipped, lm_lambda,
                                              n_hessian_corrections,
                                              n_filter_f_steps, n_filter_h_steps,
                                              n_filter_fallbacks, length(filter))
        end

        # Phase 8.2 Part A — step norm clamping.
        # Prevent pathologically large steps from a near-singular BFGS
        # Hessian or a poorly-conditioned QP from contaminating the line
        # search and the downstream (s, y) curvature sample. Clamp to
        # `step_clamp_factor · (1 + ‖x‖∞)` so normally-sized steps are
        # untouched but 100× overshoots get rescaled to a sensible range.
        step_clamped = false
        if options.numerical_safeguards
            step_norm = norm(ws.dx, Inf)
            clamp_bound = options.step_clamp_factor * (one(T) + norm(ws.x, Inf))
            if step_norm > clamp_bound && isfinite(clamp_bound) && step_norm > zero(T)
                scale = clamp_bound / step_norm
                ws.dx .*= scale
                step_clamped = true
                n_steps_clamped += 1
                if options.verbose
                    @debug "Step clamped" iteration=i step_norm clamp_bound scale
                end
            end
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
                    return SQPResult{T, typeof(ws.x)}(ws.x, ws.f_last, i, false,
                                                      constraint_violation_last, :trust_region_failed,
                                                      diagnostics, n_soc_steps,
                                                      n_steps_clamped, n_bfgs_skipped, lm_lambda,
                                                      n_hessian_corrections,
                                                      n_filter_f_steps, n_filter_h_steps,
                                                      n_filter_fallbacks, length(filter))
                end
            end
        else
            # Line search
            lookback = options.phi0_lookback
            start_idx = max(1, length(ws.phi_history) - min(i, lookback))
            phi0max = maximum(ws.phi_history[start_idx:end])

            # ─────────────────────────────────────────────────────────────
            # Phase 9.1 — Filter line search (Wächter-Biegler).
            # Tries the filter acceptance criterion first; if it rejects all
            # backtracked α values, falls through to the Schittkowski merit
            # line search. The fallback gives us a well-tested safety net
            # while avoiding the need for a full restoration phase.
            # ─────────────────────────────────────────────────────────────
            filter_accepted = false
            if use_filter
                f_k = f(ws.x)
                theta_k = theta_constraint_violation(g(ws.x), h(ws.x))
                grad_f_dot_d = dot(df(ws.x), ws.dx)

                alpha_f, accepted, step_type, f_alpha_tr, theta_alpha_tr =
                    filter_line_search(filter, f, g, h, ws.x, ws.dx,
                                       f_k, theta_k, grad_f_dot_d, options)

                if accepted
                    alpha = alpha_f
                    ls_comment = string("filter ", step_type == :f ? "f-step" : "h-step",
                                        " α=", round(alpha, digits = 4))
                    ws.pensig .+= alpha .* (ws.sigma .- ws.pensig)
                    ws.penlam .+= alpha .* (ws.lambda .- ws.penlam)
                    # Sync merit-function history so the Schittkowski
                    # fallback on later iterations has a valid phi0max.
                    push!(ws.phi_history, merit_phi(alpha))

                    if step_type == :f
                        n_filter_f_steps += 1
                    else
                        # Augment filter only on h-steps (WB 2006 convention)
                        n_filter_h_steps += 1
                        augment!(filter, f_k, theta_k,
                                 options.filter_gamma_f, options.filter_gamma_theta)
                    end

                    ws.p = alpha * ws.dx
                    filter_accepted = true
                else
                    n_filter_fallbacks += 1
                    if options.verbose
                        @debug "Filter line search fell through — using Schittkowski merit" iteration=i
                    end
                end
            end

            if !filter_accepted
                # Second-Order Correction path: only available with COSMOQPSolver
                # (the correction QP implementation is COSMO-only in v0.8.1).
                if options.use_soc && qp_solver isa COSMOQPSolver
                    soc_cb = function()
                        # Evaluate true constraints at the full-step trial point
                        x_trial = ws.x .+ ws.dx
                        c_trial = T[g(x_trial); h(x_trial)]
                        # Correction QP: drive the linearized residual toward zero
                        # using the primary-point Jacobian A. Inequality rows
                        # (first n_ineq) use one-sided bounds; equality rows use
                        # two-sided bounds.
                        d_c = solve_qp_correction(qp_solver, H_dense, c_trial, A, df(ws.x), n_ineq)
                        if d_c === nothing
                            return (merit_phi, dphi, false)
                        end
                        dx_soc = ws.dx .+ d_c

                        # Merit function along the corrected direction
                        phi_soc_fn(a_) = begin
                            x_try = ws.x .+ a_ .* dx_soc
                            pensig_try = ws.pensig .+ a_ .* (ws.sigma .- ws.pensig)
                            penlam_try = ws.penlam .+ a_ .* (ws.lambda .- ws.penlam)
                            augmented_lagrangian(f(x_try), g(x_try), h(x_try),
                                                 pensig_try, penlam_try, ws.r)
                        end
                        dphi_soc_fn(a_) = ForwardDiff.derivative(phi_soc_fn, a_)

                        # Commit the corrected direction so ws.p = α·ws.dx downstream
                        ws.dx = dx_soc
                        return (phi_soc_fn, dphi_soc_fn, true)
                    end

                    alpha, phi_val, ls_comment, used_soc = schittkowski_line_search_soc(
                        merit_phi, dphi, phi0max, one(T);
                        soc_callback = soc_cb,
                        mu = options.line_search_mu, beta = options.line_search_beta)
                    used_soc && (n_soc_steps += 1)
                else
                    alpha, phi_val, ls_comment = schittkowski_line_search(
                        merit_phi, dphi, phi0max, one(T);
                        mu = options.line_search_mu, beta = options.line_search_beta)
                end

                ws.pensig .+= alpha .* (ws.sigma .- ws.pensig)
                ws.penlam .+= alpha .* (ws.lambda .- ws.penlam)
                push!(ws.phi_history, min(phi_val, phi0max))

                if isnan(alpha) || isinf(alpha)
                    @warn "Line search failed" alpha
                    return SQPResult{T, typeof(ws.x)}(ws.x, ws.f_last, i, false,
                                                      constraint_violation_last, :line_search_failed,
                                                      diagnostics, n_soc_steps,
                                                      n_steps_clamped, n_bfgs_skipped, lm_lambda,
                                                      n_hessian_corrections,
                                                      n_filter_f_steps, n_filter_h_steps,
                                                      n_filter_fallbacks, length(filter))
                end

                ws.p = alpha * ws.dx
            end
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
                return SQPResult{T, typeof(ws.x)}(ws.x, f_new, i, true,
                                                  constraint_violation, :converged,
                                                  diagnostics, n_soc_steps,
                                                  n_steps_clamped, n_bfgs_skipped, lm_lambda,
                                                  n_hessian_corrections,
                                                  n_filter_f_steps, n_filter_h_steps,
                                                  n_filter_fallbacks, length(filter))
            end

            # Phase 8.2 Part B — decide whether to trust this (s, y) pair.
            # Only skip when:
            #  (i)  the step was explicitly clamped — direction itself was wrong, or
            #  (ii) the line search collapsed to a very tiny α — pathological direction.
            # Non-convex curvature (sy < 0) is NOT a skip signal: robust BFGS
            # handles mildly-negative curvature correctly via its z-interpolation,
            # and legitimate non-convex NLPs like HS071 routinely produce such
            # samples even on clean full steps.
            bfgs_skipped = false
            if options.numerical_safeguards
                if step_clamped || alpha < options.bfgs_skip_alpha
                    bfgs_skipped = true
                    n_bfgs_skipped += 1
                    if options.verbose
                        @debug "BFGS update skipped" iteration=i step_clamped alpha
                    end
                end
            end

            # Hessian update — store (s, y) for L-BFGS history.
            #
            # Priority order:
            #   1. external hess_lag (always tried first when provided)
            #   2. analytical ∇²L(x) via AD — always attempted when available.
            #      - PD: use directly
            #      - non-PD and use_analytical_hessian=true: Phase 9.0
            #        eigenvalue correction
            #      - non-PD and use_analytical_hessian=false: fall through
            #        to BFGS (preserves v0.8.2 behavior for :bfgs strategy)
            #   3. BFGS update (fallback)
            #
            # The analytical path runs regardless of bfgs_skipped because
            # it doesn't consume the suspect (s, y) sample. The
            # bfgs_skipped flag only gates the BFGS fallback.
            use_bfgs = false
            analytical_refreshed = false

            if has_external_hessian
                Had_new = try
                    hess_lag(ws.x, ws.sigma, ws.lambda)
                catch
                    nothing
                end
            else
                Had_new = try
                    d2l(ws.x)
                catch
                    nothing
                end
            end

            if Had_new === nothing || all(iszero, Had_new)
                use_bfgs = true
            elseif isposdef(Had_new)
                ws.H .= Matrix{T}(Had_new)
                ws.k_reset = i
                analytical_refreshed = true
            elseif use_analytical_hessian || has_external_hessian
                # Phase 9.0 eigenvalue correction: the true Hessian's
                # directions encode real curvature BFGS cannot see —
                # clip eigenvalues to a PD floor rather than discarding it.
                ws.H .= Matrix{T}(Had_new)
                _, corrected, _ = modify_eigenvalues!(ws.H;
                                    floor = options.hessian_correction_floor)
                if corrected
                    n_hessian_corrections += 1
                end
                ws.k_reset = i
                analytical_refreshed = true
            else
                use_bfgs = true
            end

            if use_bfgs && !bfgs_skipped
                if has_external_hessian
                    grad_new = df(ws.x)
                    grad_old = df(ws.x .- ws.p)
                    q = grad_new .- grad_old
                else
                    q = dl(ws.x) .- dl(ws.x .- ws.p)
                end
                ws.H, ws.k_reset = update_hessian!(ws.H, ws.p, q, i, ws.k_reset)
            end
            # else: analytical refresh not available AND sample is bad → keep H unchanged

            # Store L-BFGS history only when we trust this (s, y) pair
            if !bfgs_skipped
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
            end

            # Phase 8.2 Part C — adaptive LM damping update.
            # Grow on bad iterations (clamped or skipped).
            # Once the solver has entered "damped mode" (ever_activated = true),
            # shrink only slowly on clean steps and keep a floor. This prevents
            # LM from collapsing between pathological iterations.
            if options.numerical_safeguards
                if step_clamped || bfgs_skipped
                    lm_lambda = lm_lambda > eps(T) ?
                                min(lm_lambda * options.lm_grow, options.lm_max) :
                                options.lm_min_active
                    lm_good_streak = 0
                    lm_ever_activated = true
                elseif alpha >= T(0.9)
                    lm_good_streak += 1
                    if lm_good_streak >= 3
                        lm_lambda *= options.lm_shrink
                        # Keep a floor while the solver has shown ill-conditioning.
                        # Only fully release LM after a long clean streak.
                        if lm_ever_activated && lm_good_streak < 20
                            lm_lambda = max(lm_lambda, options.lm_min_active)
                        elseif lm_lambda < options.lm_min
                            lm_lambda = zero(T)
                        end
                    end
                else
                    # partial step — hold lm_lambda, reset the streak
                    lm_good_streak = 0
                end
            end

            # Phase 9.0 — mid-solve fallback. If the analytical-Hessian path
            # has driven LM damping to its maximum and kept it there for
            # many consecutive iterations, the analytical direction isn't
            # helping. Switch to BFGS for the rest of the solve, reset the
            # Hessian to identity (so BFGS rebuilds fresh), and clear
            # LM/L-BFGS state.
            if use_analytical_hessian && lm_lambda >= options.lm_max * T(0.99)
                lm_saturated_streak += 1
                if lm_saturated_streak >= 10
                    use_analytical_hessian = false
                    ws.H .= Matrix{T}(I, n, n)
                    ws.k_reset = i
                    empty!(ws.s_history)
                    empty!(ws.y_history)
                    lm_lambda = zero(T)
                    lm_ever_activated = false
                    lm_good_streak = 0
                    if options.verbose
                        @info "Analytical Hessian ineffective — falling back to BFGS" iteration=i
                    end
                end
            else
                lm_saturated_streak = 0
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

    return SQPResult{T, typeof(ws.x)}(ws.x, ws.f_last, options.max_iterations, false,
                                      constraint_violation_last, :max_iterations,
                                      diagnostics, n_soc_steps,
                                      n_steps_clamped, n_bfgs_skipped, lm_lambda,
                                      n_hessian_corrections,
                                      n_filter_f_steps, n_filter_h_steps,
                                      n_filter_fallbacks, length(filter))
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
