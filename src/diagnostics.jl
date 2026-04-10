"""
    diagnose_problem(problem::NLPProblem;
                     ad_backend = nothing,
                     n_samples  = 5,
                     perturbation_scale = 1e-3,
                     rng = Random.default_rng())

Probe the problem at `x0` and return a [`ProblemDiagnostics`](@ref) summarising
constraint nonlinearity, Jacobian conditioning, Hessian positive-definiteness,
and initial feasibility. Also recommends a globalization strategy.

# Arguments
- `problem::NLPProblem{T}` — the problem to diagnose.
- `ad_backend` — optional `ADTypes` backend (default: auto-select ForwardDiff).
- `n_samples::Int = 5` — number of random perturbations for nonlinearity estimate.
- `perturbation_scale::T = 1e-3` — relative perturbation size.
- `rng = Random.default_rng()` — random number generator (for reproducibility).
"""
function diagnose_problem(
    problem::NLPProblem{T};
    ad_backend = nothing,
    n_samples::Int = 8,
    perturbation_scale::T = T(1e-1),
    rng = Random.MersenneTwister(42),
) where {T <: AbstractFloat}
    (; f, g, h, x0, n, n_ineq, n_eq) = problem

    dg_fn = make_jacobian(g, x0; backend = ad_backend)
    dh_fn = make_jacobian(h, x0; backend = ad_backend)

    Jg = dg_fn(x0)
    Jh = dh_fn(x0)
    J = vcat(Jg, Jh)
    g0 = g(x0)
    h0 = h(x0)
    c0 = vcat(g0, h0)
    normJ = norm(J)

    # Metric 1 — constraint linearization error ratio.
    # Measures how much the true constraint deviates from the linear model
    # `c_lin(x0+δ) = c(x0) + J(x0)·δ`. The ratio is normalized so that a
    # perfectly linear constraint yields 0 and strong quadratic/exponential
    # nonlinearity yields large values. We take the max across random
    # directions to capture the worst case (a single bad direction is enough
    # to cause Maratos-effect failures in the line search).
    nonlin_max = zero(T)
    jac_var_max = zero(T)
    if (n_ineq + n_eq) > 0 && normJ > eps(T)
        for _ in 1:n_samples
            delta = perturbation_scale .* (2 .* rand(rng, T, n) .- one(T)) .*
                    (one(T) .+ abs.(x0))
            norm_delta = norm(delta)
            norm_delta < eps(T) && continue

            c_true = vcat(g(x0 .+ delta), h(x0 .+ delta))
            J_delta = J * delta
            residual = c_true .- c0 .- J_delta

            # Robust ratio: residual relative to (max of linear step and
            # a floor scaled by ‖J‖·‖δ‖). Floor ensures no division by near-zero
            # when J·δ happens to be orthogonal to the perturbation direction.
            ref = max(norm(J_delta), T(1e-2) * normJ * norm_delta, eps(T))
            ratio = norm(residual) / ref
            isfinite(ratio) && (nonlin_max = max(nonlin_max, ratio))

            # Metric 2 — Jacobian variability (second-order curvature proxy):
            # ‖J(x0+δ) - J(x0)‖_F / ‖J(x0)‖_F / ‖δ‖
            Jg_d = dg_fn(x0 .+ delta)
            Jh_d = dh_fn(x0 .+ delta)
            J_d = vcat(Jg_d, Jh_d)
            jac_diff = norm(J_d .- J) / (normJ * norm_delta + eps(T))
            isfinite(jac_diff) && (jac_var_max = max(jac_var_max, jac_diff))
        end
    end
    # Combine the two nonlinearity signals. Jacobian variability captures
    # second-order curvature that the linearization-error ratio may miss when
    # the perturbation direction is aligned with a nearly-linear mode.
    nonlin = max(nonlin_max, jac_var_max)

    # Metric 2 — Jacobian condition via SVD
    kappa = if size(J, 1) == 0 || size(J, 2) == 0
        one(T)
    else
        Jd = Matrix{T}(J)
        svdvals_J = try
            svdvals(Jd)
        catch
            T[]
        end
        if isempty(svdvals_J)
            T(Inf)
        else
            s_min = minimum(svdvals_J)
            s_max = maximum(svdvals_J)
            s_min > eps(T) ? s_max / s_min : T(Inf)
        end
    end

    # Metric 3 — objective Hessian positive-definiteness at x0
    posdef = try
        d2f_fn = make_hessian(f, x0; backend = ad_backend)
        Hf = d2f_fn(x0)
        isposdef(Matrix{T}(Hf))
    catch
        false
    end

    # Metric 4 — initial feasibility (∞-norm of violations)
    feas_vals = T[]
    for gi in g0
        push!(feas_vals, max(gi, zero(T)))
    end
    for hi in h0
        push!(feas_vals, abs(hi))
    end
    feas = isempty(feas_vals) ? zero(T) : norm(feas_vals, Inf)

    # Warnings and recommended strategy.
    # Nonlinearity threshold ~0.3 corresponds to "residual or Jacobian drift
    # is ~30% of the linear model over a 10% perturbation" — enough to
    # cause Maratos-effect failures. SOC is safe on linearly-behaved
    # problems (it's only triggered when dphi0 >= 0), so a permissive
    # threshold errs on the side of recommending it.
    warnings = String[]
    if nonlin > T(0.3)
        push!(warnings,
              "High constraint nonlinearity ($(round(nonlin, digits=2))): " *
              "QP linearization is a poor model. Consider use_soc = true.")
    end
    if kappa > T(1e8)
        push!(warnings,
              "Ill-conditioned constraint Jacobian (cond ≈ " *
              "$(round(kappa, sigdigits=2))).")
    end
    if !posdef
        push!(warnings,
              "Objective Hessian at x0 is not positive definite: " *
              "BFGS approximation will be used.")
    end
    if feas > T(1e3)
        push!(warnings, "Initial point is severely infeasible (‖c‖∞ = $(round(feas, digits=2))).")
    end

    strategy = if nonlin > T(0.3) && (n_ineq + n_eq) > 0
        :line_search_soc
    elseif kappa > T(1e10)
        :trust_region
    else
        :line_search
    end

    return ProblemDiagnostics{T}(
        nonlin, kappa, posdef, feas,
        n, n_ineq, n_eq,
        strategy, warnings,
    )
end
