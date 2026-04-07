const SQP = SequentialQuadraticProgramming

# ============================================================================
# Supporting types
# ============================================================================

mutable struct VariableInfo
    lower_bound::Float64
    upper_bound::Float64
    start::Union{Nothing, Float64}
end

VariableInfo() = VariableInfo(-Inf, Inf, nothing)

struct ConstraintInfo{F, S}
    func::F
    set::S
end

struct EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end
MOI.features_available(::EmptyNLPEvaluator) = [:Grad, :Jac]
MOI.initialize(::EmptyNLPEvaluator, ::Vector{Symbol}) = nothing
MOI.eval_objective(::EmptyNLPEvaluator, ::AbstractVector) = 0.0
MOI.eval_objective_gradient(::EmptyNLPEvaluator, g, ::AbstractVector) = fill!(g, 0.0)
MOI.eval_constraint(::EmptyNLPEvaluator, ::AbstractVector, ::AbstractVector) = nothing
MOI.jacobian_structure(::EmptyNLPEvaluator) = Tuple{Int, Int}[]

const EMPTY_NLP_DATA = MOI.NLPBlockData(MOI.NLPBoundsPair[], EmptyNLPEvaluator(), false)

# ============================================================================
# Optimizer
# ============================================================================

mutable struct Optimizer <: MOI.AbstractOptimizer
    variables::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{Nothing,
                     MOI.ScalarAffineFunction{Float64},
                     MOI.ScalarQuadraticFunction{Float64}}
    linear_le_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}}
    linear_ge_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}}
    linear_eq_constraints::Vector{ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}}
    silent::Bool
    options::Dict{String, Any}
    solution::Union{Nothing, Vector{Float64}}
    objective_value::Float64
    status::Symbol
    iterations::Int
    constraint_violation::Float64
    solve_time::Float64
end

function Optimizer(; kwargs...)
    model = Optimizer(
        VariableInfo[],
        EMPTY_NLP_DATA,
        MOI.MIN_SENSE,
        nothing,
        ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}[],
        ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}[],
        ConstraintInfo{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}[],
        false,
        Dict{String, Any}(),
        nothing,
        NaN,
        :not_called,
        0,
        NaN,
        NaN,
    )
    for (key, val) in kwargs
        model.options[string(key)] = val
    end
    return model
end

# ============================================================================
# MOI.AbstractOptimizer interface
# ============================================================================

MOI.get(::Optimizer, ::MOI.SolverName) = "SequentialQuadraticProgramming"

function MOI.empty!(model::Optimizer)
    empty!(model.variables)
    model.nlp_data = EMPTY_NLP_DATA
    model.sense = MOI.MIN_SENSE
    model.objective = nothing
    empty!(model.linear_le_constraints)
    empty!(model.linear_ge_constraints)
    empty!(model.linear_eq_constraints)
    model.solution = nothing
    model.objective_value = NaN
    model.status = :not_called
    model.iterations = 0
    model.constraint_violation = NaN
    model.solve_time = NaN
    return
end

function MOI.is_empty(model::Optimizer)
    return isempty(model.variables) &&
           model.nlp_data.evaluator isa EmptyNLPEvaluator &&
           model.objective === nothing &&
           isempty(model.linear_le_constraints) &&
           isempty(model.linear_ge_constraints) &&
           isempty(model.linear_eq_constraints)
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

# ============================================================================
# Variables
# ============================================================================

function MOI.add_variable(model::Optimizer)
    push!(model.variables, VariableInfo())
    return MOI.VariableIndex(length(model.variables))
end

MOI.add_variables(model::Optimizer, n::Int) = [MOI.add_variable(model) for _ in 1:n]
MOI.get(model::Optimizer, ::MOI.NumberOfVariables) = length(model.variables)

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return [MOI.VariableIndex(i) for i in 1:length(model.variables)]
end

MOI.is_valid(model::Optimizer, vi::MOI.VariableIndex) = 1 <= vi.value <= length(model.variables)

# ============================================================================
# Variable bounds
# ============================================================================

function MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.VariableIndex}, ::Type{<:Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}, MOI.Interval{Float64}}},
)
    return true
end

function MOI.add_constraint(model::Optimizer, vi::MOI.VariableIndex, set::MOI.LessThan{Float64})
    model.variables[vi.value].upper_bound = set.upper
    return MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, vi::MOI.VariableIndex, set::MOI.GreaterThan{Float64})
    model.variables[vi.value].lower_bound = set.lower
    return MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, vi::MOI.VariableIndex, set::MOI.EqualTo{Float64})
    model.variables[vi.value].lower_bound = set.value
    model.variables[vi.value].upper_bound = set.value
    return MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, vi::MOI.VariableIndex, set::MOI.Interval{Float64})
    model.variables[vi.value].lower_bound = set.lower
    model.variables[vi.value].upper_bound = set.upper
    return MOI.ConstraintIndex{MOI.VariableIndex, MOI.Interval{Float64}}(vi.value)
end

# ============================================================================
# Linear constraints
# ============================================================================

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{<:Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}}},
)
    return true
end

function MOI.add_constraint(model::Optimizer, func::MOI.ScalarAffineFunction{Float64}, set::MOI.LessThan{Float64})
    push!(model.linear_le_constraints, ConstraintInfo(func, set))
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(length(model.linear_le_constraints))
end

function MOI.add_constraint(model::Optimizer, func::MOI.ScalarAffineFunction{Float64}, set::MOI.GreaterThan{Float64})
    push!(model.linear_ge_constraints, ConstraintInfo(func, set))
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}(length(model.linear_ge_constraints))
end

function MOI.add_constraint(model::Optimizer, func::MOI.ScalarAffineFunction{Float64}, set::MOI.EqualTo{Float64})
    push!(model.linear_eq_constraints, ConstraintInfo(func, set))
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(length(model.linear_eq_constraints))
end

# ============================================================================
# Objective
# ============================================================================

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{<:Union{MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64}}},
)
    return true
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense
MOI.set(model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense) = (model.sense = sense)
MOI.get(model::Optimizer, ::MOI.Silent) = model.silent
MOI.set(model::Optimizer, ::MOI.Silent, value::Bool) = (model.silent = value)

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    model.options[attr.name] = value
    return
end

function MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    return model.options[attr.name]
end

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    model.nlp_data = nlp_data
    return
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveFunction{F}, func::F) where {F}
    model.objective = func
    return
end

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart, vi::MOI.VariableIndex, value::Union{Nothing, Float64})
    model.variables[vi.value].start = value
    return
end

# ============================================================================
# optimize!
# ============================================================================

function _eval_affine(func::MOI.ScalarAffineFunction{Float64}, x::Vector{Float64})
    val = func.constant
    for term in func.terms
        val += term.coefficient * x[term.variable.value]
    end
    return val
end

function _eval_quadratic(func::MOI.ScalarQuadraticFunction{Float64}, x::Vector{Float64})
    val = func.constant
    for term in func.affine_terms
        val += term.coefficient * x[term.variable.value]
    end
    for term in func.quadratic_terms
        coef = term.variable_1 == term.variable_2 ? term.coefficient / 2 : term.coefficient
        val += coef * x[term.variable_1.value] * x[term.variable_2.value]
    end
    return val
end

function MOI.optimize!(model::Optimizer)
    t0 = time()
    n = length(model.variables)

    # Variable bounds
    lb = [v.lower_bound for v in model.variables]
    ub = [v.upper_bound for v in model.variables]

    # Starting point
    x0 = zeros(Float64, n)
    for i in 1:n
        v = model.variables[i]
        if v.start !== nothing
            x0[i] = v.start
        elseif isfinite(v.lower_bound) && isfinite(v.upper_bound)
            x0[i] = (v.lower_bound + v.upper_bound) / 2
        elseif isfinite(v.lower_bound)
            x0[i] = v.lower_bound
        elseif isfinite(v.upper_bound)
            x0[i] = v.upper_bound
        else
            x0[i] = 0.0
        end
    end

    # Build objective function
    evaluator = model.nlp_data.evaluator
    has_nlp_obj = model.nlp_data.has_objective

    # Initialize NLP evaluator with all available features
    nlp_bounds = model.nlp_data.constraint_bounds
    n_nlp_constraints = length(nlp_bounds)

    available = !(evaluator isa EmptyNLPEvaluator) ? MOI.features_available(evaluator) : Symbol[]
    features = Symbol[]
    if has_nlp_obj && :Grad in available
        push!(features, :Grad)
    end
    if n_nlp_constraints > 0 && :Jac in available
        push!(features, :Jac)
    end
    has_hess = :Hess in available
    if has_hess
        push!(features, :Hess)
    end
    if !isempty(features)
        MOI.initialize(evaluator, features)
    end

    sign = model.sense == MOI.MAX_SENSE ? -1.0 : 1.0

    # NOTE: MOI evaluator callbacks only accept Float64 vectors, not ForwardDiff.Dual.
    # We use `Float64.(x)` to strip any Dual wrappers so FiniteDiff fallback works
    # correctly when the SQP solver's NLPProblem wraps these with bounds (vcat).
    f = if has_nlp_obj
        x -> sign * MOI.eval_objective(evaluator, Float64.(x))
    elseif model.objective isa MOI.ScalarAffineFunction{Float64}
        obj = model.objective
        x -> sign * _eval_affine(obj, x)
    elseif model.objective isa MOI.ScalarQuadraticFunction{Float64}
        obj = model.objective
        x -> sign * _eval_quadratic(obj, x)
    else
        x -> 0.0
    end

    # Classify NLP constraints into inequality (g <= 0) and equality (h = 0)
    nlp_ineq_indices = Int[]
    nlp_ineq_upper = Float64[]
    nlp_ineq_ge_indices = Int[]
    nlp_ineq_ge_lower = Float64[]
    nlp_eq_indices = Int[]
    nlp_eq_values = Float64[]

    for (i, bp) in enumerate(nlp_bounds)
        if bp.lower == bp.upper
            push!(nlp_eq_indices, i)
            push!(nlp_eq_values, bp.lower)
        else
            if bp.upper < Inf
                push!(nlp_ineq_indices, i)
                push!(nlp_ineq_upper, bp.upper)
            end
            if bp.lower > -Inf
                push!(nlp_ineq_ge_indices, i)
                push!(nlp_ineq_ge_lower, bp.lower)
            end
        end
    end

    # Build inequality constraint function g(x) <= 0
    # NOTE: Float64.(x) ensures MOI evaluator receives Float64, not Dual numbers
    nlp_buf = zeros(Float64, n_nlp_constraints)

    function g(x)
        result = Float64[]
        # Linear LE: a'x <= b  =>  a'x - b <= 0
        for info in model.linear_le_constraints
            push!(result, _eval_affine(info.func, Float64.(x)) - info.set.upper)
        end
        # Linear GE: a'x >= b  =>  -(a'x - b) <= 0  =>  b - a'x <= 0
        for info in model.linear_ge_constraints
            push!(result, info.set.lower - _eval_affine(info.func, Float64.(x)))
        end
        # NLP constraints
        if n_nlp_constraints > 0
            MOI.eval_constraint(evaluator, nlp_buf, Float64.(x))
            # NLP LE constraints: c(x) <= upper  =>  c(x) - upper <= 0
            for (j, idx) in enumerate(nlp_ineq_indices)
                push!(result, nlp_buf[idx] - nlp_ineq_upper[j])
            end
            # NLP GE constraints: c(x) >= lower  =>  lower - c(x) <= 0
            for (j, idx) in enumerate(nlp_ineq_ge_indices)
                push!(result, nlp_ineq_ge_lower[j] - nlp_buf[idx])
            end
        end
        return isempty(result) ? zeros(0) : result
    end

    # Build equality constraint function h(x) = 0
    nlp_buf_h = zeros(Float64, n_nlp_constraints)
    function h(x)
        result = Float64[]
        # Linear EQ: a'x = b  =>  a'x - b = 0
        for info in model.linear_eq_constraints
            push!(result, _eval_affine(info.func, Float64.(x)) - info.set.value)
        end
        # NLP EQ constraints: c(x) = value  =>  c(x) - value = 0
        if !isempty(nlp_eq_indices)
            MOI.eval_constraint(evaluator, nlp_buf_h, Float64.(x))
            for (j, idx) in enumerate(nlp_eq_indices)
                push!(result, nlp_buf_h[idx] - nlp_eq_values[j])
            end
        end
        return isempty(result) ? zeros(0) : result
    end

    # Build SQP options from model options
    max_iter = get(model.options, "max_iterations", 200)
    xtol = get(model.options, "xtol", 1e-6)
    ftol = get(model.options, "ftol", 1e-6)

    opts = SQP.SQPOptions(;
        max_iterations = max_iter,
        xtol = xtol,
        ftol = ftol,
        verbose = !model.silent,
    )

    # Build exact derivative functions from MOI evaluator (bypasses FiniteDiff)
    _grad_f = nothing
    if has_nlp_obj && :Grad in features
        _grad_buf = zeros(Float64, n)
        _grad_f = function (x)
            MOI.eval_objective_gradient(evaluator, _grad_buf, Float64.(x))
            return sign * copy(_grad_buf)
        end
    end

    _jac_g = nothing
    _jac_h = nothing
    if n_nlp_constraints > 0 && :Jac in features
        jac_struct = MOI.jacobian_structure(evaluator)
        jac_vals = zeros(Float64, length(jac_struct))

        # Build a function that returns the full Jacobian of g (inequalities)
        # and h (equalities) separately, accounting for linear constraints too
        n_lin_le = length(model.linear_le_constraints)
        n_lin_ge = length(model.linear_ge_constraints)
        n_lin_eq = length(model.linear_eq_constraints)
        n_nlp_ineq = length(nlp_ineq_indices) + length(nlp_ineq_ge_indices)
        n_nlp_eq = length(nlp_eq_indices)
        n_total_ineq = n_lin_le + n_lin_ge + n_nlp_ineq
        n_total_eq = n_lin_eq + n_nlp_eq

        # Note: when bounds are present, NLPProblem wraps g as
        # g_wrapped(x) = vcat(g_orig(x), lb .- x, x .- ub)
        # So the Jacobian must include [-I; I] rows for bounds.
        has_var_bounds = any(isfinite, lb) || any(isfinite, ub)
        n_bound_rows = has_var_bounds ? 2 * n : 0
        n_total_ineq_with_bounds = n_total_ineq + n_bound_rows

        _jac_g = function (x)
            J = zeros(Float64, n_total_ineq_with_bounds, n)
            row = 0
            # Linear LE Jacobian
            for info in model.linear_le_constraints
                row += 1
                for term in info.func.terms
                    J[row, term.variable.value] += term.coefficient
                end
            end
            # Linear GE Jacobian (negated)
            for info in model.linear_ge_constraints
                row += 1
                for term in info.func.terms
                    J[row, term.variable.value] -= term.coefficient
                end
            end
            # NLP constraint Jacobian
            if n_nlp_constraints > 0
                MOI.eval_constraint_jacobian(evaluator, jac_vals, Float64.(x))
                Jnlp = zeros(Float64, n_nlp_constraints, n)
                for (k, (i, j)) in enumerate(jac_struct)
                    Jnlp[i, j] += jac_vals[k]
                end
                for (_, idx) in enumerate(nlp_ineq_indices)
                    row += 1
                    J[row, :] .= Jnlp[idx, :]
                end
                for (_, idx) in enumerate(nlp_ineq_ge_indices)
                    row += 1
                    J[row, :] .= -Jnlp[idx, :]
                end
            end
            # Bounds: lb - x <= 0  →  Jacobian = -I
            #         x - ub <= 0  →  Jacobian = +I
            if has_var_bounds
                for j in 1:n
                    J[row + j, j] = -1.0       # d(lb - x)/dx = -I
                    J[row + n + j, j] = 1.0     # d(x - ub)/dx = +I
                end
            end
            return J
        end

        _jac_h = function (x)
            J = zeros(Float64, n_total_eq, n)
            row = 0
            # Linear EQ Jacobian
            for info in model.linear_eq_constraints
                row += 1
                for term in info.func.terms
                    J[row, term.variable.value] += term.coefficient
                end
            end
            # NLP EQ Jacobian
            if !isempty(nlp_eq_indices)
                MOI.eval_constraint_jacobian(evaluator, jac_vals, Float64.(x))
                Jnlp = zeros(Float64, n_nlp_constraints, n)
                for (k, (i, j)) in enumerate(jac_struct)
                    Jnlp[i, j] += jac_vals[k]
                end
                for (_, idx) in enumerate(nlp_eq_indices)
                    row += 1
                    J[row, :] .= Jnlp[idx, :]
                end
            end
            return J
        end
    end

    # Build Hessian of Lagrangian from evaluator
    _hess_lag = nothing
    if has_hess && n_nlp_constraints > 0
        hess_struct = MOI.hessian_lagrangian_structure(evaluator)
        hess_vals = zeros(Float64, length(hess_struct))

        _hess_lag = function (x, sigma_mult, lambda_mult)
            # Build full mu vector for MOI (all NLP constraints, in original order)
            mu = zeros(Float64, n_nlp_constraints)
            # Map our inequality/equality multipliers back to NLP constraint indices
            # sigma_mult corresponds to g constraints: [lin_le; lin_ge; nlp_ineq; nlp_ge]
            # We need to map nlp_ineq and nlp_ge multipliers back to NLP constraint order
            n_lin = length(model.linear_le_constraints) + length(model.linear_ge_constraints)
            for (j, idx) in enumerate(nlp_ineq_indices)
                mu[idx] += sigma_mult[n_lin + j]  # LE: c - upper <= 0
            end
            for (j, idx) in enumerate(nlp_ineq_ge_indices)
                mu[idx] -= sigma_mult[n_lin + length(nlp_ineq_indices) + j]  # GE: lower - c <= 0, so -mu
            end
            n_lin_eq_count = length(model.linear_eq_constraints)
            for (j, idx) in enumerate(nlp_eq_indices)
                mu[idx] += lambda_mult[n_lin_eq_count + j]
            end

            MOI.eval_hessian_lagrangian(evaluator, hess_vals, Float64.(x), sign, mu)
            H = zeros(Float64, n, n)
            for (k, (i, j)) in enumerate(hess_struct)
                H[i, j] += hess_vals[k]
                if i != j
                    H[j, i] += hess_vals[k]
                end
            end
            return H
        end
    end

    # Solve
    has_bounds = any(isfinite, lb) || any(isfinite, ub)
    result = if has_bounds
        SQP.sqp_solve(f, g, h, x0, lb, ub; options = opts,
                      grad_f = _grad_f, jac_g = _jac_g, jac_h = _jac_h,
                      hess_lag = _hess_lag)
    else
        SQP.sqp_solve(f, g, h, x0; options = opts,
                      grad_f = _grad_f, jac_g = _jac_g, jac_h = _jac_h,
                      hess_lag = _hess_lag)
    end

    # Store results
    model.solution = result.x
    model.objective_value = sign * result.objective  # undo sign flip for MAX
    model.status = result.status
    model.iterations = result.iterations
    model.constraint_violation = result.constraint_violation
    model.solve_time = time() - t0

    return
end

# ============================================================================
# Results
# ============================================================================

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.status == :not_called
        return MOI.OPTIMIZE_NOT_CALLED
    elseif model.status == :converged
        return MOI.LOCALLY_SOLVED
    elseif model.status == :max_iterations
        return MOI.ITERATION_LIMIT
    elseif model.status == :qp_failed
        return MOI.NUMERICAL_ERROR
    elseif model.status == :line_search_failed
        return MOI.NUMERICAL_ERROR
    else
        return MOI.OTHER_ERROR
    end
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    return string(model.status)
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.solution === nothing ? 0 : 1
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(model, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    if model.status == :converged
        return MOI.FEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    return MOI.NO_SOLUTION
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.objective_value
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    return model.solution[vi.value]
end

function MOI.get(model::Optimizer, ::MOI.SolveTimeSec)
    return model.solve_time
end
