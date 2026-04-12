"""
    Filter{T}

Filter for globalization of constrained optimization, in the style of
Wächter & Biegler (2006) / Fletcher & Leyffer (2002).

A *filter* is a set of ``(f, θ)`` pairs where:
- ``f`` is the (sign-adjusted) objective value, and
- ``θ`` is the constraint violation ``\\|[g(x)_+; h(x)]\\|_1``.

A trial point ``(f_\\mathrm{trial}, θ_\\mathrm{trial})`` is **dominated** by
a filter entry ``(f', θ')`` iff ``f' \\le f_\\mathrm{trial}`` AND
``θ' \\le θ_\\mathrm{trial}`` — i.e., the filter entry is at least as good
in both criteria. The line search accepts a trial point only if it is
not dominated by any existing filter entry.

Internally the filter stores entries that have already been shifted by
the "sufficient margin" `(γ_f·θ, (1-γ_θ)·θ)`, so the dominance check is
a pure lexicographic comparison against the raw trial values.

Reference: Wächter & Biegler, *Line Search Filter Methods for Nonlinear
Programming*, SIAM J. Optim. 16(1), 2006.
"""
mutable struct Filter{T <: AbstractFloat}
    entries::Vector{Tuple{T, T}}   # (f_entry, theta_entry)
    max_size::Int
end

Filter{T}(; max_size::Int = 50) where {T <: AbstractFloat} =
    Filter{T}(Tuple{T, T}[], max_size)

Base.length(F::Filter) = length(F.entries)
Base.isempty(F::Filter) = isempty(F.entries)

"""
    reset!(F::Filter)

Clear all entries from the filter.
"""
function reset!(F::Filter{T}) where {T <: AbstractFloat}
    empty!(F.entries)
    return F
end

"""
    is_dominated(F::Filter, f_trial, θ_trial)

Return `true` if the trial pair `(f_trial, θ_trial)` is dominated by any
entry `(f', θ')` in the filter, i.e. `f' ≤ f_trial AND θ' ≤ θ_trial`.
"""
function is_dominated(F::Filter{T}, f_trial::T, theta_trial::T) where {T <: AbstractFloat}
    for (f_e, theta_e) in F.entries
        if f_e <= f_trial && theta_e <= theta_trial
            return true
        end
    end
    return false
end

"""
    augment!(F::Filter, f_k, θ_k, γ_f, γ_θ)

Add the shifted pair `(f_k − γ_f·θ_k, (1 − γ_θ)·θ_k)` to the filter.
This is the Wächter-Biegler convention: the stored entry is already
backed off by the sufficient-decrease margin so the dominance check
can use raw trial values.

If `F` already contains an entry that dominates (or equals) the new
shifted pair, the filter is left unchanged. Existing entries that would
become dominated by the new one are removed (prunes the filter).

If the filter exceeds `F.max_size` entries, the oldest entry is evicted
(FIFO eviction, which is a common practical simplification over the
strict Wächter-Biegler version).
"""
function augment!(F::Filter{T}, f_k::T, theta_k::T,
                  gamma_f::T, gamma_theta::T) where {T <: AbstractFloat}
    f_new = f_k - gamma_f * theta_k
    theta_new = (one(T) - gamma_theta) * theta_k

    # Skip augmentation if something already dominates the new entry
    for (f_e, theta_e) in F.entries
        if f_e <= f_new && theta_e <= theta_new
            return F
        end
    end

    # Remove existing entries that the new entry would dominate
    filter!(e -> !(f_new <= e[1] && theta_new <= e[2]), F.entries)

    push!(F.entries, (f_new, theta_new))

    # FIFO eviction if the filter is over capacity
    while length(F.entries) > F.max_size
        popfirst!(F.entries)
    end

    return F
end

"""
    theta_constraint_violation(g_val, h_val)

L1 constraint violation `‖[g_+; h]‖_1`. Zero when feasible.
"""
function theta_constraint_violation(g_val::AbstractVector{T},
                                    h_val::AbstractVector{T}) where {T <: AbstractFloat}
    s = zero(T)
    for v in g_val
        if v > zero(T)
            s += v
        end
    end
    for v in h_val
        s += abs(v)
    end
    return s
end

"""
    filter_line_search(F, f, g, h, x, dx, f_k, theta_k, grad_f_dot_d, options)

Wächter-Biegler filter line search.

Backtracks on `α ∈ [0, 1]` starting at `α = 1`, accepting the first
trial point `x + α·dx` that is not dominated by the filter and that
satisfies either the f-step Armijo condition (when the switching
condition holds) or an h-step progress test.

Returns `(alpha, accepted::Bool, step_type::Symbol, f_alpha, theta_alpha)`:
- `alpha` — step length chosen (or the smallest tried, on rejection)
- `accepted` — `true` if a step was accepted
- `step_type` — `:f` (f-type Armijo step) or `:h` (h-type progress step),
  or `:rejected` if no step accepted
- `f_alpha`, `theta_alpha` — metrics at the accepted trial point, or at
  the last rejected trial when `accepted == false`

The caller is responsible for augmenting the filter when `step_type == :h`.
"""
function filter_line_search(
    F::Filter{T}, f::Function, g::Function, h::Function,
    x::AbstractVector{T}, dx::AbstractVector{T},
    f_k::T, theta_k::T, grad_f_dot_d::T,
    options::SQPOptions{T},
) where {T <: AbstractFloat}
    alpha = one(T)
    beta = options.line_search_beta

    gamma_f = options.filter_gamma_f
    gamma_theta = options.filter_gamma_theta
    eta = options.filter_armijo_eta
    s_phi = options.filter_switching_s_phi
    s_theta = options.filter_switching_s_theta
    delta = options.filter_switching_delta
    alpha_min = options.filter_alpha_min

    # Switching condition uses the step's *descent strength* on f:
    # if -∇f'd > 0 (f-descent direction) and that descent is big
    # compared to the constraint violation scale, we're in f-step mode.
    descent = -grad_f_dot_d
    f_step_mode_base = descent > zero(T)

    f_alpha_last = f_k
    theta_alpha_last = theta_k
    step_type = :rejected

    while alpha >= alpha_min
        x_trial = x .+ alpha .* dx
        g_trial = g(x_trial)
        h_trial = h(x_trial)
        f_alpha = f(x_trial)
        theta_alpha = theta_constraint_violation(g_trial, h_trial)
        f_alpha_last = f_alpha
        theta_alpha_last = theta_alpha

        # Filter dominance check
        if is_dominated(F, f_alpha, theta_alpha)
            alpha *= beta
            continue
        end

        # Switching condition (Wächter-Biegler eq 19):
        # `(-∇f'd)^{s_φ}·α^{1-s_θ} > δ·θ_k^{s_θ}` → f-step mode,
        # otherwise h-step mode.
        switching_lhs = f_step_mode_base ?
                        descent^s_phi * alpha^(one(T) - s_theta) :
                        zero(T)
        switching_rhs = delta * theta_k^s_theta
        in_f_step_mode = f_step_mode_base && switching_lhs > switching_rhs

        if in_f_step_mode
            # Armijo sufficient decrease on f
            if f_alpha <= f_k + eta * alpha * grad_f_dot_d
                step_type = :f
                return alpha, true, step_type, f_alpha, theta_alpha
            end
        else
            # h-step: accept if either feasibility or objective improves enough
            if theta_alpha <= (one(T) - gamma_theta) * theta_k ||
               f_alpha <= f_k - gamma_f * theta_k
                step_type = :h
                return alpha, true, step_type, f_alpha, theta_alpha
            end
        end

        alpha *= beta
    end

    # All backtracking exhausted — return the last-tried α so the caller
    # can decide what to do (typically fall back to a merit line search)
    return alpha, false, step_type, f_alpha_last, theta_alpha_last
end
