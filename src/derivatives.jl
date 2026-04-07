import ADTypes: AutoForwardDiff, AutoFiniteDiff

const DEFAULT_BACKEND = AutoForwardDiff()
const FALLBACK_BACKEND = AutoFiniteDiff()

"""
    _select_gradient_backend(f, x0)

Try ForwardDiff first; fall back to FiniteDiff if it fails (e.g., MOI evaluator closures).
"""
function _select_gradient_backend(f, x0)
    try
        DI.gradient(f, DEFAULT_BACKEND, x0)
        return DEFAULT_BACKEND
    catch
        return FALLBACK_BACKEND
    end
end

"""
    _select_jacobian_backend(g, x0)

Try ForwardDiff first; fall back to FiniteDiff if it fails.
"""
function _select_jacobian_backend(g, x0)
    try
        DI.jacobian(g, DEFAULT_BACKEND, x0)
        return DEFAULT_BACKEND
    catch
        return FALLBACK_BACKEND
    end
end

"""
    make_gradient(f, x0; backend=nothing)

Create a gradient function for `f` using DifferentiationInterface.
Uses `prepare_gradient` for efficient repeated evaluation.
If `backend` is `nothing`, auto-selects ForwardDiff with FiniteDiff fallback.
"""
function make_gradient(f, x0::AbstractVector; backend = nothing)
    be = backend !== nothing ? backend : _select_gradient_backend(f, x0)
    prep = DI.prepare_gradient(f, be, x0)
    return x -> DI.gradient(f, prep, be, x)
end

"""
    make_jacobian(g, x0; backend=nothing)

Create a Jacobian function for `g` using DifferentiationInterface.
Uses `prepare_jacobian` for efficient repeated evaluation.
If `backend` is `nothing`, auto-selects ForwardDiff with FiniteDiff fallback.
"""
function make_jacobian(g, x0::AbstractVector; backend = nothing)
    be = backend !== nothing ? backend : _select_jacobian_backend(g, x0)
    prep = DI.prepare_jacobian(g, be, x0)
    return x -> DI.jacobian(g, prep, be, x)
end

"""
    make_hessian(f, x0; backend=nothing)

Create a Hessian function for `f` using DifferentiationInterface.
Returns a function that produces a zero matrix if AD fails.
"""
function make_hessian(f, x0::AbstractVector{T}; backend = nothing) where {T}
    n = length(x0)
    be = backend !== nothing ? backend : _select_gradient_backend(f, x0)
    prep = try
        DI.prepare_hessian(f, be, x0)
    catch
        return x -> spzeros(T, n, n)
    end
    # Verify it produces a non-zero result
    h0 = try
        DI.hessian(f, prep, be, x0)
    catch
        return x -> spzeros(T, n, n)
    end
    if all(iszero, h0)
        return x -> spzeros(T, n, n)
    end
    return x -> DI.hessian(f, prep, be, x)
end
