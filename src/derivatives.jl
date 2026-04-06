"""
    make_gradient(f, x0)

Create a gradient function for `f`, using ForwardDiff with FiniteDiff fallback.
"""
function make_gradient(f, x0::AbstractVector)
    try
        grad_f = x -> ForwardDiff.gradient(f, x)
        grad_f(x0)  # test evaluation
        return grad_f
    catch
        return x -> FiniteDiff.finite_difference_gradient(f, x)
    end
end

"""
    make_jacobian(g, x0)

Create a Jacobian function for `g`, using ForwardDiff with FiniteDiff fallback.
"""
function make_jacobian(g, x0::AbstractVector)
    try
        jac_g = x -> ForwardDiff.jacobian(g, x)
        jac_g(x0)  # test evaluation
        return jac_g
    catch
        return x -> FiniteDiff.finite_difference_jacobian(g, x)
    end
end

"""
    make_hessian(f, x0)

Create a Hessian function for `f`, using ForwardDiff. Returns a function that
produces a zero matrix if AD fails.
"""
function make_hessian(f, x0::AbstractVector{T}) where {T}
    n = length(x0)
    try
        hess_f = x -> ForwardDiff.hessian(f, x)
        h0 = hess_f(x0)
        if all(iszero, h0)
            return x -> spzeros(T, n, n)
        end
        return hess_f
    catch
        return x -> spzeros(T, n, n)
    end
end
