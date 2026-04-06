"""
    NLPProblem{T, F, G, H, V}

Nonlinear programming problem: min f(x) s.t. g(x) <= 0, h(x) = 0.

Variable bounds are converted to inequality constraints internally.
"""
struct NLPProblem{T <: AbstractFloat, F, G, H, V <: AbstractVector{T}}
    f::F
    g::G
    h::H
    x0::V
    n::Int
    n_ineq::Int
    n_eq::Int
end

function NLPProblem(f, g, h, x0::AbstractVector{T}) where {T <: AbstractFloat}
    g0 = g(x0)
    h0 = h(x0)
    NLPProblem{T, typeof(f), typeof(g), typeof(h), typeof(x0)}(
        f, g, h, x0, length(x0), length(g0), length(h0),
    )
end

function NLPProblem(
    f, g, h,
    x0::AbstractVector{T},
    lb::AbstractVector{T},
    ub::AbstractVector{T},
) where {T <: AbstractFloat}
    g0 = g(x0)
    h0 = h(x0)
    n_orig_ineq = length(g0)
    n_eq = length(h0)
    n = length(x0)
    nb = max(length(lb), length(ub))

    g_with_bounds = if n_orig_ineq > 0 && nb > 0
        x -> vcat(g(x), lb .- x, x .- ub)
    elseif nb > 0
        x -> vcat(lb .- x, x .- ub)
    else
        g
    end

    n_ineq = n_orig_ineq + (nb > 0 ? 2 * n : 0)

    NLPProblem{T, typeof(f), typeof(g_with_bounds), typeof(h), typeof(x0)}(
        f, g_with_bounds, h, x0, n, n_ineq, n_eq,
    )
end
