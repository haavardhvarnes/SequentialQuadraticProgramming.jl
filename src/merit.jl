"""
    augmented_lagrangian(f_val, g_val, h_val, pensig, penlam, r)

Evaluate the augmented Lagrangian merit function given pre-computed function values.

Based on Schittkowski NM_SQP2.pdf.
"""
function augmented_lagrangian(
    f_val::T, g_val::AbstractVector{T}, h_val::AbstractVector{T},
    pensig::AbstractVector{T}, penlam::AbstractVector{T}, r::AbstractVector{T},
) where {T <: AbstractFloat}
    out = f_val
    n_ineq = length(pensig)
    for j in 1:n_ineq
        if g_val[j] >= pensig[j] / r[j]
            out += pensig[j] * g_val[j] + T(0.5) * r[j] * g_val[j]^2
        else
            out += T(0.5) * pensig[j]^2 / r[j]
        end
    end
    nr = n_ineq
    for j in 1:length(penlam)
        out += penlam[j] * h_val[j] + T(0.5) * r[j + nr] * h_val[j]^2
    end
    return out
end

"""
    make_merit_function(f, g, h, pensig, penlam, r)

Create a callable merit function phi(x) from problem functions and penalty parameters.
"""
function make_merit_function(f, g, h, pensig, penlam, r)
    function phi(x)
        augmented_lagrangian(f(x), g(x), h(x), pensig, penlam, r)
    end
    return phi
end
