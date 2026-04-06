"""
    schittkowski_line_search(phi, dphi, phi0max, amax, mu, beta)

Quadratic interpolation line search with Armijo-Goldstein sufficient decrease condition.

Based on Schittkowski NM_SQP2.pdf.

Returns `(alpha, phi_alpha, comment)`.
"""
function schittkowski_line_search(
    phi::Function, dphi::Function,
    phi0max::T, amax::T = one(T);
    mu::T = T(1e-4), beta::T = T(0.5),
) where {T <: AbstractFloat}
    phi0 = phi(zero(T))
    phi0_ = phi0

    dphi0 = try
        dphi(zero(T))
    catch
        hphi = T(1e-6)
        (phi(hphi) - phi0) / hphi
    end

    if isinf(dphi0) || isnan(dphi0)
        hphi = T(1e-6)
        dphi0 = (phi(hphi) - phi0) / hphi
        @warn "dphi in line search is NaN or Inf, using finite difference" phi0 dphi0
    end

    comment = ""

    if dphi0 >= zero(T)
        if phi(amax) < phi0
            return amax, phi(amax), "dphi0 > 0, alpha = amax improves merit"
        end
        phi0_ = max(phi0, phi0max)
        comment = "dphi0 > 0 -> inc. LS-tol"
        dphi0 = T(-1e-7)
    end

    alpha = amax
    count = 0

    while phi(alpha) > (phi0_ + mu * alpha * dphi0) && alpha > T(1e-5)
        alpha_ = (T(0.5) * alpha * alpha * dphi0) / (alpha * dphi0 - phi(alpha) + phi0)
        if isnan(alpha_) || isinf(alpha_)
            @warn "Line search interpolation failed" alpha phi_alpha=phi(alpha) phi0 dphi0
            return alpha, phi(alpha), "interpolation failed"
        else
            alpha = max(beta * alpha, alpha_)
        end
        count += 1
    end

    return alpha, phi(alpha), comment
end
