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

"""
    schittkowski_line_search_soc(phi, dphi, phi0max, amax;
                                 soc_callback, mu, beta)

Line search with Second-Order Correction (SOC) fallback for the Maratos effect.

Computes `dphi0 = dphi(0)`. If the direction descends (`dphi0 < 0`), this
function is equivalent to [`schittkowski_line_search`](@ref). If the merit
function appears to go uphill (`dphi0 >= 0`, the Maratos signature), it
invokes `soc_callback()` to obtain a corrected merit function/derivative
pair along the SOC direction, then runs the standard line search on those.

`soc_callback()` must return `(phi_soc, dphi_soc, accepted::Bool)`:
- `phi_soc(α)` — merit function evaluated along the corrected step
- `dphi_soc(α)` — its derivative w.r.t. `α`
- `accepted` — `true` if the correction QP solved and produced a usable
  direction; `false` to fall back to the legacy patched line search.

Returns `(alpha, phi_alpha, comment, used_soc::Bool)`.
"""
function schittkowski_line_search_soc(
    phi::Function, dphi::Function,
    phi0max::T, amax::T = one(T);
    soc_callback::Function,
    mu::T = T(1e-4), beta::T = T(0.5),
) where {T <: AbstractFloat}
    phi0 = phi(zero(T))
    dphi0 = try
        dphi(zero(T))
    catch
        hphi = T(1e-6)
        (phi(hphi) - phi0) / hphi
    end
    if isinf(dphi0) || isnan(dphi0)
        hphi = T(1e-6)
        dphi0 = (phi(hphi) - phi0) / hphi
    end

    # Downhill — no SOC needed, delegate to the standard line search.
    if dphi0 < zero(T)
        a, p, c = schittkowski_line_search(phi, dphi, phi0max, amax; mu = mu, beta = beta)
        return a, p, c, false
    end

    # Uphill — try SOC before giving up.
    phi_soc, dphi_soc, accepted = soc_callback()
    if accepted
        a, p, c = schittkowski_line_search(phi_soc, dphi_soc, phi0max, amax; mu = mu, beta = beta)
        return a, p, "SOC: " * c, true
    end

    # SOC failed — fall back to the legacy patched behaviour.
    a, p, c = schittkowski_line_search(phi, dphi, phi0max, amax; mu = mu, beta = beta)
    return a, p, "SOC failed, " * c, false
end
