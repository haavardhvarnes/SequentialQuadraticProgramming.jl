"""
    robust_bfgs_update!(H, s, y; m=1e-5, M=1e5)

Robust BFGS update from Yang et al. 2022.

Modifies and returns `H` in-place.

Reference: "A robust BFGS algorithm for unconstrained nonlinear optimization problems"
https://arxiv.org/pdf/1212.5929.pdf
"""
function robust_bfgs_update!(H::AbstractMatrix{T}, s::AbstractVector{T}, y::AbstractVector{T};
                             m::T = T(1e-5), M::T = T(1e5)) where {T <: AbstractFloat}
    mss = m * dot(s, s)
    ys = dot(y, s)
    s_y = s .- y

    gamhigh = (mss - ys) / (mss / m - ys)
    gamlow = if isequal(y, s)
        zero(T)
    else
        dsy = dot(s_y, s_y)
        num = dot(s_y, M * s .- 2 * y)
        disc = M * dot(s, s_y)^2 + 4 * (M - 1) * (dot(s, s) * dot(y, y) - ys^2)
        disc = max(disc, zero(T))  # numerical safety
        (num - sqrt(disc)) / (2 * dsy)
    end

    gam = mss > ys ? max(gamlow, gamhigh) : max(zero(T), gamlow)
    z = gam * s .+ (1 - gam) * y

    zs = dot(z, s)
    sHs = dot(s, H * s)

    if abs(zs) > eps(T) && abs(sHs) > eps(T)
        Hs = H * s
        H .= H .+ (z * z') / zs .- (Hs * Hs') / sHs
    end

    return H
end

"""
    bfgs_update!(H, s, y)

Classic BFGS update with curvature condition safeguard.
"""
function bfgs_update!(H::AbstractMatrix{T}, s::AbstractVector{T}, y::AbstractVector{T}) where {T <: AbstractFloat}
    ys = dot(y, s)
    sHs = dot(s, H * s)

    # Ensure curvature condition
    q = copy(y)
    count = 0
    cmax = 10 * length(q)
    while dot(q, s) < T(1e-5) && count < cmax
        _, idx = findmin(q)
        q[idx] *= T(0.5)
        count += 1
    end

    if count < cmax
        Hs = H * s
        H .= H .+ (q * q') / dot(q, s) .- (Hs * Hs') / dot(s, Hs)
    end

    return H
end

"""
    ensure_positive_definite!(H)

If `H` is not positive definite, replace it with the identity matrix.
Returns `(H, was_modified)`.
"""
function ensure_positive_definite!(H::AbstractMatrix{T}) where {T <: AbstractFloat}
    if isposdef(H)
        return H, false
    else
        n = size(H, 1)
        H .= Matrix{T}(I, n, n)
        return H, true
    end
end

"""
    update_hessian!(H, s, y, iteration, k_reset)

Perform BFGS Hessian update using robust BFGS, with initial scaling on first iteration.
Returns `(H, k_reset)`.
"""
function update_hessian!(H::AbstractMatrix{T}, s::AbstractVector{T}, y::AbstractVector{T},
                         iteration::Int, k_reset::Int) where {T <: AbstractFloat}
    n = length(s)
    if iteration == 1
        H_new = bfgs_update!(Matrix{T}(I, n, n), s, y)
        H .= H_new
    else
        robust_bfgs_update!(H, s, y)
    end
    return H, k_reset
end
