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
    modify_eigenvalues!(H; floor=1e-8, method=:abs)

In-place symmetric eigenvalue correction for the Hessian. Preserves the
true curvature *directions* in `H`; only modifies the sign/magnitude of
eigenvalues that are too small or negative.

Two methods:

- `:abs` (default) — replace each eigenvalue `λᵢ` with `max(|λᵢ|, floor)`.
  Negative curvature becomes positive curvature of equal magnitude. This
  is the "modified Newton" approach: the QP subproblem minimizes
  `(1/2)dᵀ|∇²L|d + gᵀd`, which makes progress along directions of
  negative curvature at the same rate as positive curvature. Best for
  highly indefinite analytical Hessians (HS071-style bilinear, HS092-style
  oscillating curvature).

- `:clip` — replace each eigenvalue `λᵢ` with `max(λᵢ, floor)`. Negative
  eigenvalues get clipped to a small positive value, losing their
  magnitude information. Cheaper but much more aggressive; use only when
  the analytical Hessian is mostly PD with a few slightly-negative
  eigenvalues.

Returns `(H, was_modified::Bool, min_eigenvalue_before::T)`.
"""
function modify_eigenvalues!(H::AbstractMatrix{T};
                             floor::T = T(1e-8),
                             method::Symbol = :abs) where {T <: AbstractFloat}
    Hsym = Symmetric(Matrix{T}((H .+ H') ./ 2))
    F = eigen(Hsym)
    λ_min = minimum(F.values)

    if method == :abs
        # Modified Newton: |λᵢ| with positive floor
        needs_mod = any(λ -> λ < floor, F.values)
        needs_mod || return H, false, λ_min
        λ_new = [max(abs(λ), floor) for λ in F.values]
    elseif method == :clip
        if λ_min >= floor
            return H, false, λ_min
        end
        λ_new = max.(F.values, floor)
    else
        error("modify_eigenvalues! method must be :abs or :clip, got :$method")
    end

    H .= F.vectors * Diagonal(λ_new) * F.vectors'
    H .= (H .+ H') ./ 2     # symmetrize to kill roundoff
    return H, true, λ_min
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

"""
    lbfgs_hessian(s_history, y_history, n)

Reconstruct an approximate Hessian from L-BFGS history using the compact
representation. Returns a dense positive definite matrix.

Uses the two-loop recursion approach to build H₀ scaled by γ = (sᵀy)/(yᵀy)
from the most recent pair, then applies all stored corrections.
"""
function lbfgs_hessian(s_history::Vector{V}, y_history::Vector{V},
                       n::Int) where {T <: AbstractFloat, V <: AbstractVector{T}}
    m = length(s_history)
    if m == 0
        return Matrix{T}(I, n, n)
    end

    # Initial scaling: γ = sₘᵀyₘ / yₘᵀyₘ (Nocedal & Wright)
    s_last = s_history[end]
    y_last = y_history[end]
    yy = dot(y_last, y_last)
    gamma = yy > eps(T) ? dot(s_last, y_last) / yy : one(T)
    gamma = max(gamma, T(1e-8))  # ensure positive

    # Build H = γI then apply BFGS updates from history
    H = gamma * Matrix{T}(I, n, n)
    for k in 1:m
        s = s_history[k]
        y = y_history[k]
        sy = dot(s, y)
        if sy > eps(T)
            Hs = H * s
            sHs = dot(s, Hs)
            if sHs > eps(T)
                H .= H .+ (y * y') / sy .- (Hs * Hs') / sHs
            end
        end
    end

    # Ensure positive definite
    if !isposdef(H)
        H .= gamma * Matrix{T}(I, n, n)
    end

    return H
end
