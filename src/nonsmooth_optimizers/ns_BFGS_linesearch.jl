"""
    linesearch_nsbfgs


Nonsmooth linesearch from *Nonsmooth optimization via quasi-Newton methods*, Lewis & Overton, 2013.
"""
function linesearch_nsbfgs(pb, xₖ, ∇fₖ, d)
    ω₁=1e-4
    ω₂=0.5
    maxit = 50
    τₑ = 2

    dh₀ = dot(∇fₖ, d)
    if dh₀ > 0
        @warn "ArmijoWolfe ns_BFGS: non negative direction provided, taking opposite direction." dot(∇fₖ, d), norm(∇fₖ), norm(d)
        d .*= -1.0
    end

    α = 1
    α_low, α_up = 0, Inf
    A_t, W_t = false, false

    F_x = F(pb, xₖ)
    F_cand = Inf
    x_cand = copy(xₖ)

    A(t) = (F_x + ω₁*α*dh₀ ≥ F_cand)
    W(t) = (is_differentiable(pb, x_cand) && dot(∂F_elt(pb, x_cand+t*d), d) > ω₂ * dh₀)

    it_ls = 0
    validpoint = false
    while !validpoint
        x_cand = xₖ + α * d
        F_cand = F(pb, x_cand)

        if F_x > F_cand > F_x - 3*eps(F_cand)
            @warn "Linesearch: reached conditionning of funtion here" it_ls
            # @printf "F_x        : %.16e\n" F_x
            # @printf "F_cand     : %.16e\n" F_cand
            # @printf "eps(F_cand): %.16e\n" eps(F_cand)
            break
        end

        A_α, W_α = A(α), W(α)
        if A_α && W_α
            # α = 1 should be accepted for superlinear convergence.
            validpoint = true
            break
        end

        if !A_α
            α_up = α

            α = (α_up + α_low) / 2
        elseif !W_α
            α_low = α

            if isinf(α_up)
                α = τₑ * α_low
            else
                α = (α_up + α_low) / 2
            end
        else
            validpoint = true
        end

        it_ls += 1
        (it_ls > maxit) && (break)
    end

    if !A_t
        @debug "Linesearch: no suficient decrease" F_cand F_x + ω₁*α*dh_0
    end
    if !W_t
        @debug "Linesearch: too small step" F_x + ω₂*α*dh_0 F_cand
    end

    return α, it_ls
end
