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

    t = 1.0
    α, β = 0.0, Inf
    A_t, W_t = false, false

    Fₖ = F(pb, xₖ)
    F_cand = Inf
    x_cand = copy(xₖ)

    it_ls = 0
    ncalls_∂F_elt = 0
    validpoint = false
    while !validpoint
        x_cand .= xₖ .+ t .* d
        F_cand = F(pb, x_cand)

        if Fₖ > F_cand > Fₖ - 3*eps(F_cand)
            @warn "Linesearch: reached conditionning of funtion here" it_ls
            break
        end

        Aₜ = (Fₖ + ω₁*t*dh₀ ≥ F_cand)
        Wₜ = false
        if is_differentiable(pb, x_cand)
            Wₜ = (dot(∂F_elt(pb, x_cand), d) > ω₂ * dh₀)
            ncalls_∂F_elt += 1
        end


        if !Aₜ
            β = t
        elseif !Wₜ
            α = t
        else
            validpoint = true
            break
        end

        if !isinf(β)
            t = (α + β) / 2
        else
            t = 2 * α
        end

        it_ls += 1
        (it_ls > maxit) && (break)
    end

    if !A_t
        @debug "Linesearch: no suficient decrease" F_cand Fₖ + ω₁*t*dh_0
    end
    if !W_t
        @debug "Linesearch: too small step" F_x + ω₂*t*dh_0 F_cand
    end

    return t, (;
               it_ls,
               F = 1+it_ls,
               ∂F_elt = ncalls_∂F_elt,
               )
end
