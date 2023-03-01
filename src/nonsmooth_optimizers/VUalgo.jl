"""
    $TYPEDSIGNATURES

Parameters:
- `ϵ`: overall precision required
- `m`: sufficient decrease parameter
- `μlow`: minimal prox parameter (μ is inverse of γ). Higher μ means smaller serious steps, but less null steps
"""
Base.@kwdef struct VUbundle{Tf} <: NonSmoothOptimizer{Tf}
    μlow::Tf = 0.05
    ϵ::Tf = 1e-6
    m::Tf = 0.5
    curvmin::Tf = 1e-6          # min curvature for quasi Newton matrix update
    Newton_accel::Bool = true
end

"""
    $TYPEDSIGNATURES

Parameters:
- `σ`: in (0, 0.5!], lower values enforce higher precision on each prox point approximation,
"""
mutable struct VUbundleState{Tf} <: OptimizerState{Tf}
    p::Vector{Tf}                                                       # point
    s::Vector{Tf}                                                       # minimal norm subgradient of the current ϵ subdifferential
    ϵ::Tf                                                               # ?
    σ::Tf
    U::Matrix{Tf}                                                       # orthonormal basis of the approximation of the current U space
    H::Matrix{Tf}  # inverse hessian (approximation) matrix
    kase::Int64    # quasi newton approximation status
    pprev::Vector{Tf}
    sprev::Vector{Tf}
    k::Int64                                                            # iteration counter
    μ::Tf                                                               # step of proximal point
    μmax::Tf
    bundle::Bundle{Tf}
    firstsnnorm::Tf
    nₖ::Int64           # Dimension of U space
    nₖ₋₁::Int64
    histys::Vector{NamedTuple{(:y, :s), Tuple{Vector{Tf}, Vector{Tf}}}} # history of BFGS steps and subgradients difference
end

function initial_state(::VUbundle{Tf}, initial_x::Vector{Tf}, pb) where Tf
    n = length(initial_x)
    v = ∂F_elt(pb, initial_x)
    return VUbundleState(
        copy(initial_x),
        v,
        Tf(1),
        Tf(0.5),
        Matrix{Tf}(1.0I, n, n),
        Matrix{Tf}(1.0I, n, n),
        0,
        copy(initial_x),
        v,
        1,
        Tf(4),
        Tf(4e4),
        initial_bundle(pb, initial_x),
        norm(v),
        -1,
        -1,
        NamedTuple{(:y, :s), Tuple{Vector{Tf}, Vector{Tf}}}[],
    )
end

#
### Printing
#
function print_header(o::VUbundle)
    println("**** VUbundle algorithm")
    println(" + Newton accel: ", o.Newton_accel)
    return
end

display_logs_header_post(gs::VUbundle) = print("μ         ϵ̂       | ŝ|          #nullsteps     nₖ   ⟨dᴺ, sₖ⟩  |dᴺ|")

function display_logs_post(os, gs::VUbundle)
    ai = os.additionalinfo
    @printf "%.2e  %.2e %.2e     %-2i             %-2i   % .1e  %.2e" ai.μ ai.ϵ̂  ai.ŝnorm ai.nnullsteps ai.nₖ ai.dotsₖNewtonstep ai.Newtonsteplength
end

#
### VUbundle method
#
function update_iterate!(state, VU::VUbundle{Tf}, pb) where Tf
    # ϵₖ = state.ϵ

    pₖ = state.p
    pₖ₋₁ = state.pprev
    sₖ = state.s
    sₖ₋₁ = state.sprev
    Uₖ = state.U
    Hₖ = state.H
    μₖ = state.μ
    σₖ = Tf(0.5)        # Tf(inv(1+state.k^2))
    bundle = state.bundle

    pₖsave = deepcopy(pₖ)
    sₖsave = deepcopy(sₖ)

    μmin = 1e-6

    dotsₖNewtonstep = 0.0
    nₖ = size(Uₖ, 2)
    ν = nₖ
    Newtonsteplength = 0.0
    nullsteps = []
    sclst = -1.
    sscale = -1.

    local bundleinfo

    printstyled("\n\n))) iteration ", state.k, "\n", color = :green)
    println("pₖ     = ", state.p)
    println("pₖ₋₁   = ", state.pprev)
    println("sₖ     = ", state.s)
    println("sₖ₋₁   = ", state.sprev)
    println("Uₖ     = ", state.U)
    println("Hₖ     = ", state.H)
    println("μₖ     = ", state.μ)
    println("σₖ     = ", σₖ)
    println("kase   = ", state.kase)
    display(bundle)
    @show nₖ

    if VU.Newton_accel
        μlast = μₖ

        ## NOTE 1. qNewton step computation
        printstyled(" 1. qNewton step computation\n", color = :green)

        if state.k == 1
            Hₖ .= μₖ * I(2)
            haveragecurvinv = 1/μₖ
        end
        if state.k > 2
            sscale = norm(sₖ₋₁) / norm(sₖ)
            sclst = sscale * μₖ
            μₖ = max(μₖ, sclst)
            @show μₖ
            @show sclst
        end

        νlow = ν

        # TODO make this in place for H
        du, hmin, haveinv, Hout, state.kase = qNewtonupdate(pₖ, pₖ₋₁, sₖ, sₖ₋₁, Uₖ, Hₖ, state.k, VU.curvmin, ν, νlow, μₖ, state.kase)
        state.H .= Hout

        @show du
        @show Hₖ
        printstyled(" 2. Stopping test\n", color = :green)

        ## NOTE 2. Stopping test
        # TODO

        printstyled(" 3. Newton step\n", color = :green)

        ## NOTE 3. Taking Newton step
        performUstep = true
        # gp = ? # out of extrapolation step # BUG here
        # if dot(gp, du) > 0.1 * dot(sₖ, du)
        #     @warn "Bad inverse hessian, no U step."
        #     performUstep = false
        # end

        if performUstep
            xᶜₖ₊₁ = pₖ + du
            gxᶜₖ₊₁ = ∂F_elt(pb, xᶜₖ₊₁)
            Fxᶜₖ₊₁ = F(pb, xᶜₖ₊₁)
            errxᶜₖ₊₁ = F(pb, pₖ) - F(pb, xᶜₖ₊₁) + dot(gxᶜₖ₊₁, xᶜₖ₊₁ - pₖ) # linearization error, centered at p


            @show xᶜₖ₊₁
            @show errxᶜₖ₊₁

            printstyled(" 4. extrapolation search\n", color = :green)
            # NOTE 4. extrapolation search

            Fpₖ = F(pb, pₖ)
            gpₖ = ∂F_elt(pb, pₖ)
            dxc = xᶜₖ₊₁ - pₖ
            gxcdxc = dot(gxᶜₖ₊₁, dxc)

            xᶜₖ₊₁, Fxᶜₖ₊₁, gxᶜₖ₊₁, erxc, muave, mufirst, nsrch = esearch!(bundle, xᶜₖ₊₁, Fxᶜₖ₊₁, gxᶜₖ₊₁, errxᶜₖ₊₁, pₖ, Fpₖ, gpₖ, dxc, gxcdxc, pb, state.k, μₖ, σₖ, sₖ)

            @show xᶜₖ₊₁
            @show Fxᶜₖ₊₁
            @show gxᶜₖ₊₁
            @show erxc
            @show muave
            @show mufirst
            @show nsrch
            display(bundle)

            printstyled(" 5. mu update\n", color = :green)
            # NOTE 5. mu update
            μₖ, state.μmax = update_mu(μₖ, state.k, mufirst, muave, nₖ, state.nₖ₋₁, haveinv, nsrch, sclst, μlast, sscale, sₖ, μmin, state.μmax, state.firstsnnorm)
            @show μₖ

            # NOTE 6. check for sufficient descent\n
            printstyled(" 6. check for sufficient descent\n", color = :green)


            xᶜprovidesdescent = (Fxᶜₖ₊₁-F(pb, pₖ) ≤ -0.5*(VU.m/μₖ) * norm(gxᶜₖ₊₁)^2)
            @show xᶜprovidesdescent
            if xᶜprovidesdescent && norm(gxᶜₖ₊₁) < norm(sₖ)
                # NOTE 6.a sufficient descent, proceding\n
                printstyled(" 6.a sufficient descent, proceding\n", color = :green)
                throw(ErrorException("Sufficient descent, to be implemented"))

            else
                # NOTE 6.b need additional bundle\n
                printstyled(" 6.b need additional bundle\n", color = :green)
                println("- centering bundle at xc from p")
                display(bundle)

                change_bundle_center!(bundle, xᶜₖ₊₁, Fxᶜₖ₊₁)

                println("- bundle centered at xc")
                display(bundle)

                # Bundle subroutine at point xᶜₖ₊₁ (ie proximal step approximation)
                ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁, bundleinfo = bundlesubroutine!(state.bundle, pb, μₖ, xᶜₖ₊₁, σₖ, VU.ϵ, haveinv)
                println("- prox bundle output")
                @show pᶜₖ₊₁
                @show sᶜₖ₊₁
                @show ϵᶜₖ₊₁
                display(Uᶜₖ₊₁)

                # change bundle center from xᶜ to p
                println("- centering bundle at p from xc")
                display(bundle)

                change_bundle_center!(bundle, pₖ, Fpₖ)

                println("- Bundle recentered:")
                display(bundle)
            end

            newtonproxsuccessful = (F(pb, pᶜₖ₊₁) ≤ F(pb, pₖ) - VU.m / (2μₖ) * norm(sᶜₖ₊₁)^2) && norm(sᶜₖ₊₁) ≤ max(1, sqrt(size(state.p, 1)))*norm(sᶜₖ₊₁) # ? delete 2nd if k=1 or 2 ?
        else # No Newton step
            newtonproxsuccessful = false
        end
    else
        newtonproxsuccessful = false
    end

    if !newtonproxsuccessful
        throw(ErrorException("Newton + prox failed, to be implemented"))
        # if F(pb, pᶜₖ₊₁) ≤ F(pb, pₖ) - VU.m / (2μₖ) * norm(sᶜₖ₊₁)^2
    end
    # else
    #     # Linesearch on line pₖ → pᶜₖ₊₁ to get an xₖ₊₁ such that F(xₖ₊₁) ≤ F(pₖ)
    #     @warn "U-Newton + approximate prox failed to provide sufficient decrease"
    #     xₖ₊₁ = F(pb, pₖ) < F(pb, pᶜₖ₊₁) ? pₖ : pᶜₖ₊₁

    #     state.ϵ, state.p, state.s, state.U, bundleinfo = bundlesubroutine!(state.bundle, pb, μₖ, xₖ₊₁, σₖ, VU.ϵ; printlev = 0)
    #     nullsteps = vcat(nullsteps, bundleinfo.phist) # log nullsteps of correction prox-bundle step
    # end


    printstyled(" 8 memory\n", color = :green)
    # NOTE 8 memory
    state.ϵ, state.p, state.s, state.U = ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁
    state.μ = μₖ
    nullsteps = vcat(nullsteps, bundleinfo.phist) # log nullsteps of prox-bundle step

    println("- centering bundle at xc from p")
    display(bundle)
    Fpᶜₖ₊₁ = F(pb, pᶜₖ₊₁)
    change_bundle_center!(bundle, pᶜₖ₊₁, Fpᶜₖ₊₁)
    println("- bundle centered at xc")
    display(bundle)

    iteration_status = iteration_failed
    if bundleinfo.subroutinestatus == :SeriousStepFound
        iteration_status = iteration_completed
    elseif  bundleinfo.subroutinestatus == :SeriousStepFound
        iteration_status = problem_solved
    end
    state.k += 1


    state.pprev .= pₖsave
    state.sprev .= sₖsave

    return (;
            μ = state.μ,
            ϵ̂ = state.ϵ,
            ŝnorm = norm(state.s),
            bundleinfo.nnullsteps,
            dotsₖNewtonstep,
            nₖ,
            Newtonsteplength,
            nullsteps
    ), iteration_status
end

get_minimizer_candidate(state::VUbundleState) = state.p
