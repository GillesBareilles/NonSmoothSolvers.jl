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
    lognullsteps::Bool = false
end

"""
    $TYPEDSIGNATURES

Parameters:
- `σ`: in (0, 0.5!], lower values enforce higher precision on each prox point approximation,
"""
mutable struct VUbundleState{Tf} <: OptimizerState{Tf}
    p::Vector{Tf}  # point
    Fp::Tf         # function value at p
    gp::Vector{Tf} # subgradient at p
    s::Vector{Tf}  # minimal norm subgradient of the current ϵ subdifferential
    ϵ::Tf          # ?
    σ::Tf
    U::Matrix{Tf}  # orthonormal basis of the approximation of the current U space
    H::Matrix{Tf}  # inverse hessian (approximation) matrix
    kase::Int64    # quasi newton approximation status
    pprev::Vector{Tf}
    sprev::Vector{Tf}
    k::Int64       # iteration counter
    μ::Tf          # step of proximal point
    μmax::Tf
    bundle::Bundle{Tf}
    firstsnnorm::Tf
    nₖ::Int64      # Dimension of U space
    nₖ₋₁::Int64
    run::Int64     # TODO: determine utility
    nullstepshist::MutableLinkedList{Vector{Tf}}
end

function initial_state(::VUbundle{Tf}, initial_x::Vector{Tf}, pb) where Tf
    n = length(initial_x)
    fx, v = blackbox_oracle(pb, initial_x)
    nullstepshist = MutableLinkedList{Vector{Tf}}(initial_x)
    push!(nullstepshist, initial_x)
    return VUbundleState(
        copy(initial_x),
        Tf(0),
        similar(initial_x),
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
        n,
        n,
        1,
        nullstepshist,
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


function toto(nullstepshist::MutableLinkedList{T}; loc = "") where T
    # println("---- len: ", length(nullstepshist))
    @show loc
    if length(nullstepshist) > 2 && nullstepshist[end] in nullstepshist[1:end-1]
        for p in nullstepshist
            @show p
        end
    end
    return
end
toto(state) = toto(state.nullstepshist)

#
### VUbundle method
#
function update_iterate!(state, VU::VUbundle{Tf}, pb) where Tf
    # ϵₖ = state.ϵ
    printlev = 0

    pₖ = state.p
    Fpₖ = state.Fp
    gpₖ = state.gp
    pₖ₋₁ = state.pprev
    sₖ = state.s
    sₖ₋₁ = state.sprev
    ϵₖ = state.ϵ
    Uₖ = state.U
    Hₖ = state.H
    μₖ = state.μ
    σₖ = Tf(0.5)        # Tf(inv(1+state.k^2))
    bundle = state.bundle

    pₖsave = deepcopy(pₖ)
    sₖsave = deepcopy(sₖ)
    nₖsave = state.nₖ

    @info "hist len" length(state.nullstepshist)
    toto(state.nullstepshist; loc = "xx")

    μmin = 1e-6

    dotsₖNewtonstep = 0.0
    nₖ = size(Uₖ, 2)
    ν = nₖ
    Newtonsteplength = 0.0
    nullsteps = []
    sclst = -1.
    sscale = -1.
    haveinv = -1.

    n = size(state.p, 1)


    local bundleinfo = (;)

    (printlev > 0) && printstyled("\n\n))) iteration ", state.k, "\n", color = :green)
    (printlev > 0) && println("pₖ     = ", state.p)
    (printlev > 1) && println("pₖ₋₁   = ", state.pprev)
    (printlev > 0) && println("sₖ     = ", state.s)
    (printlev > 1) && println("sₖ₋₁   = ", state.sprev)
    (printlev > 1) && println("Uₖ     = ", state.U)
    (printlev > 1) && println("Hₖ     = ", state.H)
    (printlev > 1) && println("μₖ     = ", state.μ)
    (printlev > 1) && println("σₖ     = ", σₖ)
    (printlev > 1) && println("kase   = ", state.kase)
    (printlev > 1) && display(bundle)
    (printlev > 1) && @show nₖ

    # NOTE missing stuff here

    if VU.Newton_accel
        μlast = μₖ

        ## NOTE 1. qNewton step computation
        (printlev > 2) && printstyled(" 1. qNewton step computation\n", color = :green)

        if state.k == 1
            Hₖ .= μₖ * I(n)
            haveinv = 1/μₖ
        end
        if state.k > 2
            sscale = norm(sₖ₋₁) / norm(sₖ)
            sclst = sscale * μₖ
            μₖ = max(μₖ, sclst)
            (printlev > 3) && @show μₖ
            (printlev > 3) && @show sclst
        end

        νlow = ν

        # TODO make this in place for H
        du, hmin, haveinv, Hout, state.kase = qNewtonupdate(pₖ, pₖ₋₁, sₖ, sₖ₋₁, Uₖ, Hₖ, state.k, VU.curvmin, ν, νlow, μₖ, state.kase)
        state.H .= Hout

        (printlev > 2) && @show du
        (printlev > 2) && @show Hₖ
        (printlev > 2) && printstyled(" 2. Stopping test\n", color = :green)

        ## NOTE 2. Stopping test
        # but stop test should be before all calculations of du
        # okay, but haveinv is updated in qnewton procedure...
        if ϵₖ + haveinv * norm(sₖ)^2 ≤ VU.ϵ^2;
            @info "\n eps+(|s|^2)/h STOPPING TEST SATISFIED in $(state.k) iterations"
            return (;
                    μ = state.μ,
                    ϵ̂ = state.ϵ,
                    ŝnorm = norm(state.s),
                    nnullsteps = 0,
                    dotsₖNewtonstep,
                    nₖ,
                    Newtonsteplength,
                    nullsteps
            ), problem_solved
        end

        (printlev > 2) && printstyled(" 3. Newton step\n", color = :green)

        ## NOTE 3. Taking Newton step
        performUstep = true
        # gp = ? # out of extrapolation step # BUG here
        # if dot(gp, du) > 0.1 * dot(sₖ, du)
        #     @warn "Bad inverse hessian, no U step."
        #     performUstep = false
        # end

        if performUstep
            xᶜₖ₊₁ = pₖ + du
            Fxᶜₖ₊₁, gxᶜₖ₊₁ = blackbox_oracle(pb, xᶜₖ₊₁)
            push!(state.nullstepshist, xᶜₖ₊₁); toto(state; loc = "aa")
            errxᶜₖ₊₁ = F(pb, pₖ) - Fxᶜₖ₊₁ + dot(gxᶜₖ₊₁, xᶜₖ₊₁ - pₖ) # linearization error, centered at p


            (printlev > 2) && @show xᶜₖ₊₁
            (printlev > 2) && @show errxᶜₖ₊₁

            (printlev > 2) && printstyled(" 4. extrapolation search\n", color = :green)
            # NOTE 4. extrapolation search

            Fpₖ, gpₖ = blackbox_oracle(pb, pₖ)
            push!(state.nullstepshist, pₖ); toto(state; loc = "bb")
            @info "adding" pₖ
            dxc = xᶜₖ₊₁ - pₖ
            gxcdxc = dot(gxᶜₖ₊₁, dxc)

            xᶜₖ₊₁, Fxᶜₖ₊₁, gxᶜₖ₊₁, errxᶜₖ₊₁, muave, mufirst, nsrch = esearch!(bundle, xᶜₖ₊₁, Fxᶜₖ₊₁, gxᶜₖ₊₁, errxᶜₖ₊₁, pₖ, Fpₖ, gpₖ, dxc, gxcdxc, pb, state.k, μₖ, σₖ, sₖ; state.nullstepshist)

            (printlev > 2) && @show xᶜₖ₊₁
            (printlev > 2) && @show Fxᶜₖ₊₁
            (printlev > 2) && @show gxᶜₖ₊₁
            (printlev > 2) && @show errxᶜₖ₊₁
            (printlev > 2) && @show muave
            (printlev > 2) && @show mufirst
            (printlev > 2) && @show nsrch
            (printlev > 2) && display(bundle)

            (printlev > 2) && printstyled(" 5. mu update\n", color = :green)
            # NOTE 5. mu update
            μₖ, state.μmax = update_mu(μₖ, state.k, mufirst, muave, nₖ, state.nₖ₋₁, haveinv, nsrch, sclst, μlast, sscale, sₖ, μmin, state.μmax, state.firstsnnorm, state.run)
            (printlev > 2) && @show μₖ

            # NOTE 6. check for sufficient descent\n
            (printlev > 2) && printstyled(" 6. check for sufficient descent\n", color = :green)
            xᶜprovidesdescent = (Fxᶜₖ₊₁-F(pb, pₖ) ≤ -0.5*(VU.m/μₖ) * norm(gxᶜₖ₊₁)^2)
            state.run = 0

            (printlev > 2) && @show xᶜprovidesdescent
            if xᶜprovidesdescent && norm(gxᶜₖ₊₁) < norm(sₖ)
                # NOTE 6.a sufficient descent, proceding\n
                (printlev > 2) && printstyled(" 6.a sufficient descent, proceding\n", color = :green)

                ϵᶜₖ₊₁ = Tf(0)
                Fpᶜₖ₊₁ = Fxᶜₖ₊₁
                pᶜₖ₊₁ = xᶜₖ₊₁
                gpᶜₖ₊₁ = gxᶜₖ₊₁
                @assert norm(gpᶜₖ₊₁ - ∂F_elt(pb, pᶜₖ₊₁)) < 1e-15
                sᶜₖ₊₁ = gxᶜₖ₊₁
                Uᶜₖ₊₁ = Matrix{Tf}(I, n, n)

                # HACK: not updated quantities
                # fphat=fxc;
                # dualsh=1.0
                # gphat=gxc;

                (printlev > 3) && @show bundle
                empty!(bundle.bpts)
                push!(bundle.bpts, BundlePoint(-1., gxᶜₖ₊₁, errxᶜₖ₊₁, [-1.])); toto(state)
                (printlev > 3) && @show bundle

                # faisp=[gxc];
                # erlinp=erxc;
                # phat=xc;
                # shat=gxc;
                # epshat=0;
                # pbest=xc; fpbest=fxc; gpbest=gxc; hesspbest=hessxc;
                # hessiansp=hessxc; hessphat=hessxc;

            else
                # NOTE 6.b need additional bundle\n
                (printlev > 2) && printstyled(" 6.b need additional bundle\n", color = :green)

                state.run = 1

                (printlev > 3) && println("- centering bundle at xc from p")
                (printlev > 3) && display(bundle)

                change_bundle_center!(bundle, xᶜₖ₊₁, Fxᶜₖ₊₁)

                (printlev > 3) && println("- bundle centered at xc")
                (printlev > 3) && display(bundle)

                # Bundle subroutine at point xᶜₖ₊₁ (ie proximal step approximation)
                ϵᶜₖ₊₁, pᶜₖ₊₁, Fpᶜₖ₊₁, gpᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁, bundleinfo = bundlesubroutine!(state.bundle, pb, μₖ, xᶜₖ₊₁, σₖ, VU.ϵ, haveinv)
                (printlev > 2) && println("- prox bundle output")
                (printlev > 2) && @show pᶜₖ₊₁
                (printlev > 2) && @show sᶜₖ₊₁
                (printlev > 2) && @show ϵᶜₖ₊₁
                (printlev > 2) && display(Uᶜₖ₊₁)

                # change bundle center from xᶜ to p
                (printlev > 3) && println("- centering bundle at p from xc")
                (printlev > 3) && display(bundle)

                change_bundle_center!(bundle, pₖ, Fpₖ)

                (printlev > 3) && println("- Bundle recentered:")
                (printlev > 3) && display(bundle)
            end

            newtonproxsuccessful = (Fpᶜₖ₊₁ ≤ F(pb, pₖ) - VU.m / (2μₖ) * norm(sᶜₖ₊₁)^2) && norm(sᶜₖ₊₁) ≤ max(1, sqrt(n))*norm(sᶜₖ₊₁) # ? delete 2nd if k=1 or 2 ?
        else # No Newton step
            newtonproxsuccessful = false
        end
    else
        newtonproxsuccessful = false
    end

    # NOTE 7. recovery from prox + Newton failure
    if !newtonproxsuccessful
        @warn "Newton + prox failed, to be implemented" (Fpᶜₖ₊₁ ≤ F(pb, pₖ) - VU.m / (2μₖ) * norm(sᶜₖ₊₁)^2) (norm(sᶜₖ₊₁) ≤ max(1, sqrt(n))*norm(sᶜₖ₊₁))

        xₖ₊₁ = copy(pₖ)
        if performUstep
            # Linesearch on line pₖ → pᶜₖ₊₁ to get an xₖ₊₁ such that F(xₖ₊₁) ≤ F(pₖ)
            # Dummy VU linesearch
            # TODO replace by linesearch for nondummies
            xₖ₊₁ = F(pb, pₖ) < Fpᶜₖ₊₁ ? pₖ : pᶜₖ₊₁
            Fxₖ₊₁ = F(pb, xₖ₊₁)

            # TODO adjust μ

            # center bundle to xₖ₊₁, if different from pₖ
            change_bundle_center!(bundle, xₖ₊₁, Fxₖ₊₁)
        end

        println(" New prox point update")
        ϵᶜₖ₊₁, pᶜₖ₊₁, Fpᶜₖ₊₁, gpᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁, bundleinfo = bundlesubroutine!(state.bundle, pb, μₖ, xₖ₊₁, σₖ, VU.ϵ, haveinv)
        nullsteps = vcat(nullsteps, bundleinfo.phist) # log nullsteps of correction prox-bundle step
        change_bundle_center!(bundle, pₖ, Fpₖ)
    end
    # else
    #     @warn "U-Newton + approximate prox failed to provide sufficient decrease"
    #     xₖ₊₁ = F(pb, pₖ) < Fpᶜₖ₊₁ ? pₖ : pᶜₖ₊₁

    #     state.ϵ, state.p, state.s, state.U, bundleinfo = bundlesubroutine!(state.bundle, pb, μₖ, xₖ₊₁, σₖ, VU.ϵ; printlev = 0)
    #     nullsteps = vcat(nullsteps, bundleinfo.phist) # log nullsteps of correction prox-bundle step
    # end


    (printlev > 2) && printstyled(" 8 memory\n", color = :green)
    # NOTE 8 memory
    state.ϵ, state.p, state.Fp, state.gp, state.s, state.U = ϵᶜₖ₊₁, pᶜₖ₊₁, Fpᶜₖ₊₁, gpᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁
    state.μ = μₖ

    (printlev > 3) && println("- centering bundle at xc from p")
    (printlev > 3) && display(bundle)
    @assert isapprox(Fpᶜₖ₊₁, F(pb, pᶜₖ₊₁))
    change_bundle_center!(bundle, pᶜₖ₊₁, Fpᶜₖ₊₁)
    (printlev > 3) && println("- bundle centered at xc")
    (printlev > 3) && display(bundle)

    iteration_status = iteration_failed
    if !haskey(bundleinfo, :subroutinestatus) # NOTE Only Newton step (!)
        iteration_status = iteration_completed
    elseif bundleinfo.subroutinestatus == :SeriousStepFound # NOTE Newton + bundle
        nullsteps = vcat(nullsteps, bundleinfo.phist) # log nullsteps of prox-bundle step
        iteration_status = iteration_completed
    elseif  bundleinfo.subroutinestatus == :ApproxMinimizerFound # NOTE Global solution found
        iteration_status = problem_solved
    end
    state.k += 1


    state.pprev .= pₖsave
    state.sprev .= sₖsave
    state.nₖ₋₁ = nₖsave

    return (;
            μ = state.μ,
            ϵ̂ = state.ϵ,
            ŝnorm = norm(state.s),
            nnullsteps = haskey(bundleinfo, :nnullsteps) ? bundleinfo.nnullsteps : 0,
            dotsₖNewtonstep,
            nₖ,
            Newtonsteplength,
            nullsteps
    ), iteration_status
end

get_minimizer_candidate(state::VUbundleState) = state.p
