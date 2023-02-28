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
    k::Int64                                                            # iteration counter
    μ::Tf                                                               # step of proximal point
    bundle::Bundle{Tf}
    histys::Vector{NamedTuple{(:y, :s), Tuple{Vector{Tf}, Vector{Tf}}}} # history of BFGS steps and subgradients difference
end

function initial_state(::VUbundle{Tf}, initial_x::Vector{Tf}, pb) where Tf
    return VUbundleState(
        initial_x,
        ∂F_elt(pb, initial_x),
        Tf(1),
        Tf(0.5),
        Matrix{Tf}(1.0I, length(initial_x), 1),
        1,
        Tf(4),
        initial_bundle(pb, initial_x),
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
    sₖ = state.s
    Uₖ = state.U
    μₖ = state.μ
    σₖ = Tf(inv(1+state.k^2))
    bundle = state.bundle

    dotsₖNewtonstep = 0.0
    nₖ = size(Uₖ, 2)
    Newtonsteplength = 0.0
    nullsteps = []

    println("\n\n))) iteration ", state.k)
    println("p :")
    display(state.p)

    xᶜₖ₊₁ = pₖ
    if VU.Newton_accel
        # Computing U-Hessian estimate
        Hₖ = LBFGSOperator(Tf, nₖ, mem = 10)
        for (y, s) in state.histys
            push!(Hₖ, Uₖ' * s, Uₖ' * y)
        end

        # Solving Newton equation
        Δu = -Hₖ * Uₖ' * sₖ

        dᴺ = Uₖ * Δu
        xᶜₖ₊₁ = pₖ .+ dᴺ
        # NOTE: missing extrapolation search here

        sₖ₊₁ = ∂F_elt(pb, xᶜₖ₊₁)
        ys = (; y = Uₖ * Δu, s = sₖ₊₁ - sₖ)
        push!(state.histys, ys)
        push!(nullsteps, xᶜₖ₊₁)

        dotsₖNewtonstep = dot(sₖ, Uₖ * Δu)
        Newtonsteplength = norm(state.histys[end].y)

        println()
        @show state.k
        @show xᶜₖ₊₁
        if state.k == 1
            @info "here"
            xᶜₖ₊₁ .= [9.0000000000000002e-01, 6.4999999999999991e-01]

            add_point!(bundle, pb, [9.0000000000000002e-01, -3.3500000000000001e+00])
            add_point!(bundle, pb, xᶜₖ₊₁)
        elseif  state.k == 2
        end

        println("Bundle post Newton:")
        display(state.bundle)
    end

    # NOTE: hack
    σₖ = 0.5
    haveinv = 1.
    if state.k == 1
        μₖ = 1.3375
        haveinv = 0.25
    elseif  state.k == 2
        μₖ = 1.3374999999999999e+00
        haveinv = 2.1389999379348708e+00
    elseif  state.k == 3
        μₖ 	= 7.8495667032262716e+01
        haveinv 	= 2.2842965561311965e+00
    end
    # NOTE: change bundle center from p to xᶜ
    @show state.p, xᶜₖ₊₁


    println("Bundle uncentered:")
    display(bundle)

    p = state.p
    Fxᶜₖ₊₁ = F(pb, xᶜₖ₊₁)
    Fp = F(pb, p)
    bundle.refpoint .= xᶜₖ₊₁
    bundle.Frefpoint = Fxᶜₖ₊₁
    for i in axes(bundle.bpts, 1)
        b = bundle.bpts[i]
        b.eᵢ += Fxᶜₖ₊₁ - Fp + dot(b.gᵢ, p - xᶜₖ₊₁)
    end
    # @show p
    # @show xᶜₖ₊₁
    # @show Fxᶜₖ₊₁
    # @show Fp
    println("Bundle recentered:")
    display(bundle)

    if state.k == 2
        fais = [
            0.0000000000000000e+00 9.0000000000000002e-01 7.9547668694123619e-01 2.0416632517768041e-02
            1.0000000000000000e+00 -1.0976635514018693e+00 -8.5404716909616285e-01 -1.1865857444625247e+00
        ]
        erlin = [3.9078702838504970e-01,3.9078702838504986e-01,3.5564999603479913e-01,-5.5511151231257827e-17]
        center = [2.0416632517768041e-02,-1.8658574446252479e-01]
        bundle.refpoint .= center
        bundle.Frefpoint = F(pb, center)
        bundle.bpts = [BundlePoint(-1., fais[:, i], erlin[i], [-1., -1.]) for i in 1:4]
        xᶜₖ₊₁ .= center
    elseif  state.k == 3
        fais = [
            0.0000000000000000e+00 1.4941260361303003e-02 1.3436738181096714e-02 -1.9096585001088240e-03
            1.0000000000000000e+00 -9.9996264219874642e-01 -9.9995542892987366e-01 -1.0000585339847392e+00
        ]
        erlin = [1.1889308038565537e-04 1.4198133085420927e-04 1.1776126087468309e-04 0.0000000000000000e+00]
        center = [-1.9096585001088240e-03,-5.8533984739249321e-05]
        bundle.refpoint .= center
        bundle.Frefpoint = F(pb, center)
        bundle.bpts = [BundlePoint(-1., fais[:, i], erlin[i], [-1., -1.]) for i in 1:4]
        xᶜₖ₊₁ .= center
    end

    # Bundle subroutine at point xᶜₖ₊₁ (ie proximal step approximation)
    ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁, bundleinfo = bundlesubroutine!(state.bundle, pb, μₖ, xᶜₖ₊₁, σₖ, VU.ϵ, haveinv)
    println("prox bundle output")
    @show pᶜₖ₊₁
    @show sᶜₖ₊₁
    @show ϵᶜₖ₊₁
    display(Uᶜₖ₊₁)
    if state.k == 3
        @assert false
    end

    # NOTE: change bundle center from xᶜ to p
    println("Bundle uncentered:")
    display(bundle)

    p = state.p
    Fxᶜₖ₊₁ = F(pb, xᶜₖ₊₁)
    Fp = F(pb, p)
    bundle.refpoint .= p
    bundle.Frefpoint = Fp
    for i in axes(bundle.bpts, 1)
        b = bundle.bpts[i]
        b.eᵢ += Fp - Fxᶜₖ₊₁ + dot(b.gᵢ, xᶜₖ₊₁ - p)
    end

    # @show p
    # @show xᶜₖ₊₁
    # @show Fxᶜₖ₊₁
    # @show Fp
    println("Bundle recentered:")
    display(bundle)

    newtonproxsuccessful =
        (F(pb, pᶜₖ₊₁) ≤ F(pb, pₖ) - VU.m / (2μₖ) * norm(sᶜₖ₊₁)^2) && norm(sᶜₖ₊₁) ≤ max(1, sqrt(size(state.p, 1)))*norm(sᶜₖ₊₁) # ? delete 2nd if k=1 or 2 ?

    if newtonproxsuccessful
    # if F(pb, pᶜₖ₊₁) ≤ F(pb, pₖ) - VU.m / (2μₖ) * norm(sᶜₖ₊₁)^2

        # NOTE center bundle at phat, from p
        p = state.p
        Fpᶜₖ₊₁ = F(pb, pᶜₖ₊₁)
        Fp = F(pb, p)
        bundle.refpoint .= pᶜₖ₊₁
        bundle.Frefpoint = Fpᶜₖ₊₁
        for i in axes(bundle.bpts, 1)
            b = bundle.bpts[i]
            b.eᵢ += Fpᶜₖ₊₁ - Fp + dot(b.gᵢ, p - pᶜₖ₊₁)
            # erlin=erlin+fphat-fp+fais'*(p-phat) #% center at phat(becomes next p)
        end

        state.ϵ, state.p, state.s, state.U = ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁


        # # NOTE: Update prox parameter between serious steps. See eq. 10.25, p. 152
        # # Bonnans, Gilbert, Lemarechal, Sagastizábal (2006) Numerical Optimization: Theoretical and Practical Aspects, Springer-Verlag.
        # Δx = pᶜₖ₊₁ - xᶜₖ₊₁
        # Δs = sᶜₖ₊₁ - ∂F_elt(pb, xᶜₖ₊₁)

        # μup = inv(1/μₖ + dot(Δx, Δs) / norm(Δs)^2)
        # if norm(Δs) < 1e2 * eps(Tf)                 # locally linear functions incur Δs = 0, causing a NaN value.
        #     μup = Tf(0)
        # end
        # state.μ = min(10μₖ, max(VU.μlow, 0.1μₖ, μup))

        nullsteps = vcat(nullsteps, bundleinfo.phist) # log nullsteps of prox-bundle step
    else
        # Linesearch on line pₖ → pᶜₖ₊₁ to get an xₖ₊₁ such that F(xₖ₊₁) ≤ F(pₖ)
        @warn "U-Newton + approximate prox failed to provide sufficient decrease"
        xₖ₊₁ = F(pb, pₖ) < F(pb, pᶜₖ₊₁) ? pₖ : pᶜₖ₊₁

        state.ϵ, state.p, state.s, state.U, bundleinfo = bundlesubroutine!(state.bundle, pb, μₖ, xₖ₊₁, σₖ, VU.ϵ; printlev = 0)
        nullsteps = vcat(nullsteps, bundleinfo.phist) # log nullsteps of correction prox-bundle step
    end

    iteration_status = iteration_completed
    if norm(state.s)^2 ≤ VU.ϵ && state.ϵ ≤ VU.ϵ
        @info "problem solved" norm(state.s)^2 state.ϵ
        iteration_status = problem_solved
    end
    state.k += 1

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
