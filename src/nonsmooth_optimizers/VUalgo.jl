"""
    $TYPEDSIGNATURES

Parameters:
- `ϵ`: overall precision required
- `m`: sufficient decrease parameter
- `μlow`: minimal prox parameter (μ is inverse of γ). Higher μ means smaller serious steps, but less null steps
"""
Base.@kwdef struct VUbundle{Tf} <: NonSmoothOptimizer{Tf}
    μlow::Tf = 0.05
    ϵ::Tf = 1e-12
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
    bundle::Vector{BundlePoint{Tf}}
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
        BundlePoint{Tf}[],
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

    dotsₖNewtonstep = 0.0
    nₖ = size(Uₖ, 2)
    Newtonsteplength = 0.0
    nullsteps = []

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

        sₖ₊₁ = ∂F_elt(pb, xᶜₖ₊₁)
        ys = (; y = Uₖ * Δu, s = sₖ₊₁ - sₖ)
        push!(state.histys, ys)
        push!(nullsteps, xᶜₖ₊₁)

        dotsₖNewtonstep = dot(sₖ, Uₖ * Δu)
        Newtonsteplength = norm(state.histys[end].y)
    end

    # Bundle subroutine at point xᶜₖ₊₁ (ie proximal step approximation)
    ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁, state.bundle, bundleinfo = bundlesubroutine(pb, μₖ, xᶜₖ₊₁, σₖ, VU.ϵ, state.bundle)

    if F(pb, pᶜₖ₊₁) ≤ F(pb, pₖ) - VU.m / (2μₖ) * norm(sᶜₖ₊₁)^2
        state.ϵ, state.p, state.s, state.U = ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁

        # NOTE: Update prox parameter between serious steps. See eq. 10.25, p. 152
        # Bonnans, Gilbert, Lemarechal, Sagastizábal (2006) Numerical Optimization: Theoretical and Practical Aspects, Springer-Verlag.
        Δx = pᶜₖ₊₁ - xᶜₖ₊₁
        Δs = sᶜₖ₊₁ - ∂F_elt(pb, xᶜₖ₊₁)

        μup = inv(1/μₖ + dot(Δx, Δs) / norm(Δs)^2)
        if norm(Δs) < 1e2 * eps(Tf)                 # locally linear functions incur Δs = 0, causing a NaN value.
            μup = Tf(0)
        end
        state.μ = min(10μₖ, max(VU.μlow, 0.1μₖ, μup))

        nullsteps = vcat(nullsteps, bundleinfo.phist) # log nullsteps of prox-bundle step
    else
        # Linesearch on line pₖ → pᶜₖ₊₁ to get an xₖ₊₁ such that F(xₖ₊₁) ≤ F(pₖ)
        @warn "U-Newton + approximate prox failed to provide sufficient decrease"
        xₖ₊₁ = F(pb, pₖ) < F(pb, pᶜₖ₊₁) ? pₖ : pᶜₖ₊₁

        state.ϵ, state.p, state.s, state.U, state.bundle, bundleinfo = bundlesubroutine(pb, μₖ, xₖ₊₁, σₖ, VU.ϵ, state.bundle; printlev = 0)
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
