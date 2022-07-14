"""
    $TYPEDSIGNATURES

Parameters:
- `σ`: in (0, 0.5!], lower values enforce higher precision on each prox point approximation,
- `ϵ`: overall precision required
- `m`: sufficient decrease parameter
- `μlow`: minimal prox parameter (μ is inverse of γ). Higher μ means smaller serious steps, but less null steps
"""
Base.@kwdef struct VUbundle{Tf} <: NonSmoothOptimizer{Tf}
    μlow::Tf = 0.5
    σ::Tf = 0.1
    ϵ::Tf = 1e-10
    m::Tf = 0.5
    Newton_accel::Bool = true
end

Base.@kwdef mutable struct VUbundleState{Tf} <: OptimizerState{Tf}
    p::Vector{Tf}                                                       # point
    s::Vector{Tf}                                                       # minimal norm subgradient of the current ϵ subdifferential
    ϵ::Tf                                                               # ?
    U::Matrix{Tf}                                                       # orthonormal basis of the approximation of the current U space
    k::Int64 = 1                                                        # iteration counter
    histys::Vector{NamedTuple{(:y, :s), Tuple{Vector{Tf}, Vector{Tf}}}} # history of BFGS steps and subgradients difference
end

function initial_state(::VUbundle{Tf}, initial_x::Vector{Tf}, pb) where Tf
    return VUbundleState(
        p = initial_x,
        s = ∂F_elt(pb, initial_x),
        ϵ = Tf(1.0),
        U = Matrix{Tf}(1.0I, length(initial_x), 1),
        histys = NamedTuple{(:y, :s), Tuple{Vector{Tf}, Vector{Tf}}}[],
    )
end


#
### Printing
#
print_header(::VUbundle) = println("**** VUbundle algorithm")

display_logs_header_post(gs::VUbundle) = print("ϵ̂        |ŝ|          #nullsteps     nₖ   ⟨dNewton, sₖ⟩ |dNewton|")

function display_logs_post(os, gs::VUbundle)
    ai = os.additionalinfo
    @printf "%.2e %.2e     %-2i             %-2i   % .1e  %.2e" ai.ϵ̂  ai.ŝnorm ai.nnullsteps ai.nₖ ai.dotsₖNewtonstep ai.Newtonsteplength
end

#
### VUbundle method
#
function update_iterate!(state, VU::VUbundle{Tf}, pb) where Tf
    ϵₖ = state.ϵ
    pₖ = state.p
    sₖ = state.s
    Uₖ = state.U

    dotsₖNewtonstep = 0.0
    nₖ = size(Uₖ, 2)
    Newtonsteplength = 0.0

    xᶜₖ₊₁ = pₖ
    if VU.Newton_accel
        # Computing U-Hessian estimate
        Hₖ = LBFGSOperator(Tf, nₖ, mem = 10)
        for (y, s) in state.histys
            push!(Hₖ, Uₖ' * s, Uₖ' * y)
        end

        # Solving Newton equation
        Δu = -Hₖ * Uₖ' * sₖ
        xᶜₖ₊₁ = pₖ + Uₖ * Δu

        sₖ₊₁ = ∂F_elt(pb, xᶜₖ₊₁)
        ys = (; y = Uₖ * Δu, s = sₖ₊₁ - sₖ)
        push!(state.histys, ys)

        dotsₖNewtonstep = dot(sₖ, Uₖ * Δu)
        Newtonsteplength = norm(state.histys[end].y)
    end

    # μₖ₊₁ = VU.μlow # prox parameter
    μₖ₊₁ = Tf(3.0) # prox parameter

    # Bundle subroutine at point xᶜₖ₊₁ (ie proximal step approximation)
    ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁, bundleinfo = bundlesubroutine(pb, μₖ₊₁, xᶜₖ₊₁, VU.σ, VU.ϵ)

    if F(pb, pᶜₖ₊₁) ≤ F(pb, pₖ) - VU.m / (2μₖ₊₁) * norm(sᶜₖ₊₁)^2
        state.ϵ, state.p, state.s, state.U = ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁
    else
        # Linesearch on line pₖ → pᶜₖ₊₁ to get an xₖ₊₁ such that F(xₖ₊₁) ≤ F(pₖ)
        @warn "U-Newton + approximate prox failed to provide sufficient decrease"
        xₖ₊₁ = F(pb, pₖ) < F(pb, pᶜₖ₊₁) ? pₖ : pᶜₖ₊₁

        state.ϵ, state.p, state.s, state.U, bundleinfo = bundlesubroutine(pb, μₖ₊₁, xₖ₊₁, VU.σ, VU.ϵ; printlev = 0)
    end

    iteration_status = iteration_completed
    if norm(state.s)^2 ≤ VU.ϵ && state.ϵ ≤ VU.ϵ
        @info "problem solved" norm(state.s)^2 state.ϵ
        iteration_status = problem_solved
    end
    state.k += 1

    return (;
            ϵ̂ = state.ϵ,
            ŝnorm = norm(state.s),
            bundleinfo.nnullsteps,
            dotsₖNewtonstep,
            nₖ,
            Newtonsteplength,
    ), iteration_status
end

get_minimizer_candidate(state::VUbundleState) = state.p
