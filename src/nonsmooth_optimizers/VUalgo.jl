struct VUbundle{Tf} <: NonSmoothOptimizer{Tf}
    μlow::Tf                # Prox parameter (inverse of γ)
    ϵ::Tf                   # Stopping criterion on the norm of `s`
    m::Tf                   # parameter of sufficient decrease for accepting U-Newton iterate
    Newton_accel::Bool
end
function VUbundle(; μlow = 0.5, ϵ = 1e-3, m = 0.5, Newton_accel = false)
    return VUbundle(μlow, ϵ, m, Newton_accel)
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
        ϵ = 1.0,
        U = Matrix(1.0I, length(initial_x), 1),
        histys = NamedTuple{(:y, :s), Tuple{Vector{Tf}, Vector{Tf}}}[],
    )
end


#
### Printing
#
print_header(::VUbundle) = println("**** VUbundle algorithm")

display_logs_header_post(gs::VUbundle) = print("#nullsteps     nₖ   ⟨dNewton, sₖ⟩  |dNewton|")

function display_logs_post(os, gs::VUbundle)
    ai = os.additionalinfo
    @printf "%-2i             %-2i   % .1e  %2e" ai.nnullsteps ai.nₖ ai.dotsₖstep ai.Newtonsteplength
end

#
### VUbundle method
#
function update_iterate!(state, o::VUbundle{Tf}, pb) where Tf
    μlow = 0.5
    σ = 1e-5
    ϵ = 1e-3
    m = 0.5

    ϵₖ = state.ϵ
    pₖ = state.p
    sₖ = state.s
    Uₖ = state.U

    # @show pₖ
    xᶜₖ₊₁ = pₖ
    if o.Newton_accel
        # Computing U-Hessian estimate
        nₖ = size(Uₖ, 2)
        Hₖ = Matrix{Tf}(1.0I, nₖ, nₖ)

        # InverseLBFGSOperator
        Hₖ = InverseLBFGSOperator(Tf, nₖ)
        for (i, ys) in enumerate(state.histys)
            y, s = ys
            push!(Hₖ, Uₖ' * s, Uₖ' * y)
        end

        # Solving Newton equation
        Δu = - Hₖ * Uₖ' * sₖ
        xᶜₖ₊₁ = pₖ + Uₖ * Δu

        sₖ₊₁ = ∂F_elt(pb, xᶜₖ₊₁)
        ys = (;y = Uₖ * Δu, s = sₖ₊₁ - sₖ)
        push!(state.histys, ys)
    end
    # @show xᶜₖ₊₁

    μₖ₊₁ = μlow # prox parameter
    μₖ₊₁ = 3.0 # prox parameter

    # Bundle subroutine at point xᶜₖ₊₁
    # aka proximal step approximation
    ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁, bundleinfo = bundlesubroutine(pb, μₖ₊₁, xᶜₖ₊₁, σ)


    if F(pb, pᶜₖ₊₁) ≤ F(pb, pₖ) - m / (2μₖ₊₁) * norm(sᶜₖ₊₁)^2
        ϵₖ₊₁, pₖ₊₁, sₖ₊₁, Uₖ₊₁ = ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁

        state.ϵ = ϵₖ₊₁
        state.p = pₖ₊₁
        state.s = sₖ₊₁
        state.U = Uₖ₊₁
    else
        # Linesearch on line pₖ → pᶜₖ₊₁ to get an xₖ₊₁ such that F(xₖ₊₁) ≤ F(pₖ)
        @warn "U-Newton + approximate prox failed to provide sufficient decrease"
        xₖ₊₁ = F(pb, pₖ) < F(pb, pᶜₖ₊₁) ? pₖ : pᶜₖ₊₁

        ϵₖ₊₁, pₖ₊₁, sₖ₊₁, Uₖ₊₁, bundleinfo = bundlesubroutine(pb, μₖ₊₁, xₖ₊₁, σ)

        state.ϵ = ϵₖ₊₁
        state.p = pₖ₊₁
        state.s = sₖ₊₁
        state.U = Uₖ₊₁
    end


    state.k += 1

    return (;
            bundleinfo.nnullsteps,
            dotsₖstep = dot(sₖ, Uₖ * Δu),
            nₖ,
            Newtonsteplength = norm(state.histys[end].y),
    ), iteration_completed
end


using JuMP, OSQP


get_minimizer_candidate(state::VUbundleState) = state.p
