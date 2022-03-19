struct VUalg{Tf} <: NonSmoothOptimizer
    μlow::Tf                # Prox parameter (inverse of γ)
    ϵ::Tf                   # Stopping criterion on the norm of `s`
    m::Tf                   # parameter of sufficient decrease for accepting U-Newton iterate
    Newton_accel::Bool
end
function VUalg(; μlow = 0.5, ϵ = 1e-3, m = 0.5, Newton_accel = false)
    return VUalg(μlow, ϵ, m, Newton_accel)
end

Base.@kwdef mutable struct VUalgState{Tf}
    p::Vector{Tf}           # point
    s::Vector{Tf}           # minimal norm subgradient of the current ϵ subdifferential
    ϵ::Tf                   # ?
    U::Matrix{Tf}           # orthonormal basis of the approximation of the current U space
    k::Int64 = 1            # iteration counter
end

function initial_state(::VUalg, initial_x, pb)
    return VUalgState(
        p = initial_x,
        s = ∂F_elt(pb, initial_x),
        ϵ = 1.0,
        U = Matrix(1.0I, length(initial_x), 1)
    )
end


#
### Printing
#
print_header(::VUalg) = println("**** VUalg algorithm")


#
### VUalg method
#
function update_iterate!(state, opt::VUalg, pb)
    Tf = Float64

    μlow = 0.5
    σ = 1e-5
    ϵ = 1e-3
    m = 0.5

    ϵₖ = state.ϵ
    pₖ = state.p
    sₖ = state.s
    Uₖ = state.U

    xᶜₖ₊₁ = pₖ
    if opt.Newton_accel
        # Computing U-Hessian estimate
        nₖ = size(Uₖ, 2)
        Hₖ = Matrix{Tf}(1.0I, nₖ, nₖ)

        # Solving Newton equation
        Δu = - inv(Hₖ) * Uₖ' * sₖ

        xᶜₖ₊₁ = pₖ + Uₖ * Δu
    end

    μₖ₊₁ = μlow # prox parameter

    # Bundle subroutine at point xᶜₖ₊₁
    # aka proximal step approximation
    ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁ = bundlesubroutine(pb, μₖ₊₁, xᶜₖ₊₁, σ)


    if F(pb, pᶜₖ₊₁) ≤ F(pb, pₖ) - m / (2μₖ₊₁) * norm(sᶜₖ₊₁)^2
        @info "U-Newton + approximate prox achieved sufficient decrease"
        ϵₖ₊₁, pₖ₊₁, sₖ₊₁, Uₖ₊₁ = ϵᶜₖ₊₁, pᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁

        state.ϵ = ϵₖ₊₁
        state.p = pₖ₊₁
        state.s = sₖ₊₁
        state.U = Uₖ₊₁
    else
        # Linesearch on line pₖ → pᶜₖ₊₁ to get an xₖ₊₁ such that F(xₖ₊₁) ≤ F(pₖ)
        @info "U-Newton + approximate prox failed to provide sufficient decrease"
        xₖ₊₁ = F(pb, pₖ) < F(pb, pᶜₖ₊₁) ? pₖ : pᶜₖ₊₁

        ϵₖ₊₁, pₖ₊₁, sₖ₊₁, Uₖ₊₁ = bundlesubroutine(pb, μₖ₊₁, xₖ₊₁, σ)

        state.ϵ = ϵₖ₊₁
        state.p = pₖ₊₁
        state.s = sₖ₊₁
        state.U = Uₖ₊₁
    end


    state.k += 1

    return NamedTuple(), iteration_completed
end


function bundlepoint(pb, yᵢ, x)
    fᵢ = F(pb, yᵢ)
    gᵢ = ∂F_elt(pb, yᵢ)
    eᵢ = F(pb, x) - fᵢ - dot(gᵢ, x - yᵢ)

    return (; fᵢ, gᵢ, eᵢ, yᵢ)
end

raw"""
    $TYPEDSIGNATURES

Compute one serious step of the proximal bundle algorithm:
$\arg\min_z F(z) + 0.5 * μ \|z - x\|^2$.
Return ...
"""
function bundlesubroutine(pb, μ, x, σ)
    bundle = [bundlepoint(pb, x, x)]
    μ = 1
    σ = 1e-5

    α̂minnormelt = nothing
    activebundle = nothing

    ϵ̂ = Inf
    p̂ = similar(x)
    ŝ = similar(x)

    @printf "it |B| |Bact|          |ŝ|        |ϵ̂|  tol(μ, σ)\n"
    it = 0
    while true
        r̂, p̂, α̂ = solve_χQP(pb, μ, x, bundle)
        ϵ̂ = F(pb, p̂) - r̂

        activebundle = form_active_bundle(pb, bundle, α̂, p̂, x, r̂)

        ŝ, α̂minnormelt = solve_γQP(activebundle)

        @printf "%2i  %2i  %2i        %.2e  % .2e  %.2e\n" it length(bundle) length(activebundle) norm(ŝ) ϵ̂ (σ / μ * norm(ŝ)^2)
        bundle = deepcopy(activebundle)

        (ϵ̂ < σ / μ * norm(ŝ)^2) && break
        it += 1
        it > 50 && break
    end

    Û = get_Uorthonormalbasis(activebundle, α̂minnormelt)
    return ϵ̂, p̂, ŝ, Û
end

using JuMP, OSQP

function get_Uorthonormalbasis(activebundle, α̂minnormelt)
    n = length(first(activebundle).gᵢ)
    actindices = α̂minnormelt .> 1e-13
    nactgᵢ = sum(actindices)

    if nactgᵢ == 1
        return zeros(n, 0)
    end

    actgᵢs = map(bndl -> bndl.gᵢ, activebundle[actindices])

    V̂ = zeros(n, nactgᵢ-1)
    for i in 1:nactgᵢ-1
        V̂[:, i] .= actgᵢs[i] - actgᵢs[end]
    end

    @show rank(V̂)

    Û = nullspace(V̂')
    @show size(Û)
    return Û
end

function solve_χQP(pb, μ, x, bundle)
    nbundle = length(bundle)

    model = Model(optimizer_with_attributes(OSQP.Optimizer,
                                            "eps_abs" => 1e-12,
                                            "eps_rel" => 1e-12,
                                            "polish" => true
                                            ))
    set_silent(model)

    α = @variable(model, α[1:nbundle])

    # Simplex constraints
    @constraint(model, α .≥ 0)
    @constraint(model, sum(α) == 1.)

    # Objective
    eᵢαᵢ = sum(bundleelt.eᵢ * α[i] for (i, bundleelt) in enumerate(bundle))
    gᵢαᵢ = sum(bundleelt.gᵢ * α[i] for (i, bundleelt) in enumerate(bundle))
    @objective(model, Min, 1/(2μ) * dot(gᵢαᵢ, gᵢαᵢ) + eᵢαᵢ)
    JuMP.optimize!(model)
    if termination_status(model) ∉ Set([MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
        @warn "solve_χQP: subproblem was not solved to optimality" termination_status(model) primal_status(model) dual_status(model)
    end
    #TODO: check optimality of solution

    α̂ = value.(α)
    p̂ = x - (1/μ) * sum(α̂[i] * bndlelt.gᵢ for (i, bndlelt) in enumerate(bundle))
    r̂ = F(pb, x) + maximum(-bndl.eᵢ + dot(bndl.gᵢ, p̂ - x) for (i, bndl) in enumerate(bundle))
    return r̂, p̂, α̂
end

function form_active_bundle(pb, bundle, α̂, p̂, x, r̂)
    actbndl = []
    for (i, bndlelt) in enumerate(bundle)
        αcond = α̂[i] > 1e-14
        modelcond = norm(r̂ - (F(pb, x) - bndlelt.eᵢ  + dot(bndlelt.gᵢ, p̂ - x))) < 1e-13
        # @show αcond, modelcond

        if αcond
            push!(actbndl, bndlelt)
        end
    end
    push!(actbndl, bundlepoint(pb, p̂, x))
    return actbndl
end

function solve_γQP(activebundle)
    nbundle = length(activebundle)

    model = Model(optimizer_with_attributes(OSQP.Optimizer,
                                            "eps_abs" => 1e-12,
                                            "eps_rel" => 1e-12,
                                            "polish" => true
                                            ))
    set_silent(model)
    α = @variable(model, α[1:nbundle])

    # Simplex constraints
    @constraint(model, α .≥ 0)
    @constraint(model, sum(α) == 1.)

    # Objective
    gᵢαᵢ = sum(bundleelt.gᵢ * α[i] for (i, bundleelt) in enumerate(activebundle))
    @objective(model, Min, 1/2 * dot(gᵢαᵢ, gᵢαᵢ))
    JuMP.optimize!(model)
    if termination_status(model) ∉ Set([MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
        @warn "solve_γQP: subproblem was not solved to optimality" termination_status(model) primal_status(model) dual_status(model)
    end

    JuMP.optimize!(model)

    α̂ = value.(α)
    ŝ = sum(α̂[i] * bndlelt.gᵢ for (i, bndlelt) in enumerate(activebundle))

    return ŝ, α̂
end


get_minimizer_candidate(state::VUalgState) = state.p
