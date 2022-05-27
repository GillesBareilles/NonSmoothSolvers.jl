using ConvexHullProjection

import ConvexHullProjection: f, ∇f!, ∇²f!, g, prox_γg!

raw"""
    SimplexShadow{Tf}
Models the convex hull of vectors gs.
"""
struct PrimalQPPb{Tf} <: CHP.StructuredSet{Tf}
    A::Matrix{Tf}
    b::Vector{Tf}
    μ::Tf
end

f(ch::PrimalQPPb, α, ::CHP.AmbRepr) = 1/(2 * ch.μ) * norm(ch.A * α)^2 + dot(ch.b, α)

∇f!(res, ch::PrimalQPPb, α, ::CHP.AmbRepr) = (res .= (1/ch.μ) .* ch.A' * ch.A * α .+ ch.b)

∇²f!(res, ch::PrimalQPPb, x, d, ::CHP.AmbRepr) = (res .= (1/ch.μ) .* ch.A' * ch.A * d)

g(::PrimalQPPb, α, ::CHP.AmbRepr) = sum(α) == 1 && sum(α .>= 0) == length(α)


"""
    prox_γg!(res, ch, α)
Computes the prox of the indicator of the simplex, which amounts to projecting onto the simplex.
"""
function prox_γg!(res, ::PrimalQPPb{Tf}, α, ::CHP.AmbRepr) where Tf
    M = CHP.project_simplex!(res, α)
    return M
end



################################################################################
## Primal bundle step
################################################################################
function solve_χQP(pb, μ, x, bundle)
    nbundle = length(bundle)

    # model = Model(optimizer_with_attributes(OSQP.Optimizer,
    #                                         "eps_abs" => 1e-12,
    #                                         "eps_rel" => 1e-12,
    #                                         "polish" => true
    #                                         ))
    # set_silent(model)

    # α = @variable(model, α[1:nbundle])

    # # Simplex constraints
    # @constraint(model, α .≥ 0)
    # @constraint(model, sum(α) == 1.)

    # # Objective
    # eᵢαᵢ = sum(bundleelt.eᵢ * α[i] for (i, bundleelt) in enumerate(bundle))
    # gᵢαᵢ = sum(bundleelt.gᵢ * α[i] for (i, bundleelt) in enumerate(bundle))
    # @objective(model, Min, 1/(2μ) * dot(gᵢαᵢ, gᵢαᵢ) + eᵢαᵢ)
    # JuMP.optimize!(model)
    # if termination_status(model) ∉ Set([MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
    #     @warn "solve_χQP: subproblem was not solved to optimality" termination_status(model) primal_status(model) dual_status(model)
    # end

    #     @show termination_status(model)
    #     @show primal_status(model)
    #     @show dual_status(model)

    # α̂ = value.(α)


    nbundle = length(bundle)
    n = length(first(bundle).gᵢ)
    P = zeros(n, nbundle)
    for (i, bundleelt) in enumerate(bundle)
        P[:, i] = bundleelt.gᵢ
    end
    b = [bundleelt.eᵢ for bundleelt in bundle]
    χpb = PrimalQPPb(P, b, μ)

    α̂, manifold = optimize(χpb, ones(nbundle)./nbundle;
                             showtermination = false,
                             showtrace = false,
                             newtonaccel = false,
                             showls = false)

    p̂ = x - (1/μ) * sum(α̂[i] * bndlelt.gᵢ for (i, bndlelt) in enumerate(bundle))
    r̂ = F(pb, x) + maximum(-bndl.eᵢ + dot(bndl.gᵢ, p̂ - x) for (i, bndl) in enumerate(bundle))

    return r̂, p̂, α̂
end

################################################################################
## Dual bundle step
################################################################################
function solve_γQP(activebundle)
    nbundle = length(activebundle)

    # model = Model(optimizer_with_attributes(OSQP.Optimizer,
    #                                         "eps_abs" => 1e-12,
    #                                         "eps_rel" => 1e-12,
    #                                         "polish" => true
    #                                         ))
    # set_silent(model)
    # α = @variable(model, α[1:nbundle])

    # # Simplex constraints
    # @constraint(model, α .≥ 0)
    # @constraint(model, sum(α) == 1.)

    # # Objective
    # gᵢαᵢ = sum(bundleelt.gᵢ * α[i] for (i, bundleelt) in enumerate(activebundle))
    # @objective(model, Min, 1/2 * dot(gᵢαᵢ, gᵢαᵢ))
    # JuMP.optimize!(model)
    # if termination_status(model) ∉ Set([MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
    #     @warn "solve_γQP: subproblem was not solved to optimality" termination_status(model) primal_status(model) dual_status(model)
    # end
    # JuMP.optimize!(model)
    # α̂ = value.(α)

    #### Solve with Wolfe's algo
    nbundle = length(activebundle)
    n = length(first(activebundle).gᵢ)
    P = zeros(n, nbundle)
    for (i, bundleelt) in enumerate(activebundle)
        P[:, i] = bundleelt.gᵢ
    end
    α̂ = nearest_point_polytope(P; show_trace = false)

    ŝ = sum(α̂[i] * bndlelt.gᵢ for (i, bndlelt) in enumerate(activebundle))

    return ŝ, α̂
end
