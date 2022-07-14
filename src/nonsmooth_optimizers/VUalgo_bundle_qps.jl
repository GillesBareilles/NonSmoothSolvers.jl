################################################################################
## Primal bundle step
################################################################################
abstract type AbstractχQPSolver end
struct χOSQP <: AbstractχQPSolver end
struct χCHP <: AbstractχQPSolver end
struct χQPSimplex <: AbstractχQPSolver end

function solve_χQP(pb, μ, x, bundle)
    # α̂  = solve_χQP(μ, bundle, χOSQP())
    α̂  = solve_χQP(μ, bundle, χQPSimplex())

    p̂ = x - (1/μ) * sum(α̂[i] * bndlelt.gᵢ for (i, bndlelt) in enumerate(bundle))
    r̂ = F(pb, x) + maximum(-bndl.eᵢ + dot(bndl.gᵢ, p̂ - x) for (i, bndl) in enumerate(bundle))

    return r̂, p̂, α̂
end

function solve_χQP(μ, bundle, ::χOSQP)
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

    α̂ = value.(α)
    return α̂
end

function solve_χQP(μ::Tf, bundle, ::χQPSimplex) where Tf
    nbundle = length(bundle)
    n = length(first(bundle).gᵢ)
    P = zeros(Tf, n, nbundle)
    for (i, bundleelt) in enumerate(bundle)
        P[:, i] = bundleelt.gᵢ
    end
    b = μ * [bundleelt.eᵢ for bundleelt in bundle]

    α̂ = qpsimplex(P, b; check_optimality = false)
    return α̂
end

################################################################################
## Dual bundle step
################################################################################

abstract type AbstractγQPSolver end
struct γOSQP <: AbstractγQPSolver end
struct γNPP <: AbstractγQPSolver end

function solve_γQP(activebundle)
    # α̂  = solve_γQP(activebundle, γOSQP())
    α̂  = solve_γQP(activebundle, γNPP())
    ŝ = sum(α̂[i] * bndlelt.gᵢ for (i, bndlelt) in enumerate(activebundle))

    return ŝ, α̂
end

function solve_γQP(activebundle, ::γOSQP)
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
    return α̂ = value.(α)
end

function solve_γQP(activebundle, ::γNPP)
    nbundle = length(activebundle)
    n = length(first(activebundle).gᵢ)
    P = zeros(n, nbundle)
    for (i, bundleelt) in enumerate(activebundle)
        P[:, i] = bundleelt.gᵢ
    end
    return α̂ = nearest_point_polytope(P; show_trace = false)
end
