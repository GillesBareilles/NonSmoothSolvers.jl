
Base.@kwdef struct GradientSampling <: NonSmoothOptimizer
    m::Int64
    β::Float64 = 1e-4
    γ::Float64 = 0.5
    ϵ_opt::Float64 = 1e-2
    ν_opt::Float64 = 1e-6
    θ_ϵ::Float64 = 0.1
    θ_ν::Float64 = 0.1
    ls_maxit::Int64 = 50
end

GradientSampling(initial_x::AbstractVector) = GradientSampling(m=length(initial_x)+1)

Base.@kwdef mutable struct GradientSamplingState{Tx}
    x::Tx
    xs::Vector{Tx}
    ϵₖ::Float64 = 0.1
    νₖ::Float64 = 0.1
    k::Int64 = 1
end

function initial_state(gs::GradientSampling, initial_x, pb)
    return GradientSamplingState(
        x = initial_x,
        xs = Vector([initial_x for i in 1:gs.m]),
    )
end


#
### Printing
#
print_header(gs::GradientSampling) = println("**** GradientSampling algorithm\nm = $(gs.m)")

display_logs_header_post(gs::GradientSampling) = print("||gᵏ||     ϵₖ       νₖ        it_ls")
function display_logs_post(os, gs::GradientSampling)
    @printf "%.3e  %.1e  %.1e  %2i" os.additionalinfo.gᵏ_norm os.additionalinfo.ϵₖ os.additionalinfo.νₖ os.additionalinfo.it_ls
end


#
### GradientSampling method
#
function update_iterate!(state, gs::GradientSampling, pb)
    iteration_status = iteration_completed

    ## 1. Sample m points in 𝔹(x, ϵₖ)
    Random.seed!(123 + state.k)
    for i in 1:gs.m
        state.xs[i] .= rand(Normal(), size(state.x))
        state.xs[i] .*= rand()^(1/length(state.x)) / norm(state.xs[i])
        state.xs[i] .*= state.ϵₖ
    end

    ## 2. Find minimal norm element of convex hull at gradients of previous points.
    model = Model(with_optimizer(OSQP.Optimizer; polish=true, verbose=false, max_iter=1e8, eps_abs=1e-8, eps_rel=1e-8))

    # TODO: check complexity of this part
    t = @variable(model, 0 <= t[1:gs.m+1] <= 1)
    gconvhull = t[end] .* ∂F_elt(pb, state.x) .+ sum(t[i] .* ∂F_elt(pb, state.xs[i]) for i in 1:gs.m)
    @objective(model, Min, dot(gconvhull, gconvhull))
    @constraint(model, sum(t) == 1)

    optimize!(model)

    if termination_status(model) ∉ Set([MOI.OPTIMAL, MOI.SLOW_PROGRESS])
        @warn "ComProx: subproblem of it $n was not solved to optimality" termination_status(model) primal_status(model) dual_status(model)
    end

    gᵏ = value.(gconvhull)
    gᵏ_norm = norm(gᵏ)

    ## 3. termination
    if gᵏ_norm ≤ gs.ν_opt && state.ϵₖ ≤ gs.ϵ_opt
        iteration_status = problem_solved
    end

    ## 4. Update parameters
    ν_next = state.νₖ
    ϵ_next = state.ϵₖ
    tₖ = 1.0
    it_ls = 0
    if gᵏ_norm ≤ state.νₖ
        ν_next = gs.θ_ν * state.νₖ
        ϵ_next = gs.θ_ϵ * state.ϵₖ
        tₖ = 0.0
    else
        ν_next = state.νₖ
        ϵ_next = state.ϵₖ
        tₖ = 1.0

        fₖ = F(pb, state.x)
        while !(F(pb, state.x - tₖ * gᵏ) < fₖ - gs.β * tₖ * gᵏ_norm^2) && (it_ls < gs.ls_maxit)
            tₖ *= gs.γ
            it_ls += 1
        end

        if it_ls == gs.ls_maxit
            @warn("GradientSampling(): linesearch exceeded $(gs.ls_maxit) iterations, no suitable steplength found.")
        end
    end

    x_next = state.x - tₖ * gᵏ
    if !is_differentiable(pb, x_next)
        @warn("Gradient sampling: F not differentiable at next point, portion to be imùplemented.")
        # throw(error("Gradient sampling: F not differentiable at next point, portion to be imùplemented."))
    end

    state.ϵₖ = ϵ_next
    state.νₖ = ν_next
    state.x = x_next
    state.k += 1

    return (ϵₖ=state.ϵₖ, νₖ=state.νₖ, it_ls=it_ls, gᵏ_norm=gᵏ_norm), iteration_status
end



get_minimizer_candidate(state::GradientSamplingState) = state.x