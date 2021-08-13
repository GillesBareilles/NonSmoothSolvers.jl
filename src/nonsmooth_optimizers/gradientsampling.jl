
Base.@kwdef struct GradientSampling <: NonSmoothOptimizer
    m::Int64
    β::Float64 = 1e-4
    γ::Float64 = 0.5
    ϵ_opt::Float64 = 1e-3
    ν_opt::Float64 = 1e-5
    θ_ϵ::Float64 = 0.1
    θ_ν::Float64 = 0.1
    ls_maxit::Int64 = 70
end

GradientSampling(initial_x::AbstractVector) = GradientSampling(m=length(initial_x)+1)

Base.@kwdef mutable struct GradientSamplingState{Tx}
    x::Tx
    xs::Vector{Tx}
    ϵₖ::Float64 = 1e-1
    νₖ::Float64 = 0.1
    k::Int64 = 1
end

function initial_state(gs::GradientSampling, initial_x, pb)
    return GradientSamplingState(
        x = copy(initial_x),
        xs = Vector([zeros(size(initial_x)) for i in 1:gs.m]),
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
"""
    update_iterate!(state, gs::GradientSampling, pb)

NOTE: each iteration is costly. This can be explored with NonSmoothProblems.to.
───────────────────────────────────────────────────────────────────────────────────────────
Time                   Allocations
──────────────────────   ───────────────────────
Tot / % measured:                  45.5s / 2.89%           1.63GiB / 14.5%

Section                            ncalls     time   %tot     avg     alloc   %tot      avg
───────────────────────────────────────────────────────────────────────────────────────────
GS 2. minimum norm (sub)gradient       20    1.30s  98.6%  64.9ms    240MiB  99.0%  12.0MiB
GS 4. Update parameters                20   12.8ms  0.97%   641μs   1.63MiB  0.67%  83.6KiB
GS 5. diff check                       20   3.04ms  0.23%   152μs    428KiB  0.17%  21.4KiB
GS 1. point sampling                   20   2.49ms  0.19%   124μs    497KiB  0.20%  24.9KiB
GS 3. Termination                      20   33.8μs  0.00%  1.69μs      320B  0.00%    16.0B
───────────────────────────────────────────────────────────────────────────────────────────
"""
function update_iterate!(state, gs::GradientSampling, pb)
    iteration_status = iteration_completed

    @timeit_debug "GS 1. point sampling" begin
    ## 1. Sample m points in 𝔹(x, ϵₖ)
    Random.seed!(123 + state.k)
    for i in 1:gs.m
        state.xs[i] .= rand(Normal(), size(state.x))
        state.xs[i] .*= rand()^(1/length(state.x)) / norm(state.xs[i])
        state.xs[i] .*= state.ϵₖ
        state.xs[i] .+= state.x
    end
    end

    ## 2. Find minimal norm element of convex hull at gradients of previous points.
    @timeit_debug "GS 2. minimum norm (sub)gradient" begin
    model = Model(with_optimizer(Ipopt.Optimizer; print_level=0))

    t = @variable(model, 0 <= t[1:gs.m+1] <= 1)
    gconvhull = t[end] .* ∂F_elt(pb, state.x) .+ sum(t[i] .* ∂F_elt(pb, state.xs[i]) for i in 1:gs.m)
    @objective(model, Min, dot(gconvhull, gconvhull))
    @constraint(model, sum(t) == 1)

    optimize!(model)

    if termination_status(model) ∉ Set([MOI.OPTIMAL, MOI.SLOW_PROGRESS, MOI.LOCALLY_SOLVED])
        @warn "ComProx: subproblem was not solved to optimality" termination_status(model) primal_status(model) dual_status(model)
    end

    gᵏ = value.(gconvhull)
    end
    gᵏ_norm = norm(gᵏ)

    ## 3. termination
    @timeit_debug "GS 3. Termination" begin
    if gᵏ_norm ≤ gs.ν_opt && state.ϵₖ ≤ gs.ϵ_opt
        iteration_status = problem_solved
    end
    end

    ## 4. Update parameters
    @timeit_debug "GS 4. Update parameters" begin
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
    end

    @timeit_debug "GS 5. diff check" begin
    x_next = state.x - tₖ * gᵏ
    if !is_differentiable(pb, x_next)
        @warn("Gradient sampling: F not differentiable at next point, portion to be implemented.")
    end

    state.ϵₖ = ϵ_next
    state.νₖ = ν_next
    state.x = x_next
    state.k += 1
    end

    return (ϵₖ=state.ϵₖ, νₖ=state.νₖ, it_ls=it_ls, gᵏ_norm=gᵏ_norm), iteration_status
end



get_minimizer_candidate(state::GradientSamplingState) = state.x
