
Base.@kwdef struct NSBFGS <: NonSmoothOptimizer
    ϵ_opt::Float64 = 1e-8
end


Base.@kwdef mutable struct NSBFGSState{Tf}
    x::Vector{Tf}
    ∇f::Vector{Tf}
    ∇f_next::Vector{Tf}
    Hₖ::Matrix{Tf}
end

function initial_state(::NSBFGS, initial_x::Vector{Tf}, pb) where {Tf}
    n = length(initial_x)
    return NSBFGSState(
        x = initial_x,
        ∇f = ∂F_elt(pb, initial_x),
        ∇f_next = zeros(Tf, n),
        Hₖ = Matrix{Tf}(I, n, n),
    )
end


#
### Printing
#
print_header(gs::NSBFGS) = println("**** NSBFGS algorithm")

display_logs_header_post(gs::NSBFGS) = print("⟨yₖ,sₖ⟩    |d|      ||∇f||    nit_ls")

function display_logs_post(os, gs::NSBFGS)
    ai = os.additionalinfo
    @printf "%.3e  %.1e  %.1e  %2i" ai.dot_yₖ_sₖ ai.dnorm ai.∇F_nextnorm ai.ls_niter
end

#
### NSBFGS method
#
function update_iterate!(state, bfgs::NSBFGS, pb)
    iteration_status = iteration_completed

    dₖ = similar(state.∇f)
    x_next = similar(state.x)

    ## 1. Compute descent direction
    @timeit_debug "1. Descent direction" begin
    dₖ .= -1 .* state.Hₖ * state.∇f
    end

    ## 2. Execute linesearch
    @timeit_debug "2. Linesearch" begin
    tₖ, ls_ncalls = linesearch_nsbfgs(pb, state.x, state.∇f, dₖ)
    end

    @timeit_debug "3. Step, subgradient" begin
    x_next .= state.x .+ tₖ .* dₖ
    state.∇f_next .= ∂F_elt(pb, x_next)
    end

    ## 3. Check diff at new point
    @timeit_debug "4. Diff check, stop test" begin
    if !is_differentiable(pb, x_next)
        @warn "Algorithm breaks down (in theory)"
        @warn "Linesearch returned point of nondifferentiability."
        iteration_status = iteration_failed
    end

    if norm(state.∇f_next) ≤ bfgs.ϵ_opt
        iteration_status = problem_solved
    end
    end

    ## 4. Update BFGS matrix
    @timeit_debug "5. BFGS update" begin
    sₖ = tₖ .* dₖ
    yₖ = state.∇f_next - state.∇f

    dot_yₖ_sₖ = dot(yₖ, sₖ)

    if dot_yₖ_sₖ > 0
        # if o.initial_rescale || o.iterative_rescale
        #     o.Hₖ *= dot_yₖ_sₖ / dot(o.Hₖ*yₖ, yₖ)
        #     o.initial_rescale = false
        # end

        uₖ = state.Hₖ * yₖ

        c1 = (dot(yₖ, uₖ) + dot_yₖ_sₖ) / dot_yₖ_sₖ^2
        c2 = 1/dot_yₖ_sₖ

        state.Hₖ = state.Hₖ + c1 * (sₖ*sₖ') - c2 * (sₖ*uₖ' + uₖ*sₖ')
    else
        @warn "No update of BFGS inverse hessian approximation here" dot_yₖ_sₖ
    end
    end


    state.x .= x_next
    state.∇f .= state.∇f_next

    return (;dot_yₖ_sₖ,
            ls_niter = ls_ncalls.it_ls,
            ∇F_nextnorm=norm(state.∇f),
            dnorm=norm(dₖ),
            F = ls_ncalls.F,                # oracle_calls
            ∂F_elt = 1 + ls_ncalls.∂F_elt,
            is_differentiable = 1
            ), iteration_status
end


get_minimizer_candidate(state::NSBFGSState) = state.x
