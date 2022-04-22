"""
    $TYPEDSIGNATURES

Gradient sampling algorthm.
"""
Base.@kwdef struct GradientSampling{Tf} <: NonSmoothOptimizer{Tf}
    m::Int64
    β::Tf = 1e-4
    γ::Tf = 0.5
    ϵ_opt::Tf = 1e-6
    ν_opt::Tf = 1e-6
    θ_ϵ::Tf = 0.1
    θ_ν::Tf = 0.1
    ls_maxit::Int64 = 70
    nppopt::NearestPointPolytope{Tf} = NearestPointPolytope()
end

GradientSampling(initial_x::AbstractVector) = GradientSampling(m = length(initial_x) * 2)

Base.@kwdef mutable struct GradientSamplingState{Tf} <: OptimizerState{Tf}
    x::Vector{Tf}
    ∂gᵢs::Matrix{Tf}
    ϵₖ::Tf
    νₖ::Tf
    nppstate::NearestPointPolytopeState{Tf}
    k::Int64 = 1
end

function initial_state(gs::GradientSampling{Tf}, initial_x::Vector{Tf}, pb) where {Tf}
    ∂gᵢs = zeros(Tf, length(initial_x), gs.m + 1)
    return GradientSamplingState(;
        x = initial_x,
        ∂gᵢs,
        nppstate = initial_state(∂gᵢs),
        ϵₖ = Tf(0.1),
        νₖ = Tf(0.1),
    )
end


#
### Printing
#
print_header(gs::GradientSampling) = println("**** GradientSampling algorithm\nm = $(gs.m)")

display_logs_header_post(gs::GradientSampling) =
    print("||gᵏ||     ϵₖ       νₖ        it_ls")
function display_logs_post(os, gs::GradientSampling)
    @printf "%.3e  %.1e  %.1e  %2i" os.additionalinfo.gᵏ_norm os.additionalinfo.ϵₖ os.additionalinfo.νₖ os.additionalinfo.it_ls
end


#
### GradientSampling method
#
"""
    update_iterate!(state, gs::GradientSampling, pb)

NOTE: each iteration is costly. This can be explored with NonSmoothProblems.to.
On the maxquadBGLS problem
 ────────────────────────────────────────────────────────────────────────────────────────────────────
                                                            Time                    Allocations
                                                   ───────────────────────   ────────────────────────
                 Tot / % measured:                      103ms /  93.9%           9.50MiB /  96.1%

 Section                                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────────────────────
 update_iterate!                              100   95.6ms   98.7%   956μs   8.92MiB   97.8%  91.4KiB
   GS 2. minimum norm (sub)gradient           100   83.3ms   86.0%   833μs   3.56MiB   39.0%  36.4KiB
   GS 1. sampling points, eval gradients      100   7.39ms    7.6%  73.9μs   3.85MiB   42.2%  39.4KiB
   GS 4. Update parameters                    100   4.24ms    4.4%  42.4μs   1.40MiB   15.3%  14.3KiB
   GS 5. diff check                           100    288μs    0.3%  2.88μs    117KiB    1.3%  1.17KiB
   GS 3. Termination                          100   41.1μs    0.0%   411ns     0.00B    0.0%    0.00B
 build_optimstate                             100   1.19ms    1.2%  11.9μs    207KiB    2.2%  2.07KiB
 CV check                                     100   47.0μs    0.0%   470ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────────────────────
"""
function update_iterate!(
    state::GradientSamplingState{Tf},
    gs::GradientSampling,
    pb,
) where {Tf}
    iteration_status = iteration_completed
    ∂gᵢs = state.∂gᵢs

    ## 1. Sample m points in 𝔹(x, ϵₖ)
    samplegradients!(∂gᵢs, pb, state.x, state.ϵₖ)

    ## 2. Find minimal norm element of convex hull at gradients of previous points.
    # alphaOSQP = find_minimumnormelt_OSQP(∂gᵢs)
    # gᵏ = ∂gᵢs * alphaOSQP
    # alphaWolfe = nearest_point_polytope(∂gᵢs)
    # gᵏ = ∂gᵢs * alphaWolfe

    initialize_state!(state.nppstate, ∂gᵢs)
    nearest_point_polytope!(state.nppstate, gs.nppopt, ∂gᵢs; show_trace = false)
    gᵏ = state.nppstate.x

    gᵏ_norm = norm(gᵏ)


    ## 3. termination
    if gᵏ_norm ≤ gs.ν_opt && state.ϵₖ ≤ gs.ϵ_opt
        iteration_status = problem_solved
        @info "termination condition satisfied"
    end

    ## 4. Update parameters
    ν_next = 0.
    ϵ_next = 0.
    tₖ = 1.0
    it_ls = 0
    if gᵏ_norm ≤ state.νₖ
        ν_next = gs.θ_ν * state.νₖ
        ϵ_next = gs.θ_ϵ * state.ϵₖ
        tₖ = 0.0
        @info "reducing sampling size"
    else
        # This test is not costly, and may help detect difficult cases
        gtd = dot(gᵏ, ∂F_elt(pb, state.x))
        if gtd <= 0
            @warn "not descent direction, gtd = $gtd"
        end

        ν_next = state.νₖ
        ϵ_next = state.ϵₖ

        fₖ = F(pb, state.x)
        while !(F(pb, state.x - tₖ * gᵏ) < fₖ - gs.β * tₖ * gᵏ_norm^2)
            tₖ *= gs.γ
            it_ls += 1

            if it_ls > gs.ls_maxit
                @warn "GradientSampling(): linesearch exceeded $(gs.ls_maxit) iterations, no suitable steplength found."
                break
            end
            if (norm(gᵏ) * tₖ < 10 * eps(Tf))
                @warn("Linesearch reached numerical precision")
                break
            end
        end
    end

    x_next = state.x - tₖ * gᵏ
    if !is_differentiable(pb, x_next)
        @error(
            "Gradient sampling: F not differentiable at next point, portion to be implemented."
        )
    end

    state.ϵₖ = ϵ_next
    state.νₖ = ν_next
    state.x = x_next
    state.k += 1

    return (;
        ϵₖ = state.ϵₖ,
        νₖ = state.νₖ,
        it_ls,
        gᵏ_norm,
        F = 2 + it_ls,              # orcale calls
        ∂F_elt = gs.m + 1,
        is_differentiable = 1,
    ),
    iteration_status
end

get_minimizer_candidate(state::GradientSamplingState) = state.x


function find_minimumnormelt_CHP(∂gᵢs::Matrix{Tf}) where {Tf}
    set = CHP.SimplexShadow(∂gᵢs)
    x0 = zeros(Tf, size(∂gᵢs, 2))

    showtermination = true
    showtrace = true
    showls = false
    α, str = CHP.optimize(set, x0; showtermination, showtrace, showls, maxiter = 13)
    return α
end

function samplegradients!(∂gᵢs, pb, x, ϵₖ)
    n = size(∂gᵢs, 1)
    nsamples = size(∂gᵢs, 2) - 1

    for i = 1:nsamples
        ∂gᵢ = @view ∂gᵢs[:, i]
        ∂gᵢ .= rand(MvNormal(zeros(n), ScalMat(n, 1.0)))
        ∂gᵢ .*= ϵₖ * rand()^(1 / n) / norm(∂gᵢ)
        ∂gᵢ .+= x
        ∂gᵢ .= ∂F_elt(pb, ∂gᵢ)
    end
    ∂gᵢs[:, end] .= ∂F_elt(pb, x)
    return
end
