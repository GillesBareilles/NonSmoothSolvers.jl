struct Polytope{Tm}
    P::Tm
end
F(pb::Polytope, w) = norm(pb.P * w)


Base.@kwdef struct NearestPointPolytope{Tf} <: Optimizer{Tf}
    Z₁::Tf = 1e-10
    Z₂::Tf = 1e-10
    Z₃::Tf = 1e-10
end

Base.@kwdef mutable struct NearestPointPolytopeState{Tf} <: OptimizerState{Tf}
    w::Vector{Tf}
    x::Vector{Tf}
    S::BitArray
    it::Int64 = 1
    norm2Pᵢs::Vector{Tf}
end

function initial_state(::NearestPointPolytope, initial_w, pb)
    norm2Pᵢs = [norm(pb.P[:, i])^2 for i in axes(pb.P, 2)]
    j = argmin(norm2Pᵢs)

    m = size(pb.P, 2)
    S = BitArray(zeros(m))
    S[j] = 1

    # Sa = BitVector(zeros(m))
    # Sa[j] = 1
    w = zeros(m)
    w[j] = 1
    return NearestPointPolytopeState(;
        w,
        x = pb.P * initial_w,
        S,
        norm2Pᵢs,
    )
end


#
### Printing
#
print_header(::NearestPointPolytope) = println("**** NearestPointPolytope algorithm")
display_logs_header_common(::NearestPointPolytope) =
    print("it.   time      F(x)                     step       ")

display_logs_common(os, ::NearestPointPolytope) =
    @printf "%4i  %.1e  % .16e  % .3e  " os.it os.time os.Fx os.norm_step


function build_initoptimstate(
    state,
    optimizer::NearestPointPolytope{Tf},
    pb;
    optimstate_extensions,
) where {Tf}
    return OptimizationState(
        it = 0,
        time = 0.0,
        Fx = F(pb, get_minimizer_candidate(state)),
        norm_step = Tf(0.0),
        ncalls_F = 0,
        ncalls_∂F_elt = 0,
        additionalinfo = NamedTuple(
            indname => indcallback(optimizer, state) for
            (indname, indcallback) in optimstate_extensions
        ),
    )
end

#
### NearestPointPolytope method
#
function update_iterate!(state, npp::NearestPointPolytope{Tf}, pb) where Tf
    state.x .= pb.P * state.w
    x = state.x
    S = state.S
    P = pb.P
    w = state.w


    # j = argmin([dot(P[:, i], x) for i in axes(P, 2)])
    j = argmin(P' * x)

    if dot(x, P[:, j]) >
        norm(x)^2 - npp.Z₁ * max(state.norm2Pᵢs[j], maximum(state.norm2Pᵢs[S]))

        return NamedTuple(), problem_solved
    end
    if j ∈ S
        @info "disaster, stopping"
        return NamedTuple(), iteration_failed
    end

    # push!(S, j)
    S[j] = 1
    w[j] = 0

    removed_pts = SortedSet{Int64}()
    innerit = 0
    while true
        # Step 2
        p = sum(S)

        A = ones(p + 1, p + 1)
        A[1, 1] = 0
        Pₛ = @view P[:, S]
        A[2:end, 2:end] .= Pₛ' * Pₛ
        b = zeros(p + 1)
        b[1] = 1
        res = A \ b
        v = res[2:end]


        if sum(v .> npp.Z₂) == length(v)
            w[S] .= v
            @debug "Point in the ri of current convex hull"
            break
        end

        # Step 3
        wₛ = @view w[collect(S)]
        POS = filter(i -> wₛ[i] > v[i] + npp.Z₃, 1:length(S))
        θ = min(1, minimum(i -> wₛ[i] / (wₛ[i] - v[i]), POS))
        @. wₛ = (1 - θ) * wₛ + θ * v

        w[w.<npp.Z₂] .= 0
        k::Int64 = findfirst(i -> (w[i] == 0), collect(S))
        k = collect(S)[k]
        delete!(S, k)
        push!(removed_pts, k)
        innerit += 1
        innerit > 10 && @assert false
    end

    return NamedTuple(), iteration_completed
end

function display_optimizerstatus(
    pb,
    ::NearestPointPolytope,
    state,
    initial_x,
    stopped_by_updatefailure,
    stopped_by_time_limit,
    iteration,
    time_count,
) where {Tf}
    x_final = get_minimizer_candidate(state)
    println("
* status:
    final point value:      $(F(pb, x_final))
    stopped by it failure:  $(stopped_by_updatefailure)
    stopped by time:        $(stopped_by_time_limit)
* Counters:
    Iterations:  $iteration
    Time:        $time_count")

    S = state.S
    w = state.w
    x = state.x
    println("S                    \t", collect(S))
    println("1 - eᵀw               \t", sum(w) - 1)
    println("|x - P*w|             \t", norm(x - pb.P * w))
    println("Max{|xᵀPⱼ - xᵀx|, j∈S} \t", maximum([ abs(dot(x, pb.P[:, j]) - dot(x, x)) for j in S ]))
    println("Min{xᵀPⱼ - xᵀx, j}     \t", minimum([ dot(x, pb.P[:, j]) - dot(x, x) for j in axes(pb.P, 2) ]))
    return
end

get_minimizer_candidate(state::NearestPointPolytopeState) = state.w
