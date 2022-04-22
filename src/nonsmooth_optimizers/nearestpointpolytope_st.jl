Base.@kwdef struct NearestPointPolytope{Tf} <: Optimizer{Tf}
    Z₁::Tf = 1e-10
    Z₂::Tf = 1e-10
    Z₃::Tf = 1e-10
end

Base.@kwdef mutable struct NearestPointPolytopeState{Tf} <: OptimizerState{Tf}
    w::Vector{Tf}
    x::Vector{Tf}
    S::Set{Int64} = Set{Int64}()
    it::Int64 = 1
    norm2Pᵢs::Vector{Tf}
end

function initial_state(P)
    n, m = size(P)
    return NearestPointPolytopeState(w = zeros(m), x = zeros(n), norm2Pᵢs = zeros(m))
end
function initialize_state!(state, P)
    # for i in axes(P, 2)
    #     state.norm2Pᵢs[i] = norm(P[:, i])^2
    # end

    j = argmin(state.norm2Pᵢs)
    state.w .= 0
    state.w[j] = 1
    state.x .= P * state.w
    empty!(state.S)
    push!(state.S, j)
    return nothing
end


#
### Printing
#
print_header(::NearestPointPolytope) = println("**** NearestPointPolytope algorithm")
display_logs_header(::NearestPointPolytope) =
    print("it.   time      F(x)                     step       \n")
function display_logs(state, ::NearestPointPolytope; time_count)
    @printf "%4i  %.1e  %s\n" state.it time_count collect(state.S)
end

function nearest_point_polytope(P; show_trace = false)
    state = initial_state(P)
    initialize_state!(state, P)
    o = NearestPointPolytope()
    nearest_point_polytope!(state, o, P; show_trace)
    return get_minimizer_candidate(state)
end

function nearest_point_polytope!(
    state,
    optimizer,
    P;
    show_trace = false,
    iterations_limit = 10,
)
    time_count = 0.0
    iteration = 0
    converged = false
    stopped = false

    show_trace && print_header(optimizer)
    show_trace && display_logs_header(optimizer)

    while !converged && !stopped && iteration < iterations_limit
        iteration += 1
        _time = time()
        iterationstatus = update_iterate!(state, optimizer, P)
        time_count += time() - _time

        show_trace && display_logs(state, optimizer; time_count)
        stopped = (iterationstatus == iteration_failed)
        converged = (iterationstatus == problem_solved)
    end
end

#
### NearestPointPolytope method
#
raw"""
    $TYPEDSIGNATURES

## TODOs
- save `w` as a sparse vector?
- better heuristic of selection for j
- better LP solve strategy

## Benchmark
```
julia> @benchmark NSS.nearest_point_polytope(P)
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  32.888 μs …  7.961 ms  ┊ GC (min … max): 0.00% … 99.20%
 Time  (median):     37.116 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   41.617 μs ± 79.586 μs  ┊ GC (mean ± σ):  1.90% ±  0.99%

     █▂
  ▃▄▅██▅▅▃▃▂▂▂▂▃▃▃▄▄▄▄▄▄▄▄▄▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂ ▃
  32.9 μs         Histogram: frequency by time        68.4 μs <

 Memory estimate: 23.05 KiB, allocs estimate: 125.
```
"""
function update_iterate!(state, npp::NearestPointPolytope{Tf}, P) where {Tf}
    state.x .= P * state.w
    x = state.x
    S = state.S
    w = state.w
    state.it += 1

    # NOTE: this requires a full matrix vector product. That's a lot.
    # There is probably better than this rule, see Wolfe, Note 1
    state.norm2Pᵢs .= P' * x
    j = argmin(state.norm2Pᵢs)

    # NOTE: We don't follow the theoretical stopping condition as it is too costly.
    if dot(x, P[:, j]) >
        norm(x)^2 - npp.Z₁ * max(norm(P[:, j]), maximum(norm(P[:, i]) for i in S))^2
    # if dot(x, P[:, j]) > prevfloat(norm(x)^2)
    # if state.norm2Pᵢs[j] > norm(x)^2 - 1e3 * eps(Tf)
        # Optimality condition met, problem solved
        return problem_solved
    end
    if j ∈ S
        @info "disaster, stopping" state.norm2Pᵢs[j] norm(x)^2 dot(x, x)

        # To quote Wolfe: "disaster happened", stopping
        return iteration_failed
    end

    push!(S, j)
    w[j] = 0

    innerit = 0
    while true
        # Step 2

        # NOTE: this is another clear place where we coudl do better.
        v = solveLP(P, S)

        if sum(v .> npp.Z₂) == length(v)
            # Point in the ri of current convex hull
            w[collect(S)] .= v
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
        innerit += 1
        innerit > 10 && @assert false
    end


    return iteration_completed
end

function solveLP(P, S)
    p = length(S)
    A = ones(p + 1, p + 1)
    A[1, 1] = 0
    Pₛ = @view P[:, collect(S)]
    A[2:end, 2:end] .= Pₛ' * Pₛ
    b = zeros(p + 1)
    b[1] = 1
    res = A \ b
    v = res[2:end]
    return v
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
    println(
        "Max{|xᵀPⱼ - xᵀx|, j∈S} \t",
        maximum([abs(dot(x, pb.P[:, j]) - dot(x, x)) for j in S]),
    )
    println(
        "Min{xᵀPⱼ - xᵀx, j}     \t",
        minimum([dot(x, pb.P[:, j]) - dot(x, x) for j in axes(pb.P, 2)]),
    )
    return
end

get_minimizer_candidate(state::NearestPointPolytopeState) = state.w
