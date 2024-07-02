#
## Print and logs
#
display_logs_header_pre(o) = nothing
display_logs_header_common(o::To) where {Tf,To<:NonSmoothOptimizer{Tf}} =
    print("it.   time      F(x)                     step       ")
display_logs_header_post(o) = nothing

function display_logs_header(o::Optimizer, pb)
    display_logs_header_pre(o)
    display_logs_header_common(o)
    display_logs_header_post(o)
    println()
    return
end


display_logs_pre(os, o) = nothing
display_logs_common(os, o::To) where {To<:NonSmoothOptimizer} =
    @printf "%4i  %.1e  % .16e  % .3e  " os.it os.time os.Fx os.norm_step
display_logs_post(os, o) = nothing

function display_logs(os::OptimizationState, optimizer)
    display_logs_pre(os, optimizer)
    display_logs_common(os, optimizer)
    display_logs_post(os, optimizer)

    println()
end

function build_optimstate(
    state,
    optimizer,
    pb,
    it,
    time,
    x_prev,
    optimstate_additionalinfo;
    optimstate_extensions = OrderedDict{Symbol,Function}(),
)
    osextensions = NamedTuple(
        indname => indcallback(optimizer, state, optimstate_additionalinfo) for
        (indname, indcallback) in optimstate_extensions
    )

    return OptimizationState(
        it = it,
        time = time,
        Fx = F(pb, get_minimizer_candidate(state)),
        norm_step = norm(x_prev - get_minimizer_candidate(state)),
        ncalls_F = 0,
        ncalls_∂F_elt = 1,
        additionalinfo = merge(optimstate_additionalinfo, osextensions),
    )
end

function build_initoptimstate(
    state::Ts,
    optimizer::To,
    pb;
    optimstate_extensions,
) where {Tf,Ts<:OptimizerState{Tf},To<:NonSmoothOptimizer{Tf}}
    return OptimizationState(
        it = 0,
        time = 0.0,
        Fx = F(pb, get_minimizer_candidate(state)),
        norm_step = Tf(0.0),
        ncalls_F = 0,
        ncalls_∂F_elt = 0,
        additionalinfo = NamedTuple(
            indname => indcallback(optimizer, state, nothing) for
            (indname, indcallback) in optimstate_extensions
        ),
    )
end

@enum IterationStatus begin
    iteration_completed
    iteration_failed
    problem_solved
end

"""
    $TYPEDSIGNATURES

Call the `optimizer` on problem `pb`, with initial point `initial_x`. Returns a
tuple containing the final iterate vector and a trace.

Features:
- timing of the `update_iterate` method only;
- saves basic information of each iteration in a vector of `OptimizationState`,
  the so-called trace;
- the information saved at each iterate may be enriched by the user by providing
  a name and callback function via the `optimstate_extension` argument.

### Example
```julia
getx(o, os) = os.x
optimstate_extensions = OrderedDict{Symbol, Function}(:x => getx)

optimize!(pb, o, xclose; optparams, optimstate_extensions)
```
"""
function optimize!(
    pb,
    optimizer::O,
    initial_x;
    state = nothing,
    optimstate_extensions::OrderedDict{Symbol,Function} = OrderedDict{Symbol,Function}(),
    optparams = OptimizerParams(),
) where {O<:Optimizer}

    if getfield(NonSmoothSolvers, :timeit_debug_enabled)()
        reset_timer!()
    end

    ## Collecting parameters
    iterations_limit = optparams.iterations_limit
    show_trace = optparams.show_trace
    show_final_status = show_trace

    isnothing(state) && (state = initial_state(optimizer, copy(initial_x), pb))
    iteration = 0
    converged = false
    stopped = false
    time_count = 0.0
    stopped_by_updatefailure = false
    stopped_by_iterationpbsolved = false
    stopped_by_time_limit = false

    x_prev = copy(initial_x)

    show_trace && print_header(optimizer)
    show_trace && display_logs_header(optimizer, pb)

    tr = Vector{OptimizationState}([
        build_initoptimstate(state, optimizer, pb; optimstate_extensions),
    ])

    if show_trace
        display_logs_common(tr[1], optimizer)
        println()
    end

    while !converged && !stopped && iteration < iterations_limit
        iteration += 1

        _time = time()
        # @timeit_debug "update_iterate!" optimstate_additionalinfo, iterationstatus = update_iterate!(state, optimizer, pb)
        optimstate_additionalinfo, iterationstatus = update_iterate!(state, optimizer, pb)
        time_count += time() - _time

        # @timeit_debug "build_optimstate" begin
        optimizationstate = build_optimstate(
            state,
            optimizer,
            pb,
            iteration,
            time_count,
            x_prev,
            optimstate_additionalinfo;
            optimstate_extensions,
        )
        push!(tr, optimizationstate)
        # end

        ## Display logs and save iteration information
        if show_trace && (
            mod(iteration, ceil(iterations_limit / optparams.trace_length)) == 0 ||
            iteration == iterations_limit
        )
            display_logs(optimizationstate, optimizer)
        end

        stopped_by_updatefailure = (iterationstatus == iteration_failed)
        stopped_by_iterationpbsolved = (iterationstatus == problem_solved)
        stopped_by_time_limit = time_count > optparams.time_limit
        stopped = stopped_by_time_limit || stopped_by_updatefailure
        x_prev .= get_minimizer_candidate(state)
    end

    x_final = get_minimizer_candidate(state)

    # Display status of optimizer:
    show_final_status && display_optimizerstatus(
        pb,
        optimizer,
        state,
        initial_x,
        stopped_by_updatefailure,
        stopped_by_time_limit,
        stopped_by_iterationpbsolved,
        iteration,
        time_count,
    )


    if getfield(NonSmoothSolvers, :timeit_debug_enabled)()
        printstyled("\n\n")
        print_timer()
        printstyled("\n\n")
    end

    return x_final, tr
end

function display_optimizerstatus(
    pb,
    ::To,
    state,
    initial_x,
    stopped_by_updatefailure,
    stopped_by_time_limit,
    stopped_by_iterationpbsolved,
    iteration,
    time_count,
) where {Tf,To<:NonSmoothOptimizer{Tf}}
    x_final = get_minimizer_candidate(state)
    println("
* status:
    initial point value:    $(F(pb, initial_x))
    final point value:      $(F(pb, x_final))
    optimality condition:   $(stopped_by_iterationpbsolved)
    stopped by it failure:  $(stopped_by_updatefailure)
    stopped by time:        $(stopped_by_time_limit)
* Counters:
    Iterations:  $iteration
    Time:        $time_count")
    return
end
