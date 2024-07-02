#
## Print and logs
#
# function build_optimstate(
#     state,
#     optimizer,
#     pb,
#     it,
#     time,
#     x_prev,
#     optimstate_additionalinfo;
#     optimstate_extensions = OrderedDict{Symbol,Function}(),
# )
#     osextensions = NamedTuple(
#         indname => indcallback(optimizer, state, optimstate_additionalinfo) for
#         (indname, indcallback) in optimstate_extensions
#     )

#     return OptimizationState(
#         it = it,
#         time = time,
#         Fx = F(pb, get_minimizer_candidate(state)),
#         norm_step = norm(x_prev - get_minimizer_candidate(state)),
#         ncalls_F = 0,
#         ncalls_∂F_elt = 1,
#         additionalinfo = merge(optimstate_additionalinfo, osextensions),
#     )
# end

function build_initoptimstate(
    state::Ts,
    optimizer::To,
    pb,
    optimstate_extensions
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


abstract type AbstractTraceStrategy end
struct DefaultTraceStrategy <: AbstractTraceStrategy end

struct DefaultTraceItem{Tx, Tfx}
    x::Tx
    Fx::Tfx
    it::Int64
    time::Float64
end
function build_inittrace(::DefaultTraceStrategy, state)
    return DefaultTraceItem(get_minimizer_candidate(state), get_minval_candidate(state), 0, 0.0)
end
function build_traceitem(::DefaultTraceStrategy, o, state, iteration, time_count)
    return DefaultTraceItem(get_minimizer_candidate(state), get_minval_candidate(state), iteration, time_count)
end
# TODO: passer les structs en NamedTuple


"""

TODO: ENTRY POINT
"""
function optimize(
    pb,
    optimizer::O,
    initial_x::Tx;
    optparams = OptimizerParams(),
    tracestrategy = DefaultTraceStrategy(),
) where {O<:Optimizer, Tx}
    state = initial_state(optimizer, copy(initial_x), pb)
    return optimize!(state, pb, optimizer, optparams, tracestrategy)
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
    state,
    pb,
    optimizer,
    optparams,
    tracestrategy,
)
    # if getfield(NonSmoothSolvers, :timeit_debug_enabled)()
    #     reset_timer!()
    # end

    x_prev = get_minimizer_candidate(state)
    x_init = get_minimizer_candidate(state)

    ## Collecting parameters
    iterations_limit = optparams.iterations_limit
    show_trace = optparams.show_trace
    show_final_status = show_trace

    iteration = 0
    converged = false
    stopped = false
    time_count = 0.0
    stopped_by_updatefailure = false
    stopped_by_iterationpbsolved = false
    stopped_by_time_limit = false

    show_trace && print_header(optimizer)
    show_trace && display_logs_header(optimizer, pb)

    tr_ = [
        build_inittrace(tracestrategy, state)
    ]

    show_trace && display_logs(state, iteration, time_count)

    while !converged && !stopped && iteration < iterations_limit
        iteration += 1

        _time = time()
        # @timeit_debug "update_iterate!" updateinformation, iterationstatus = update_iterate!(state, optimizer, pb)
        updateinformation, iterationstatus = update_iterate!(state, optimizer, pb)
        time_count += time() - _time

        # @timeit_debug "build_optimstate" begin
        traceitem = build_traceitem(tracestrategy, optimizer, state, iteration, time_count)
        push!(tr_, traceitem)
        # end

        ## Display logs and save iteration information
        if show_trace && (
            mod(iteration, ceil(iterations_limit / optparams.trace_length)) == 0 ||
            iteration == iterations_limit
        )
            display_logs(state, updateinformation, iteration, time_count)
        end

        # @timeit_debug "CV check" begin
            converged = (iterationstatus == problem_solved)
            for cvchecker in optparams.cvcheckers
                converged = converged || hasconverged(cvchecker, pb, optimizer, state, updateinformation)
            end
        # end

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
        x_init,
        stopped_by_updatefailure,
        stopped_by_time_limit,
        stopped_by_iterationpbsolved,
        iteration,
        time_count,
    )


    # if getfield(NonSmoothSolvers, :timeit_debug_enabled)()
    #     printstyled("\n\n")
    #     print_timer()
    #     printstyled("\n\n")
    # end

    return x_final, tr_
end

function display_optimizerstatus(
    pb,
    ::To,
    state,
    x_init,
    stopped_by_updatefailure,
    stopped_by_time_limit,
    stopped_by_iterationpbsolved,
    iteration,
    time_count,
) where {Tf,To<:NonSmoothOptimizer{Tf}}
    x_final = get_minimizer_candidate(state)
    println("
* status:
    initial point value:    $(F(pb, x_init))
    final point value:      $(F(pb, x_final))
    optimality condition:   $(stopped_by_iterationpbsolved)
    stopped by it failure:  $(stopped_by_updatefailure)
    stopped by time:        $(stopped_by_time_limit)
* Counters:
    Iterations:  $iteration
    Time:        $time_count")
    return
end
