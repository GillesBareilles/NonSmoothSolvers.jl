#
## Print and logs
#
display_logs_header_pre(o) = nothing
display_logs_header_post(o) = nothing

function display_logs_header(o::Optimizer, pb::NSP.NonSmoothPb)
    display_logs_header_pre(o::Optimizer)
    print("it.   time      F(x)                     step       ")
    display_logs_header_post(o::Optimizer)
    println()
    return
end


display_logs_post(os, optimizer) = nothing
function display_logs(os::OptimizationState, optimizer)
    print("\033[0m")

    @printf "%4i  %.1e  % .16e  % .3e  " os.it os.time os.F_x os.norm_step
    display_logs_post(os, optimizer)

    print("\033[0m")
    println()
end

function build_optimstate(state, optimizer, pb, it, time, x_prev, optimstate_additionalinfo)
    ## TODO: add possibility for user to log additional information contained in state into the NamedTuple.
    return OptimizationState(
        it = it,
        time = time,
        F_x = F(pb, get_minimizer_candidate(state)),
        norm_step = norm(x_prev - get_minimizer_candidate(state)),
        ncalls_F = 0,
        ncalls_∂F_elt = 1,
        additionalinfo = optimstate_additionalinfo,
    )
end

@enum IterationStatus begin
    iteration_completed
    iteration_failed
    problem_solved
end

function optimize!(
    pb,
    optimizer::O,
    initial_x;
    state = nothing,
    optimstate_extensions = [],
    optparams = OptimizerParams()
) where {O<:Optimizer}

    if getfield(NonSmoothSolvers, :timeit_debug_enabled)()
        reset_timer!()
    end

    ## Collecting parameters
    iterations_limit = optparams.iterations_limit
    show_trace = optparams.show_trace

    isnothing(state) && (state = initial_state(optimizer, copy(initial_x), pb))
    iteration = 0
    converged = false
    stopped = false
    time_count = 0.0

    x_prev = copy(initial_x)

    show_trace && print_header(optimizer)
    show_trace && display_logs_header(optimizer, pb)
    tr = Vector{OptimizationState}([OptimizationState(F_x = F(pb, initial_x))])

    if show_trace
        @printf "%4i  %.1e  % .16e\n" iteration time_count F(pb, get_minimizer_candidate(state))
    end

    while !converged && !stopped && iteration < iterations_limit
        iteration += 1

        _time = time()
        optimstate_additionalinfo, iterationstatus = update_iterate!(state, optimizer, pb)
        time_count += time() - _time

        optimizationstate = build_optimstate(state, optimizer, pb, iteration, time_count, x_prev, optimstate_additionalinfo)
        push!(tr, optimizationstate)

        ## Display logs and save iteration information
        if show_trace && (mod(iteration, ceil(iterations_limit / optparams.trace_length)) == 0 || iteration==iterations_limit)
            display_logs(optimizationstate, optimizer)
        end

        # Check convergence
        converged = (iterationstatus == problem_solved)
        for cvchecker in optparams.cvcheckers
            converged = converged || has_converged(cvchecker, pb, optimizer, optimizationstate)
        end

        stopped_by_time_limit = time_count > optparams.time_limit
        stopped = stopped_by_time_limit
        x_prev .= get_minimizer_candidate(state)
    end

    x_final = get_minimizer_candidate(state)

    # Display status of optimizer:
    println("
* status:
    initial point value:    $(F(pb, initial_x))
    final point value:      $(F(pb, x_final))
* Counters:
    Iterations:  $iteration
    Time:        $time_count")


    if getfield(NonSmoothSolvers, :timeit_debug_enabled)()
        printstyled("\n\n")
        print_timer()
        printstyled("\n\n")
    end

    return x_final, tr
end
