using NonSmoothSolvers
using NonSmoothProblems
using LinearAlgebra
using DataStructures


include("plot_utils.jl")


function run_VUplots()
    problems = []

    push!(problems, (; name="maxquadBGLS", pb=MaxQuadBGLS(), xinit=ones(10), iterations_limit=10, Fopt = prevfloat(-8.4140833459641462e-01)))
    for ν in 0:3
        pb, xopt, Fopt, Mopt = NSP.F3d_U(ν)
        xinit = xopt .+ Float64[100, 33, -100]
        push!(problems, (; name="F3d-U$(ν)", pb, xinit, iterations_limit=20, Fopt))
    end
    pb, xopt, Fopt, Mopt = NSP.F2d()
    push!(problems, (; name="F2d", pb, xinit=[0.9, 1.9], iterations_limit=20, Fopt))


    getpoint(o, os, osext) = deepcopy(os.p)
    plot = TikzDocument()

    for problem in problems
        @info "++ Problem $(problem.name)"
        x = problem.xinit
        pb = problem.pb
        iterations_limit = problem.iterations_limit
        Fopt = problem.Fopt

        # NOTE: Bundle without Newton accel
        o = VUbundle(Newton_accel = false)
        optparams = OptimizerParams(;iterations_limit, trace_length=35)
        try
            sol, tr = optimize!(pb, o, x; optparams, optimstate_extensions = OrderedDict{Symbol,Function}(:p => getpoint))
            push!(plot, plot_seriousnullsteps(pb, tr, Fopt; title = problem.name*"- without Newton"))
            write_nullsteps(pb, tr, Fopt, joinpath("output", problem.name*"_without_Newton"*".txt"))
        catch e
            @error e
        end

        # NOTE: Bundle with Newton accel
        o = VUbundle(Newton_accel = true)
        optparams = OptimizerParams(;iterations_limit, trace_length=35)
        try
            sol, tr = optimize!(pb, o, x; optparams, optimstate_extensions = OrderedDict{Symbol,Function}(:p => getpoint))
            push!(plot, plot_seriousnullsteps(pb, tr, Fopt; title = problem.name*"- with qNewton"))
            write_nullsteps(pb, tr, Fopt, joinpath("output", problem.name*"_with_qNewton"*".txt"))
        catch e
            @error e
        end
    end

    savefig(plot, joinpath("output", "VU_seriousnullsteps"))
    return nothing
end
