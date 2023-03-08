using NonSmoothSolvers
using NonSmoothProblems
using LinearAlgebra
using DataStructures


include("plot_utils.jl")


function run_VUplots()
    problems = []

    xopt=[-0.12625658060784,-0.03437830253436,-0.00685719866363, 0.02636065787685, 0.06729492268391,-0.27839950072585, 0.07421866459664, 0.13852404793680, 0.08403122337963, 0.03858030992197]
    pb = MaxQuadAlt()
    push!(problems, (; name="maxquadBGLS", pb, xinit=ones(10), iterations_limit=40, Fopt = prevfloat(F(pb, xopt))))


    # for ν in 0:3
    #     pb, xopt, Fopt, Mopt = NSP.F3d_U(ν)
    #     xinit = xopt .+ Float64[100, 33, -100]
    #     push!(problems, (; name="F3d-U$(ν)", pb, xinit, iterations_limit=20, Fopt))
    # end
    # pb, xopt, Fopt, Mopt = NSP.F2d()
    # push!(problems, (; name="F2d", pb, xinit=[0.9, 1.9], iterations_limit=20, Fopt))

    getpoint(o, os, osext) = deepcopy(os.p)
    plot = TikzDocument()

    for problem in problems
        @info "++ Problem $(problem.name)"
        x = problem.xinit
        pb = problem.pb
        iterations_limit = problem.iterations_limit
        Fopt = problem.Fopt

        # NOTE: Bundle without Newton accel
        # o = VUbundle(Newton_accel = false)
        # optparams = OptimizerParams(;iterations_limit, trace_length=35)
        # try
        #     sol, tr = optimize!(pb, o, x; optparams, optimstate_extensions = OrderedDict{Symbol,Function}(:p => getpoint))
        #     push!(plot, plot_seriousnullsteps(pb, tr, Fopt; title = problem.name*"- without Newton"))
        #     write_nullsteps(pb, tr, joinpath("output", problem.name*"_without_Newton"*".txt"))
        # catch e
        #     @error e
        # end

        # NOTE: Bundle with Newton accel
        o = VUbundle(Newton_accel = true)
        optparams = OptimizerParams(;iterations_limit, trace_length=35)
        try
            sol, tr = optimize!(pb, o, x; optparams, optimstate_extensions = OrderedDict{Symbol,Function}(:p => getpoint))
            push!(plot, plot_seriousnullsteps(pb, tr, Fopt; title = problem.name*"- with qNewton"))
            write_nullsteps(pb, tr, joinpath("output", problem.name*"_with_qNewton"*".txt"))
            @show sol
        catch e
            @error e
        end
    end

    savefig(plot, joinpath("output", "VU_seriousnullsteps"))
    return nothing
end
