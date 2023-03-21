using NonSmoothSolvers
using NonSmoothProblems
using DataStructures
using LinearAlgebra
using JLD2

function main()
    problems = OrderedDict{String, Any}()

    # f2d
    pb, xopt, Fopt, Mopt = F2d()
    xinit = [1., 1.]
    problems["f2d"] = pb, xinit, xopt, Fopt


    # f3d
    for ν in 0:3
        pb, xopt, Fopt, Mopt = F3d_U(ν)
        xinit = xopt + 1e2 .* [1, 1/3, -1]
        problems["f3d-U$ν"] = pb, xinit, xopt, Fopt
    end

    pb_data = Dict{String, Any}()

    for (pbname, (pb, xinit, xopt, Fopt)) in problems

        # Run VU newton on the problem
        o = VUbundle(Newton_accel = true)
        optparams = OptimizerParams(iterations_limit=40, trace_length=0)
        sol, tr = optimize!(pb, o, xinit; optparams)

        stats = @timed sol, tr = optimize!(pb, o, xinit; optparams)

        # Extract relevent information
        pb_data[pbname] = OrderedDict{String, Any}(
            "||x - xopt||" => norm(sol - xopt),
            "F(x) - F(xopt)" => F(pb, sol) - Fopt,
            "nbboxcalls" => tr[end].additionalinfo.bboxcalls,
            "niter" => length(tr) - 1,
            "time" => stats.time,
            "gctime" => stats.gctime,
        )
    end

    # Save all information
    jldsave("benchmark/data_ref.jld2"; pb_data)
end

function compare_run(filea, fileb)
    # load datas
    pb_dataa = load("benchmark/data_ref.jld2", "pb_data")

    pb_datab = load("benchmark/data_ref.jld2", "pb_data")

    @show pb_dataa

    # produce performance profiles
    # TODO

end
