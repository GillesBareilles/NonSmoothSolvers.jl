using NonSmoothSolvers
using NonSmoothProblems
using DataStructures
using LinearAlgebra
using JLD2
using Dates

include("matlabbaseline.jl")

function main(; writedate = false)
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

    # maxquad
    pb = MaxQuadBGLS()
    xinit = ones(Float64, 10)
    xopt = xinit
    Fopt = -1
    problems["maxquadBGLS"] = pb, xinit, xopt, Fopt

    # maxquad 2
    pb = MaxQuadAlt()
    xinit = ones(Float64, 10)
    xopt = xinit
    Fopt = -1
    problems["maxquadAlt"] = pb, xinit, xopt, Fopt

    pb_data = Dict{String, Any}()

    for (pbname, (pb, xinit, xopt, Fopt)) in problems
        # Run VU newton on the problem
        o = VUbundle(Newton_accel = true)
        optparams = OptimizerParams(iterations_limit=40, trace_length=0)
        sol, tr = optimize!(pb, o, xinit; optparams)

        stats = @timed sol, tr = optimize!(pb, o, xinit; optparams)

        # Extract relevent information
        pb_data[pbname] = OrderedDict{String, Any}(
            "nbboxcalls    " => tr[end].additionalinfo.bboxcalls,
            "niter         " => length(tr) - 1,
            "x             " => sol,
            "F(x)          " => F(pb, sol),
            "||x - xopt||  " => norm(sol - xopt),
            "F(x) - F(xopt)" => F(pb, sol) - Fopt,
            "time          " => stats.time,
        )
    end

    gitcommit = readchomp(`git --git-dir .git rev-parse HEAD`)
    date = now()
    filename = "benchmark/VUjulia_curr.jld2"
    if writedate
        filename = "benchmark/VUjulia_$(date).jld2"
    end
    jldsave(filename*".jld2"; pb_data, gitcommit, date)
    @info "Wrote $(filename).jld2"

    write_data_txt(pb_data, gitcommit, date, filename*".txt")
    return
end

function write_data_txt(pb_data, gitcommit, date, filename)
    open(filename, "w") do io
        println(io, "Date   : ", date)
        println(io, "commit : ", gitcommit)

        for pbname in ["f2d", "f3d-U0", "f3d-U1", "f3d-U2", "f3d-U3", "maxquad", "maxquadBGLS", "maxquadAlt"]
            !haskey(pb_data, pbname) && continue
            pbdata = pb_data[pbname]
            println(io, "> ", pbname)
            for (kpiname, kpival) in pbdata
                println(io, "  > ", kpiname, " : ", kpival)
            end
        end
    end
    @info "wrote $filename"
    return
end

