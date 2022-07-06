using NonSmoothSolvers
using NonSmoothProblems

function main()
    pb = Halfhalf()
    x = 1 .+ zeros(pb.n)

    @show F(pb, x)

    o = VUbundle()
    optparams = OptimizerParams(iterations_limit=20, trace_length=20)
    sol, tr = optimize!(pb, o, x; optparams)

    @show sol

    println("\n\n\n")
    pb = MaxQuadBGLS()
    x = ones(10)

    o = VUbundle(Newton_accel = true)
    optparams = OptimizerParams(iterations_limit=20, trace_length=20)
    sol, tr = optimize!(pb, o, x; optparams)


    return
end

main()
