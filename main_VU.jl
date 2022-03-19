using NonSmoothSolvers
using NonSmoothProblems

function main()
    pb = Halfhalf()
    x = 1 .+ zeros(pb.n)

    # pb = MaxQuadBGLS()
    # x = ones(10)

    @show F(pb, x)

    o = VUalg()
    optparams = OptimizerParams(iterations_limit=5, trace_length=20)
    sol, tr = optimize!(pb, o, x; optparams)
    return
end

main()
