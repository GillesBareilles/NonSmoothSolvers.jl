using NonSmoothSolvers
using NonSmoothProblems
using LinearAlgebra
using DataStructures

function main(; ν = 0)
    # pb, xopt, Fopt, Mopt = NSP.F2d()
    # x=[0.9, 1.9]

    # pb = NSP.MaxQuadBGLS()
    # x = ones(10)

    pb, xopt, Fopt, Mopt = NSP.F3d_U(ν)
    x = xopt .+ Float64[100, 33, -100]

    iterations_limit=20

    # NOTE: Bundle without Newton accel
    o = VUbundle(Newton_accel = true)
    optparams = OptimizerParams(;iterations_limit, trace_length=35, time_limit = 100)

    sol, tr = optimize!(pb, o, x; optparams)
    @show sol
    return
end
