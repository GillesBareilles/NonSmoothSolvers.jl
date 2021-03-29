using NonSmoothProblems
using NonSmoothSolvers

const NSS = NonSmoothSolvers

function main()
    pb = NonSmoothProblems.SmoothQuad2d_2()
    x = [2.0, 3.5]

    # pb = MaxQuad2d()
    # x = [2.0, 2.0]

    pb = MaxQuadBGLS()
    x = zeros(10) .+ 1

    @show F(pb, x)

    println()
    println()
    o = GradientSampling(m=20, β=1e-4)
    xfinal_gs, tr = optimize!(pb, o, x, optparams=OptimizerParams(iterations_limit=100, trace_length=50))

    println()
    println()
    o = NSBFGS()
    xfinal_nsbfgs, tr = optimize!(pb, o, xfinal_gs, optparams=OptimizerParams(iterations_limit=100, trace_length=50))

    println()
    display(xfinal_gs)
    display(xfinal_nsbfgs)
    return
end

res = main()