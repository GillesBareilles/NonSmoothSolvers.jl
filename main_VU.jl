using NonSmoothSolvers
using NonSmoothProblems

function main()
    pb = Halfhalf()
    x = 1 .+ zeros(pb.n)

    @show F(pb, x)

    o = VUbundle(Newton_accel = true)
    optparams = OptimizerParams(iterations_limit=20, trace_length=20)
    sol, tr = optimize!(pb, o, x; optparams)

    @show sol

    println("\n\n\n")
    pb = MaxQuadBGLS()
    x = ones(10)

    o = VUbundle(Newton_accel = true)
    optparams = OptimizerParams(iterations_limit=20, trace_length=20)
    sol, tr = optimize!(pb, o, x; optparams)

    println("\n\n\n")
    Tf = Float64
    n = 25
    m = 50
    pb = get_eigmax_affine(; m, n, seed=1864)
    # Initial point is perturbed minimum
    xopt = [
        0.18843321959369272,
        0.31778063128576134,
        0.34340066698932187,
        -0.27805652811628134,
        -0.1340243453861452,
        -0.12921798176305369,
        -0.5566692206939368,
        -0.6007421833719635,
        0.05910386724008742,
        0.17705864693916648,
        0.08556420932871216,
        -0.026666254662448905,
        -0.23677377353260096,
        -0.48199437746045676,
        0.06585075102257752,
        0.04851608933735588,
        -0.3925094708809553,
        -0.24927524067693352,
        0.5381266955502098,
        0.2599737695610786,
        -0.5646166025020284,
        0.1550051571713463,
        -0.2641217487440864,
        0.3668468331373211,
        -0.2080390109713874,
    ]
    x = xopt + 1e-3 * ones(n)
    # x = ones(n)

    sol, tr = optimize!(pb, o, x; optparams)
    @show norm(sol - xopt)
    @show eigvals(g(pb, sol))

    return
end

main()
