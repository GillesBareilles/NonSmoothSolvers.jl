using NonSmoothSolvers
using NonSmoothProblems
using LinearAlgebra
using PlotsOptim
using DataStructures

function main()
    pb = Halfhalf()
    x = 6 .+ zeros(pb.n)

    # @show F(pb, x)

    o = VUbundle(Newton_accel = true)
    optparams = OptimizerParams(iterations_limit=20, trace_length=20)
    sol, tr = optimize!(pb, o, x; optparams)

    # @show sol
    # function getnullsteps(o, os, osaddinfo)
    #     return isnothing(osaddinfo) ? [] : osaddinfo.nullsteps
    # end
    # optimstate_extensions = OrderedDict{Symbol,Function}(:nsteps => getnullsteps)

    # println("\n\n\n")
    pb = MaxQuadBGLS()
    x = ones(10)

    o = VUbundle(Newton_accel = true)
    optparams = OptimizerParams(iterations_limit=35, trace_length=35)
    sol, tr = optimize!(pb, o, x; optparams)


    # Plot...
    # Fhist = [ F(pb, p) for os in tr for p in os.additionalinfo.nsteps ]
    # acthist = [ argmax(NSP.g(pb, p)) for os in tr for p in os.additionalinfo.nsteps ]
    # display(hcat(0:length(Fhist)-1, Fhist, acthist))
    Fopt = prevfloat(-8.4140833459641462e-01)

    optimdata = OrderedDict(
        "VU" => tr
    )
    getabsc_time(optimizer, trace) = [os.it for os in (trace)]
    getord_subopt(optimizer, trace) = [os.Fx - Fopt for os in trace]

    fig = plot_curves(
        optimdata,
        getabsc_time,
        getord_subopt;
        xlabel="it",
        ylabel=raw"subopt",
        nmarks=1000,
        includelegend=false,
    )
    savefig(fig, joinpath(".", "VU" * "_time_subopt"))
    return

    # for tr
    # tr[2].additionalinfo.nsteps
    # return tr

    # println("\n\n\n")
    # Tf = Float64
    # n = 25
    # m = 50
    # pb = get_eigmax_affine(; m, n, seed=1864)
    # # Initial point is perturbed minimum
    # xopt = [
    #     0.18843321959369272,
    #     0.31778063128576134,
    #     0.34340066698932187,
    #     -0.27805652811628134,
    #     -0.1340243453861452,
    #     -0.12921798176305369,
    #     -0.5566692206939368,
    #     -0.6007421833719635,
    #     0.05910386724008742,
    #     0.17705864693916648,
    #     0.08556420932871216,
    #     -0.026666254662448905,
    #     -0.23677377353260096,
    #     -0.48199437746045676,
    #     0.06585075102257752,
    #     0.04851608933735588,
    #     -0.3925094708809553,
    #     -0.24927524067693352,
    #     0.5381266955502098,
    #     0.2599737695610786,
    #     -0.5646166025020284,
    #     0.1550051571713463,
    #     -0.2641217487440864,
    #     0.3668468331373211,
    #     -0.2080390109713874,
    # ]
    # x = xopt + 1e-3 * ones(n)
    # # x = ones(n)

    # optparams = OptimizerParams(iterations_limit=60, trace_length=60)
    # sol, tr = optimize!(pb, o, x; optparams)
    # @show norm(sol - xopt)
    # @show eigvals(g(pb, sol))

    return
end

main()
