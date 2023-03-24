using Test
using NonSmoothProblems
using LinearAlgebra

function build_bundle(pb, fais, erlin, center)
        bundle = NSS.initial_bundle(pb, center)
        bundle.refpoint .= center
        bundle.Frefpoint = F(pb, center)
        bundle.bpts = [NSS.BundlePoint(-1., fais[:, i], erlin[i], Float64[-1., -1.]) for i in 1:length(erlin)]
    return bundle
end
function build_faiserlin(bundle)
    erlin = [pt.eᵢ for pt in bundle.bpts]
    fais = zeros(2, length(bundle.bpts))
    for i in 1:length(bundle.bpts)
        pt = bundle.bpts[i]
        fais[:, i] = pt.gᵢ
    end
    return fais, erlin
end

@testset "VUalg - bundle" begin
    pb, xopt, Fopt, Mopt = NSP.F2d()

    σₖ = 0.5
    ϵ = 1e-6

    @testset "Iteration 1 of fsiam" begin
        μₖ = 1.3374999999999999e+00
        haveinv = 2.5000000000000000e-01
        center = [9.0000000000000002e-01,6.4999999999999991e-01]
        bundle = build_bundle(pb,
            [
                0.0000000000000000e+00 0.0000000000000000e+00 9.0000000000000002e-01
                1.0000000000000000e+00 1.0000000000000000e+00 -4.3499999999999996e+00
            ],
            [0.0000000000000000e+00,0.0000000000000000e+00,8.6837499999999981e+00],
            center
        )

        ϵᶜₖ₊₁, pᶜₖ₊₁, Fpᶜₖ₊₁, gpᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁, hist = NSS.bundlesubroutine!(bundle, pb, μₖ, center, σₖ, ϵ, haveinv)

        @test isapprox(ϵᶜₖ₊₁, 3.5137032350250452e-02)
        @test isapprox(pᶜₖ₊₁, [7.9547668694123619e-01,1.4595283090383715e-01])
        @test isapprox(sᶜₖ₊₁, [3.6234692702785271e-01,1.5546450912355636e-01])
        @test isapprox(Uᶜₖ₊₁, [9.1898612115550682e-01,3.9428987956014822e-01])

        fais, erlin = build_faiserlin(bundle)


        @test isapprox(erlin, [0.0000000000000000e+00,9.6325039304742766e-01,8.1624433582331790e-01])
        @test isapprox(fais, [
            0.0000000000000000e+00 9.0000000000000002e-01 7.9547668694123619e-01
            1.0000000000000000e+00 -1.0976635514018693e+00 -8.5404716909616285e-01
        ])
    end



    @testset "Iteration 2 of fsiam" begin
        μₖ = 1.3374999999999999e+00
        haveinv = 2.1389999379348708e+00
        center = [2.0416632517768041e-02,-1.8658574446252479e-01]
        bundle = build_bundle(pb,
            [
                0.0000000000000000e+00 9.0000000000000002e-01 7.9547668694123619e-01 2.0416632517768041e-02
                1.0000000000000000e+00 -1.0976635514018693e+00 -8.5404716909616285e-01 -1.1865857444625247e+00
            ],
            [3.9078702838504970e-01,3.9078702838504986e-01,3.5564999603479913e-01,-5.5511151231257827e-17],
            center
        )

        ϵᶜₖ₊₁, pᶜₖ₊₁, Fpᶜₖ₊₁, gpᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁, hist = NSS.bundlesubroutine!(bundle, pb, μₖ, center, σₖ, ϵ, haveinv)

        @test isapprox(ϵᶜₖ₊₁, 1.1318195110213538e-06)
        @test isapprox(pᶜₖ₊₁, [1.3436738181096714e-02,4.4571070126370183e-05])
        @test isapprox(sᶜₖ₊₁, [6.7182155661946949e-03,4.5136457693606413e-05])
        @test isapprox(Uᶜₖ₊₁, [9.9997743151648499e-01,6.7183671895537896e-03])

        fais, erlin = build_faiserlin(bundle)
        @test isapprox(erlin, [3.9078702838504970e-01,1.7429080999404239e-02,1.7439796800384016e-02])
        @test isapprox(fais, [
            0.0000000000000000e+00 1.4941260361303003e-02 1.3436738181096714e-02
            1.0000000000000000e+00 -9.9996264219874642e-01 -9.9995542892987366e-01
        ])
    end

    @testset "Iteration 3 of fsiam" begin
        μₖ = 7.8495667032262716e+01
        haveinv = 2.2842965561311965e+00
        center = [-1.9096585001088240e-03,-5.8533984739249321e-05]
        bundle = build_bundle(pb,
            [
                0.0000000000000000e+00 1.4941260361303003e-02 1.3436738181096714e-02 -1.9096585001088240e-03
                1.0000000000000000e+00 -9.9996264219874642e-01 -9.9995542892987366e-01 -1.0000585339847392e+00
            ],
            [1.1889308038565537e-04 1.4198133085420927e-04 1.1776126087468309e-04 0.0000000000000000e+00],
            center
        )

        ϵᶜₖ₊₁, pᶜₖ₊₁, Fpᶜₖ₊₁, gpᶜₖ₊₁, sᶜₖ₊₁, Uᶜₖ₊₁, hist = NSS.bundlesubroutine!(bundle, pb, μₖ, center, σₖ, ϵ, haveinv)

        @test isapprox(ϵᶜₖ₊₁, 1.8408188198065270e-09)
        @test isapprox(pᶜₖ₊₁, [-1.8974380079133488e-03,8.9914754115239052e-07])
        @test isapprox(sᶜₖ₊₁, [-9.4871857656428544e-04,9.0006774722262151e-07])
        @test isapprox(Uᶜₖ₊₁, [-9.9999954996602503e-01,9.4871900351996903e-04])

        fais, erlin = build_faiserlin(bundle)
        @test isapprox(erlin, [1.1889308038565537e-04,0.0000000000000000e+00,1.8408188210851232e-09])
        @test isapprox(fais, [
            0.0000000000000000e+00 -1.9096585001088240e-03 -1.8974380079133488e-03
            1.0000000000000000e+00 -1.0000585339847392e+00 -9.9999910085245880e-01
        ])
    end
end
