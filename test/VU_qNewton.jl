using Test
using NonSmoothProblems
using LinearAlgebra

@testset "VUalg - qNewton - fsiam" begin
    pb, xopt, Fopt, Mopt = NSP.F2d()

    @testset "it. 1" begin
        pₖ = [0.9, 1.9]
        pₖ₋₁ = [0.9, 1.9]
        sₖ = [0.0, 1.0]
        sₖ₋₁ = [0.0, 1.0]
        Uₖ = [1.0 0.0; 0.0 1.0]
        Hₖ = [4.0 0.0; 0.0 4.0]
        k = 1
        curvmin = 1.0e-6
        ν = 2.0
        νlow = 2.0
        μₖ = 4.0
        kase = 0.0

        du, hmin, haveinv, Hout, kase = NSS.qNewtonupdate(pₖ, pₖ₋₁, sₖ, sₖ₋₁, Uₖ, Hₖ, k, curvmin, ν, νlow, μₖ, kase)

        @test isapprox(du, [0.0, -0.25])
        @test isapprox(haveinv, 0.25)
        @test isapprox(hmin, 0.0)
        @test isapprox(Hout, [4.0 0.0; 0.0 4.0])
        @test isapprox(kase, 0.0)
    end

    @testset "it. 2" begin
        pₖ = [0.7954766869412362, 0.14595283090383726]
        pₖ₋₁ = [0.9, 1.9]
        sₖ = [0.36234692702785265, 0.15546450912355625]
        sₖ₋₁ = [0.0, 1.0]
        Uₖ = [0.9189861211555068; 0.39428987956014827;;]
        Hₖ = [4.0 0.0; 0.0 4.0]
        k = 2
        curvmin = 1.0e-6
        ν = 1.0
        νlow = 1.0
        μₖ = 1.3375
        kase = 0

        du, hmin, haveinv, H, kase = NSS.qNewtonupdate(pₖ, pₖ₋₁, sₖ, sₖ₋₁, Uₖ, Hₖ, k, curvmin, ν, νlow, μₖ, kase)

        @test isapprox(du, [-0.7750600544234679, -0.33253857536636194])
        @test isapprox(haveinv, 2.1389999379348703)
        @test isapprox(hmin, 0.0)
        @test isapprox(H, [0.4675081949583715 0.0; 0.0 0.4675081949583715])
        @test isapprox(kase, 1)
    end

    @testset "it. 3" begin
        pₖ = [0.013436738181096714, 4.457107012645345e-5]
        pₖ₋₁ = [0.7954766869412362, 0.14595283090383715]
        sₖ = [0.006718215566194694, 4.5136457693661924e-5]
        sₖ₋₁ = [0.3623469270278527, 0.15546450912355636]
        Uₖ = [0.999977431516485; 0.0067183671895537644;;]
        Hₖ = [0.4675081949583715 0.0; 0.0 0.4675081949583715]
        k = 3
        curvmin = 1.0e-6
        ν = 1.0
        νlow = 1.0
        μₖ = 78.49566703226272
        kase = 1

        du, hmin, haveinv, H, kase = NSS.qNewtonupdate(pₖ, pₖ₋₁, sₖ, sₖ₋₁, Uₖ, Hₖ, k, curvmin, ν, νlow, μₖ, kase)

        @test isapprox(du, [-0.015346396681205538, -0.00010310505486561911])
        @test isapprox(haveinv, 2.2842965561311965)
        @test isapprox(hmin, 0.0)
        @test isapprox(H, [0.4364492826600879 0.09806118467912484; 0.09806118467912484 0.539596654183735])
        @test isapprox(kase, 2)
    end

    @testset "it. 4" begin
        pₖ = [-0.0018974380079133488, 8.991475411523905e-7]
        pₖ₋₁ = [0.013436738181096714, 4.457107012637018e-5]
        sₖ = [-0.0009487185765642851, 9.000677476111996e-7]
        sₖ₋₁ = [0.006718215566194695, 4.513645769360641e-5]
        Uₖ = [-0.999999549966025; 0.0009487190035200044;;]
        Hₖ = [0.4364492826600879 0.09806118467912484; 0.09806118467912484 0.539596654183735]
        k = 4
        curvmin = 1.0e-6
        ν = 1.0
        νlow = 1.0
        μₖ = 1.3375
        kase = 2

        du, hmin, haveinv, H, kase = NSS.qNewtonupdate(pₖ, pₖ₋₁, sₖ, sₖ₋₁, Uₖ, Hₖ, k, curvmin, ν, νlow, μₖ, kase)

        @test isapprox(du, [0.001897500636589409, -1.800195723272915e-6])
        @test isapprox(haveinv, 2.000066914954979)
        @test isapprox(hmin, 0.0)
        @test isapprox(H, [0.49998593698974375 0.0014126451805965882; 0.0014126451805965882 0.5169142670584466])
        @test isapprox(kase, 2)
    end
end
