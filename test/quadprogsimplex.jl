using Test
using Random
using NonSmoothSolvers
using LinearAlgebra
using SparseArrays
using OSQP


objval(P, a, w) = 0.5*norm(P * w)^2 + dot(a, w)


@testset "QuadProgSimplex" begin
    @testset "Float64" begin
        P = Float64[
            0 3 -2
            2 0 1
        ]
        a = zeros(3)

        w_Kiwiel = quadprogsimplex(P, a)
        w_OSQP = find_minimumnormelt_OSQP(P, a)
        @test w_Kiwiel ≈ w_OSQP
        @test prevfloat(norm(P * w_Kiwiel)) ≤ nextfloat(norm(P * w_OSQP))

        @show res = NSS.checkoptimality(P, a, w_Kiwiel)

        P = rand(10, 20)
        a = rand(20)
        w_Kiwiel = quadprogsimplex(P, a)
        w_OSQP = find_minimumnormelt_OSQP(P, a)
        @test w_Kiwiel ≈ w_OSQP
        @test prevfloat(objval(P, a, w_Kiwiel)) ≤ nextfloat(objval(P, a, w_OSQP))


        @show res = NSS.checkoptimality(P, a, w_Kiwiel)
    end

    @testset "gradient sampling" begin
        # A case close to gradient sampling:
        @testset "pb $npb" for npb in 1:10
            Random.seed!(1643 + npb)
            P = getbundlelikeproblem(n=20, ngroups=4, nvecpergroup=6)
            a = zeros(6*4)
            w_Kiwiel = quadprogsimplex(P, a; show_trace = true)
            w_OSQP = find_minimumnormelt_OSQP(P, a)
            @test norm(P * w_Kiwiel) ≤ nextfloat(norm(P * w_OSQP))
        end
    end

    @testset "BigFloat" begin
        P = BigFloat[
            0 3 -2
            2 0 1
        ]
        a = zeros(BigFloat, 3)

        w_Kiwiel = quadprogsimplex(P, a)
        @test isa(w_Kiwiel, Vector{BigFloat})
        @test w_Kiwiel ≈ BigFloat[0.0, 11//26, 15//26]
    end

    # @testset "Rational" begin
    #     P = Rational[
    #         0 3 -2
    #         2 0 1
    #     ]
    #     a = zeros(Rational, 3)

    #     w_Kiwiel = quadprogsimplex(P, a)
    #     @test isa(w_Kiwiel, Vector{Rational})
    #     @test w_Kiwiel == [0//1, 11//26, 15//26]
    # end

    @testset "Kiwiel's instances" begin
        @testset "Tf = $Tf" for Tf in [Float64, BigFloat]
            @testset "j = $jₐ" for jₐ in 1:30
                P, a, x̄, Ĵ = getKiwieltestpb(;jₐ, n = 15, Tf)

                w_Kiwiel = quadprogsimplex(P, a)
                res = NSS.checkoptimality(P, a, w_Kiwiel)
                @test res <= 100*eps(Tf) broken=(Tf == Float64 && jₐ == 18)
            end
        end
    end
end
