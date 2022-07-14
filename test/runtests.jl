using Test
using Random
using NonSmoothSolvers
using NonSmoothProblems
using LinearAlgebra


function getpb(Tf)
    ε = 0.2
    return MaxQuadPb{Tf}(2, 2,
        Vector{Matrix{Tf}}([Diagonal([1, 0]), Diagonal([2, 1])]),
        Vector{Vector{Tf}}([[1 - ε, 0], [-ε, 0]]),
        Vector{Tf}([0, 0])
    )
end

@testset "Simple max of quad example" begin
    optparams = OptimizerParams(; iterations_limit = 60, trace_length = 5, time_limit = 1.0)
    @testset "Tf = $Tf" for Tf in [Float64, BigFloat]
        pb = getpb(Tf)
        xinit = Tf[1, 1]

        @testset "Optimizer $(typeof(o))" for (o, xtol) in [
            (GradientSampling(xinit), 1e-2),
            (NSBFGS{Tf}(), 1e-8),
            (Subgradient{Tf}(), 1e-2),
        ]

            xfinal_gs, tr = NSS.optimize!(pb, o, xinit; optparams)
            @test isa(xfinal_gs, Vector{Tf})
            @test xfinal_gs ≈ Tf[0, 0] atol = xtol
            @test xinit == Tf[1, 1]
        end
    end
end
