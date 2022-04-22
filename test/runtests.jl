using Test
using Random
using NonSmoothSolvers
using LinearAlgebra
using SparseArrays
using JuMP, OSQP

function find_minimumnormelt_OSQP(∂gᵢs)
    n, nsamples = size(∂gᵢs)

    P = sparse(∂gᵢs' * ∂gᵢs)
    q = zeros(nsamples)
    A = sparse(vcat(Diagonal(1.0I, nsamples), ones(nsamples)'))
    l = zeros(nsamples + 1)
    l[end] = 1
    u = Inf * ones(nsamples + 1)
    u[end] = 1

    # Solve problem
    options = Dict(
        :verbose => false,
        :polish => true,
        :eps_abs => 1e-06,
        :eps_rel => 1e-06,
        :max_iter => 5000,
    )
    model = OSQP.Model()
    OSQP.setup!(model; P = P, q = q, A = A, l = l, u = u, options...)
    results = OSQP.solve!(model)
    return results.x
end



@testset "Nearest point of polytope" begin
    P = Float64[
        0 3 -2
        2 0 1
    ]

    w_Wolfe = nearest_point_polytope(P)
    w_OSQP = find_minimumnormelt_OSQP(P)
    @test w_Wolfe ≈ w_OSQP
    @test norm(P * w_Wolfe) ≤ nextfloat(norm(P * w_OSQP))


    P = rand(10, 20)
    w_Wolfe = nearest_point_polytope(P)
    w_OSQP = find_minimumnormelt_OSQP(P)
    @test w_Wolfe ≈ w_OSQP
    @test norm(P * w_Wolfe) ≤ nextfloat(norm(P * w_OSQP))

    # A case close to gradient sampling:
    n = 20
    basevecs = rand(n, 4)
    P = zeros(n, 4 * 6)
    for i = 1:4
        for j = 1+(i-1)*6:i*6
            P[:, j] .= basevecs[:, i] + 1e-6 * randn(n)
        end
    end

    w_Wolfe, x = nearest_point_polytope(P)
    w_OSQP = find_minimumnormelt_OSQP(P)
    # @test w_Wolfe ≈ w_OSQP
    @test norm(P * w_Wolfe) ≤ nextfloat(norm(P * w_OSQP))
end

