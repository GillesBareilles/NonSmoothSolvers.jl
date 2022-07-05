using Test
using Random
using NonSmoothSolvers
using LinearAlgebra
using SparseArrays
using OSQP

"""
    find_minimumnormelt_OSQP(P, b)

Solve minimize the quadratic `0.5*xᵀPᵀPx + bᵀx` such that `x` lives in the
simplex set.
"""
function find_minimumnormelt_OSQP(P, b)
    n, nsamples = size(P)

    P = sparse(P' * P)
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
    OSQP.setup!(model; P = P, q = b, A = A, l = l, u = u, options...)
    results = OSQP.solve!(model)
    return results.x
end

function find_minimumnormelt_OSQP(P)
    return find_minimumnormelt_OSQP(P, zeros(size(P, 2)))
end

function getbundlelikeproblem(;n=20, ngroups=4, nvecpergroup=6)
    basevecs = rand(n, ngroups)
    P = zeros(n, ngroups * nvecpergroup)
    for i = 1:ngroups
        for j = 1+(i-1)*nvecpergroup:i*nvecpergroup
            P[:, j] .= basevecs[:, i] + 1e-6 * randn(n)
        end
    end
    return P
end

function getKiwieltestpb(; n=5, Tf=Float64, b = 1e5)
    m = 2n+2
    P = Tf[j / (i + j) for i in 1:n, j in 1:m]
    Ĵ = 1:floor(n/2)

    x̄ = Tf[ j in Ĵ ? 1/(j+1) : 0 for j in 1:m]
    v̄ = minimum([- P[:, j]' * P * x̄ for j in 1:m])
    a = [ -v̄ - P[:, j]' * P * x̄ + (j in Ĵ ? 0 : b) for j in 1:m]

    return P, a, x̄, Ĵ
end


@testset "Nearest point of polytope - Float64" begin
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
    @testset "pb $npb" for npb in 1:10
        Random.seed!(1643 + npb)
        P = getbundlelikeproblem(n=20, ngroups=4, nvecpergroup=6)
        w_Wolfe = nearest_point_polytope(P)
        w_OSQP = find_minimumnormelt_OSQP(P)
        @test norm(P * w_Wolfe) ≤ nextfloat(norm(P * w_OSQP))
    end
end

@testset "Nearest point of polytope - BigFloat" begin
    P = BigFloat[
        0 3 -2
        2 0 1
    ]

    w_Wolfe = nearest_point_polytope(P)
    @test isa(w_Wolfe, Vector{BigFloat})
    @test w_Wolfe ≈ BigFloat[0.0, 11//26, 15//26]
end

@testset "Nearest point of polytope - Rational" begin
    P = Rational[
        0 3 -2
        2 0 1
    ]

    w_Wolfe = nearest_point_polytope(P)
    @test isa(w_Wolfe, Vector{Rational})
    @test w_Wolfe == [0//1, 11//26, 15//26]
end
