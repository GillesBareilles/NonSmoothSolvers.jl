export test1, nearest_point_polytope
export main

function test1()
    return [
        1 1 2
        1 2 1
    ]
end

function test2()
    return Float64[
        0 3 -2
        2 0 1
    ]
end

function test_large()
    n = 40

    nbasevecs = 10
    noccvecs = 6
    basevecs = rand(n, nbasevecs)
    P = zeros(n, nbasevecs * noccvecs)
    for i = 1:nbasevecs
        for j = 1+(i-1)*noccvecs:i*noccvecs
            P[:, j] .= basevecs[:, i] + 1e-6 * randn(n)
        end
    end
    return P
end

raw"""
    $TYPEDSIGNATURES

Implement the algorithm from Wolfe's paper.

## Note
The convex combination of step 3c seems contradictory to note 6.
We replaced $w = θw + (1-θ)v$ by $w = (1-θ)w + θv$.

## Reference
- Wolfe (1976) Finding the Nearest Point in A Polytope, Mathematical Programming.
"""
function nearest_point_polytope(P)
    n, m = size(P)

    norm2Pᵢs = [norm(P[:, i])^2 for i in axes(P, 2)]
    Z₁ = 1e-10
    Z₂ = 1e-10
    Z₃ = 1e-10

    # Step 0
    j = argmin(norm2Pᵢs)
    S = SortedSet{Int64}(j)
    Sa = BitVector(zeros(m))
    Sa[j] = 1
    @show S
    @show Sa

    w = zeros(m)
    w[j] = 1
    x = P * w

    showtrace = false

    showtrace && println("it    j   S   removed elements")

    converged = false
    keepon = true
    it = 1
    while !converged && keepon
        # Step 1
        x = P * w
        j = argmin([dot(P[:, i], x) for i in axes(P, 2)])


        if dot(x, P[:, j]) >
           norm(x)^2 - Z₁ * max(norm2Pᵢs[j], maximum(norm2Pᵢs[collect(S)]))

            converged = true
            break
        end
        if j ∈ S
            keepon = false
            @info "disaster, stopping"
        end

        push!(S, j)
        @show S
        Sa[j] = 1
        @show Sa

        w[j] = 0

        removed_pts = SortedSet{Int64}()
        innerit = 0
        while true
            # Step 2
            p = length(S)
            A = ones(p + 1, p + 1)
            A[1, 1] = 0
            Pₛ = @view P[:, collect(S)]
            A[2:end, 2:end] .= Pₛ' * Pₛ
            b = zeros(p + 1)
            b[1] = 1
            res = A \ b
            v = res[2:end]


            if sum(v .> Z₂) == length(v)
                w[collect(S)] .= v
                @debug "Point in the ri of current convex hull"
                break
            end

            # Step 3
            wₛ = @view w[collect(S)]
            POS = filter(i -> wₛ[i] > v[i] + Z₃, 1:length(S))
            θ = min(1, minimum(i -> wₛ[i] / (wₛ[i] - v[i]), POS))
            @. wₛ = (1 - θ) * wₛ + θ * v

            w[w.<Z₂] .= 0
            k::Int64 = findfirst(i -> (w[i] == 0), collect(S))
            k = collect(S)[k]
            delete!(S, k)
            push!(removed_pts, k)

            @show S
            Sa[k] = 0
            @show Sa


            innerit += 1
            innerit > 10 && @assert false
        end

        showtrace && @printf "%2i  %i   %s    %s  " it j collect(S) collect(removed_pts)

        it += 1
        # it > 500 && (keepon = false)
    end

    showtrace && show_final_status(S, w, P, x)

    return w, x
end

function show_final_status(S, w, P, x)
    println()
    println("S                    \t", collect(S))
    println("1 - eᵀw               \t", sum(w) - 1)
    println("|x - P*w|             \t", norm(x - P * w))
    println(
        "Max{|xᵀPⱼ - xᵀx|, j∈S} \t",
        maximum([abs(dot(x, P[:, j]) - dot(x, x)) for j in S]),
    )
    println(
        "Min{xᵀPⱼ - xᵀx, j}     \t",
        minimum([dot(x, P[:, j]) - dot(x, x) for j in axes(P, 2)]),
    )
    return
end

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

function main()
    P = test2()
    nearest_point_polytope(P)
end
