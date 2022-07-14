
function solveqpsimplex(P::Matrix{Tf}, a::Vector{Tf}) where Tf
    n, m = size(P)
    J = BitVector(zeros(m))
    ŷdense = zeros(Tf, m)

    # Init
    l = argmin([0.5*norm(P[:, j])^2 + a[j] for j in axes(a, 1)])

    J[l] = 1
    v = -(norm(P[:, l])^2 + a[l])
    ŷdense[l] = 1

    it = 1
    while true
        println("---> it: $it")
        @show findall(J)
        ## Termination test, update of active set
        J̄ = findall((!).(J))
        P̂, â, ŷ = getactiveproblem(P, a, ŷdense, J)
        multipliers = [ v .+ P[:, j]' * P̂ * ŷ + a[j] for j in J̄ ]

        indrel_negmultiplier = findfirst(y->y<0, multipliers)
        if isnothing(indrel_negmultiplier)
            @info "Problem solved" multipliers
            break
        end

        l = J̄[indrel_negmultiplier]
        @show l

        ## Testing if rank of next matrix is still full
        nextsyspsd = isnextsystempsd(P, P̂, J, l)

        if nextsyspsd
            ## Handle rank deficiency case
            @error "rank deficiency case, TBD"
            throw(error( "rank deficiency case, TBD"))

        else
            ## Increase active set size
            @info "increasing active set size" l
            J[l] = 1
            ŷdense[l] = 0
        end

        itint = 1
        while true
            ## Solve problem on active set
            # HACK: brutal linear solve here, should do stuff with linear operators and iterative solvers
            # @info "Solving active QP"
            P̂, â, ŷ = getactiveproblem(P, a, ŷdense, J)
            k = sum(J)

            y, v = solve_activeqp_expl(P̂, â, k)


            if isnothing(findfirst(t->t<=0, y))
                # @info "feasible solution for the active problem found"
                ŷ .= y
                break
            end

            ## If solution is not positive, reduce active set
            # TODO: Check this section against ill-conditioned instances of Kiwiel
            cminval = Inf
            cindminval = 0
            for (i, yᵢ) in enumerate(y)
                yᵢ ≥ 0 && continue

                cval = ŷ[i] / (ŷ[i] - y[i])
                if cval < cminval
                    cminval = cval
                    cindminval = i
                end
            end
            @assert cindminval != 0
            @assert cminval < 1

            t = min(1, cminval)
            ŷ .+= t .* (y .- ŷ)

            indrm = findall(J)[cindminval]
            ŷ[cindminval] = 0
            J[indrm] = 0
            # @info "removing coord $indrm"
            @info "active set reduction" indrm

            itint += 1
            (itint > 500) && break
        end

        it +=1
        (it > 50) && break
    end

    return ŷdense
end
