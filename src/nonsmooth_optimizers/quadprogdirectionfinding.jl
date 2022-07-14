struct QuadProgSimplex{Tf} <: Optimizer{Tf}
end
QuadProgSimplex(Tf = Float64) = QuadProgSimplex{Tf}()

Base.@kwdef mutable struct QuadProgSimplexState{Tf} <: OptimizerState{Tf}
    ydense::Vector{Tf}
    J::BitVector
    v::Tf
    multipliers::Vector{Tf}
    it::Int64 = -1
end

function initial_state(::QuadProgSimplex{Tf}, P::Matrix{Tf}) where Tf
    n, m = size(P)
    return QuadProgSimplexState(
        J = BitVector(zeros(m)),
        ydense = zeros(Tf, m),
        multipliers = zeros(Tf, m),
        v = Tf(0),
    )
end

function initialize_state!(state::QuadProgSimplexState{Tf}, P, a) where {Tf}
    l = argmin([0.5*norm(P[:, j])^2 + a[j] for j in axes(a, 1)])

    state.J[l] = 1
    state.v = -(norm(P[:, l])^2 + a[l])
    state.ydense[l] = 1
    return nothing
end


#
### Printing
#
print_header(::QuadProgSimplex) = println("**** QuadProgSimplex algorithm")
display_logs_header(::QuadProgSimplex) = print("it.   time      0.5|Px|² + ⟨a, x⟩        act. set\n")
function display_logs(state, ::QuadProgSimplex; time_count, P, a)
    x = get_minimizer_candidate(state)
    @printf "%4i  %.1e   %-.16e   %s\n" state.it time_count 0.5*norm(P*x)^2+dot(a,x) findall(state.J)
end

raw"""
    $TYPEDSIGNATURES

Solve the problem $min_{x \in \Delta} 0.5 \|Px\|^2 + \langle a, x\rangle$ where $x$
is constrained to the simplex set $\Delta$ (non-negative coordinates that sum
to one).

Reference:
- Kiwiel (1986) A Method for Solving Certain Quadratic Programming Problems
  Arising in Nonsmooth Optimization, IMA Journal of Numerical Analysis.
"""
function quadprogsimplex(P::Tm, a::Tv; show_trace = false, check_optimality=false) where {Tf, Tm<:AbstractMatrix{Tf}, Tv<:AbstractVector{Tf}}
    o = QuadProgSimplex(Tf)
    state = initial_state(o, P)
    initialize_state!(state, P, a)
    quadprogsimplex!(state, o, P, a; show_trace)

    if check_optimality
        checkoptimality(P, a, state.ydense, state.J; print=true)
    end
    return get_minimizer_candidate(state)
end

function quadprogsimplex!(
    state,
    optimizer,
    P,
    a;
    show_trace = false,
    iterations_limit = 100,
)
    time_count = 0.0
    iteration = 0
    converged = false
    stopped = false

    show_trace && print_header(optimizer)
    show_trace && display_logs_header(optimizer)

    while !converged && !stopped && iteration < iterations_limit
        iteration += 1
        _time = time()
        iterationstatus = update_iterate!(state, optimizer, P, a)
        time_count += time() - _time

        show_trace && display_logs(state, optimizer; time_count, P, a)
        stopped = (iterationstatus == iteration_failed)
        converged = (iterationstatus == problem_solved)
    end
end

#
### QuadProgSimplex method
#
raw"""
    $TYPEDSIGNATURES
"""
function update_iterate!(state, ::QuadProgSimplex{Tf}, P, a) where {Tf}
    J = state.J
    state.it += 1
    multipliers = state.multipliers
    v = state.v

    ## Termination test, update of active set
    J̄ = findall((!).(J))
    P̂, â, ŷ = getactiveproblem(P, a, state.ydense, J)

    multipliers .= v .+ a .+ P' * P̂ * ŷ
    indrel_negmultiplier = findfirst(t -> t<0, @view multipliers[J̄])

    if isnothing(indrel_negmultiplier)
        return problem_solved
    end
    l = J̄[indrel_negmultiplier]

    ## Testing if rank of next matrix is still full
    nextsyspsd = isnextsystempsd(P, P̂, J, l)
    # println()
    # @show P, P̂, J, l
    # println()

    if nextsyspsd
        ## Handle rank deficiency case
        @error "rank deficiency case, TBD"
        throw(error("rank deficiency case, TBD"))
    else
        ## Increase active set size
        J[l] = 1
        state.ydense[l] = 0
    end

    itint = 1
    while true
        ## Solve problem on active set
        # HACK: brutal linear solve here, should do stuff with linear operators and iterative solvers
        P̂, â, ŷ = getactiveproblem(P, a, state.ydense, J)
        k = sum(J)

        y, state.v = solve_activeqp(P̂, â, k)

        if isnothing(findfirst(t -> t <= 0, y))
            ŷ .= y
            break
        end

        ## If solution is not positive, reduce active set
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

        itint += 1
        if (itint > 500)
            @info "breaking from inner iterations..."
            break
        end
    end

    return iteration_completed
end

function checkoptimality(P, a, x)
    J = x .> eps(Float64)
    return checkoptimality(P, a, x, J)
end

raw"""
    $TYPEDSIGNATURES

Check optimality of point `x` with nonnull coordinates `Jac` by checking tolerences
 on the KKT system (2.1) of Kiwiel's paper.

The multiplier `v` is computed so as to solve the second line of the system. We
thus check that $x \ge 0$, $\sum x_i = 1$ and that, for each nonnull coordinate
$x_j$, $v + P_j^{\top} P x + a_j = 0$.
"""
function checkoptimality(P, a, x, Jact; print=false)
    v = - minimum(P' * P * x + a)

    res_posx = max(-minimum(x[Jact]), 0)
    res_sumone = abs(sum(x[Jact]) - 1)
    res_Riemanngrad = norm([P[:, j]' * P * x + a[j] + v for j in findall(Jact)])

    if print
        println("x ≥ 0                  : ", res_posx)
        println("∑ x - 1                : ", res_sumone)
        println("gradient on active face: ", res_Riemanngrad)
    end

    return max(res_posx, res_sumone, res_Riemanngrad)
end

get_minimizer_candidate(state::QuadProgSimplexState) = state.ydense
get_activeface_candidate(state::QuadProgSimplexState) = findall(state.J)



function getactiveproblem(P, a, y, J)
    P̂ = @view P[:, J]
    ŷ = @view y[J]
    â = @view a[J]
    return P̂, â, ŷ
end

# function isnextsystempsd(P::Matrix{Tf}, P̂, J, l) where Tf
#     k = sum(J)
#     P̂ₑ = vcat(ones(Tf, k)', P̂)
#     ỹ = (P̂ₑ' * P̂ₑ) \ P̂ₑ' * vcat(Tf(1), P[:, l])
#     res = (sum(ỹ) == 1) && (P̂ * ỹ == P[:, l])
#     return res
# end

function isnextsystempsd(P::Matrix{Tf}, P̂, J, l) where Tf
    k = sum(J)
    P̂ₑ = vcat(ones(Tf, k)', P̂)
    rhs = vcat(Tf(1), P[:, l])

    ỹ = IterativeSolvers.lsmr(P̂ₑ, rhs)

    res = (sum(ỹ) == 1) && (P̂ * ỹ == P[:, l])
    return res
end

function solve_activeqp(P̂::Tm, â, k) where {Tf, Tm <: AbstractMatrix{Tf}}
    A = zeros(Tf, k+1, k+1)
    A[1, 1:end-1] .= 1
    A[1, end] = 0
    A[2:end, 1:end-1] .= P̂' * P̂
    A[2:end, end] .= 1

    # res = IterativeSolvers.cg(A, vcat([1], -â))
    res = A \ vcat([1], -â)
    return res[1:end-1], res[end]
end
