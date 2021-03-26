struct Subgradient <: NonSmoothOptimizer end

Base.@kwdef mutable struct SubgradientState{Tx}
    x::Tx           # point
    v::Tx           # subgradient
    k::Int64 = 1
end

initial_state(::Subgradient, initial_x, pb) = SubgradientState(x=initial_x, v=initial_x .* 0)


#
### Printing
#
print_header(::Subgradient) = println("**** Subgradient algorithm")


#
### Subgradient method
#
function update_iterate!(state, ::Subgradient, pb)
    state.v = ∂F_elt(pb, state.x)

    state.x -= 1/state.k * state.v
    state.k += 1

    return NamedTuple(), iteration_completed
end



get_minimizer_candidate(state::SubgradientState) = state.x