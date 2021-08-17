"""
    OptimizationState

Stores information after one iteration of the optimizer. Generic information is stored explicitly in the struct,
custom information may be stored in the field `additionalinfo::NamedTuple`.
"""
Base.@kwdef struct OptimizationState{Tf, Ti}
    it::Int64 = 0
    time::Float64 = 0.0
    Fx::Tf
    norm_step::Tf
    ncalls_F::Int64 = 0
    ncalls_∂F_elt::Int64 = 0
    additionalinfo::Ti = NamedTuple()
end

const OptimizationTrace{T} = Vector{OptimizationState{T}}

"""
    OptimizerParams

Generic parameters for optim algs
"""
Base.@kwdef struct OptimizerParams
    iterations_limit::Int64 = 1e3
    time_limit::Float64 = 30.0
    show_trace::Bool = true
    trace_length::Int64 = 20
    cvcheckers::Set{ConvergenceChecker} = Set([])
end

