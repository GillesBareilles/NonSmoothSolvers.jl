module NonSmoothSolvers

import Base.length

using NonSmoothProblems
const NSP = NonSmoothProblems
using LinearAlgebra
using Printf
using Random
using Distributions
using PDMats
using TimerOutputs
using DataStructures

using Infiltrator
using DocStringExtensions

using QuadProgSimplex

using LinearOperators

const NSS = NonSmoothSolvers

import TimerOutputs: enable_debug_timings, disable_debug_timings

export enable_debug_timings, disable_debug_timings

using JuMP, OSQP

#
### solvers
#
abstract type ConvergenceChecker end
function hasconverged(cvchecker::ConvergenceChecker, pb, optimizer, updateinformation)
    throw(
        error(
            "hasconverged: not defined for types &(typeof(cvchecker)), &(typeof(pb)), &(typeof(optimizer)), &(typeof(optimizationstate)).",
        ),
    )
end

abstract type Optimizer{Tf} end
abstract type OptimizerState{Tf} end
abstract type NonSmoothOptimizer{Tf} <: Optimizer{Tf} end

include("solver_types.jl")
include("optimize.jl")

include("nonsmooth_optimizers/subgradient.jl")

include("nonsmooth_optimizers/gradientsampling.jl")

include("nonsmooth_optimizers/ns_BFGS_linesearch.jl")
include("nonsmooth_optimizers/ns_BFGS.jl")

include("nonsmooth_optimizers/VUalgo_utils.jl")
include("nonsmooth_optimizers/VUalgo_qNewton.jl")
include("nonsmooth_optimizers/VUalgo_esearch.jl")
include("nonsmooth_optimizers/VUalgo_isearch.jl")
include("nonsmooth_optimizers/VUalgo_bundle_qps.jl")
include("nonsmooth_optimizers/VUalgo_bundle.jl")
include("nonsmooth_optimizers/VUalgo.jl")

export NSS

export optimize, optimize!
export OptimizerParams

export Subgradient
export GradientSampling
export NSBFGS

export VUbundle, VUbundleState

end # module
