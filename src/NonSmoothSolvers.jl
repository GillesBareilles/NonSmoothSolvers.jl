module NonSmoothSolvers

import NonSmoothProblems as NSP
using LinearAlgebra
using Printf
using Random
using Distributions
using TimerOutputs
using DataStructures

using JuMP
using Ipopt

import JuMP.optimize!
import TimerOutputs: enable_debug_timings, disable_debug_timings

export enable_debug_timings, disable_debug_timings

#
### solvers
#
abstract type ConvergenceChecker end
abstract type Optimizer end
abstract type OptimizerState end
abstract type NonSmoothOptimizer <: Optimizer end

include("solver_types.jl")
include("optimize.jl")

include("nonsmooth_optimizers/subgradient.jl")
include("nonsmooth_optimizers/gradientsampling.jl")
include("nonsmooth_optimizers/ns_BFGS_linesearch.jl")
include("nonsmooth_optimizers/ns_BFGS.jl")


export OptimizerParams
export optimize!


export Subgradient
export GradientSampling
export NSBFGS

end # module
