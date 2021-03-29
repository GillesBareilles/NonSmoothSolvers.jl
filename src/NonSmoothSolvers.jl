module NonSmoothSolvers

using NonSmoothProblems
using LinearAlgebra
using Printf
using Random
using Distributions
using TimerOutputs

using JuMP
using OSQP
using Mosek, MosekTools
using Ipopt

import JuMP.optimize!

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

export SimpleQuad
greet() = print("Hello World!")

end # module
