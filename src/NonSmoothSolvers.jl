module NonSmoothSolvers

using LinearAlgebra
using JuMP
using Printf
using NonSmoothProblems
using Random
using Distributions

using OSQP

import JuMP.optimize!

# #
# ### NonSmooth Problem interface
# #
# """
#     NonSmoothPb

# Abstract type for generic nonsmooth problem.
# """
# abstract type NonSmoothPb end

# F(pb::NonSmoothPb, x) = throw(error("F(): Not implemented for problem type $(typeof(pb))."))
# ∂F_elt(pb::NonSmoothPb, x) = throw(error("subgradient(): Not implemented for problem type $(typeof(pb))."))
# ∂F_minnormelt(pb::NonSmoothPb, x) = throw(error("minnormsubgradient(): Not implemented for problem type $(typeof(pb))."))


# struct SimpleQuad <: NonSmoothPb end
# F(::SimpleQuad, x) = norm(x, 2)^2
# ∂F_elt(::SimpleQuad, x) = 2*x
# ∂F_minnormelt(::SimpleQuad, x) = 2*x



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
