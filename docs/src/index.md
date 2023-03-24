```@meta
CurrentModule = NonSmoothSolvers
```

# NonSmoothSolvers

Documentation for [NonSmoothSolvers](https://github.com/GillesBareilles/NonSmoothSolvers.jl).

```@index
```

```@docs
NonSmoothSolvers.OptimizerParams
NonSmoothSolvers.OptimizationState
NonSmoothSolvers.optimize!
```

## Gradient sampling
```@docs
NonSmoothSolvers.GradientSampling
```
<!-- NonSmoothSolvers.update_iterate!(state, gs::GradientSampling, pb) -->

## Non smooth BFGS
```@docs
NonSmoothSolvers.linesearch_nsbfgs
```
<!-- NonSmoothSolvers.update_iterate! -->


<!-- ```@autodocs -->
<!-- Modules = [NonSmoothSolvers] -->
<!-- pages = ["NonSmoothSolvers.jl", "optimize.jl", "solver_types.jl"] -->
<!-- ``` -->
