var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = NonSmoothSolvers","category":"page"},{"location":"#NonSmoothSolvers","page":"Home","title":"NonSmoothSolvers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for NonSmoothSolvers.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [NonSmoothSolvers]","category":"page"},{"location":"#NonSmoothSolvers.GradientSampling","page":"Home","title":"NonSmoothSolvers.GradientSampling","text":"Gradient sampling algorthm.\n\n\n\n\n\n","category":"type"},{"location":"#NonSmoothSolvers.OptimizationState","page":"Home","title":"NonSmoothSolvers.OptimizationState","text":"OptimizationState\n\nStores information after one iteration of the optimizer. Generic information is stored explicitly in the struct, custom information may be stored in the field additionalinfo::NamedTuple.\n\n\n\n\n\n","category":"type"},{"location":"#NonSmoothSolvers.OptimizerParams","page":"Home","title":"NonSmoothSolvers.OptimizerParams","text":"OptimizerParams\n\nGeneric parameters for optim algs\n\n\n\n\n\n","category":"type"},{"location":"#NonSmoothSolvers.VUbundle","page":"Home","title":"NonSmoothSolvers.VUbundle","text":"Parameters:\n\nϵ: overall precision required\nm: sufficient decrease parameter\nμlow: minimal prox parameter (μ is inverse of γ). Higher μ means smaller serious steps, but less null steps\n\n\n\n\n\n","category":"type"},{"location":"#NonSmoothSolvers.VUbundleState","page":"Home","title":"NonSmoothSolvers.VUbundleState","text":"Parameters:\n\nσ: in (0, 0.5!], lower values enforce higher precision on each prox point approximation,\n\n\n\n\n\n","category":"type"},{"location":"#NonSmoothSolvers.linesearch_nsbfgs-NTuple{5, Any}","page":"Home","title":"NonSmoothSolvers.linesearch_nsbfgs","text":"linesearch_nsbfgs\n\nNonsmooth linesearch from Nonsmooth optimization via quasi-Newton methods, Lewis & Overton, 2013.\n\n\n\n\n\n","category":"method"},{"location":"#NonSmoothSolvers.optimize!-Union{Tuple{O}, Tuple{Any, O, Any}} where O<:NonSmoothSolvers.Optimizer","page":"Home","title":"NonSmoothSolvers.optimize!","text":"optimize!(pb, optimizer::NonSmoothSolvers.Optimizer, initial_x; state, optimstate_extensions, optparams) -> Tuple{Any, Any}\n\n\nCall the optimizer on problem pb, with initial point initial_x. Returns a tuple containing the final iterate vector and a trace.\n\nFeatures:\n\ntiming of the update_iterate method only;\nsaves basic information of each iteration in a vector of OptimizationState, the so-called trace;\nthe information saved at each iterate may be enriched by the user by providing a name and callback function via the optimstate_extension argument.\n\nExample\n\ngetx(o, os) = os.x\noptimstate_extensions = OrderedDict{Symbol, Function}(:x => getx)\n\noptimize!(pb, o, xclose; optparams, optimstate_extensions)\n\n\n\n\n\n","category":"method"},{"location":"#NonSmoothSolvers.update_iterate!-Union{Tuple{Tf}, Tuple{NonSmoothSolvers.GradientSamplingState{Tf}, GradientSampling, Any}} where Tf","page":"Home","title":"NonSmoothSolvers.update_iterate!","text":"update_iterate!(state, gs::GradientSampling, pb)\n\nNOTE: each iteration is costly. This can be explored with NonSmoothProblems.to. On the maxquadBGLS problem  ────────────────────────────────────────────────────────────────────────────────────────────────────                                                             Time                    Allocations                                                    ───────────────────────   ────────────────────────                  Tot / % measured:                      103ms /  93.9%           9.50MiB /  96.1%\n\nSection                                   ncalls     time    %tot     avg     alloc    %tot      avg  ────────────────────────────────────────────────────────────────────────────────────────────────────  updateiterate!                              100   95.6ms   98.7%   956μs   8.92MiB   97.8%  91.4KiB    GS 2. minimum norm (sub)gradient           100   83.3ms   86.0%   833μs   3.56MiB   39.0%  36.4KiB    GS 1. sampling points, eval gradients      100   7.39ms    7.6%  73.9μs   3.85MiB   42.2%  39.4KiB    GS 4. Update parameters                    100   4.24ms    4.4%  42.4μs   1.40MiB   15.3%  14.3KiB    GS 5. diff check                           100    288μs    0.3%  2.88μs    117KiB    1.3%  1.17KiB    GS 3. Termination                          100   41.1μs    0.0%   411ns     0.00B    0.0%    0.00B  buildoptimstate                             100   1.19ms    1.2%  11.9μs    207KiB    2.2%  2.07KiB  CV check                                     100   47.0μs    0.0%   470ns     0.00B    0.0%    0.00B  ────────────────────────────────────────────────────────────────────────────────────────────────────\n\n\n\n\n\n","category":"method"}]
}
