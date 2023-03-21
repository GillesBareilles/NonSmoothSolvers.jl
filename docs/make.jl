using NonSmoothSolvers
using Documenter

DocMeta.setdocmeta!(
    NonSmoothSolvers,
    :DocTestSetup,
    :(using NonSmoothSolvers);
    recursive = true,
)

makedocs(;
    modules = [NonSmoothSolvers],
    authors = "Gilles Bareilles <gilles.bareilles@protonmail.com>",
    sitename = "NonSmoothSolvers.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://GillesBareilles.github.io/NonSmoothSolvers.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "VU-algorithm" => "VUalg.md"
    ],
)

deploydocs(; repo = "github.com/GillesBareilles/NonSmoothSolvers.jl", devbranch = "master")
