using Documenter
using SequentialQuadraticProgramming

makedocs(
    sitename = "SequentialQuadraticProgramming.jl",
    modules = [SequentialQuadraticProgramming],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials" => [
            "Basic Constrained Optimization" => "tutorials/basic_constrained.md",
            "Nonlinear Constraints" => "tutorials/nonlinear_constraints.md",
            "JuMP Integration" => "tutorials/jump_integration.md",
            "AD Backends" => "tutorials/ad_backends.md",
            "Globalization Strategies" => "tutorials/globalization.md",
        ],
        "API Reference" => [
            "Problem Definition" => "api/problem.md",
            "Solver Interface" => "api/solver.md",
            "QP Subsolvers" => "api/qp_solvers.md",
            "Hessian Updates" => "api/hessian.md",
            "Derivatives" => "api/derivatives.md",
        ],
        "Design & Theory" => [
            "Algorithm Overview" => "design/algorithm.md",
            "Convergence & Tuning" => "design/convergence.md",
        ],
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs, :docs_block, :cross_references],
    remotes = nothing,
)

deploydocs(
    repo = "github.com/haavardhvarnes/SequentialQuadraticProgramming.jl.git",
    push_preview = true,
)
