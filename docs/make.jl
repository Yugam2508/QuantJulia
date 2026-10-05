# Build the documentation site:
#     julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#     julia --project=docs docs/make.jl
# The theory notes stay where they are (docs/notes, docs/*.md) so they read
# fine on GitHub; they are copied into the Documenter source tree at build time.

using Documenter
using QuantJulia

const DOCS = @__DIR__
const GEN = joinpath(DOCS, "src", "generated")
rm(GEN; recursive = true, force = true)
mkpath(GEN)
for f in ("roadmap.md", "rough_heston.md", "rough_heston_spec.md")
    cp(joinpath(DOCS, f), joinpath(GEN, f))
end
notes = sort(filter(endswith(".md"), readdir(joinpath(DOCS, "notes"))))
for f in notes
    cp(joinpath(DOCS, "notes", f), joinpath(GEN, f))
end

makedocs(
    sitename = "QuantJulia",
    modules = [QuantJulia],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             edit_link = "main"),
    pages = [
        "Home" => "index.md",
        "User guide" => "guide.md",
        "API reference" => "api.md",
        "Results: rough vs classical" => "generated/rough_heston.md",
        "Theory notes" => ["generated/" * f for f in notes],
        "Roadmap" => "generated/roadmap.md",
        "Rough Heston spec" => "generated/rough_heston_spec.md",
    ],
    checkdocs = :exports,
    # The notes link to repository files (src/…, results/…) by relative path;
    # those resolve on GitHub but not inside the site.
    warnonly = [:cross_references],
)

deploydocs(repo = "github.com/Yugam2508/QuantJulia.git", devbranch = "main",
           push_preview = false)
