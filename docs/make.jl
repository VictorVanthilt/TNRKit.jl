using TNRKit
using Documenter
using DocumenterCitations

bibpath = joinpath(@__DIR__, "src", "assets", "tnrkit.bib")
bib = CitationBibliography(bibpath; style=:authoryear)

makedocs(
    sitename="Documentation",
    pages = [
        "Home" => "index.md"
        "Library" => "lib/lib.md"
    ],
    
    plugins = [bib]
)

# deploydocs(; repo="github.com/VictorVanthilt/TNRKit.jl.git")
