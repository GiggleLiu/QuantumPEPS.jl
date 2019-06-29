using Documenter, QuantumPEPS

makedocs(;
    modules=[QuantumPEPS],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/GiggleLiu/QuantumPEPS.jl/blob/{commit}{path}#L{line}",
    sitename="QuantumPEPS.jl",
    authors="JinGuo Liu",
    assets=String[],
)

deploydocs(;
    repo="github.com/GiggleLiu/QuantumPEPS.jl",
)
