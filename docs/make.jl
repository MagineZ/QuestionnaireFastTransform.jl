using QuestionnaireFastTransform
using Documenter

DocMeta.setdocmeta!(QuestionnaireFastTransform, :DocTestSetup, :(using QuestionnaireFastTransform); recursive=true)

makedocs(;
    modules=[QuestionnaireFastTransform],
    authors="Pei-Chun Su <b94401079@gmail.com>",
    sitename="QuestionnaireFastTransform.jl",
    format=Documenter.HTML(;
        canonical="https://MagineZ.github.io/QuestionnaireFastTransform.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MagineZ/QuestionnaireFastTransform.jl",
    devbranch="main",
)
