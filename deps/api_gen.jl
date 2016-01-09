using Docile
using Lexicon
using ParallelAccelerator

index = Index()
pa_mods = Docile.Collector.submodules(ParallelAccelerator)

for mod in pa_mods
    update!(index, save("../doc/api/$mod.md",mod))
end
save("../doc/api/api-index.md", index)
