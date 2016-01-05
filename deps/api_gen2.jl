using Docile
using Lexicon
using ParallelAccelerator

index = Index()
pa_mods = Docile.Collector.submodules(ParallelAccelerator)
ct_mods = Docile.Collector.submodules(CompilerTools)
mods = union(ct_mods, pa_mods)

for mod in mods
    update!(index, save("../doc/api/$mod.md",mod))
end
save("../doc/api/api-index.md", index)
