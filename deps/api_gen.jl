using Lexicon
using ParallelAccelerator

index = Index()
update!(index, save("../doc/api/ParallelAccelerator.md",ParallelAccelerator))
update!(index, save("../doc/api/DomainIR.md",ParallelAccelerator.DomainIR))
update!(index, save("../doc/api/ParallelIR.md",ParallelAccelerator.ParallelIR))
save("../doc/api/api-index.md", index)
