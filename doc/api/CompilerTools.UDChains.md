# CompilerTools.UDChains

## Internal

---

<a id="method__getorcreate.1" class="lexicon_definition"></a>
#### getOrCreate(live::Dict{Symbol, Set{T}},  s::Symbol) [¶](#method__getorcreate.1)
Get the set of definition blocks reaching this block for a given symbol "s".


*source:*
[CompilerTools/src/udchains.jl:49](file:///home/etotoni/.julia/v0.4/CompilerTools/src/udchains.jl)

---

<a id="method__getorcreate.2" class="lexicon_definition"></a>
#### getOrCreate(udchains::Dict{CompilerTools.LivenessAnalysis.BasicBlock, CompilerTools.UDChains.UDInfo},  bb::CompilerTools.LivenessAnalysis.BasicBlock) [¶](#method__getorcreate.2)
Get the UDInfo for a specified basic block "bb" or create one if it doesn't already exist.


*source:*
[CompilerTools/src/udchains.jl:59](file:///home/etotoni/.julia/v0.4/CompilerTools/src/udchains.jl)

---

<a id="method__getudchains.1" class="lexicon_definition"></a>
#### getUDChains(bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__getudchains.1)
Get the Use-Definition chains at a basic block level given LivenessAnalysis.BlockLiveness as input in "bl".


*source:*
[CompilerTools/src/udchains.jl:105](file:///home/etotoni/.julia/v0.4/CompilerTools/src/udchains.jl)

---

<a id="method__printlabels.1" class="lexicon_definition"></a>
#### printLabels(level,  dict) [¶](#method__printlabels.1)
Print a live in or live out dictionary in a nice way if the debug level is set high enough.


*source:*
[CompilerTools/src/udchains.jl:82](file:///home/etotoni/.julia/v0.4/CompilerTools/src/udchains.jl)

---

<a id="method__printset.1" class="lexicon_definition"></a>
#### printSet(level,  s) [¶](#method__printset.1)
Print the set part of a live in or live out dictiononary in a nice way if the debug level is set high enough.


*source:*
[CompilerTools/src/udchains.jl:69](file:///home/etotoni/.julia/v0.4/CompilerTools/src/udchains.jl)

---

<a id="method__printudinfo.1" class="lexicon_definition"></a>
#### printUDInfo(level,  ud) [¶](#method__printudinfo.1)
Print UDChains in a nice way if the debug level is set high enough.


*source:*
[CompilerTools/src/udchains.jl:93](file:///home/etotoni/.julia/v0.4/CompilerTools/src/udchains.jl)

---

<a id="type__udinfo.1" class="lexicon_definition"></a>
#### CompilerTools.UDChains.UDInfo [¶](#type__udinfo.1)
Contains the UDchains for one basic block.


*source:*
[CompilerTools/src/udchains.jl:37](file:///home/etotoni/.julia/v0.4/CompilerTools/src/udchains.jl)

