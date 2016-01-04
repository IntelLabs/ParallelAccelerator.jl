# ParallelAccelerator.DomainIR

## Internal

---

<a id="method__lookupconstdef.1" class="lexicon_definition"></a>
#### lookupConstDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode}) [¶](#method__lookupconstdef.1)
Look up a definition of a variable only when it is const or assigned once.
Return nothing If none is found.


*source:*
[ParallelAccelerator/src/domain-ir.jl:259](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/domain-ir.jl)

---

<a id="method__lookupconstdefforarg.1" class="lexicon_definition"></a>
#### lookupConstDefForArg(state::ParallelAccelerator.DomainIR.IRState,  s) [¶](#method__lookupconstdefforarg.1)
Look up a definition of a variable recursively until the RHS is no-longer just a variable.
Return the last rhs If found, or the input variable itself otherwise.


*source:*
[ParallelAccelerator/src/domain-ir.jl:273](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/domain-ir.jl)

---

<a id="method__lookupdef.1" class="lexicon_definition"></a>
#### lookupDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode}) [¶](#method__lookupdef.1)
Look up a definition of a variable.
Return nothing If none is found.


*source:*
[ParallelAccelerator/src/domain-ir.jl:250](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/domain-ir.jl)

---

<a id="method__lookupdefinallscopes.1" class="lexicon_definition"></a>
#### lookupDefInAllScopes(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode}) [¶](#method__lookupdefinallscopes.1)
Look up a definition of a variable throughout nested states until a definition is found.
Return nothing If none is found.


*source:*
[ParallelAccelerator/src/domain-ir.jl:286](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/domain-ir.jl)

---

<a id="method__updatedef.1" class="lexicon_definition"></a>
#### updateDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode},  rhs) [¶](#method__updatedef.1)
Update the definition of a variable.


*source:*
[ParallelAccelerator/src/domain-ir.jl:235](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/domain-ir.jl)

