# API-INDEX


## MODULE: ParallelAccelerator

---

## Internal

[__init__()](ParallelAccelerator.md#method____init__.1)  Called when the package is loaded to do initialization.

[embed()](ParallelAccelerator.md#method__embed.1)  This version of embed tries to use JULIA_HOME to find the root of the source distribution.

[embed(julia_root)](ParallelAccelerator.md#method__embed.2)  Call this function if you want to embed binary-code of ParallelAccelerator into your Julia build to speed-up @acc compilation time.

[getPackageRoot()](ParallelAccelerator.md#method__getpackageroot.1)  Generate a file path to the directory above the one containing this source file.

[getPseMode()](ParallelAccelerator.md#method__getpsemode.1)  Return internal mode number by looking up environment variable "PROSPECT_MODE".

[getTaskMode()](ParallelAccelerator.md#method__gettaskmode.1)  Return internal mode number by looking up environment variable "PROSPECT_TASK_MODE".

## MODULE: ParallelAccelerator.DomainIR

---

## Internal

[lookupConstDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode})](DomainIR.md#method__lookupconstdef.1)  Look up a definition of a variable only when it is const or assigned once.

[lookupConstDefForArg(state::ParallelAccelerator.DomainIR.IRState,  s)](DomainIR.md#method__lookupconstdefforarg.1)  Look up a definition of a variable recursively until the RHS is no-longer just a variable.

[lookupDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode})](DomainIR.md#method__lookupdef.1)  Look up a definition of a variable.

[lookupDefInAllScopes(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode})](DomainIR.md#method__lookupdefinallscopes.1)  Look up a definition of a variable throughout nested states until a definition is found.

[updateDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode},  rhs)](DomainIR.md#method__updatedef.1)  Update the definition of a variable.

