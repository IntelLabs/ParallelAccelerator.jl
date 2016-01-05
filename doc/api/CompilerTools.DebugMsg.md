# CompilerTools.DebugMsg

## Exported

---

<a id="method__init.1" class="lexicon_definition"></a>
#### init() [¶](#method__init.1)
A module using DebugMsg must call DebugMsg.init(), which expands to several local definitions
that provide three functions: set_debug_level, dprint, dprintln.


*source:*
[CompilerTools/src/debug.jl:40](file:///home/etotoni/.julia/v0.4/CompilerTools/src/debug.jl)

## Internal

---

<a id="global__prospect_dev_mode.1" class="lexicon_definition"></a>
#### PROSPECT_DEV_MODE [¶](#global__prospect_dev_mode.1)
When this module is first loaded, we check if PROSPECT_DEV_MODE is set in environment.
If it is not, then all debug messages will be surpressed.


*source:*
[CompilerTools/src/debug.jl:34](file:///home/etotoni/.julia/v0.4/CompilerTools/src/debug.jl)

