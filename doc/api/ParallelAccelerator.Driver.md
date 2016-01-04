# ParallelAccelerator.Driver

## Exported

---

<a id="method__captureoperators.1" class="lexicon_definition"></a>
#### captureOperators(func,  ast,  sig) [¶](#method__captureoperators.1)
A pass that translates supported operators and function calls to
those defined in ParallelAccelerator.API.


*source:*
[ParallelAccelerator/src/driver.jl:98](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/driver.jl)

---

<a id="method__runstencilmacro.1" class="lexicon_definition"></a>
#### runStencilMacro(func,  ast,  sig) [¶](#method__runstencilmacro.1)
Pass that translates runStencil call in the same way as a macro would do.
This is only used when PROSPECT_MODE is off.


*source:*
[ParallelAccelerator/src/driver.jl:107](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/driver.jl)

---

<a id="method__tocartesianarray.1" class="lexicon_definition"></a>
#### toCartesianArray(func,  ast,  sig) [¶](#method__tocartesianarray.1)
Pass for comprehension to cartesianarray translation.


*source:*
[ParallelAccelerator/src/driver.jl:114](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/driver.jl)

