# ParallelAccelerator.Driver

## Exported

---

<a id="method__captureoperators.1" class="lexicon_definition"></a>
#### captureOperators(func,  ast,  sig) [¶](#method__captureoperators.1)
A pass that translates supported operators and function calls to
those defined in ParallelAccelerator.API.


*source:*
[ParallelAccelerator/src/driver.jl:98](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/driver.jl#L98)

---

<a id="method__runstencilmacro.1" class="lexicon_definition"></a>
#### runStencilMacro(func,  ast,  sig) [¶](#method__runstencilmacro.1)
Pass that translates runStencil call in the same way as a macro would do.
This is only used when PROSPECT_MODE is off.


*source:*
[ParallelAccelerator/src/driver.jl:107](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/driver.jl#L107)

---

<a id="method__tocartesianarray.1" class="lexicon_definition"></a>
#### toCartesianArray(func,  ast,  sig) [¶](#method__tocartesianarray.1)
Pass for comprehension to cartesianarray translation.


*source:*
[ParallelAccelerator/src/driver.jl:114](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/driver.jl#L114)

