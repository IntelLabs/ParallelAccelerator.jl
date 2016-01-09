# ParallelAccelerator.API.Stencil

## Exported

---

<a id="method__runstencil.1" class="lexicon_definition"></a>
#### runStencil(inputs...) [¶](#method__runstencil.1)
"runStencil" takes arguments in the form of "(kernel_function, A, B, C, ...,
iteration, border_style)" where "kernel_function" is a lambda that represent
stencil kernel, and "A", "B", "C", ... are arrays (of same dimension and size)
that will be traversed by the stencil, and they must match the number of
arguments of the stencil kernel. "iteration" and "border_style" are optional,
and if present, "iteration" is the number of steps to repeat the stencil
computation, and "border_style" is a symbol of the following:

    :oob_src_zero, returns zero from a read when it is out-of-bound. 

    :oob_dst_zero, writes zero to destination array in an assigment should any read in its right-hand-side become out-of-bound.

    :oob_wraparound, wraps around the index (with respect to source array dimension and sizes) of a read operation when it is out-of-bound.

    :oob_skip, skips write to destination array in an assignment should any read in its right-hand-side become out-of-bound.

The "kernel_function" should take a set of arrays as input, and in the function
body only index them with relative indices as if there is a cursor (at index 0)
traversing them. It may contain more than one assignment to any input arrays,
but such writes must always be indexed at 0 to guarantee write operation never
goes out-of-bound. Also care must be taken when the same array is both read
from and written into in the kernel function, as they'll result in
non-deterministic behavior when the stencil is parallelized by ParallelAccelerator. 

The "kernel_function" may optionally access scalar variables from outerscope,
but no write is permitted.  It may optinally contain a return statement, which
acts as a specification for buffer swapping when it is an interative stencil
loop (ISL). The number of arrays returned must match the input arrays.  If it
is not an ISL, it should always return nothing.

This function is a reference implementation of stencil in native Julia for two
purposes: to verify expected return result, and to make sure user code type
checks. It runs very very slow, so any real usage should go through ParallelAccelerator
optimizations.

"runStencil" always returns nothing.


*source:*
[ParallelAccelerator/src/api-stencil.jl:142](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/api-stencil.jl#L142)

## Internal

---

<a id="method__process_node.1" class="lexicon_definition"></a>
#### process_node(node,  state,  top_level_number,  is_top_level,  read) [¶](#method__process_node.1)
This function is a AstWalker callback.


*source:*
[ParallelAccelerator/src/api-stencil.jl:377](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/api-stencil.jl#L377)

---

<a id="macro___comprehend.1" class="lexicon_definition"></a>
#### @comprehend(ast) [¶](#macro___comprehend.1)
Translate all comprehension in an AST into equivalent code that uses cartesianarray call.


*source:*
[ParallelAccelerator/src/api-stencil.jl:392](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/api-stencil.jl#L392)

