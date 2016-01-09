# ParallelAccelerator.Comprehension

## Exported

---

<a id="macro___comprehend.1" class="lexicon_definition"></a>
#### @comprehend(ast) [¶](#macro___comprehend.1)
Translate all comprehension in an AST into equivalent code that uses cartesianarray call.


*source:*
[ParallelAccelerator/src/comprehension.jl:82](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/comprehension.jl#L82)

## Internal

---

<a id="method__comprehension_to_cartesianarray.1" class="lexicon_definition"></a>
#### comprehension_to_cartesianarray(ast) [¶](#method__comprehension_to_cartesianarray.1)
Translate an ast whose head is :comprehension into equivalent code that uses cartesianarray call.


*source:*
[ParallelAccelerator/src/comprehension.jl:35](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/comprehension.jl#L35)

---

<a id="method__process_node.1" class="lexicon_definition"></a>
#### process_node(node,  state,  top_level_number,  is_top_level,  read) [¶](#method__process_node.1)
This function is a AstWalker callback.


*source:*
[ParallelAccelerator/src/comprehension.jl:64](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/comprehension.jl#L64)

