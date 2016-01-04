# ParallelAccelerator.Comprehension

## Exported

---

<a id="macro___comprehend.1" class="lexicon_definition"></a>
#### @comprehend(ast) [¶](#macro___comprehend.1)
Translate all comprehension in an AST into equivalent code that uses cartesianarray call.


*source:*
[ParallelAccelerator/src/comprehension.jl:82](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/comprehension.jl)

## Internal

---

<a id="method__comprehension_to_cartesianarray.1" class="lexicon_definition"></a>
#### comprehension_to_cartesianarray(ast) [¶](#method__comprehension_to_cartesianarray.1)
Translate an ast whose head is :comprehension into equivalent code that uses cartesianarray call.


*source:*
[ParallelAccelerator/src/comprehension.jl:35](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/comprehension.jl)

---

<a id="method__process_node.1" class="lexicon_definition"></a>
#### process_node(node,  state,  top_level_number,  is_top_level,  read) [¶](#method__process_node.1)
This function is a AstWalker callback.


*source:*
[ParallelAccelerator/src/comprehension.jl:64](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/comprehension.jl)

