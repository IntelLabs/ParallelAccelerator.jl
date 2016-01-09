# ParallelAccelerator.ParallelIR

## Exported

---

<a id="method__astwalk.1" class="lexicon_definition"></a>
#### AstWalk(ast,  callback,  cbdata) [¶](#method__astwalk.1)
ParallelIR version of AstWalk.
Invokes the DomainIR version of AstWalk and provides the parallel IR AstWalk callback AstWalkCallback.

Parallel IR AstWalk calls Domain IR AstWalk which in turn calls CompilerTools.AstWalker.AstWalk.
For each AST node, CompilerTools.AstWalker.AstWalk calls Domain IR callback to give it a chance to handle the node if it is a Domain IR node.
Likewise, Domain IR callback first calls Parallel IR callback to give it a chance to handle Parallel IR nodes.
The Parallel IR callback similarly first calls the user-level callback to give it a chance to process the node.
If a callback returns "nothing" it means it didn't modify that node and that the previous code should process it.
The Parallel IR callback will return "nothing" if the node isn't a Parallel IR node.
The Domain IR callback will return "nothing" if the node isn't a Domain IR node.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:5229](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L5229)

---

<a id="method__pirinplace.1" class="lexicon_definition"></a>
#### PIRInplace(x) [¶](#method__pirinplace.1)
If set to non-zero, perform the phase where non-inplace maps are converted to inplace maps to reduce allocations.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4139](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4139)

---

<a id="method__pirnumsimplify.1" class="lexicon_definition"></a>
#### PIRNumSimplify(x) [¶](#method__pirnumsimplify.1)
Specify the number of passes over the AST that do things like hoisting and other rearranging to maximize fusion.
DEPRECATED.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1950](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1950)

---

<a id="method__pirrunastasks.1" class="lexicon_definition"></a>
#### PIRRunAsTasks(x) [¶](#method__pirrunastasks.1)
Debugging feature to specify the number of tasks to create and to stop thereafter.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2140](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2140)

---

<a id="method__pirsetfuselimit.1" class="lexicon_definition"></a>
#### PIRSetFuseLimit(x) [¶](#method__pirsetfuselimit.1)
Control how many parfor can be fused for testing purposes.
    -1 means fuse all possible parfors.
    0  means don't fuse any parfors.
    1+ means fuse the specified number of parfors but then stop fusing beyond that.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1944](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1944)

---

<a id="method__pirshortcutarrayassignment.1" class="lexicon_definition"></a>
#### PIRShortcutArrayAssignment(x) [¶](#method__pirshortcutarrayassignment.1)
Enables an experimental mode where if there is a statement a = b and they are arrays and b is not live-out then 
use a special assignment node like a move assignment in C++.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4165](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4165)

---

<a id="method__pirtaskgraphmode.1" class="lexicon_definition"></a>
#### PIRTaskGraphMode(x) [¶](#method__pirtaskgraphmode.1)
Control how blocks of code are made into tasks.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:589](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L589)

---

<a id="method__from_exprs.1" class="lexicon_definition"></a>
#### from_exprs(ast::Array{Any, 1},  depth,  state) [¶](#method__from_exprs.1)
Process an array of expressions.
Differentiate between top-level arrays of statements and arrays of expression that may occur elsewhere than the :body Expr.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2567](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2567)

---

<a id="type__pirloopnest.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.PIRLoopNest [¶](#type__pirloopnest.1)
Holds the information about a loop in a parfor node.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:89](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L89)

---

<a id="type__pirparforast.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.PIRParForAst [¶](#type__pirparforast.1)
The parfor AST node type.
While we are lowering domain IR to parfors and fusing we use this representation because it
makes it easier to associate related statements before and after the loop to the loop itself.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:333](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L333)

---

<a id="type__pirreduction.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.PIRReduction [¶](#type__pirreduction.1)
Holds the information about a reduction in a parfor node.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:99](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L99)

## Internal

---

<a id="method__astwalkcallback.1" class="lexicon_definition"></a>
#### AstWalkCallback(x::Expr,  dw::ParallelAccelerator.ParallelIR.DirWalk,  top_level_number::Int64,  is_top_level::Bool,  read::Bool) [¶](#method__astwalkcallback.1)
AstWalk callback that handles ParallelIR AST node types.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:5054](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L5054)

---

<a id="method__equivalenceclassesadd.1" class="lexicon_definition"></a>
#### EquivalenceClassesAdd(ec::ParallelAccelerator.ParallelIR.EquivalenceClasses,  sym::Symbol) [¶](#method__equivalenceclassesadd.1)
Add a symbol as part of a new equivalence class if the symbol wasn't already in an equivalence class.
Return the equivalence class for the symbol.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:152](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L152)

---

<a id="method__equivalenceclassesclear.1" class="lexicon_definition"></a>
#### EquivalenceClassesClear(ec::ParallelAccelerator.ParallelIR.EquivalenceClasses) [¶](#method__equivalenceclassesclear.1)
Clear an equivalence class.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:166](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L166)

---

<a id="method__equivalenceclassesmerge.1" class="lexicon_definition"></a>
#### EquivalenceClassesMerge(ec::ParallelAccelerator.ParallelIR.EquivalenceClasses,  merge_to::Symbol,  merge_from::Symbol) [¶](#method__equivalenceclassesmerge.1)
At some point we realize that two arrays must have the same dimensions but up until that point
we might not have known that.  In which case they will start in different equivalence classes,
merge_to and merge_from, but need to be combined into one equivalence class.
Go through the equivalence class dictionary and for any symbol belonging to the merge_from
equivalence class, change it to now belong to the merge_to equivalence class.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:136](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L136)

---

<a id="method__pirbbreorder.1" class="lexicon_definition"></a>
#### PIRBbReorder(x) [¶](#method__pirbbreorder.1)
If set to non-zero, perform the bubble-sort like reordering phase to coalesce more parfor nodes together for fusion.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4155](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4155)

---

<a id="method__pirhoistallocation.1" class="lexicon_definition"></a>
#### PIRHoistAllocation(x) [¶](#method__pirhoistallocation.1)
If set to non-zero, perform the rearrangement phase that tries to moves alllocations outside of loops.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4147](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4147)

---

<a id="method__typedexpr.1" class="lexicon_definition"></a>
#### TypedExpr(typ,  rest...) [¶](#method__typedexpr.1)
This should pretty always be used instead of Expr(...) to form an expression as it forces the typ to be provided.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:80](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L80)

---

<a id="method__addunknownarray.1" class="lexicon_definition"></a>
#### addUnknownArray(x::Union{GenSym, Symbol},  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__addunknownarray.1)
Given an array whose name is in "x", allocate a new equivalence class for this array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:866](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L866)

---

<a id="method__addunknownrange.1" class="lexicon_definition"></a>
#### addUnknownRange(x::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1},  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__addunknownrange.1)
Given an array of RangeExprs describing loop nest ranges, allocate a new equivalence class for this range.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:876](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L876)

---

<a id="method__add_merge_correlations.1" class="lexicon_definition"></a>
#### add_merge_correlations(old_sym::Union{GenSym, Symbol},  new_sym::Union{GenSym, Symbol},  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__add_merge_correlations.1)
If we somehow determine that two arrays must be the same length then 
get the equivalence classes for the two arrays and merge those equivalence classes together.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:917](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L917)

---

<a id="method__asarray.1" class="lexicon_definition"></a>
#### asArray(x) [¶](#method__asarray.1)
Return one element array with element x.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:5045](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L5045)

---

<a id="method__augment_sn.1" class="lexicon_definition"></a>
#### augment_sn(dim::Int64,  index_vars,  range::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1}) [¶](#method__augment_sn.1)
Make sure the index parameters to arrayref or arrayset are Int64 or SymbolNode.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:60](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L60)

---

<a id="method__call_instruction_count.1" class="lexicon_definition"></a>
#### call_instruction_count(args,  state::ParallelAccelerator.ParallelIR.eic_state,  debug_level) [¶](#method__call_instruction_count.1)
Generate an instruction count estimate for a call instruction.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:352](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L352)

---

<a id="method__checkandaddsymbolcorrelation.1" class="lexicon_definition"></a>
#### checkAndAddSymbolCorrelation(lhs::Union{GenSym, Symbol},  state,  dim_array) [¶](#method__checkandaddsymbolcorrelation.1)
Make sure all the dimensions are SymbolNodes.
Make sure each dimension variable is assigned to only once in the function.
Extract just the dimension variables names into dim_names and then register the correlation from lhs to those dimension names.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3577](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3577)

---

<a id="method__convertunsafe.1" class="lexicon_definition"></a>
#### convertUnsafe(stmt) [¶](#method__convertunsafe.1)
Remove unsafe array access Symbols from the incoming "stmt".
Returns the updated statement if something was modifed, else returns "nothing".


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:950](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L950)

---

<a id="method__convertunsafeorelse.1" class="lexicon_definition"></a>
#### convertUnsafeOrElse(stmt) [¶](#method__convertunsafeorelse.1)
Try to remove unsafe array access Symbols from the incoming "stmt".  If successful, then return the updated
statement, else return the unmodified statement.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:969](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L969)

---

<a id="method__convertunsafewalk.1" class="lexicon_definition"></a>
#### convertUnsafeWalk(x::Expr,  state,  top_level_number,  is_top_level,  read) [¶](#method__convertunsafewalk.1)
The AstWalk callback to find unsafe arrayset and arrayref variants and
replace them with the regular Julia versions.  Sets the "found" flag
in the state when such a replacement is performed.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:918](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L918)

---

<a id="method__copy_propagate.1" class="lexicon_definition"></a>
#### copy_propagate(node::ANY,  data::ParallelAccelerator.ParallelIR.CopyPropagateState,  top_level_number,  is_top_level,  read) [¶](#method__copy_propagate.1)
In each basic block, if there is a "copy" (i.e., something of the form "a = b") then put
that in copies as copies[a] = b.  Then, later in the basic block if you see the symbol
"a" then replace it with "b".  Note that this is not SSA so "a" may be written again
and if it is then it must be removed from copies.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3148](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3148)

---

<a id="method__count_assignments.1" class="lexicon_definition"></a>
#### count_assignments(x,  symbol_assigns::Dict{Symbol, Int64},  top_level_number,  is_top_level,  read) [¶](#method__count_assignments.1)
AstWalk callback to count the number of static times that a symbol is assigne within a method.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1197](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1197)

---

<a id="method__create1d_array_access_desc.1" class="lexicon_definition"></a>
#### create1D_array_access_desc(array::SymbolNode) [¶](#method__create1d_array_access_desc.1)
Create an array access descriptor for "array".
Presumes that for point "i" in the iteration space that only index "i" is accessed.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:124](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L124)

---

<a id="method__create2d_array_access_desc.1" class="lexicon_definition"></a>
#### create2D_array_access_desc(array::SymbolNode) [¶](#method__create2d_array_access_desc.1)
Create an array access descriptor for "array".
Presumes that for points "(i,j)" in the iteration space that only indices "(i,j)" is accessed.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:134](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L134)

---

<a id="method__createinstructioncountestimate.1" class="lexicon_definition"></a>
#### createInstructionCountEstimate(the_parfor::ParallelAccelerator.ParallelIR.PIRParForAst,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__createinstructioncountestimate.1)
Takes a parfor and walks the body of the parfor and estimates the number of instruction needed for one instance of that body.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:557](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L557)

---

<a id="method__createloweredaliasmap.1" class="lexicon_definition"></a>
#### createLoweredAliasMap(dict1) [¶](#method__createloweredaliasmap.1)
Take a single-step alias map, e.g., a=>b, b=>c, and create a lowered dictionary, a=>c, b=>c, that
maps each array to the transitively lowered array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2127](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2127)

---

<a id="method__createmaplhstoparfor.1" class="lexicon_definition"></a>
#### createMapLhsToParfor(parfor_assignment,  the_parfor,  is_multi::Bool,  sym_to_type::Dict{Union{GenSym, Symbol}, DataType},  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__createmaplhstoparfor.1)
Creates a mapping between variables on the left-hand side of an assignment where the right-hand side is a parfor
and the arrays or scalars in that parfor that get assigned to the corresponding parts of the left-hand side.
Returns a tuple where the first element is a map for arrays between left-hand side and parfor and the second
element is a map for reduction scalars between left-hand side and parfor.
is_multi is true if the assignment is a fusion assignment.
parfor_assignment is the AST of the whole expression.
the_parfor is the PIRParForAst type part of the incoming assignment.
sym_to_type is an out parameter that maps symbols in the output mapping to their types.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2061](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2061)

---

<a id="method__createstatevar.1" class="lexicon_definition"></a>
#### createStateVar(state,  name,  typ,  access) [¶](#method__createstatevar.1)
Add a local variable to the current function's lambdaInfo.
Returns a symbol node of the new variable.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:759](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L759)

---

<a id="method__createtempforarray.1" class="lexicon_definition"></a>
#### createTempForArray(array_sn::Union{GenSym, Symbol, SymbolNode},  unique_id::Int64,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__createtempforarray.1)
Create a temporary variable that is parfor private to hold the value of an element of an array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:767](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L767)

---

<a id="method__createtempforarray.2" class="lexicon_definition"></a>
#### createTempForArray(array_sn::Union{GenSym, Symbol, SymbolNode},  unique_id::Int64,  state::ParallelAccelerator.ParallelIR.expr_state,  temp_type) [¶](#method__createtempforarray.2)
Create a temporary variable that is parfor private to hold the value of an element of an array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:767](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L767)

---

<a id="method__createtempforrangeoffset.1" class="lexicon_definition"></a>
#### createTempForRangeOffset(num_used,  ranges::Array{ParallelAccelerator.ParallelIR.RangeData, 1},  unique_id::Int64,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__createtempforrangeoffset.1)
Create a variable to hold the offset of a range offset from the start of the array.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:452](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L452)

---

<a id="method__createtempforrangedarray.1" class="lexicon_definition"></a>
#### createTempForRangedArray(array_sn::Union{GenSym, Symbol, SymbolNode},  range::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1},  unique_id::Int64,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__createtempforrangedarray.1)
Create a temporary variable that is parfor private to hold the value of an element of an array.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:479](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L479)

---

<a id="method__create_array_access_desc.1" class="lexicon_definition"></a>
#### create_array_access_desc(array::SymbolNode) [¶](#method__create_array_access_desc.1)
Create an array access descriptor for "array".


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:144](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L144)

---

<a id="method__create_equivalence_classes.1" class="lexicon_definition"></a>
#### create_equivalence_classes(node::Expr,  state::ParallelAccelerator.ParallelIR.expr_state,  top_level_number::Int64,  is_top_level::Bool,  read::Bool) [¶](#method__create_equivalence_classes.1)
AstWalk callback to determine the array equivalence classes.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3727](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3727)

---

<a id="method__dfsvisit.1" class="lexicon_definition"></a>
#### dfsVisit(swd::ParallelAccelerator.ParallelIR.StatementWithDeps,  vtime::Int64,  topo_sort::Array{ParallelAccelerator.ParallelIR.StatementWithDeps, N}) [¶](#method__dfsvisit.1)
Construct a topological sort of the dependence graph.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4186](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4186)

---

<a id="method__estimateinstrcount.1" class="lexicon_definition"></a>
#### estimateInstrCount(ast::Expr,  state::ParallelAccelerator.ParallelIR.eic_state,  top_level_number,  is_top_level,  read) [¶](#method__estimateinstrcount.1)
AstWalk callback for estimating the instruction count.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:474](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L474)

---

<a id="method__extractarrayequivalencies.1" class="lexicon_definition"></a>
#### extractArrayEquivalencies(node::Expr,  state) [¶](#method__extractarrayequivalencies.1)
"node" is a domainIR node.  Take the arrays used in this node, create an array equivalence for them if they 
don't already have one and make sure they all share one equivalence class.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3516](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3516)

---

<a id="method__findselecteddimensions.1" class="lexicon_definition"></a>
#### findSelectedDimensions(inputInfo::Array{ParallelAccelerator.ParallelIR.InputInfo, 1},  state) [¶](#method__findselecteddimensions.1)
Given all the InputInfo for a Domain IR operation being lowered to Parallel IR,
determine the number of output dimensions for those arrays taking into account
that singly selected trailing dimensinos are eliminated.  Make sure that all such
arrays have the same output dimensions because this will match the loop nest size.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:959](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L959)

---

<a id="method__flattenparfor.1" class="lexicon_definition"></a>
#### flattenParfor(new_body,  the_parfor::ParallelAccelerator.ParallelIR.PIRParForAst) [¶](#method__flattenparfor.1)
Takes a new array of body statements in the process of construction in "new_body" and takes a parfor to add to that
body.  This parfor is in the nested (parfor code is in the parfor node itself) temporary form we use for fusion although 
pre-statements and post-statements are already elevated by this point.  We replace this nested form with a non-nested
form where we have a parfor_start and parfor_end to delineate the parfor code.


*source:*
[ParallelAccelerator/src/parallel-ir-flatten.jl:94](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-flatten.jl#L94)

---

<a id="method__form_and_simplify.1" class="lexicon_definition"></a>
#### form_and_simplify(ranges::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1}) [¶](#method__form_and_simplify.1)
For each entry in ranges, form a range length expression and simplify them.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1034](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1034)

---

<a id="method__form_and_simplify.2" class="lexicon_definition"></a>
#### form_and_simplify(rd::ParallelAccelerator.ParallelIR.RangeData) [¶](#method__form_and_simplify.2)
Convert one RangeData to some length expression and then simplify it.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1010](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1010)

---

<a id="method__from_asserteqshape.1" class="lexicon_definition"></a>
#### from_assertEqShape(node::Expr,  state) [¶](#method__from_asserteqshape.1)
Create array equivalences from an assertEqShape AST node.
There are two arrays in the args to assertEqShape.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2901](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2901)

---

<a id="method__from_assignment.1" class="lexicon_definition"></a>
#### from_assignment(lhs,  rhs,  depth,  state) [¶](#method__from_assignment.1)
Process an assignment expression.
Starts by recurisvely processing the right-hand side of the assignment.
Eliminates the assignment of a=b if a is dead afterwards and b has no side effects.
    Does some array equivalence class work which may be redundant given that we now run a separate equivalence class pass so consider removing that part of this code.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2984](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2984)

---

<a id="method__from_call.1" class="lexicon_definition"></a>
#### from_call(ast::Array{Any, 1},  depth,  state) [¶](#method__from_call.1)
Process a call AST node.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3106](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3106)

---

<a id="method__from_expr.1" class="lexicon_definition"></a>
#### from_expr(ast::Expr,  depth,  state::ParallelAccelerator.ParallelIR.expr_state,  top_level) [¶](#method__from_expr.1)
The main ParallelIR function for processing some node in the AST.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4876](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4876)

---

<a id="method__from_lambda.1" class="lexicon_definition"></a>
#### from_lambda(lambda::Expr,  depth,  state) [¶](#method__from_lambda.1)
Process a :lambda Expr.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1223](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1223)

---

<a id="method__from_root.1" class="lexicon_definition"></a>
#### from_root(function_name,  ast::Expr) [¶](#method__from_root.1)
The main ENTRY point into ParallelIR.
1) Do liveness analysis.
2) Convert mmap to mmap! where possible.
3) Do some code rearrangement (e.g., hoisting) to maximize later fusion.
4) Create array equivalence classes within the function.
5) Rearrange statements within a basic block to push domain operations to the bottom so more fusion.
6) Call the main from_expr to process the AST for the function.  This will
a) Lower domain IR to parallel IR AST nodes.
b) Fuse parallel IR nodes where possible.
c) Convert to task IR nodes if task mode enabled.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4535](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4535)

---

<a id="method__fullyloweralias.1" class="lexicon_definition"></a>
#### fullyLowerAlias(dict::Dict{Union{GenSym, Symbol}, Union{GenSym, Symbol}},  input::Union{GenSym, Symbol}) [¶](#method__fullyloweralias.1)
Given an "input" Symbol, use that Symbol as key to a dictionary.  While such a Symbol is present
in the dictionary replace it with the corresponding value from the dict.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2116](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2116)

---

<a id="method__fuse.1" class="lexicon_definition"></a>
#### fuse(body,  body_index,  cur,  state) [¶](#method__fuse.1)
Test whether we can fuse the two most recent parfor statements and if so to perform that fusion.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2158](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2158)

---

<a id="method__generate_instr_count.1" class="lexicon_definition"></a>
#### generate_instr_count(function_name,  signature) [¶](#method__generate_instr_count.1)
Try to figure out the instruction count for a given call.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:398](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L398)

---

<a id="method__getarrayelemtype.1" class="lexicon_definition"></a>
#### getArrayElemType(array::GenSym,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__getarrayelemtype.1)
Returns the element type of an Array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:731](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L731)

---

<a id="method__getarrayelemtype.2" class="lexicon_definition"></a>
#### getArrayElemType(array::SymbolNode,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__getarrayelemtype.2)
Returns the element type of an Array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:724](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L724)

---

<a id="method__getarrayelemtype.3" class="lexicon_definition"></a>
#### getArrayElemType(atyp::DataType) [¶](#method__getarrayelemtype.3)
Returns the element type of an Array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:711](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L711)

---

<a id="method__getarraynumdims.1" class="lexicon_definition"></a>
#### getArrayNumDims(array::GenSym,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__getarraynumdims.1)
Return the number of dimensions of an Array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:748](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L748)

---

<a id="method__getarraynumdims.2" class="lexicon_definition"></a>
#### getArrayNumDims(array::SymbolNode,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__getarraynumdims.2)
Return the number of dimensions of an Array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:739](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L739)

---

<a id="method__getconstdims.1" class="lexicon_definition"></a>
#### getConstDims(num_dim_inputs,  inputInfo::ParallelAccelerator.ParallelIR.InputInfo) [¶](#method__getconstdims.1)
In the case where a domain IR operation on an array creates a lower dimensional output,
the indexing expression needs the expression that selects those constant trailing dimensions
that are being dropped.  This function returns an array of those constant expressions for
the trailing dimensions.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:980](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L980)

---

<a id="method__getcorrelation.1" class="lexicon_definition"></a>
#### getCorrelation(sng::Union{GenSym, Symbol, SymbolNode},  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__getcorrelation.1)
Get the equivalence class of a domain IR input in inputInfo.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2024](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2024)

---

<a id="method__getfirstarraylens.1" class="lexicon_definition"></a>
#### getFirstArrayLens(prestatements,  num_dims) [¶](#method__getfirstarraylens.1)
Get the variable which holds the length of the first input array to a parfor.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1715](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1715)

---

<a id="method__getio.1" class="lexicon_definition"></a>
#### getIO(stmt_ids,  bb_statements) [¶](#method__getio.1)
Given a set of statement IDs and liveness information for the statements of the function, determine
which symbols are needed at input and which symbols are purely local to the functio.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:840](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L840)

---

<a id="method__getinputset.1" class="lexicon_definition"></a>
#### getInputSet(node::ParallelAccelerator.ParallelIR.PIRParForAst) [¶](#method__getinputset.1)
Returns a Set with all the arrays read by this parfor.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1383](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1383)

---

<a id="method__getlhsfromassignment.1" class="lexicon_definition"></a>
#### getLhsFromAssignment(assignment) [¶](#method__getlhsfromassignment.1)
Get the left-hand side of an assignment expression.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1354](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1354)

---

<a id="method__getlhsoutputset.1" class="lexicon_definition"></a>
#### getLhsOutputSet(lhs,  assignment) [¶](#method__getlhsoutputset.1)
Get the real outputs of an assignment statement.
If the assignment expression is normal then the output is just the left-hand side.
If the assignment expression is augmented with a FusionSentinel then the real outputs
are the 4+ arguments to the expression.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1398](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1398)

---

<a id="method__getmaxlabel.1" class="lexicon_definition"></a>
#### getMaxLabel(max_label,  stmts::Array{Any, 1}) [¶](#method__getmaxlabel.1)
Scan the body of a function in "stmts" and return the max label in a LabelNode AST seen in the body.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4349](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4349)

---

<a id="method__getnonblock.1" class="lexicon_definition"></a>
#### getNonBlock(head_preds,  back_edge) [¶](#method__getnonblock.1)
Find the basic block before the entry to a loop.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:5](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L5)

---

<a id="method__getoraddarraycorrelation.1" class="lexicon_definition"></a>
#### getOrAddArrayCorrelation(x::Union{GenSym, Symbol},  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__getoraddarraycorrelation.1)
Return a correlation set for an array.  If the array was not previously added then add it and return it.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:929](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L929)

---

<a id="method__getoraddrangecorrelation.1" class="lexicon_definition"></a>
#### getOrAddRangeCorrelation(array,  ranges::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1},  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__getoraddrangecorrelation.1)
Gets (or adds if absent) the range correlation for the given array of RangeExprs.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1071](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1071)

---

<a id="method__getoraddsymbolcorrelation.1" class="lexicon_definition"></a>
#### getOrAddSymbolCorrelation(array::Union{GenSym, Symbol},  state::ParallelAccelerator.ParallelIR.expr_state,  dims::Array{Union{GenSym, Symbol}, 1}) [¶](#method__getoraddsymbolcorrelation.1)
A new array is being created with an explicit size specification in dims.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1109](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1109)

---

<a id="method__getparforcorrelation.1" class="lexicon_definition"></a>
#### getParforCorrelation(parfor,  state) [¶](#method__getparforcorrelation.1)
Get the equivalence class of the first array who length is extracted in the pre-statements of the specified "parfor".


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2017](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2017)

---

<a id="method__getparfornode.1" class="lexicon_definition"></a>
#### getParforNode(node) [¶](#method__getparfornode.1)
Get the parfor object from either a bare parfor or one part of an assignment.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1336](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1336)

---

<a id="method__getpastindex.1" class="lexicon_definition"></a>
#### getPastIndex(arrays::Dict{Union{GenSym, Symbol}, Array{Array{Any, 1}, 1}}) [¶](#method__getpastindex.1)
Look at the arrays that are accessed and see if they use a forward index, i.e.,
an index that could be greater than 1.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:30](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L30)

---

<a id="method__getprivateset.1" class="lexicon_definition"></a>
#### getPrivateSet(body::Array{Any, 1}) [¶](#method__getprivateset.1)
Go through the body of a parfor and collect those Symbols, GenSyms, etc. that are assigned to within the parfor except reduction variables.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1175](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1175)

---

<a id="method__getprivatesetinner.1" class="lexicon_definition"></a>
#### getPrivateSetInner(x::Expr,  state::Set{Union{GenSym, Symbol, SymbolNode}},  top_level_number::Int64,  is_top_level::Bool,  read::Bool) [¶](#method__getprivatesetinner.1)
The AstWalk callback function for getPrivateSet.
For each AST in a parfor body, if the node is an assignment or loop head node then add the written entity to the state.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1145](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1145)

---

<a id="method__getrhsfromassignment.1" class="lexicon_definition"></a>
#### getRhsFromAssignment(assignment) [¶](#method__getrhsfromassignment.1)
Get the right-hand side of an assignment expression.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1347](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1347)

---

<a id="method__getsname.1" class="lexicon_definition"></a>
#### getSName(ssn::Symbol) [¶](#method__getsname.1)
Get the name of a symbol whether the input is a Symbol or SymbolNode or :(::) Expr.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2530](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2530)

---

<a id="method__get_one.1" class="lexicon_definition"></a>
#### get_one(ast::Array{T, N}) [¶](#method__get_one.1)
Take something returned from AstWalk and assert it should be an array but in this
context that the array should also be of length 1 and then return that single element.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:5030](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L5030)

---

<a id="method__get_unique_num.1" class="lexicon_definition"></a>
#### get_unique_num() [¶](#method__get_unique_num.1)
If we need to generate a name and make sure it is unique then include an monotonically increasing number.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1130](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1130)

---

<a id="method__hasnosideeffects.1" class="lexicon_definition"></a>
#### hasNoSideEffects(node::Union{GenSym, LambdaStaticData, Number, Symbol, SymbolNode}) [¶](#method__hasnosideeffects.1)
Sometimes statements we exist in the AST of the form a=Expr where a is a Symbol that isn't live past the assignment
and we'd like to eliminate the whole assignment statement but we have to know that the right-hand side has no
side effects before we can do that.  This function says whether the right-hand side passed into it has side effects
or not.  Several common function calls that otherwise we wouldn't know are safe are explicitly checked for.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2840](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2840)

---

<a id="method__hassymbol.1" class="lexicon_definition"></a>
#### hasSymbol(ssn::Symbol) [¶](#method__hassymbol.1)
Returns true if the incoming AST node can be interpreted as a Symbol.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2511](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2511)

---

<a id="method__hoistallocation.1" class="lexicon_definition"></a>
#### hoistAllocation(ast::Array{Any, 1},  lives,  domLoop::CompilerTools.Loops.DomLoops,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__hoistallocation.1)
Try to hoist allocations outside the loop if possible.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4021](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4021)

---

<a id="method__insert_no_deps_beginning.1" class="lexicon_definition"></a>
#### insert_no_deps_beginning(node,  data::ParallelAccelerator.ParallelIR.RemoveNoDepsState,  top_level_number,  is_top_level,  read) [¶](#method__insert_no_deps_beginning.1)
Works with remove_no_deps below to move statements with no dependencies to the beginning of the AST.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3355](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3355)

---

<a id="method__intermediate_from_exprs.1" class="lexicon_definition"></a>
#### intermediate_from_exprs(ast::Array{Any, 1},  depth,  state) [¶](#method__intermediate_from_exprs.1)
Process an array of expressions that aren't from a :body Expr.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2581](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2581)

---

<a id="method__isarraytype.1" class="lexicon_definition"></a>
#### isArrayType(typ) [¶](#method__isarraytype.1)
Returns true if the incoming type in "typ" is an array type.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:704](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L704)

---

<a id="method__isarraytype.2" class="lexicon_definition"></a>
#### isArrayType(x::SymbolNode) [¶](#method__isarraytype.2)
Returns true if a given SymbolNode "x" is an Array type.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2609](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2609)

---

<a id="method__isarrayref.1" class="lexicon_definition"></a>
#### isArrayref(x) [¶](#method__isarrayref.1)
Is a node an arrayref node?


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1633](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1633)

---

<a id="method__isarrayrefcall.1" class="lexicon_definition"></a>
#### isArrayrefCall(x::Expr) [¶](#method__isarrayrefcall.1)
Is a node a call to arrayref.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1654](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1654)

---

<a id="method__isarrayset.1" class="lexicon_definition"></a>
#### isArrayset(x) [¶](#method__isarrayset.1)
Is a node an arrayset node?


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1623](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1623)

---

<a id="method__isarraysetcall.1" class="lexicon_definition"></a>
#### isArraysetCall(x::Expr) [¶](#method__isarraysetcall.1)
Is a node a call to arrayset.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1643](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1643)

---

<a id="method__isassignmentnode.1" class="lexicon_definition"></a>
#### isAssignmentNode(node::Expr) [¶](#method__isassignmentnode.1)
Is a node an assignment expression node.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1264](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1264)

---

<a id="method__isbareparfor.1" class="lexicon_definition"></a>
#### isBareParfor(node::Expr) [¶](#method__isbareparfor.1)
Is this a parfor node not part of an assignment statement.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1286](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1286)

---

<a id="method__isdomainnode.1" class="lexicon_definition"></a>
#### isDomainNode(ast::Expr) [¶](#method__isdomainnode.1)
Returns true if the given "ast" node is a DomainIR operation.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4205](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4205)

---

<a id="method__isfusionassignment.1" class="lexicon_definition"></a>
#### isFusionAssignment(x::Expr) [¶](#method__isfusionassignment.1)
Check if an assignement is a fusion assignment.
    In regular assignments, there are only two args, the left and right hand sides.
    In fusion assignments, we introduce a third arg that is marked by an object of FusionSentinel type.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1977](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1977)

---

<a id="method__isloopheadnode.1" class="lexicon_definition"></a>
#### isLoopheadNode(node::Expr) [¶](#method__isloopheadnode.1)
Is a node a loophead expression node (a form of assignment).


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1275](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1275)

---

<a id="method__isparforassignmentnode.1" class="lexicon_definition"></a>
#### isParforAssignmentNode(node::Expr) [¶](#method__isparforassignmentnode.1)
Is a node an assignment expression with a parfor node as the right-hand side.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1310](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1310)

---

<a id="method__issymbolsused.1" class="lexicon_definition"></a>
#### isSymbolsUsed(vars,  top_level_numbers::Array{Int64, 1},  state) [¶](#method__issymbolsused.1)
Returns true if any variable in the collection "vars" is used in any statement whose top level number is in "top_level_numbers".
    We use expr_state "state" to get the block liveness information from which we use "def" and "use" to determine if a variable
        usage is present.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1993](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1993)

---

<a id="method__is_eliminated_arraylen.1" class="lexicon_definition"></a>
#### is_eliminated_arraylen(x::Expr) [¶](#method__is_eliminated_arraylen.1)
Returns true if the input node is an assignment node where the right-hand side is a call to arraysize.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1866](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1866)

---

<a id="method__isbitstuple.1" class="lexicon_definition"></a>
#### isbitstuple(a::Tuple) [¶](#method__isbitstuple.1)
Returns true if input "a" is a tuple and each element of the tuple of isbits type.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4823](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4823)

---

<a id="method__iterations_equals_inputs.1" class="lexicon_definition"></a>
#### iterations_equals_inputs(node::ParallelAccelerator.ParallelIR.PIRParForAst) [¶](#method__iterations_equals_inputs.1)
Returns true if the domain operation mapped to this parfor has the property that the iteration space
is identical to the dimenions of the inputs.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1363](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1363)

---

<a id="method__lambdafromdomainlambda.1" class="lexicon_definition"></a>
#### lambdaFromDomainLambda(domain_lambda,  dl_inputs) [¶](#method__lambdafromdomainlambda.1)
Form a Julia :lambda Expr from a DomainLambda.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4361](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4361)

---

<a id="method__makeprivateparfor.1" class="lexicon_definition"></a>
#### makePrivateParfor(var_name::Symbol,  state) [¶](#method__makeprivateparfor.1)
Takes an existing variable whose name is in "var_name" and adds the descriptor flag ISPRIVATEPARFORLOOP to declare the
variable to be parfor loop private and eventually go in an OMP private clause.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:781](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L781)

---

<a id="method__maketasks.1" class="lexicon_definition"></a>
#### makeTasks(start_index,  stop_index,  body,  bb_live_info,  state,  task_graph_mode) [¶](#method__maketasks.1)
For a given start and stop index in some body and liveness information, form a set of tasks.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:759](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L759)

---

<a id="method__maxfusion.1" class="lexicon_definition"></a>
#### maxFusion(bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__maxfusion.1)
For every basic block, try to push domain IR statements down and non-domain IR statements up so that domain nodes
are next to each other and can be fused.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4249](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4249)

---

<a id="method__mergelambdaintoouterstate.1" class="lexicon_definition"></a>
#### mergeLambdaIntoOuterState(state,  inner_lambda::Expr) [¶](#method__mergelambdaintoouterstate.1)
Pull the information from the inner lambda into the outer lambda.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1495](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1495)

---

<a id="method__merge_correlations.1" class="lexicon_definition"></a>
#### merge_correlations(state,  unchanging,  eliminate) [¶](#method__merge_correlations.1)
If we somehow determine that two sets of correlations are actually the same length then merge one into the other.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:885](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L885)

---

<a id="method__mk_alloc_array_1d_expr.1" class="lexicon_definition"></a>
#### mk_alloc_array_1d_expr(elem_type,  atype,  length) [¶](#method__mk_alloc_array_1d_expr.1)
Return an expression that allocates and initializes a 1D Julia array that has an element type specified by
"elem_type", an array type of "atype" and a "length".


*source:*
[ParallelAccelerator/src/parallel-ir.jl:616](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L616)

---

<a id="method__mk_alloc_array_2d_expr.1" class="lexicon_definition"></a>
#### mk_alloc_array_2d_expr(elem_type,  atype,  length1,  length2) [¶](#method__mk_alloc_array_2d_expr.1)
Return an expression that allocates and initializes a 2D Julia array that has an element type specified by
"elem_type", an array type of "atype" and two dimensions of length in "length1" and "length2".


*source:*
[ParallelAccelerator/src/parallel-ir.jl:654](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L654)

---

<a id="method__mk_alloc_array_3d_expr.1" class="lexicon_definition"></a>
#### mk_alloc_array_3d_expr(elem_type,  atype,  length1,  length2,  length3) [¶](#method__mk_alloc_array_3d_expr.1)
Return an expression that allocates and initializes a 3D Julia array that has an element type specified by
"elem_type", an array type of "atype" and two dimensions of length in "length1" and "length2" and "length3".


*source:*
[ParallelAccelerator/src/parallel-ir.jl:680](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L680)

---

<a id="method__mk_arraylen_expr.1" class="lexicon_definition"></a>
#### mk_arraylen_expr(x::ParallelAccelerator.ParallelIR.InputInfo,  dim::Int64) [¶](#method__mk_arraylen_expr.1)
Create an expression whose value is the length of the input array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:562](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L562)

---

<a id="method__mk_arraylen_expr.2" class="lexicon_definition"></a>
#### mk_arraylen_expr(x::Union{GenSym, Symbol, SymbolNode},  dim::Int64) [¶](#method__mk_arraylen_expr.2)
Create an expression whose value is the length of the input array.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:555](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L555)

---

<a id="method__mk_arrayref1.1" class="lexicon_definition"></a>
#### mk_arrayref1(num_dim_inputs,  array_name,  index_vars,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__mk_arrayref1.1)
Return an expression that corresponds to getting the index_var index from the array array_name.
If "inbounds" is true then use the faster :unsafe_arrayref call that doesn't do a bounds check.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:89](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L89)

---

<a id="method__mk_arrayref1.2" class="lexicon_definition"></a>
#### mk_arrayref1(num_dim_inputs,  array_name,  index_vars,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state,  range::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1}) [¶](#method__mk_arrayref1.2)
Return an expression that corresponds to getting the index_var index from the array array_name.
If "inbounds" is true then use the faster :unsafe_arrayref call that doesn't do a bounds check.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:89](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L89)

---

<a id="method__mk_arrayset1.1" class="lexicon_definition"></a>
#### mk_arrayset1(num_dim_inputs,  array_name,  index_vars,  value,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__mk_arrayset1.1)
Return a new AST node that corresponds to setting the index_var index from the array "array_name" with "value".
The paramater "inbounds" is true if this access is known to be within the bounds of the array.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:194](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L194)

---

<a id="method__mk_arrayset1.2" class="lexicon_definition"></a>
#### mk_arrayset1(num_dim_inputs,  array_name,  index_vars,  value,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state,  range::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1}) [¶](#method__mk_arrayset1.2)
Return a new AST node that corresponds to setting the index_var index from the array "array_name" with "value".
The paramater "inbounds" is true if this access is known to be within the bounds of the array.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:194](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L194)

---

<a id="method__mk_assignment_expr.1" class="lexicon_definition"></a>
#### mk_assignment_expr(lhs::Union{GenSym, Symbol, SymbolNode},  rhs,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__mk_assignment_expr.1)
Create an assignment expression AST node given a left and right-hand side.
The left-hand side has to be a symbol node from which we extract the type so as to type the new Expr.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:515](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L515)

---

<a id="method__mk_colon_expr.1" class="lexicon_definition"></a>
#### mk_colon_expr(start_expr,  skip_expr,  end_expr) [¶](#method__mk_colon_expr.1)
Returns an expression to construct a :colon object that contains the start of a range, the end and the skip expression.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:878](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L878)

---

<a id="method__mk_convert.1" class="lexicon_definition"></a>
#### mk_convert(new_type,  ex) [¶](#method__mk_convert.1)
Returns an expression that convert "ex" into a another type "new_type".


*source:*
[ParallelAccelerator/src/parallel-ir.jl:593](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L593)

---

<a id="method__mk_gotoifnot_expr.1" class="lexicon_definition"></a>
#### mk_gotoifnot_expr(cond,  goto_label) [¶](#method__mk_gotoifnot_expr.1)
Returns a :gotoifnot Expr given a condition "cond" and a label "goto_label".


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:899](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L899)

---

<a id="method__mk_mask_arrayref1.1" class="lexicon_definition"></a>
#### mk_mask_arrayref1(cur_dimension,  num_dim_inputs,  array_name,  index_vars,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__mk_mask_arrayref1.1)
Return an expression that corresponds to getting the index_var index from the array array_name.
If "inbounds" is true then use the faster :unsafe_arrayref call that doesn't do a bounds check.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:156](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L156)

---

<a id="method__mk_mask_arrayref1.2" class="lexicon_definition"></a>
#### mk_mask_arrayref1(cur_dimension,  num_dim_inputs,  array_name,  index_vars,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state,  range::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1}) [¶](#method__mk_mask_arrayref1.2)
Return an expression that corresponds to getting the index_var index from the array array_name.
If "inbounds" is true then use the faster :unsafe_arrayref call that doesn't do a bounds check.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:156](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L156)

---

<a id="method__mk_next_expr.1" class="lexicon_definition"></a>
#### mk_next_expr(colon_sym,  start_sym) [¶](#method__mk_next_expr.1)
Returns a :next call Expr that gets the next element of an iteration range from a :colon object.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:892](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L892)

---

<a id="method__mk_parallelir_ref.1" class="lexicon_definition"></a>
#### mk_parallelir_ref(sym) [¶](#method__mk_parallelir_ref.1)
Create an expression that references something inside ParallelIR.
In other words, returns an expression the equivalent of ParallelAccelerator.ParallelIR.sym where sym is an input argument to this function.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:585](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L585)

---

<a id="method__mk_parallelir_ref.2" class="lexicon_definition"></a>
#### mk_parallelir_ref(sym,  ref_type) [¶](#method__mk_parallelir_ref.2)
Create an expression that references something inside ParallelIR.
In other words, returns an expression the equivalent of ParallelAccelerator.ParallelIR.sym where sym is an input argument to this function.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:585](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L585)

---

<a id="method__mk_parfor_args_from_mmap.1" class="lexicon_definition"></a>
#### mk_parfor_args_from_mmap!(input_arrays::Array{T, N},  dl::ParallelAccelerator.DomainIR.DomainLambda,  with_indices,  domain_oprs,  state) [¶](#method__mk_parfor_args_from_mmap.1)
The main routine that converts a mmap! AST node to a parfor AST node.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:690](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L690)

---

<a id="method__mk_parfor_args_from_mmap.2" class="lexicon_definition"></a>
#### mk_parfor_args_from_mmap(input_arrays::Array{T, N},  dl::ParallelAccelerator.DomainIR.DomainLambda,  domain_oprs,  state) [¶](#method__mk_parfor_args_from_mmap.2)
The main routine that converts a mmap AST node to a parfor AST node.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:1005](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L1005)

---

<a id="method__mk_parfor_args_from_reduce.1" class="lexicon_definition"></a>
#### mk_parfor_args_from_reduce(input_args::Array{Any, 1},  state) [¶](#method__mk_parfor_args_from_reduce.1)
The main routine that converts a reduce AST node to a parfor AST node.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:234](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L234)

---

<a id="method__mk_return_expr.1" class="lexicon_definition"></a>
#### mk_return_expr(outs) [¶](#method__mk_return_expr.1)
Given an array of outputs in "outs", form a return expression.
If there is only one out then the args of :return is just that expression.
If there are multiple outs then form a tuple of them and that tuple goes in :return args.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:500](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L500)

---

<a id="method__mk_start_expr.1" class="lexicon_definition"></a>
#### mk_start_expr(colon_sym) [¶](#method__mk_start_expr.1)
Returns an expression to get the start of an iteration range from a :colon object.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:885](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L885)

---

<a id="method__mk_svec_expr.1" class="lexicon_definition"></a>
#### mk_svec_expr(parts...) [¶](#method__mk_svec_expr.1)
Make a svec expression.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:607](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L607)

---

<a id="method__mk_tuple_expr.1" class="lexicon_definition"></a>
#### mk_tuple_expr(tuple_fields,  typ) [¶](#method__mk_tuple_expr.1)
Return an expression which creates a tuple.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1428](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1428)

---

<a id="method__mk_tupleref_expr.1" class="lexicon_definition"></a>
#### mk_tupleref_expr(tuple_var,  index,  typ) [¶](#method__mk_tupleref_expr.1)
Create an expression which returns the index'th element of the tuple whose name is contained in tuple_var.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:600](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L600)

---

<a id="method__mk_untyped_assignment.1" class="lexicon_definition"></a>
#### mk_untyped_assignment(lhs,  rhs) [¶](#method__mk_untyped_assignment.1)
Only used to create fake expression to force lhs to be seen as written rather than read.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:532](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L532)

---

<a id="method__mmapinline.1" class="lexicon_definition"></a>
#### mmapInline(ast::Expr,  lives,  uniqSet) [¶](#method__mmapinline.1)
# If a definition of a mmap is only used once and not aliased, it can be inlined into its
# use side as long as its dependencies have not been changed.
# FIXME: is the implementation still correct when branches are present?


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3944](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3944)

---

<a id="method__mmaptommap.1" class="lexicon_definition"></a>
#### mmapToMmap!(ast,  lives,  uniqSet) [¶](#method__mmaptommap.1)
Performs the mmap to mmap! phase.
If the arguments of a mmap dies aftewards, and is not aliased, then
we can safely change the mmap to mmap!.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4087](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4087)

---

<a id="method__mustremainlaststatementinblock.1" class="lexicon_definition"></a>
#### mustRemainLastStatementInBlock(node::GotoNode) [¶](#method__mustremainlaststatementinblock.1)
Returns true if the given AST "node" must remain the last statement in a basic block.
This is true if the node is a GotoNode or a :gotoifnot Expr.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4233](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4233)

---

<a id="method__nametosymbolnode.1" class="lexicon_definition"></a>
#### nameToSymbolNode(name::Symbol,  sym_to_type) [¶](#method__nametosymbolnode.1)
Forms a SymbolNode given a symbol in "name" and get the type of that symbol from the incoming dictionary "sym_to_type".


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1436](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1436)

---

<a id="method__nested_function_exprs.1" class="lexicon_definition"></a>
#### nested_function_exprs(max_label,  domain_lambda,  dl_inputs) [¶](#method__nested_function_exprs.1)
A routine similar to the main parallel IR entry put but designed to process the lambda part of
domain IR AST nodes.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4390](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4390)

---

<a id="method__next_label.1" class="lexicon_definition"></a>
#### next_label(state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__next_label.1)
Returns the next usable label for the current function.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:858](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L858)

---

<a id="method__nonexactrangesearch.1" class="lexicon_definition"></a>
#### nonExactRangeSearch(ranges::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1},  range_correlations) [¶](#method__nonexactrangesearch.1)
We can only do exact matches in the range correlation dict but there can still be non-exact matches
where the ranges are different but equivalent in length.  In this function, we can the dictionary
and look for equivalent ranges.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1045](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1045)

---

<a id="method__oneifonly.1" class="lexicon_definition"></a>
#### oneIfOnly(x) [¶](#method__oneifonly.1)
Returns a single element of an array if there is only one or the array otherwise.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2147](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2147)

---

<a id="method__parfortotask.1" class="lexicon_definition"></a>
#### parforToTask(parfor_index,  bb_statements,  body,  state) [¶](#method__parfortotask.1)
Given a parfor statement index in "parfor_index" in the "body"'s statements, create a TaskInfo node for this parfor.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:1210](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L1210)

---

<a id="method__pirprintdl.1" class="lexicon_definition"></a>
#### pirPrintDl(dbg_level,  dl) [¶](#method__pirprintdl.1)
Debug print the parts of a DomainLambda.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4340](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4340)

---

<a id="method__pir_alias_cb.1" class="lexicon_definition"></a>
#### pir_alias_cb(ast::Expr,  state,  cbdata) [¶](#method__pir_alias_cb.1)
An AliasAnalysis callback (similar to LivenessAnalysis callback) that handles ParallelIR introduced AST node types.
For each ParallelIR specific node type, form an array of expressions that AliasAnalysis
    can analyze to reflect the aliases of the given AST node.
    If we read a symbol it is sufficient to just return that symbol as one of the expressions.
    If we write a symbol, then form a fake mk_assignment_expr just to get liveness to realize the symbol is written.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:5237](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L5237)

---

<a id="method__pir_live_cb.1" class="lexicon_definition"></a>
#### pir_live_cb(ast::Expr,  cbdata::ANY) [¶](#method__pir_live_cb.1)
A LivenessAnalysis callback that handles ParallelIR introduced AST node types.
For each ParallelIR specific node type, form an array of expressions that liveness
can analysis to reflect the read/write set of the given AST node.
If we read a symbol it is sufficient to just return that symbol as one of the expressions.
If we write a symbol, then form a fake mk_assignment_expr just to get liveness to realize the symbol is written.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2705](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2705)

---

<a id="method__pir_live_cb_def.1" class="lexicon_definition"></a>
#### pir_live_cb_def(x) [¶](#method__pir_live_cb_def.1)
Just call the AST walker for symbol for parallel IR nodes with no state.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1216](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1216)

---

<a id="method__printbody.1" class="lexicon_definition"></a>
#### printBody(dlvl,  body::Array{Any, 1}) [¶](#method__printbody.1)
Pretty print the args part of the "body" of a :lambda Expr at a given debug level in "dlvl".


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2620](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2620)

---

<a id="method__printlambda.1" class="lexicon_definition"></a>
#### printLambda(dlvl,  node::Expr) [¶](#method__printlambda.1)
Pretty print a :lambda Expr in "node" at a given debug level in "dlvl".


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2633](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2633)

---

<a id="method__processandupdatebody.1" class="lexicon_definition"></a>
#### processAndUpdateBody(lambda::Expr,  f::Function,  state) [¶](#method__processandupdatebody.1)
Apply a function "f" that takes the :body from the :lambda and returns a new :body that is stored back into the :lambda.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3596](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3596)

---

<a id="method__rangesize.1" class="lexicon_definition"></a>
#### rangeSize(start,  skip,  last) [¶](#method__rangesize.1)
Compute size of a range.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:547](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L547)

---

<a id="method__rangetorangedata.1" class="lexicon_definition"></a>
#### rangeToRangeData(range::Expr,  arr,  range_num::Int64,  state) [¶](#method__rangetorangedata.1)
Convert a :range Expr introduced by Domain IR into a Parallel IR data structure RangeData.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:495](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L495)

---

<a id="method__recreateloops.1" class="lexicon_definition"></a>
#### recreateLoops(new_body,  the_parfor::ParallelAccelerator.ParallelIR.PIRParForAst,  state,  newLambdaInfo) [¶](#method__recreateloops.1)
In threads mode, we can't have parfor_start and parfor_end in the code since Julia has to compile the code itself and so
we have to reconstruct a loop infrastructure based on the parfor's loop nest information.  This function takes a parfor
and outputs that parfor to the new function body as regular Julia loops.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:1155](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L1155)

---

<a id="method__recreateloopsinternal.1" class="lexicon_definition"></a>
#### recreateLoopsInternal(new_body,  the_parfor::ParallelAccelerator.ParallelIR.PIRParForAst,  loop_nest_level,  next_available_label,  state,  newLambdaInfo) [¶](#method__recreateloopsinternal.1)
This is a recursive routine to reconstruct a regular Julia loop nest from the loop nests described in PIRParForAst.
One call of this routine handles one level of the loop nest.
If the incoming loop nest level is more than the number of loops nests in the parfor then that is the spot to
insert the body of the parfor into the new function body in "new_body".


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:1019](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L1019)

---

<a id="method__remembertypeforsym.1" class="lexicon_definition"></a>
#### rememberTypeForSym(sym_to_type::Dict{Union{GenSym, Symbol}, DataType},  sym::Union{GenSym, Symbol},  typ::DataType) [¶](#method__remembertypeforsym.1)
Add to the map of symbol names to types.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1956](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1956)

---

<a id="method__removeasserteqshape.1" class="lexicon_definition"></a>
#### removeAssertEqShape(args::Array{Any, 1},  state) [¶](#method__removeasserteqshape.1)
Implements one of the main ParallelIR passes to remove assertEqShape AST nodes from the body if they are statically known to be in the same equivalence class.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:2885](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L2885)

---

<a id="method__removenothingstmts.1" class="lexicon_definition"></a>
#### removeNothingStmts(args::Array{Any, 1},  state) [¶](#method__removenothingstmts.1)
Empty statements can be added to the AST by some passes in ParallelIR.
This pass over the statements of the :body excludes such "nothing" statements from the new :body.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3607](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3607)

---

<a id="method__remove_dead.1" class="lexicon_definition"></a>
#### remove_dead(node,  data::ParallelAccelerator.ParallelIR.RemoveDeadState,  top_level_number,  is_top_level,  read) [¶](#method__remove_dead.1)
An AstWalk callback that uses liveness information in "data" to remove dead stores.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3287](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3287)

---

<a id="method__remove_extra_allocs.1" class="lexicon_definition"></a>
#### remove_extra_allocs(ast) [¶](#method__remove_extra_allocs.1)
removes extra allocations


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4693](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4693)

---

<a id="method__remove_no_deps.1" class="lexicon_definition"></a>
#### remove_no_deps(node::ANY,  data::ParallelAccelerator.ParallelIR.RemoveNoDepsState,  top_level_number,  is_top_level,  read) [¶](#method__remove_no_deps.1)
# This routine gathers up nodes that do not use
# any variable and removes them from the AST into top_level_no_deps.  This works in conjunction with
# insert_no_deps_beginning above to move these statements with no dependencies to the beginning of the AST
# where they can't prevent fusion.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3371](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3371)

---

<a id="method__replaceparforwithdict.1" class="lexicon_definition"></a>
#### replaceParforWithDict(parfor::ParallelAccelerator.ParallelIR.PIRParForAst,  gensym_map) [¶](#method__replaceparforwithdict.1)
Not currently used but might need it at some point.
Search a whole PIRParForAst object and replace one SymAllGen with another.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:378](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L378)

---

<a id="method__run_as_task.1" class="lexicon_definition"></a>
#### run_as_task() [¶](#method__run_as_task.1)
Return true if run_as_task_decrement would return true but don't update the run_as_tasks count.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:229](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L229)

---

<a id="method__run_as_task_decrement.1" class="lexicon_definition"></a>
#### run_as_task_decrement() [¶](#method__run_as_task_decrement.1)
If run_as_tasks is positive then convert this parfor to a task and decrement the count so that only the
original number run_as_tasks if the number of tasks created.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:215](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L215)

---

<a id="method__selecttorangedata.1" class="lexicon_definition"></a>
#### selectToRangeData(select::Expr,  pre_offsets::Array{Expr, 1},  state) [¶](#method__selecttorangedata.1)
Convert the range(s) part of a :select Expr introduced by Domain IR into an array of Parallel IR data structures RangeData.


*source:*
[ParallelAccelerator/src/parallel-ir-mk-parfor.jl:531](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-mk-parfor.jl#L531)

---

<a id="method__seqtask.1" class="lexicon_definition"></a>
#### seqTask(body_indices,  bb_statements,  body,  state) [¶](#method__seqtask.1)
Form a task out of a range of sequential statements.
This is not currently implemented.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:1558](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L1558)

---

<a id="method__show.1" class="lexicon_definition"></a>
#### show(io::IO,  pnode::ParallelAccelerator.ParallelIR.PIRParForAst) [¶](#method__show.1)
Overload of Base.show to pretty print for parfor AST nodes.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:435](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L435)

---

<a id="method__simpleindex.1" class="lexicon_definition"></a>
#### simpleIndex(dict) [¶](#method__simpleindex.1)
Returns true if all array references use singular index variables and nothing more complicated involving,
for example, addition or subtraction by a constant.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:790](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L790)

---

<a id="method__simplify_internal.1" class="lexicon_definition"></a>
#### simplify_internal(x::Expr,  state,  top_level_number::Int64,  is_top_level::Bool,  read::Bool) [¶](#method__simplify_internal.1)
Do some simplification to expressions that are part of ranges.
For example, the range 2:s-1 becomes a length (s-1)-2 which this function in turn transforms to s-3.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:946](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L946)

---

<a id="method__sub_arraylen_walk.1" class="lexicon_definition"></a>
#### sub_arraylen_walk(x::Expr,  replacement,  top_level_number,  is_top_level,  read) [¶](#method__sub_arraylen_walk.1)
AstWalk callback that does the work of substitute_arraylen on a node-by-node basis.
replacement is an array containing the length of the dimensions of the arrays a part of this parfor.
If we see a call to create an array, replace the length params with those in the common set in "replacement".


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1897](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1897)

---

<a id="method__sub_arrayset_walk.1" class="lexicon_definition"></a>
#### sub_arrayset_walk(x::Expr,  cbd,  top_level_number,  is_top_level,  read) [¶](#method__sub_arrayset_walk.1)
AstWalk callback that does the work of substitute_arrayset on a node-by-node basis.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1665](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1665)

---

<a id="method__sub_cur_body_walk.1" class="lexicon_definition"></a>
#### sub_cur_body_walk(x::Expr,  cbd::ParallelAccelerator.ParallelIR.cur_body_data,  top_level_number::Int64,  is_top_level::Bool,  read::Bool) [¶](#method__sub_cur_body_walk.1)
AstWalk callback that does the work of substitute_cur_body on a node-by-node basis.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1748](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1748)

---

<a id="method__substitute_arraylen.1" class="lexicon_definition"></a>
#### substitute_arraylen(x,  replacement) [¶](#method__substitute_arraylen.1)
replacement is an array containing the length of the dimensions of the arrays a part of this parfor.
If we see a call to create an array, replace the length params with those in the common set in "replacement".


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1929](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1929)

---

<a id="method__substitute_arrayset.1" class="lexicon_definition"></a>
#### substitute_arrayset(x,  arrays_set_in_cur_body,  output_items_with_aliases) [¶](#method__substitute_arrayset.1)
Modify the body of a parfor.
temp_map holds a map of array names whose arraysets should be turned into a mapped variable instead of the arrayset. a[i] = b. a=>c. becomes c = b
map_for_non_eliminated holds arrays for which we need to add a variable to save the value but we can't eiminate the arrayset. a[i] = b. a=>c. becomes c = a[i] = b
    map_drop_arrayset drops the arrayset without replacing with a variable.  This is because a variable was previously added here with a map_for_non_eliminated case.
    a[i] = b. becomes b


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1710](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1710)

---

<a id="method__substitute_cur_body.1" class="lexicon_definition"></a>
#### substitute_cur_body(x,  temp_map::Dict{Union{GenSym, Symbol}, Union{GenSym, SymbolNode}},  index_map::Dict{Union{GenSym, Symbol}, Union{GenSym, Symbol}},  arrays_set_in_cur_body::Set{Union{GenSym, Symbol}},  replace_array_name_in_arrayset::Dict{Union{GenSym, Symbol}, Union{GenSym, Symbol}},  state::ParallelAccelerator.ParallelIR.expr_state) [¶](#method__substitute_cur_body.1)
Make changes to the second parfor body in the process of parfor fusion.
temp_map holds array names for which arrayrefs should be converted to a variable.  a[i].  a=>b. becomes b
    index_map holds maps between index variables.  The second parfor is modified to use the index variable of the first parfor.
    arrays_set_in_cur_body           # Used as output.  Collects the arrays set in the current body.
    replace_array_name_in_arrayset   # Map from one array to another.  Replace first array with second when used in arrayset context.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1852](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1852)

---

<a id="method__taskableparfor.1" class="lexicon_definition"></a>
#### taskableParfor(node) [¶](#method__taskableparfor.1)
Returns true if the "node" is a parfor and the task limit hasn't been exceeded.
Also controls whether stencils or reduction can become tasks.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:278](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L278)

---

<a id="method__tosngen.1" class="lexicon_definition"></a>
#### toSNGen(x::Symbol,  typ) [¶](#method__tosngen.1)
If we have the type, convert a Symbol to SymbolNode.
If we have a GenSym then we have to keep it.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3087](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3087)

---

<a id="method__tosymgen.1" class="lexicon_definition"></a>
#### toSymGen(x::Symbol) [¶](#method__tosymgen.1)
In various places we need a SymGen type which is the union of Symbol and GenSym.
This function takes a Symbol, SymbolNode, or GenSym and return either a Symbol or GenSym.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:819](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L819)

---

<a id="method__tosymnodegen.1" class="lexicon_definition"></a>
#### toSymNodeGen(x::Symbol,  typ) [¶](#method__tosymnodegen.1)
Form a SymbolNode with the given typ if possible or a GenSym if that is what is passed in.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:838](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L838)

---

<a id="method__uncompressed_ast.1" class="lexicon_definition"></a>
#### uncompressed_ast(l::LambdaStaticData) [¶](#method__uncompressed_ast.1)
Convert a compressed LambdaStaticData format into the uncompressed AST format.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1191](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1191)

---

<a id="type__copypropagatestate.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.CopyPropagateState [¶](#type__copypropagatestate.1)
State to aide in the copy propagation phase.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3130](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3130)

---

<a id="type__delayedfunc.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.DelayedFunc [¶](#type__delayedfunc.1)
Ad-hoc support to mimic closures when we want the arguments to be processed during AstWalk.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:67](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L67)

---

<a id="type__dirwalk.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.DirWalk [¶](#type__dirwalk.1)
Wraps the callback and opaque data passed from the user of ParallelIR's AstWalk.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:5037](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L5037)

---

<a id="type__domainoperation.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.DomainOperation [¶](#type__domainoperation.1)
Holds information about domain operations part of a parfor node.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:108](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L108)

---

<a id="type__equivalenceclasses.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.EquivalenceClasses [¶](#type__equivalenceclasses.1)
Holds a dictionary from an array symbol to an integer corresponding to an equivalence class.
All array symbol in the same equivalence class are known to have the same shape.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:118](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L118)

---

<a id="type__fusionsentinel.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.FusionSentinel [¶](#type__fusionsentinel.1)
Just used to hold a spot in an array to indicate the this is a special assignment expression with embedded real array output names from a fusion.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1967](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1967)

---

<a id="type__inprogress.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.InProgress [¶](#type__inprogress.1)
A sentinel in the instruction count estimation process.
Before recursively processing a call, we add a sentinel for that function so that if we see that
sentinel later we know we've tried to recursively process it and so can bail out by setting
fully_analyzed to false.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:346](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L346)

---

<a id="type__inputinfo.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.InputInfo [¶](#type__inputinfo.1)
Type used by mk_parfor_args... functions to hold information about input arrays.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:298](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L298)

---

<a id="type__inserttasknode.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.InsertTaskNode [¶](#type__inserttasknode.1)
A data type containing the information that CGen uses to generate a call to pert_insert_divisible_task.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:197](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L197)

---

<a id="type__pirparforstartend.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.PIRParForStartEnd [¶](#type__pirparforstartend.1)
After lowering, it is necessary to make the parfor body top-level statements so that basic blocks
can be correctly identified and labels correctly found.  There is a phase in parallel IR where we 
take a PIRParForAst node and split it into a parfor_start node followed by the body as top-level
statements followed by parfor_end (also a top-level statement).


*source:*
[ParallelAccelerator/src/parallel-ir.jl:400](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L400)

---

<a id="type__rangedata.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.RangeData [¶](#type__rangedata.1)
Holds the information from one Domain IR :range Expr.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:182](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L182)

---

<a id="type__removedeadstate.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.RemoveDeadState [¶](#type__removedeadstate.1)
Holds liveness information for the remove_dead AstWalk phase.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3280](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3280)

---

<a id="type__removenodepsstate.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.RemoveNoDepsState [¶](#type__removenodepsstate.1)
State for the remove_no_deps and insert_no_deps_beginning phases.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:3340](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L3340)

---

<a id="type__replacedregion.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.ReplacedRegion [¶](#type__replacedregion.1)
Store information about a section of a body that will be translated into a task.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:21](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L21)

---

<a id="type__rhsdead.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.RhsDead [¶](#type__rhsdead.1)
Marks an assignment statement where the left-hand side can take over the storage from the right-hand side.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:577](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L577)

---

<a id="type__statementwithdeps.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.StatementWithDeps [¶](#type__statementwithdeps.1)
Type for dependence graph creation and topological sorting.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:4171](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L4171)

---

<a id="type__taskinfo.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.TaskInfo [¶](#type__taskinfo.1)
Structure for storing information about task formation.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:36](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L36)

---

<a id="type__cur_body_data.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.cur_body_data [¶](#type__cur_body_data.1)
Holds the data for substitute_cur_body AST walk.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1737](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1737)

---

<a id="type__cuw_state.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.cuw_state [¶](#type__cuw_state.1)
Just to hold the "found" Bool that says whether a unsafe variant was replaced with a regular version.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:906](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L906)

---

<a id="type__expr_state.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.expr_state [¶](#type__expr_state.1)
State passed around while converting an AST from domain to parallel IR.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:407](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L407)

---

<a id="type__pir_arg_metadata.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.pir_arg_metadata [¶](#type__pir_arg_metadata.1)
A Julia representation of the argument metadata that will be passed to the runtime.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:157](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L157)

---

<a id="type__pir_array_access_desc.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.pir_array_access_desc [¶](#type__pir_array_access_desc.1)
Describes an array.
row_major is true if the array is stored in row major format.
dim_info describes which portion of the array is accessed for a given point in the iteration space.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:111](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L111)

---

<a id="type__pir_grain_size.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.pir_grain_size [¶](#type__pir_grain_size.1)
A Julia representation of the grain size that will be passed to the runtime.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:178](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L178)

---

<a id="type__pir_range.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.pir_range [¶](#type__pir_range.1)
Translated to pert_range_Nd_t in the task runtime.
This represents an iteration space.
dim is the number of dimensions in the iteration space.
lower_bounds contains the lower bound of the iteration space in each dimension.
upper_bounds contains the upper bound of the iteration space in each dimension.
lower_bounds and upper_bounds can be expressions.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:56](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L56)

---

<a id="type__pir_range_actual.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.pir_range_actual [¶](#type__pir_range_actual.1)
Similar to pir_range but used in circumstances where the expressions must have already been evaluated.
Therefore the arrays are typed as Int64.
Up to 3 dimensional iteration space constructors are supported to make it easier to do code generation later.


*source:*
[ParallelAccelerator/src/parallel-ir-task.jl:70](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir-task.jl#L70)

---

<a id="type__sub_arrayset_data.1" class="lexicon_definition"></a>
#### ParallelAccelerator.ParallelIR.sub_arrayset_data [¶](#type__sub_arrayset_data.1)
Holds data for modifying arrayset calls.


*source:*
[ParallelAccelerator/src/parallel-ir.jl:1615](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/parallel-ir.jl#L1615)

