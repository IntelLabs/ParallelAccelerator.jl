# API-INDEX


## MODULE: ParallelAccelerator.DistributedIR

---

## Internal

[checkParforsForDistribution(state::ParallelAccelerator.DistributedIR.DistIrState)](ParallelAccelerator.DistributedIR.md#method__checkparforsfordistribution.1)  All arrays of a parfor should distributable for it to be distributable.

[get_arr_dist_info(node::Expr,  state,  top_level_number,  is_top_level,  read)](ParallelAccelerator.DistributedIR.md#method__get_arr_dist_info.1)  mark sequential arrays

## MODULE: ParallelAccelerator.Comprehension

---

## Exported

[@comprehend(ast)](ParallelAccelerator.Comprehension.md#macro___comprehend.1)  Translate all comprehension in an AST into equivalent code that uses cartesianarray call.

---

## Internal

[comprehension_to_cartesianarray(ast)](ParallelAccelerator.Comprehension.md#method__comprehension_to_cartesianarray.1)  Translate an ast whose head is :comprehension into equivalent code that uses cartesianarray call.

[process_node(node,  state,  top_level_number,  is_top_level,  read)](ParallelAccelerator.Comprehension.md#method__process_node.1)  This function is a AstWalker callback.

## MODULE: ParallelAccelerator

---

## Internal

[__init__()](ParallelAccelerator.md#method____init__.1)  Called when the package is loaded to do initialization.

[embed()](ParallelAccelerator.md#method__embed.1)  This version of embed tries to use JULIA_HOME to find the root of the source distribution.

[embed(julia_root)](ParallelAccelerator.md#method__embed.2)  Call this function if you want to embed binary-code of ParallelAccelerator into your Julia build to speed-up @acc compilation time.

[getPackageRoot()](ParallelAccelerator.md#method__getpackageroot.1)  Generate a file path to the directory above the one containing this source file.

[getPseMode()](ParallelAccelerator.md#method__getpsemode.1)  Return internal mode number by looking up environment variable "PROSPECT_MODE".

[getTaskMode()](ParallelAccelerator.md#method__gettaskmode.1)  Return internal mode number by looking up environment variable "PROSPECT_TASK_MODE".

## MODULE: ParallelAccelerator.Driver

---

## Exported

[captureOperators(func,  ast,  sig)](ParallelAccelerator.Driver.md#method__captureoperators.1)  A pass that translates supported operators and function calls to

[runStencilMacro(func,  ast,  sig)](ParallelAccelerator.Driver.md#method__runstencilmacro.1)  Pass that translates runStencil call in the same way as a macro would do.

[toCartesianArray(func,  ast,  sig)](ParallelAccelerator.Driver.md#method__tocartesianarray.1)  Pass for comprehension to cartesianarray translation.

## MODULE: ParallelAccelerator.API.Stencil

---

## Exported

[runStencil(inputs...)](ParallelAccelerator.API.Stencil.md#method__runstencil.1)  "runStencil" takes arguments in the form of "(kernel_function, A, B, C, ...,

---

## Internal

[process_node(node,  state,  top_level_number,  is_top_level,  read)](ParallelAccelerator.API.Stencil.md#method__process_node.1)  This function is a AstWalker callback.

[@comprehend(ast)](ParallelAccelerator.API.Stencil.md#macro___comprehend.1)  Translate all comprehension in an AST into equivalent code that uses cartesianarray call.

## MODULE: ParallelAccelerator.DomainIR

---

## Internal

[lookupConstDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode})](ParallelAccelerator.DomainIR.md#method__lookupconstdef.1)  Look up a definition of a variable only when it is const or assigned once.

[lookupConstDefForArg(state::ParallelAccelerator.DomainIR.IRState,  s)](ParallelAccelerator.DomainIR.md#method__lookupconstdefforarg.1)  Look up a definition of a variable recursively until the RHS is no-longer just a variable.

[lookupDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode})](ParallelAccelerator.DomainIR.md#method__lookupdef.1)  Look up a definition of a variable.

[lookupDefInAllScopes(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode})](ParallelAccelerator.DomainIR.md#method__lookupdefinallscopes.1)  Look up a definition of a variable throughout nested states until a definition is found.

[updateDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode},  rhs)](ParallelAccelerator.DomainIR.md#method__updatedef.1)  Update the definition of a variable.

## MODULE: ParallelAccelerator.ParallelIR

---

## Exported

[AstWalk(ast,  callback,  cbdata)](ParallelAccelerator.ParallelIR.md#method__astwalk.1)  ParallelIR version of AstWalk.

[PIRInplace(x)](ParallelAccelerator.ParallelIR.md#method__pirinplace.1)  If set to non-zero, perform the phase where non-inplace maps are converted to inplace maps to reduce allocations.

[PIRNumSimplify(x)](ParallelAccelerator.ParallelIR.md#method__pirnumsimplify.1)  Specify the number of passes over the AST that do things like hoisting and other rearranging to maximize fusion.

[PIRRunAsTasks(x)](ParallelAccelerator.ParallelIR.md#method__pirrunastasks.1)  Debugging feature to specify the number of tasks to create and to stop thereafter.

[PIRSetFuseLimit(x)](ParallelAccelerator.ParallelIR.md#method__pirsetfuselimit.1)  Control how many parfor can be fused for testing purposes.

[PIRShortcutArrayAssignment(x)](ParallelAccelerator.ParallelIR.md#method__pirshortcutarrayassignment.1)  Enables an experimental mode where if there is a statement a = b and they are arrays and b is not live-out then 

[PIRTaskGraphMode(x)](ParallelAccelerator.ParallelIR.md#method__pirtaskgraphmode.1)  Control how blocks of code are made into tasks.

[from_exprs(ast::Array{Any, 1},  depth,  state)](ParallelAccelerator.ParallelIR.md#method__from_exprs.1)  Process an array of expressions.

[ParallelAccelerator.ParallelIR.PIRLoopNest](ParallelAccelerator.ParallelIR.md#type__pirloopnest.1)  Holds the information about a loop in a parfor node.

[ParallelAccelerator.ParallelIR.PIRParForAst](ParallelAccelerator.ParallelIR.md#type__pirparforast.1)  The parfor AST node type.

[ParallelAccelerator.ParallelIR.PIRReduction](ParallelAccelerator.ParallelIR.md#type__pirreduction.1)  Holds the information about a reduction in a parfor node.

---

## Internal

[AstWalkCallback(x::Expr,  dw::ParallelAccelerator.ParallelIR.DirWalk,  top_level_number::Int64,  is_top_level::Bool,  read::Bool)](ParallelAccelerator.ParallelIR.md#method__astwalkcallback.1)  AstWalk callback that handles ParallelIR AST node types.

[EquivalenceClassesAdd(ec::ParallelAccelerator.ParallelIR.EquivalenceClasses,  sym::Symbol)](ParallelAccelerator.ParallelIR.md#method__equivalenceclassesadd.1)  Add a symbol as part of a new equivalence class if the symbol wasn't already in an equivalence class.

[EquivalenceClassesClear(ec::ParallelAccelerator.ParallelIR.EquivalenceClasses)](ParallelAccelerator.ParallelIR.md#method__equivalenceclassesclear.1)  Clear an equivalence class.

[EquivalenceClassesMerge(ec::ParallelAccelerator.ParallelIR.EquivalenceClasses,  merge_to::Symbol,  merge_from::Symbol)](ParallelAccelerator.ParallelIR.md#method__equivalenceclassesmerge.1)  At some point we realize that two arrays must have the same dimensions but up until that point

[PIRBbReorder(x)](ParallelAccelerator.ParallelIR.md#method__pirbbreorder.1)  If set to non-zero, perform the bubble-sort like reordering phase to coalesce more parfor nodes together for fusion.

[PIRHoistAllocation(x)](ParallelAccelerator.ParallelIR.md#method__pirhoistallocation.1)  If set to non-zero, perform the rearrangement phase that tries to moves alllocations outside of loops.

[TypedExpr(typ,  rest...)](ParallelAccelerator.ParallelIR.md#method__typedexpr.1)  This should pretty always be used instead of Expr(...) to form an expression as it forces the typ to be provided.

[addUnknownArray(x::Union{GenSym, Symbol},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__addunknownarray.1)  Given an array whose name is in "x", allocate a new equivalence class for this array.

[addUnknownRange(x::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__addunknownrange.1)  Given an array of RangeExprs describing loop nest ranges, allocate a new equivalence class for this range.

[add_merge_correlations(old_sym::Union{GenSym, Symbol},  new_sym::Union{GenSym, Symbol},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__add_merge_correlations.1)  If we somehow determine that two arrays must be the same length then 

[asArray(x)](ParallelAccelerator.ParallelIR.md#method__asarray.1)  Return one element array with element x.

[augment_sn(dim::Int64,  index_vars,  range::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1})](ParallelAccelerator.ParallelIR.md#method__augment_sn.1)  Make sure the index parameters to arrayref or arrayset are Int64 or SymbolNode.

[call_instruction_count(args,  state::ParallelAccelerator.ParallelIR.eic_state,  debug_level)](ParallelAccelerator.ParallelIR.md#method__call_instruction_count.1)  Generate an instruction count estimate for a call instruction.

[checkAndAddSymbolCorrelation(lhs::Union{GenSym, Symbol},  state,  dim_array)](ParallelAccelerator.ParallelIR.md#method__checkandaddsymbolcorrelation.1)  Make sure all the dimensions are SymbolNodes.

[convertUnsafe(stmt)](ParallelAccelerator.ParallelIR.md#method__convertunsafe.1)  Remove unsafe array access Symbols from the incoming "stmt".

[convertUnsafeOrElse(stmt)](ParallelAccelerator.ParallelIR.md#method__convertunsafeorelse.1)  Try to remove unsafe array access Symbols from the incoming "stmt".  If successful, then return the updated

[convertUnsafeWalk(x::Expr,  state,  top_level_number,  is_top_level,  read)](ParallelAccelerator.ParallelIR.md#method__convertunsafewalk.1)  The AstWalk callback to find unsafe arrayset and arrayref variants and

[copy_propagate(node::ANY,  data::ParallelAccelerator.ParallelIR.CopyPropagateState,  top_level_number,  is_top_level,  read)](ParallelAccelerator.ParallelIR.md#method__copy_propagate.1)  In each basic block, if there is a "copy" (i.e., something of the form "a = b") then put

[count_assignments(x,  symbol_assigns::Dict{Symbol, Int64},  top_level_number,  is_top_level,  read)](ParallelAccelerator.ParallelIR.md#method__count_assignments.1)  AstWalk callback to count the number of static times that a symbol is assigne within a method.

[create1D_array_access_desc(array::SymbolNode)](ParallelAccelerator.ParallelIR.md#method__create1d_array_access_desc.1)  Create an array access descriptor for "array".

[create2D_array_access_desc(array::SymbolNode)](ParallelAccelerator.ParallelIR.md#method__create2d_array_access_desc.1)  Create an array access descriptor for "array".

[createInstructionCountEstimate(the_parfor::ParallelAccelerator.ParallelIR.PIRParForAst,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__createinstructioncountestimate.1)  Takes a parfor and walks the body of the parfor and estimates the number of instruction needed for one instance of that body.

[createLoweredAliasMap(dict1)](ParallelAccelerator.ParallelIR.md#method__createloweredaliasmap.1)  Take a single-step alias map, e.g., a=>b, b=>c, and create a lowered dictionary, a=>c, b=>c, that

[createMapLhsToParfor(parfor_assignment,  the_parfor,  is_multi::Bool,  sym_to_type::Dict{Union{GenSym, Symbol}, DataType},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__createmaplhstoparfor.1)  Creates a mapping between variables on the left-hand side of an assignment where the right-hand side is a parfor

[createStateVar(state,  name,  typ,  access)](ParallelAccelerator.ParallelIR.md#method__createstatevar.1)  Add a local variable to the current function's lambdaInfo.

[createTempForArray(array_sn::Union{GenSym, Symbol, SymbolNode},  unique_id::Int64,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__createtempforarray.1)  Create a temporary variable that is parfor private to hold the value of an element of an array.

[createTempForArray(array_sn::Union{GenSym, Symbol, SymbolNode},  unique_id::Int64,  state::ParallelAccelerator.ParallelIR.expr_state,  temp_type)](ParallelAccelerator.ParallelIR.md#method__createtempforarray.2)  Create a temporary variable that is parfor private to hold the value of an element of an array.

[createTempForRangeOffset(num_used,  ranges::Array{ParallelAccelerator.ParallelIR.RangeData, 1},  unique_id::Int64,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__createtempforrangeoffset.1)  Create a variable to hold the offset of a range offset from the start of the array.

[createTempForRangedArray(array_sn::Union{GenSym, Symbol, SymbolNode},  range::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1},  unique_id::Int64,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__createtempforrangedarray.1)  Create a temporary variable that is parfor private to hold the value of an element of an array.

[create_array_access_desc(array::SymbolNode)](ParallelAccelerator.ParallelIR.md#method__create_array_access_desc.1)  Create an array access descriptor for "array".

[create_equivalence_classes(node::Expr,  state::ParallelAccelerator.ParallelIR.expr_state,  top_level_number::Int64,  is_top_level::Bool,  read::Bool)](ParallelAccelerator.ParallelIR.md#method__create_equivalence_classes.1)  AstWalk callback to determine the array equivalence classes.

[dfsVisit(swd::ParallelAccelerator.ParallelIR.StatementWithDeps,  vtime::Int64,  topo_sort::Array{ParallelAccelerator.ParallelIR.StatementWithDeps, N})](ParallelAccelerator.ParallelIR.md#method__dfsvisit.1)  Construct a topological sort of the dependence graph.

[estimateInstrCount(ast::Expr,  state::ParallelAccelerator.ParallelIR.eic_state,  top_level_number,  is_top_level,  read)](ParallelAccelerator.ParallelIR.md#method__estimateinstrcount.1)  AstWalk callback for estimating the instruction count.

[extractArrayEquivalencies(node::Expr,  state)](ParallelAccelerator.ParallelIR.md#method__extractarrayequivalencies.1)  "node" is a domainIR node.  Take the arrays used in this node, create an array equivalence for them if they 

[findSelectedDimensions(inputInfo::Array{ParallelAccelerator.ParallelIR.InputInfo, 1},  state)](ParallelAccelerator.ParallelIR.md#method__findselecteddimensions.1)  Given all the InputInfo for a Domain IR operation being lowered to Parallel IR,

[flattenParfor(new_body,  the_parfor::ParallelAccelerator.ParallelIR.PIRParForAst)](ParallelAccelerator.ParallelIR.md#method__flattenparfor.1)  Takes a new array of body statements in the process of construction in "new_body" and takes a parfor to add to that

[form_and_simplify(ranges::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1})](ParallelAccelerator.ParallelIR.md#method__form_and_simplify.1)  For each entry in ranges, form a range length expression and simplify them.

[form_and_simplify(rd::ParallelAccelerator.ParallelIR.RangeData)](ParallelAccelerator.ParallelIR.md#method__form_and_simplify.2)  Convert one RangeData to some length expression and then simplify it.

[from_assertEqShape(node::Expr,  state)](ParallelAccelerator.ParallelIR.md#method__from_asserteqshape.1)  Create array equivalences from an assertEqShape AST node.

[from_assignment(lhs,  rhs,  depth,  state)](ParallelAccelerator.ParallelIR.md#method__from_assignment.1)  Process an assignment expression.

[from_call(ast::Array{Any, 1},  depth,  state)](ParallelAccelerator.ParallelIR.md#method__from_call.1)  Process a call AST node.

[from_expr(ast::Expr,  depth,  state::ParallelAccelerator.ParallelIR.expr_state,  top_level)](ParallelAccelerator.ParallelIR.md#method__from_expr.1)  The main ParallelIR function for processing some node in the AST.

[from_lambda(lambda::Expr,  depth,  state)](ParallelAccelerator.ParallelIR.md#method__from_lambda.1)  Process a :lambda Expr.

[from_root(function_name,  ast::Expr)](ParallelAccelerator.ParallelIR.md#method__from_root.1)  The main ENTRY point into ParallelIR.

[fullyLowerAlias(dict::Dict{Union{GenSym, Symbol}, Union{GenSym, Symbol}},  input::Union{GenSym, Symbol})](ParallelAccelerator.ParallelIR.md#method__fullyloweralias.1)  Given an "input" Symbol, use that Symbol as key to a dictionary.  While such a Symbol is present

[fuse(body,  body_index,  cur,  state)](ParallelAccelerator.ParallelIR.md#method__fuse.1)  Test whether we can fuse the two most recent parfor statements and if so to perform that fusion.

[generate_instr_count(function_name,  signature)](ParallelAccelerator.ParallelIR.md#method__generate_instr_count.1)  Try to figure out the instruction count for a given call.

[getArrayElemType(array::GenSym,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__getarrayelemtype.1)  Returns the element type of an Array.

[getArrayElemType(array::SymbolNode,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__getarrayelemtype.2)  Returns the element type of an Array.

[getArrayElemType(atyp::DataType)](ParallelAccelerator.ParallelIR.md#method__getarrayelemtype.3)  Returns the element type of an Array.

[getArrayNumDims(array::GenSym,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__getarraynumdims.1)  Return the number of dimensions of an Array.

[getArrayNumDims(array::SymbolNode,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__getarraynumdims.2)  Return the number of dimensions of an Array.

[getConstDims(num_dim_inputs,  inputInfo::ParallelAccelerator.ParallelIR.InputInfo)](ParallelAccelerator.ParallelIR.md#method__getconstdims.1)  In the case where a domain IR operation on an array creates a lower dimensional output,

[getCorrelation(sng::Union{GenSym, Symbol, SymbolNode},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__getcorrelation.1)  Get the equivalence class of a domain IR input in inputInfo.

[getFirstArrayLens(prestatements,  num_dims)](ParallelAccelerator.ParallelIR.md#method__getfirstarraylens.1)  Get the variable which holds the length of the first input array to a parfor.

[getIO(stmt_ids,  bb_statements)](ParallelAccelerator.ParallelIR.md#method__getio.1)  Given a set of statement IDs and liveness information for the statements of the function, determine

[getInputSet(node::ParallelAccelerator.ParallelIR.PIRParForAst)](ParallelAccelerator.ParallelIR.md#method__getinputset.1)  Returns a Set with all the arrays read by this parfor.

[getLhsFromAssignment(assignment)](ParallelAccelerator.ParallelIR.md#method__getlhsfromassignment.1)  Get the left-hand side of an assignment expression.

[getLhsOutputSet(lhs,  assignment)](ParallelAccelerator.ParallelIR.md#method__getlhsoutputset.1)  Get the real outputs of an assignment statement.

[getMaxLabel(max_label,  stmts::Array{Any, 1})](ParallelAccelerator.ParallelIR.md#method__getmaxlabel.1)  Scan the body of a function in "stmts" and return the max label in a LabelNode AST seen in the body.

[getNonBlock(head_preds,  back_edge)](ParallelAccelerator.ParallelIR.md#method__getnonblock.1)  Find the basic block before the entry to a loop.

[getOrAddArrayCorrelation(x::Union{GenSym, Symbol},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__getoraddarraycorrelation.1)  Return a correlation set for an array.  If the array was not previously added then add it and return it.

[getOrAddRangeCorrelation(array,  ranges::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__getoraddrangecorrelation.1)  Gets (or adds if absent) the range correlation for the given array of RangeExprs.

[getOrAddSymbolCorrelation(array::Union{GenSym, Symbol},  state::ParallelAccelerator.ParallelIR.expr_state,  dims::Array{Union{GenSym, Symbol}, 1})](ParallelAccelerator.ParallelIR.md#method__getoraddsymbolcorrelation.1)  A new array is being created with an explicit size specification in dims.

[getParforCorrelation(parfor,  state)](ParallelAccelerator.ParallelIR.md#method__getparforcorrelation.1)  Get the equivalence class of the first array who length is extracted in the pre-statements of the specified "parfor".

[getParforNode(node)](ParallelAccelerator.ParallelIR.md#method__getparfornode.1)  Get the parfor object from either a bare parfor or one part of an assignment.

[getPastIndex(arrays::Dict{Union{GenSym, Symbol}, Array{Array{Any, 1}, 1}})](ParallelAccelerator.ParallelIR.md#method__getpastindex.1)  Look at the arrays that are accessed and see if they use a forward index, i.e.,

[getPrivateSet(body::Array{Any, 1})](ParallelAccelerator.ParallelIR.md#method__getprivateset.1)  Go through the body of a parfor and collect those Symbols, GenSyms, etc. that are assigned to within the parfor except reduction variables.

[getPrivateSetInner(x::Expr,  state::Set{Union{GenSym, Symbol, SymbolNode}},  top_level_number::Int64,  is_top_level::Bool,  read::Bool)](ParallelAccelerator.ParallelIR.md#method__getprivatesetinner.1)  The AstWalk callback function for getPrivateSet.

[getRhsFromAssignment(assignment)](ParallelAccelerator.ParallelIR.md#method__getrhsfromassignment.1)  Get the right-hand side of an assignment expression.

[getSName(ssn::Symbol)](ParallelAccelerator.ParallelIR.md#method__getsname.1)  Get the name of a symbol whether the input is a Symbol or SymbolNode or :(::) Expr.

[get_one(ast::Array{T, N})](ParallelAccelerator.ParallelIR.md#method__get_one.1)  Take something returned from AstWalk and assert it should be an array but in this

[get_unique_num()](ParallelAccelerator.ParallelIR.md#method__get_unique_num.1)  If we need to generate a name and make sure it is unique then include an monotonically increasing number.

[hasNoSideEffects(node::Union{GenSym, LambdaStaticData, Number, Symbol, SymbolNode})](ParallelAccelerator.ParallelIR.md#method__hasnosideeffects.1)  Sometimes statements we exist in the AST of the form a=Expr where a is a Symbol that isn't live past the assignment

[hasSymbol(ssn::Symbol)](ParallelAccelerator.ParallelIR.md#method__hassymbol.1)  Returns true if the incoming AST node can be interpreted as a Symbol.

[hoistAllocation(ast::Array{Any, 1},  lives,  domLoop::CompilerTools.Loops.DomLoops,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__hoistallocation.1)  Try to hoist allocations outside the loop if possible.

[insert_no_deps_beginning(node,  data::ParallelAccelerator.ParallelIR.RemoveNoDepsState,  top_level_number,  is_top_level,  read)](ParallelAccelerator.ParallelIR.md#method__insert_no_deps_beginning.1)  Works with remove_no_deps below to move statements with no dependencies to the beginning of the AST.

[intermediate_from_exprs(ast::Array{Any, 1},  depth,  state)](ParallelAccelerator.ParallelIR.md#method__intermediate_from_exprs.1)  Process an array of expressions that aren't from a :body Expr.

[isArrayType(typ)](ParallelAccelerator.ParallelIR.md#method__isarraytype.1)  Returns true if the incoming type in "typ" is an array type.

[isArrayType(x::SymbolNode)](ParallelAccelerator.ParallelIR.md#method__isarraytype.2)  Returns true if a given SymbolNode "x" is an Array type.

[isArrayref(x)](ParallelAccelerator.ParallelIR.md#method__isarrayref.1)  Is a node an arrayref node?

[isArrayrefCall(x::Expr)](ParallelAccelerator.ParallelIR.md#method__isarrayrefcall.1)  Is a node a call to arrayref.

[isArrayset(x)](ParallelAccelerator.ParallelIR.md#method__isarrayset.1)  Is a node an arrayset node?

[isArraysetCall(x::Expr)](ParallelAccelerator.ParallelIR.md#method__isarraysetcall.1)  Is a node a call to arrayset.

[isAssignmentNode(node::Expr)](ParallelAccelerator.ParallelIR.md#method__isassignmentnode.1)  Is a node an assignment expression node.

[isBareParfor(node::Expr)](ParallelAccelerator.ParallelIR.md#method__isbareparfor.1)  Is this a parfor node not part of an assignment statement.

[isDomainNode(ast::Expr)](ParallelAccelerator.ParallelIR.md#method__isdomainnode.1)  Returns true if the given "ast" node is a DomainIR operation.

[isFusionAssignment(x::Expr)](ParallelAccelerator.ParallelIR.md#method__isfusionassignment.1)  Check if an assignement is a fusion assignment.

[isLoopheadNode(node::Expr)](ParallelAccelerator.ParallelIR.md#method__isloopheadnode.1)  Is a node a loophead expression node (a form of assignment).

[isParforAssignmentNode(node::Expr)](ParallelAccelerator.ParallelIR.md#method__isparforassignmentnode.1)  Is a node an assignment expression with a parfor node as the right-hand side.

[isSymbolsUsed(vars,  top_level_numbers::Array{Int64, 1},  state)](ParallelAccelerator.ParallelIR.md#method__issymbolsused.1)  Returns true if any variable in the collection "vars" is used in any statement whose top level number is in "top_level_numbers".

[is_eliminated_arraylen(x::Expr)](ParallelAccelerator.ParallelIR.md#method__is_eliminated_arraylen.1)  Returns true if the input node is an assignment node where the right-hand side is a call to arraysize.

[isbitstuple(a::Tuple)](ParallelAccelerator.ParallelIR.md#method__isbitstuple.1)  Returns true if input "a" is a tuple and each element of the tuple of isbits type.

[iterations_equals_inputs(node::ParallelAccelerator.ParallelIR.PIRParForAst)](ParallelAccelerator.ParallelIR.md#method__iterations_equals_inputs.1)  Returns true if the domain operation mapped to this parfor has the property that the iteration space

[lambdaFromDomainLambda(domain_lambda,  dl_inputs)](ParallelAccelerator.ParallelIR.md#method__lambdafromdomainlambda.1)  Form a Julia :lambda Expr from a DomainLambda.

[makePrivateParfor(var_name::Symbol,  state)](ParallelAccelerator.ParallelIR.md#method__makeprivateparfor.1)  Takes an existing variable whose name is in "var_name" and adds the descriptor flag ISPRIVATEPARFORLOOP to declare the

[makeTasks(start_index,  stop_index,  body,  bb_live_info,  state,  task_graph_mode)](ParallelAccelerator.ParallelIR.md#method__maketasks.1)  For a given start and stop index in some body and liveness information, form a set of tasks.

[maxFusion(bl::CompilerTools.LivenessAnalysis.BlockLiveness)](ParallelAccelerator.ParallelIR.md#method__maxfusion.1)  For every basic block, try to push domain IR statements down and non-domain IR statements up so that domain nodes

[mergeLambdaIntoOuterState(state,  inner_lambda::Expr)](ParallelAccelerator.ParallelIR.md#method__mergelambdaintoouterstate.1)  Pull the information from the inner lambda into the outer lambda.

[merge_correlations(state,  unchanging,  eliminate)](ParallelAccelerator.ParallelIR.md#method__merge_correlations.1)  If we somehow determine that two sets of correlations are actually the same length then merge one into the other.

[mk_alloc_array_1d_expr(elem_type,  atype,  length)](ParallelAccelerator.ParallelIR.md#method__mk_alloc_array_1d_expr.1)  Return an expression that allocates and initializes a 1D Julia array that has an element type specified by

[mk_alloc_array_2d_expr(elem_type,  atype,  length1,  length2)](ParallelAccelerator.ParallelIR.md#method__mk_alloc_array_2d_expr.1)  Return an expression that allocates and initializes a 2D Julia array that has an element type specified by

[mk_alloc_array_3d_expr(elem_type,  atype,  length1,  length2,  length3)](ParallelAccelerator.ParallelIR.md#method__mk_alloc_array_3d_expr.1)  Return an expression that allocates and initializes a 3D Julia array that has an element type specified by

[mk_arraylen_expr(x::ParallelAccelerator.ParallelIR.InputInfo,  dim::Int64)](ParallelAccelerator.ParallelIR.md#method__mk_arraylen_expr.1)  Create an expression whose value is the length of the input array.

[mk_arraylen_expr(x::Union{GenSym, Symbol, SymbolNode},  dim::Int64)](ParallelAccelerator.ParallelIR.md#method__mk_arraylen_expr.2)  Create an expression whose value is the length of the input array.

[mk_arrayref1(num_dim_inputs,  array_name,  index_vars,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__mk_arrayref1.1)  Return an expression that corresponds to getting the index_var index from the array array_name.

[mk_arrayref1(num_dim_inputs,  array_name,  index_vars,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state,  range::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1})](ParallelAccelerator.ParallelIR.md#method__mk_arrayref1.2)  Return an expression that corresponds to getting the index_var index from the array array_name.

[mk_arrayset1(num_dim_inputs,  array_name,  index_vars,  value,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__mk_arrayset1.1)  Return a new AST node that corresponds to setting the index_var index from the array "array_name" with "value".

[mk_arrayset1(num_dim_inputs,  array_name,  index_vars,  value,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state,  range::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1})](ParallelAccelerator.ParallelIR.md#method__mk_arrayset1.2)  Return a new AST node that corresponds to setting the index_var index from the array "array_name" with "value".

[mk_assignment_expr(lhs::Union{GenSym, Symbol, SymbolNode},  rhs,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__mk_assignment_expr.1)  Create an assignment expression AST node given a left and right-hand side.

[mk_colon_expr(start_expr,  skip_expr,  end_expr)](ParallelAccelerator.ParallelIR.md#method__mk_colon_expr.1)  Returns an expression to construct a :colon object that contains the start of a range, the end and the skip expression.

[mk_convert(new_type,  ex)](ParallelAccelerator.ParallelIR.md#method__mk_convert.1)  Returns an expression that convert "ex" into a another type "new_type".

[mk_gotoifnot_expr(cond,  goto_label)](ParallelAccelerator.ParallelIR.md#method__mk_gotoifnot_expr.1)  Returns a :gotoifnot Expr given a condition "cond" and a label "goto_label".

[mk_mask_arrayref1(cur_dimension,  num_dim_inputs,  array_name,  index_vars,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__mk_mask_arrayref1.1)  Return an expression that corresponds to getting the index_var index from the array array_name.

[mk_mask_arrayref1(cur_dimension,  num_dim_inputs,  array_name,  index_vars,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state,  range::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1})](ParallelAccelerator.ParallelIR.md#method__mk_mask_arrayref1.2)  Return an expression that corresponds to getting the index_var index from the array array_name.

[mk_next_expr(colon_sym,  start_sym)](ParallelAccelerator.ParallelIR.md#method__mk_next_expr.1)  Returns a :next call Expr that gets the next element of an iteration range from a :colon object.

[mk_parallelir_ref(sym)](ParallelAccelerator.ParallelIR.md#method__mk_parallelir_ref.1)  Create an expression that references something inside ParallelIR.

[mk_parallelir_ref(sym,  ref_type)](ParallelAccelerator.ParallelIR.md#method__mk_parallelir_ref.2)  Create an expression that references something inside ParallelIR.

[mk_parfor_args_from_mmap!(input_arrays::Array{T, N},  dl::ParallelAccelerator.DomainIR.DomainLambda,  with_indices,  domain_oprs,  state)](ParallelAccelerator.ParallelIR.md#method__mk_parfor_args_from_mmap.1)  The main routine that converts a mmap! AST node to a parfor AST node.

[mk_parfor_args_from_mmap(input_arrays::Array{T, N},  dl::ParallelAccelerator.DomainIR.DomainLambda,  domain_oprs,  state)](ParallelAccelerator.ParallelIR.md#method__mk_parfor_args_from_mmap.2)  The main routine that converts a mmap AST node to a parfor AST node.

[mk_parfor_args_from_reduce(input_args::Array{Any, 1},  state)](ParallelAccelerator.ParallelIR.md#method__mk_parfor_args_from_reduce.1)  The main routine that converts a reduce AST node to a parfor AST node.

[mk_return_expr(outs)](ParallelAccelerator.ParallelIR.md#method__mk_return_expr.1)  Given an array of outputs in "outs", form a return expression.

[mk_start_expr(colon_sym)](ParallelAccelerator.ParallelIR.md#method__mk_start_expr.1)  Returns an expression to get the start of an iteration range from a :colon object.

[mk_svec_expr(parts...)](ParallelAccelerator.ParallelIR.md#method__mk_svec_expr.1)  Make a svec expression.

[mk_tuple_expr(tuple_fields,  typ)](ParallelAccelerator.ParallelIR.md#method__mk_tuple_expr.1)  Return an expression which creates a tuple.

[mk_tupleref_expr(tuple_var,  index,  typ)](ParallelAccelerator.ParallelIR.md#method__mk_tupleref_expr.1)  Create an expression which returns the index'th element of the tuple whose name is contained in tuple_var.

[mk_untyped_assignment(lhs,  rhs)](ParallelAccelerator.ParallelIR.md#method__mk_untyped_assignment.1)  Only used to create fake expression to force lhs to be seen as written rather than read.

[mmapInline(ast::Expr,  lives,  uniqSet)](ParallelAccelerator.ParallelIR.md#method__mmapinline.1)  # If a definition of a mmap is only used once and not aliased, it can be inlined into its

[mmapToMmap!(ast,  lives,  uniqSet)](ParallelAccelerator.ParallelIR.md#method__mmaptommap.1)  Performs the mmap to mmap! phase.

[mustRemainLastStatementInBlock(node::GotoNode)](ParallelAccelerator.ParallelIR.md#method__mustremainlaststatementinblock.1)  Returns true if the given AST "node" must remain the last statement in a basic block.

[nameToSymbolNode(name::Symbol,  sym_to_type)](ParallelAccelerator.ParallelIR.md#method__nametosymbolnode.1)  Forms a SymbolNode given a symbol in "name" and get the type of that symbol from the incoming dictionary "sym_to_type".

[nested_function_exprs(max_label,  domain_lambda,  dl_inputs)](ParallelAccelerator.ParallelIR.md#method__nested_function_exprs.1)  A routine similar to the main parallel IR entry put but designed to process the lambda part of

[next_label(state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__next_label.1)  Returns the next usable label for the current function.

[nonExactRangeSearch(ranges::Array{Union{ParallelAccelerator.ParallelIR.MaskSelector, ParallelAccelerator.ParallelIR.RangeData, ParallelAccelerator.ParallelIR.SingularSelector}, 1},  range_correlations)](ParallelAccelerator.ParallelIR.md#method__nonexactrangesearch.1)  We can only do exact matches in the range correlation dict but there can still be non-exact matches

[oneIfOnly(x)](ParallelAccelerator.ParallelIR.md#method__oneifonly.1)  Returns a single element of an array if there is only one or the array otherwise.

[parforToTask(parfor_index,  bb_statements,  body,  state)](ParallelAccelerator.ParallelIR.md#method__parfortotask.1)  Given a parfor statement index in "parfor_index" in the "body"'s statements, create a TaskInfo node for this parfor.

[pirPrintDl(dbg_level,  dl)](ParallelAccelerator.ParallelIR.md#method__pirprintdl.1)  Debug print the parts of a DomainLambda.

[pir_alias_cb(ast::Expr,  state,  cbdata)](ParallelAccelerator.ParallelIR.md#method__pir_alias_cb.1)  An AliasAnalysis callback (similar to LivenessAnalysis callback) that handles ParallelIR introduced AST node types.

[pir_live_cb(ast::Expr,  cbdata::ANY)](ParallelAccelerator.ParallelIR.md#method__pir_live_cb.1)  A LivenessAnalysis callback that handles ParallelIR introduced AST node types.

[pir_live_cb_def(x)](ParallelAccelerator.ParallelIR.md#method__pir_live_cb_def.1)  Just call the AST walker for symbol for parallel IR nodes with no state.

[printBody(dlvl,  body::Array{Any, 1})](ParallelAccelerator.ParallelIR.md#method__printbody.1)  Pretty print the args part of the "body" of a :lambda Expr at a given debug level in "dlvl".

[printLambda(dlvl,  node::Expr)](ParallelAccelerator.ParallelIR.md#method__printlambda.1)  Pretty print a :lambda Expr in "node" at a given debug level in "dlvl".

[processAndUpdateBody(lambda::Expr,  f::Function,  state)](ParallelAccelerator.ParallelIR.md#method__processandupdatebody.1)  Apply a function "f" that takes the :body from the :lambda and returns a new :body that is stored back into the :lambda.

[rangeSize(start,  skip,  last)](ParallelAccelerator.ParallelIR.md#method__rangesize.1)  Compute size of a range.

[rangeToRangeData(range::Expr,  arr,  range_num::Int64,  state)](ParallelAccelerator.ParallelIR.md#method__rangetorangedata.1)  Convert a :range Expr introduced by Domain IR into a Parallel IR data structure RangeData.

[recreateLoops(new_body,  the_parfor::ParallelAccelerator.ParallelIR.PIRParForAst,  state,  newLambdaInfo)](ParallelAccelerator.ParallelIR.md#method__recreateloops.1)  In threads mode, we can't have parfor_start and parfor_end in the code since Julia has to compile the code itself and so

[recreateLoopsInternal(new_body,  the_parfor::ParallelAccelerator.ParallelIR.PIRParForAst,  loop_nest_level,  next_available_label,  state,  newLambdaInfo)](ParallelAccelerator.ParallelIR.md#method__recreateloopsinternal.1)  This is a recursive routine to reconstruct a regular Julia loop nest from the loop nests described in PIRParForAst.

[rememberTypeForSym(sym_to_type::Dict{Union{GenSym, Symbol}, DataType},  sym::Union{GenSym, Symbol},  typ::DataType)](ParallelAccelerator.ParallelIR.md#method__remembertypeforsym.1)  Add to the map of symbol names to types.

[removeAssertEqShape(args::Array{Any, 1},  state)](ParallelAccelerator.ParallelIR.md#method__removeasserteqshape.1)  Implements one of the main ParallelIR passes to remove assertEqShape AST nodes from the body if they are statically known to be in the same equivalence class.

[removeNothingStmts(args::Array{Any, 1},  state)](ParallelAccelerator.ParallelIR.md#method__removenothingstmts.1)  Empty statements can be added to the AST by some passes in ParallelIR.

[remove_dead(node,  data::ParallelAccelerator.ParallelIR.RemoveDeadState,  top_level_number,  is_top_level,  read)](ParallelAccelerator.ParallelIR.md#method__remove_dead.1)  An AstWalk callback that uses liveness information in "data" to remove dead stores.

[remove_extra_allocs(ast)](ParallelAccelerator.ParallelIR.md#method__remove_extra_allocs.1)  removes extra allocations

[remove_no_deps(node::ANY,  data::ParallelAccelerator.ParallelIR.RemoveNoDepsState,  top_level_number,  is_top_level,  read)](ParallelAccelerator.ParallelIR.md#method__remove_no_deps.1)  # This routine gathers up nodes that do not use

[replaceParforWithDict(parfor::ParallelAccelerator.ParallelIR.PIRParForAst,  gensym_map)](ParallelAccelerator.ParallelIR.md#method__replaceparforwithdict.1)  Not currently used but might need it at some point.

[run_as_task()](ParallelAccelerator.ParallelIR.md#method__run_as_task.1)  Return true if run_as_task_decrement would return true but don't update the run_as_tasks count.

[run_as_task_decrement()](ParallelAccelerator.ParallelIR.md#method__run_as_task_decrement.1)  If run_as_tasks is positive then convert this parfor to a task and decrement the count so that only the

[selectToRangeData(select::Expr,  pre_offsets::Array{Expr, 1},  state)](ParallelAccelerator.ParallelIR.md#method__selecttorangedata.1)  Convert the range(s) part of a :select Expr introduced by Domain IR into an array of Parallel IR data structures RangeData.

[seqTask(body_indices,  bb_statements,  body,  state)](ParallelAccelerator.ParallelIR.md#method__seqtask.1)  Form a task out of a range of sequential statements.

[show(io::IO,  pnode::ParallelAccelerator.ParallelIR.PIRParForAst)](ParallelAccelerator.ParallelIR.md#method__show.1)  Overload of Base.show to pretty print for parfor AST nodes.

[simpleIndex(dict)](ParallelAccelerator.ParallelIR.md#method__simpleindex.1)  Returns true if all array references use singular index variables and nothing more complicated involving,

[simplify_internal(x::Expr,  state,  top_level_number::Int64,  is_top_level::Bool,  read::Bool)](ParallelAccelerator.ParallelIR.md#method__simplify_internal.1)  Do some simplification to expressions that are part of ranges.

[sub_arraylen_walk(x::Expr,  replacement,  top_level_number,  is_top_level,  read)](ParallelAccelerator.ParallelIR.md#method__sub_arraylen_walk.1)  AstWalk callback that does the work of substitute_arraylen on a node-by-node basis.

[sub_arrayset_walk(x::Expr,  cbd,  top_level_number,  is_top_level,  read)](ParallelAccelerator.ParallelIR.md#method__sub_arrayset_walk.1)  AstWalk callback that does the work of substitute_arrayset on a node-by-node basis.

[sub_cur_body_walk(x::Expr,  cbd::ParallelAccelerator.ParallelIR.cur_body_data,  top_level_number::Int64,  is_top_level::Bool,  read::Bool)](ParallelAccelerator.ParallelIR.md#method__sub_cur_body_walk.1)  AstWalk callback that does the work of substitute_cur_body on a node-by-node basis.

[substitute_arraylen(x,  replacement)](ParallelAccelerator.ParallelIR.md#method__substitute_arraylen.1)  replacement is an array containing the length of the dimensions of the arrays a part of this parfor.

[substitute_arrayset(x,  arrays_set_in_cur_body,  output_items_with_aliases)](ParallelAccelerator.ParallelIR.md#method__substitute_arrayset.1)  Modify the body of a parfor.

[substitute_cur_body(x,  temp_map::Dict{Union{GenSym, Symbol}, Union{GenSym, SymbolNode}},  index_map::Dict{Union{GenSym, Symbol}, Union{GenSym, Symbol}},  arrays_set_in_cur_body::Set{Union{GenSym, Symbol}},  replace_array_name_in_arrayset::Dict{Union{GenSym, Symbol}, Union{GenSym, Symbol}},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__substitute_cur_body.1)  Make changes to the second parfor body in the process of parfor fusion.

[taskableParfor(node)](ParallelAccelerator.ParallelIR.md#method__taskableparfor.1)  Returns true if the "node" is a parfor and the task limit hasn't been exceeded.

[toSNGen(x::Symbol,  typ)](ParallelAccelerator.ParallelIR.md#method__tosngen.1)  If we have the type, convert a Symbol to SymbolNode.

[toSymGen(x::Symbol)](ParallelAccelerator.ParallelIR.md#method__tosymgen.1)  In various places we need a SymGen type which is the union of Symbol and GenSym.

[toSymNodeGen(x::Symbol,  typ)](ParallelAccelerator.ParallelIR.md#method__tosymnodegen.1)  Form a SymbolNode with the given typ if possible or a GenSym if that is what is passed in.

[uncompressed_ast(l::LambdaStaticData)](ParallelAccelerator.ParallelIR.md#method__uncompressed_ast.1)  Convert a compressed LambdaStaticData format into the uncompressed AST format.

[ParallelAccelerator.ParallelIR.CopyPropagateState](ParallelAccelerator.ParallelIR.md#type__copypropagatestate.1)  State to aide in the copy propagation phase.

[ParallelAccelerator.ParallelIR.DelayedFunc](ParallelAccelerator.ParallelIR.md#type__delayedfunc.1)  Ad-hoc support to mimic closures when we want the arguments to be processed during AstWalk.

[ParallelAccelerator.ParallelIR.DirWalk](ParallelAccelerator.ParallelIR.md#type__dirwalk.1)  Wraps the callback and opaque data passed from the user of ParallelIR's AstWalk.

[ParallelAccelerator.ParallelIR.DomainOperation](ParallelAccelerator.ParallelIR.md#type__domainoperation.1)  Holds information about domain operations part of a parfor node.

[ParallelAccelerator.ParallelIR.EquivalenceClasses](ParallelAccelerator.ParallelIR.md#type__equivalenceclasses.1)  Holds a dictionary from an array symbol to an integer corresponding to an equivalence class.

[ParallelAccelerator.ParallelIR.FusionSentinel](ParallelAccelerator.ParallelIR.md#type__fusionsentinel.1)  Just used to hold a spot in an array to indicate the this is a special assignment expression with embedded real array output names from a fusion.

[ParallelAccelerator.ParallelIR.InProgress](ParallelAccelerator.ParallelIR.md#type__inprogress.1)  A sentinel in the instruction count estimation process.

[ParallelAccelerator.ParallelIR.InputInfo](ParallelAccelerator.ParallelIR.md#type__inputinfo.1)  Type used by mk_parfor_args... functions to hold information about input arrays.

[ParallelAccelerator.ParallelIR.InsertTaskNode](ParallelAccelerator.ParallelIR.md#type__inserttasknode.1)  A data type containing the information that CGen uses to generate a call to pert_insert_divisible_task.

[ParallelAccelerator.ParallelIR.PIRParForStartEnd](ParallelAccelerator.ParallelIR.md#type__pirparforstartend.1)  After lowering, it is necessary to make the parfor body top-level statements so that basic blocks

[ParallelAccelerator.ParallelIR.RangeData](ParallelAccelerator.ParallelIR.md#type__rangedata.1)  Holds the information from one Domain IR :range Expr.

[ParallelAccelerator.ParallelIR.RemoveDeadState](ParallelAccelerator.ParallelIR.md#type__removedeadstate.1)  Holds liveness information for the remove_dead AstWalk phase.

[ParallelAccelerator.ParallelIR.RemoveNoDepsState](ParallelAccelerator.ParallelIR.md#type__removenodepsstate.1)  State for the remove_no_deps and insert_no_deps_beginning phases.

[ParallelAccelerator.ParallelIR.ReplacedRegion](ParallelAccelerator.ParallelIR.md#type__replacedregion.1)  Store information about a section of a body that will be translated into a task.

[ParallelAccelerator.ParallelIR.RhsDead](ParallelAccelerator.ParallelIR.md#type__rhsdead.1)  Marks an assignment statement where the left-hand side can take over the storage from the right-hand side.

[ParallelAccelerator.ParallelIR.StatementWithDeps](ParallelAccelerator.ParallelIR.md#type__statementwithdeps.1)  Type for dependence graph creation and topological sorting.

[ParallelAccelerator.ParallelIR.TaskInfo](ParallelAccelerator.ParallelIR.md#type__taskinfo.1)  Structure for storing information about task formation.

[ParallelAccelerator.ParallelIR.cur_body_data](ParallelAccelerator.ParallelIR.md#type__cur_body_data.1)  Holds the data for substitute_cur_body AST walk.

[ParallelAccelerator.ParallelIR.cuw_state](ParallelAccelerator.ParallelIR.md#type__cuw_state.1)  Just to hold the "found" Bool that says whether a unsafe variant was replaced with a regular version.

[ParallelAccelerator.ParallelIR.expr_state](ParallelAccelerator.ParallelIR.md#type__expr_state.1)  State passed around while converting an AST from domain to parallel IR.

[ParallelAccelerator.ParallelIR.pir_arg_metadata](ParallelAccelerator.ParallelIR.md#type__pir_arg_metadata.1)  A Julia representation of the argument metadata that will be passed to the runtime.

[ParallelAccelerator.ParallelIR.pir_array_access_desc](ParallelAccelerator.ParallelIR.md#type__pir_array_access_desc.1)  Describes an array.

[ParallelAccelerator.ParallelIR.pir_grain_size](ParallelAccelerator.ParallelIR.md#type__pir_grain_size.1)  A Julia representation of the grain size that will be passed to the runtime.

[ParallelAccelerator.ParallelIR.pir_range](ParallelAccelerator.ParallelIR.md#type__pir_range.1)  Translated to pert_range_Nd_t in the task runtime.

[ParallelAccelerator.ParallelIR.pir_range_actual](ParallelAccelerator.ParallelIR.md#type__pir_range_actual.1)  Similar to pir_range but used in circumstances where the expressions must have already been evaluated.

[ParallelAccelerator.ParallelIR.sub_arrayset_data](ParallelAccelerator.ParallelIR.md#type__sub_arrayset_data.1)  Holds data for modifying arrayset calls.

## MODULE: ParallelAccelerator.API.Capture

---

## Internal

[process_node(node::Expr,  state,  top_level_number,  is_top_level,  read)](ParallelAccelerator.API.Capture.md#method__process_node.1)  At macro level, we translate function calls and operators that matches operator names

