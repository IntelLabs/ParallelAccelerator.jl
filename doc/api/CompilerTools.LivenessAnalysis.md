# CompilerTools.LivenessAnalysis

## Exported

---

<a id="method__find_bb_for_statement.1" class="lexicon_definition"></a>
#### find_bb_for_statement(top_number::Int64,  bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__find_bb_for_statement.1)
Search for a basic block containing a statement with the given top-level number in the liveness information.
Returns a basic block label of a block having that top-level number or "nothing" if no such statement could be found.


*source:*
[CompilerTools/src/liveness.jl:395](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__show.1" class="lexicon_definition"></a>
#### show(io::IO,  bb::CompilerTools.LivenessAnalysis.BasicBlock) [¶](#method__show.1)
Overload of Base.show to pretty-print a LivenessAnalysis.BasicBlock.


*source:*
[CompilerTools/src/liveness.jl:137](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__show.2" class="lexicon_definition"></a>
#### show(io::IO,  bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__show.2)
Overload of Base.show to pretty-print BlockLiveness type.


*source:*
[CompilerTools/src/liveness.jl:349](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__show.3" class="lexicon_definition"></a>
#### show(io::IO,  tls::CompilerTools.LivenessAnalysis.TopLevelStatement) [¶](#method__show.3)
Overload of Base.show to pretty-print a LivenessAnalysis.TopLevelStatement.


*source:*
[CompilerTools/src/liveness.jl:76](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="type__blockliveness.1" class="lexicon_definition"></a>
#### CompilerTools.LivenessAnalysis.BlockLiveness [¶](#type__blockliveness.1)
The main return type from LivenessAnalysis.
Contains a dictionary that maps CFG basic block to liveness basic blocks.
Also contains the corresponding CFG.


*source:*
[CompilerTools/src/liveness.jl:281](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

## Internal

---

<a id="method__typedexpr.1" class="lexicon_definition"></a>
#### TypedExpr(typ,  rest...) [¶](#method__typedexpr.1)
Convenience function to create an Expr and make sure the type is filled in as well.
The first arg is the type of the Expr and the rest of the args are the constructors args to Expr.


*source:*
[CompilerTools/src/liveness.jl:43](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__addunmodifiedparams.1" class="lexicon_definition"></a>
#### addUnmodifiedParams(func,  signature::Array{DataType, 1},  unmodifieds,  state::CompilerTools.LivenessAnalysis.expr_state) [¶](#method__addunmodifiedparams.1)
Add an entry the dictionary of which arguments can be modified by which functions.


*source:*
[CompilerTools/src/liveness.jl:591](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__add_access.1" class="lexicon_definition"></a>
#### add_access(bb,  sym,  read) [¶](#method__add_access.1)
Called when AST traversal finds some Symbol "sym" in a basic block "bb".
"read" is true if the symbol is being used and false if it is being defined.


*source:*
[CompilerTools/src/liveness.jl:191](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__compute_live_ranges.1" class="lexicon_definition"></a>
#### compute_live_ranges(state::CompilerTools.LivenessAnalysis.expr_state,  dfn) [¶](#method__compute_live_ranges.1)
Compute the live_in and live_out information for each basic block and statement.


*source:*
[CompilerTools/src/liveness.jl:432](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__countsymboldefs.1" class="lexicon_definition"></a>
#### countSymbolDefs(s,  lives) [¶](#method__countsymboldefs.1)
Count the number of times that the symbol in "s" is defined in all the basic blocks.


*source:*
[CompilerTools/src/liveness.jl:837](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__create_unmodified_args_dict.1" class="lexicon_definition"></a>
#### create_unmodified_args_dict() [¶](#method__create_unmodified_args_dict.1)
Convert the function_descriptions table into a dictionary that can be passed to
LivenessAnalysis to indicate which args are unmodified by which functions.


*source:*
[CompilerTools/src/function-descriptions.jl:257](file:///home/etotoni/.julia/v0.4/CompilerTools/src/function-descriptions.jl)

---

<a id="method__def.1" class="lexicon_definition"></a>
#### def(x,  bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__def.1)
Get the def information for "x" where x can be a liveness or CFG basic block or a liveness or CFG statement.


*source:*
[CompilerTools/src/liveness.jl:335](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__dump_bb.1" class="lexicon_definition"></a>
#### dump_bb(bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__dump_bb.1)
Dump a bunch of debugging information about BlockLiveness.


*source:*
[CompilerTools/src/liveness.jl:501](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__find_top_number.1" class="lexicon_definition"></a>
#### find_top_number(top_number::Int64,  bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__find_top_number.1)
Search for a statement with the given top-level number in the liveness information.
Returns a LivenessAnalysis.TopLevelStatement having that top-level number or "nothing" if no such statement could be found.


*source:*
[CompilerTools/src/liveness.jl:376](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__fromcfg.1" class="lexicon_definition"></a>
#### fromCFG(live_res,  cfg::CompilerTools.CFGs.CFG,  callback::Function,  cbdata::ANY) [¶](#method__fromcfg.1)
Extract liveness information from the CFG.


*source:*
[CompilerTools/src/liveness.jl:878](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_assignment.1" class="lexicon_definition"></a>
#### from_assignment(ast::Array{Any, 1},  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY) [¶](#method__from_assignment.1)
Walk through an assignment expression.


*source:*
[CompilerTools/src/liveness.jl:566](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_call.1" class="lexicon_definition"></a>
#### from_call(ast::Array{Any, 1},  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY) [¶](#method__from_call.1)
Walk through a call expression.


*source:*
[CompilerTools/src/liveness.jl:780](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_expr.1" class="lexicon_definition"></a>
#### from_expr(ast::Expr) [¶](#method__from_expr.1)
This function gives you the option of calling the ENTRY point from_expr with an ast and several optional named arguments.


*source:*
[CompilerTools/src/liveness.jl:871](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_expr.2" class="lexicon_definition"></a>
#### from_expr(ast::Expr,  callback) [¶](#method__from_expr.2)
ENTRY point to liveness analysis.
You must pass a :lambda Expr as "ast".
If you have non-standard AST nodes, you may pass a callback that will be given a chance to process the non-standard node first.


*source:*
[CompilerTools/src/liveness.jl:859](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_expr.3" class="lexicon_definition"></a>
#### from_expr(ast::Expr,  callback,  cbdata::ANY) [¶](#method__from_expr.3)
ENTRY point to liveness analysis.
You must pass a :lambda Expr as "ast".
If you have non-standard AST nodes, you may pass a callback that will be given a chance to process the non-standard node first.


*source:*
[CompilerTools/src/liveness.jl:859](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_expr.4" class="lexicon_definition"></a>
#### from_expr(ast::Expr,  callback,  cbdata::ANY,  no_mod) [¶](#method__from_expr.4)
ENTRY point to liveness analysis.
You must pass a :lambda Expr as "ast".
If you have non-standard AST nodes, you may pass a callback that will be given a chance to process the non-standard node first.


*source:*
[CompilerTools/src/liveness.jl:859](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_expr.5" class="lexicon_definition"></a>
#### from_expr(ast::LambdaStaticData,  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY) [¶](#method__from_expr.5)
Generic routine for how to walk most AST node types.


*source:*
[CompilerTools/src/liveness.jl:932](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_exprs.1" class="lexicon_definition"></a>
#### from_exprs(ast::Array{Any, 1},  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY) [¶](#method__from_exprs.1)
Walk through an array of expressions.
Just recursively call from_expr for each expression in the array.


*source:*
[CompilerTools/src/liveness.jl:552](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_if.1" class="lexicon_definition"></a>
#### from_if(args,  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY) [¶](#method__from_if.1)
Process a gotoifnot which is just a recursive processing of its first arg which is the conditional.


*source:*
[CompilerTools/src/liveness.jl:918](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_lambda.1" class="lexicon_definition"></a>
#### from_lambda(ast::Expr,  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY) [¶](#method__from_lambda.1)
Walk through a lambda expression.
We just need to extract the ref_params because liveness needs to keep those ref_params live at the end of the function.
We don't recurse into the body here because from_expr handles that with fromCFG.


*source:*
[CompilerTools/src/liveness.jl:542](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__from_return.1" class="lexicon_definition"></a>
#### from_return(args,  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY) [¶](#method__from_return.1)
Process a return Expr node which is just a recursive processing of all of its args.


*source:*
[CompilerTools/src/liveness.jl:909](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__getunmodifiedargs.1" class="lexicon_definition"></a>
#### getUnmodifiedArgs(func::ANY,  args,  arg_type_tuple::Array{DataType, 1},  state::CompilerTools.LivenessAnalysis.expr_state) [¶](#method__getunmodifiedargs.1)
For a given function and signature, return which parameters can be modified by the function.
If we have cached this information previously then return that, else cache the information for some
well-known functions or default to presuming that all arguments could be modified.


*source:*
[CompilerTools/src/liveness.jl:707](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__get_function_from_string.1" class="lexicon_definition"></a>
#### get_function_from_string(mod::AbstractString,  func::AbstractString) [¶](#method__get_function_from_string.1)
Takes a module and a function both as Strings. Looks up the specified module as
part of the "Main" module and then looks and returns the Function object
corresponding to the "func" String in that module.


*source:*
[CompilerTools/src/function-descriptions.jl:239](file:///home/etotoni/.julia/v0.4/CompilerTools/src/function-descriptions.jl)

---

<a id="method__get_info_internal.1" class="lexicon_definition"></a>
#### get_info_internal(x::Union{CompilerTools.LivenessAnalysis.BasicBlock, CompilerTools.LivenessAnalysis.TopLevelStatement},  bl::CompilerTools.LivenessAnalysis.BlockLiveness,  field) [¶](#method__get_info_internal.1)
The live_in, live_out, def, and use routines are all effectively the same but just extract a different field name.
Here we extract this common behavior where x can be a liveness or CFG basic block or a liveness or CFG statement.
bl is BlockLiveness type as returned by a previous LivenessAnalysis.
field is the name of the field requested.


*source:*
[CompilerTools/src/liveness.jl:297](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__isdef.1" class="lexicon_definition"></a>
#### isDef(x::Union{GenSym, Symbol},  live_info) [¶](#method__isdef.1)
Query if the symbol in argument "x" is defined in live_info which can be a BasicBlock or TopLevelStatement.


*source:*
[CompilerTools/src/liveness.jl:367](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__ispassedbyref.1" class="lexicon_definition"></a>
#### isPassedByRef(x,  state::CompilerTools.LivenessAnalysis.expr_state) [¶](#method__ispassedbyref.1)
Returns true if a parameter is passed by reference.
isbits types are not passed by ref but everything else is (is this always true..any exceptions?)


*source:*
[CompilerTools/src/liveness.jl:639](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__live_in.1" class="lexicon_definition"></a>
#### live_in(x,  bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__live_in.1)
Get the live_in information for "x" where x can be a liveness or CFG basic block or a liveness or CFG statement.


*source:*
[CompilerTools/src/liveness.jl:321](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__live_out.1" class="lexicon_definition"></a>
#### live_out(x,  bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__live_out.1)
Get the live_out information for "x" where x can be a liveness or CFG basic block or a liveness or CFG statement.


*source:*
[CompilerTools/src/liveness.jl:328](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__not_handled.1" class="lexicon_definition"></a>
#### not_handled(a,  b) [¶](#method__not_handled.1)
The default callback that processes no non-standard Julia AST nodes.


*source:*
[CompilerTools/src/liveness.jl:830](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__recompute_live_ranges.1" class="lexicon_definition"></a>
#### recompute_live_ranges(state,  dfn) [¶](#method__recompute_live_ranges.1)
Clear the live_in and live_out data corresponding to all basic blocks and statements and then recompute liveness information.


*source:*
[CompilerTools/src/liveness.jl:414](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__typeofopr.1" class="lexicon_definition"></a>
#### typeOfOpr(x::ANY,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__typeofopr.1)
Get the type of some AST node.


*source:*
[CompilerTools/src/liveness.jl:598](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__uncompressed_ast.1" class="lexicon_definition"></a>
#### uncompressed_ast(l::LambdaStaticData) [¶](#method__uncompressed_ast.1)
Convert a compressed LambdaStaticData format into the uncompressed AST format.


*source:*
[CompilerTools/src/liveness.jl:532](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="method__use.1" class="lexicon_definition"></a>
#### use(x,  bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__use.1)
Get the use information for "x" where x can be a liveness or CFG basic block or a liveness or CFG statement.


*source:*
[CompilerTools/src/liveness.jl:342](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="type__accesssummary.1" class="lexicon_definition"></a>
#### CompilerTools.LivenessAnalysis.AccessSummary [¶](#type__accesssummary.1)
Sometimes if new AST nodes are introduced then we need to ask for their def and use set as a whole
and then incorporate that into our liveness analysis directly.


*source:*
[CompilerTools/src/liveness.jl:111](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="type__basicblock.1" class="lexicon_definition"></a>
#### CompilerTools.LivenessAnalysis.BasicBlock [¶](#type__basicblock.1)
Liveness information for a BasicBlock.
Contains a pointer to the corresponding CFG BasicBlock.
Contains def, use, live_in, and live_out for this basic block.
Contains an array of liveness information for the top level statements in this block.


*source:*
[CompilerTools/src/liveness.jl:124](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="type__toplevelstatement.1" class="lexicon_definition"></a>
#### CompilerTools.LivenessAnalysis.TopLevelStatement [¶](#type__toplevelstatement.1)
Liveness information for a TopLevelStatement in the CFG.
Contains a pointer to the corresponding CFG TopLevelStatement.
Contains def, use, live_in, and live_out for the current statement.


*source:*
[CompilerTools/src/liveness.jl:64](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

---

<a id="type__expr_state.1" class="lexicon_definition"></a>
#### CompilerTools.LivenessAnalysis.expr_state [¶](#type__expr_state.1)
Holds the state during the AST traversal.
cfg = the control flow graph from the CFGs module.
map = our own map of CFG basic blocks to our own basic block information with liveness in it.
cur_bb = the current basic block in which we are processing AST nodes.
read = whether the sub-tree we are currently processing is being read or written.
ref_params = those arguments to the function that are passed by reference.


*source:*
[CompilerTools/src/liveness.jl:265](file:///home/etotoni/.julia/v0.4/CompilerTools/src/liveness.jl)

