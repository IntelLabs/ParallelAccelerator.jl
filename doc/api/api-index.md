# API-INDEX


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

## MODULE: CompilerTools.ReadWriteSet

---

## Exported

[from_exprs(ast::Array{T, N})](CompilerTools.ReadWriteSet.md#method__from_exprs.1)  Walk through an array of expressions.

[from_exprs(ast::Array{T, N},  callback::Union{Function, Void},  cbdata::ANY)](CompilerTools.ReadWriteSet.md#method__from_exprs.2)  Walk through an array of expressions.

[from_exprs(ast::Array{T, N},  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY)](CompilerTools.ReadWriteSet.md#method__from_exprs.3)  Walk through an array of expressions.

[isRead(sym::Union{GenSym, Symbol},  rws::CompilerTools.ReadWriteSet.ReadWriteSetType)](CompilerTools.ReadWriteSet.md#method__isread.1)  Return true if some symbol in "sym" is read either as a scalar or array within the computed ReadWriteSetType.

[isWritten(sym::Union{GenSym, Symbol},  rws::CompilerTools.ReadWriteSet.ReadWriteSetType)](CompilerTools.ReadWriteSet.md#method__iswritten.1)  Return true if some symbol in "sym" is written either as a scalar or array within the computed ReadWriteSetType.

[CompilerTools.ReadWriteSet.AccessSet](CompilerTools.ReadWriteSet.md#type__accessset.1)  Holds which scalars and which array are accessed and for array which index expressions are used.

[CompilerTools.ReadWriteSet.ReadWriteSetType](CompilerTools.ReadWriteSet.md#type__readwritesettype.1)  Stores which scalars and arrays are read or written in some code region.

---

## Internal

[addIndexExpr!(this_dict,  array_name,  index_expr)](CompilerTools.ReadWriteSet.md#method__addindexexpr.1)  Takes a dictionary of symbol to an array of index expression.

[from_assignment(ast::Array{Any, 1},  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY)](CompilerTools.ReadWriteSet.md#method__from_assignment.1)  Process an assignment AST node.

[from_call(ast::Array{Any, 1},  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY)](CompilerTools.ReadWriteSet.md#method__from_call.1)  Process :call Expr nodes to find arrayref and arrayset calls and adding the corresponding index expressions to the read and write sets respectively.

[from_coloncolon(ast::Array{T, N},  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY)](CompilerTools.ReadWriteSet.md#method__from_coloncolon.1)  Process a :(::) AST node.

[from_expr(ast::ANY)](CompilerTools.ReadWriteSet.md#method__from_expr.1)  Walk through one AST node.

[from_expr(ast::ANY,  callback::Union{Function, Void},  cbdata::ANY)](CompilerTools.ReadWriteSet.md#method__from_expr.2)  Walk through one AST node.

[from_expr(ast::LambdaStaticData,  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY)](CompilerTools.ReadWriteSet.md#method__from_expr.3)  The main routine that switches on all the various AST node types.

[from_lambda(ast::Expr,  depth,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY)](CompilerTools.ReadWriteSet.md#method__from_lambda.1)  Walk through a lambda expression.

[from_tuple(ast::Array{T, N},  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY)](CompilerTools.ReadWriteSet.md#method__from_tuple.1)  Walk through a tuple.

[toSymGen(x::Union{GenSym, Symbol})](CompilerTools.ReadWriteSet.md#method__tosymgen.1)  In various places we need a SymGen type which is the union of Symbol and GenSym.

[tryCallback(ast::ANY,  callback::Union{Function, Void},  cbdata::ANY,  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType)](CompilerTools.ReadWriteSet.md#method__trycallback.1)  If an AST node is not recognized then we try the passing the node to the callback to see if 

[uncompressed_ast(l::LambdaStaticData)](CompilerTools.ReadWriteSet.md#method__uncompressed_ast.1)  Convert a compressed LambdaStaticData format into the uncompressed AST format.

## MODULE: CompilerTools.AstWalker

---

## Exported

[AstWalk(ast::ANY,  callback,  cbdata::ANY)](CompilerTools.AstWalker.md#method__astwalk.1)  Entry point into the code to perform an AST walk.

---

## Internal

[from_assignment(ast::Array{Any, 1},  depth,  callback,  cbdata::ANY,  top_level_number,  read)](CompilerTools.AstWalker.md#method__from_assignment.1)  AstWalk through an assignment expression.

[from_body(ast::Array{Any, 1},  depth,  callback,  cbdata::ANY,  top_level_number,  read)](CompilerTools.AstWalker.md#method__from_body.1)  AstWalk through a function body.

[from_call(ast::Array{Any, 1},  depth,  callback,  cbdata::ANY,  top_level_number,  read)](CompilerTools.AstWalker.md#method__from_call.1)  AstWalk through a call expression.

[from_expr(ast::ANY,  depth,  callback,  cbdata::ANY,  top_level_number,  is_top_level,  read)](CompilerTools.AstWalker.md#method__from_expr.1)  The main routine that switches on all the various AST node types.

[from_exprs(ast::Array{Any, 1},  depth,  callback,  cbdata::ANY,  top_level_number,  read)](CompilerTools.AstWalker.md#method__from_exprs.1)  AstWalk through an array of expressions.

[from_lambda(ast::Array{Any, 1},  depth,  callback,  cbdata::ANY,  top_level_number,  read)](CompilerTools.AstWalker.md#method__from_lambda.1)  AstWalk through a lambda expression.

[uncompressed_ast(l::LambdaStaticData)](CompilerTools.AstWalker.md#method__uncompressed_ast.1)  Convert a compressed LambdaStaticData format into the uncompressed AST format.

## MODULE: CompilerTools.CFGs

---

## Exported

[find_bb_for_statement(top_number::Int64,  bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__find_bb_for_statement.1)  Find the basic block that contains a given statement number.

[from_exprs(ast::Array{Any, 1},  depth,  state,  callback,  cbdata)](CompilerTools.CFGs.md#method__from_exprs.1)  Process an array of expressions.

[show(io::IO,  bb::CompilerTools.CFGs.BasicBlock)](CompilerTools.CFGs.md#method__show.1)  Overload of Base.show to pretty-print a CFGS.BasicBlock object.

[show(io::IO,  bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__show.2)  Overload of Base.show to pretty-print a CFG object.

[show(io::IO,  tls::CompilerTools.CFGs.TopLevelStatement)](CompilerTools.CFGs.md#method__show.3)  Overload of Base.show to pretty-print a TopLevelStatement.

---

## Internal

[TypedExpr(typ,  rest...)](CompilerTools.CFGs.md#method__typedexpr.1)  Creates a typed Expr AST node.

[addStatement(top_level,  state,  ast::ANY)](CompilerTools.CFGs.md#method__addstatement.1)  Adds a top-level statement just encountered during a partial walk of the AST.

[addStatementToEndOfBlock(bl::CompilerTools.CFGs.CFG,  block,  stmt)](CompilerTools.CFGs.md#method__addstatementtoendofblock.1)  Given a CFG "bl" and a basic "block", add statement "stmt" to the end of that block.

[changeEndingLabel(bb,  after::CompilerTools.CFGs.BasicBlock,  new_bb::CompilerTools.CFGs.BasicBlock)](CompilerTools.CFGs.md#method__changeendinglabel.1)  BasicBlock bb currently is known to contain a jump to the BasicBlock after.

[compute_dfn(basic_blocks)](CompilerTools.CFGs.md#method__compute_dfn.1)  Computes the depth first numbering of the basic block graph.

[compute_dfn_internal(basic_blocks,  cur_bb,  cur_dfn,  visited,  bbs_df_order)](CompilerTools.CFGs.md#method__compute_dfn_internal.1)  The recursive heart of depth first numbering.

[compute_dominators(bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__compute_dominators.1)  Compute the dominators of the CFG.

[compute_inverse_dominators(bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__compute_inverse_dominators.1)  Compute the inverse dominators of the CFG.

[connect(from,  to,  fallthrough)](CompilerTools.CFGs.md#method__connect.1)  Connect the "from" input argument basic block to the "to" input argument basic block.

[connect_finish(state)](CompilerTools.CFGs.md#method__connect_finish.1)  Connect the current basic block as a fallthrough to the final invisible basic block (-2).

[createFunctionBody(bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__createfunctionbody.1)  Create the array of statements that go in a :body Expr given a CFG "bl".

[dump_bb(bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__dump_bb.1)  Prints a CFG "bl" with varying degrees of verbosity from debug level 2 up to 4.

[findReachable(reachable,  cur::Int64,  bbs::Dict{Int64, CompilerTools.CFGs.BasicBlock})](CompilerTools.CFGs.md#method__findreachable.1)  Process a basic block and add its successors to the set of reachable blocks

[find_top_number(top_number::Int64,  bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__find_top_number.1)  Search for a statement with the given number in the CFG "bl".

[from_ast(ast)](CompilerTools.CFGs.md#method__from_ast.1)  The main entry point to construct a control-flow graph.

[from_expr(ast,  callback,  cbdata)](CompilerTools.CFGs.md#method__from_expr.1)  Another entry point to construct a control-flow graph but one that allows you to pass a callback and some opaque object

[from_expr(ast::LambdaStaticData,  depth,  state,  top_level,  callback,  cbdata)](CompilerTools.CFGs.md#method__from_expr.2)  The main routine that switches on all the various AST node types.

[from_goto(label,  state,  callback,  cbdata)](CompilerTools.CFGs.md#method__from_goto.1)  Process a GotoNode for CFG construction.

[from_if(args,  depth,  state,  callback,  cbdata)](CompilerTools.CFGs.md#method__from_if.1)  Process a :gotoifnot Expr not for CFG construction.

[from_label(label,  state,  callback,  cbdata)](CompilerTools.CFGs.md#method__from_label.1)  Process LabelNode for CFG construction.

[from_lambda(ast::Array{Any, 1},  depth,  state,  callback,  cbdata)](CompilerTools.CFGs.md#method__from_lambda.1)  To help construct the CFG given a lambda, we recursively process the body of the lambda.

[from_return(args,  depth,  state,  callback,  cbdata)](CompilerTools.CFGs.md#method__from_return.1)  Process a :return Expr for CFG construction.

[getBbBodyOrder(bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__getbbbodyorder.1)  Determine a valid and reasonable order of basic blocks in which to reconstruct a :body Expr.

[getDistinctStatementNum(bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__getdistinctstatementnum.1)  Get a possible new statement number by finding the maximum statement value in any BasicBlock in the given CFG and adding 1.

[getMaxBB(bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__getmaxbb.1)  Returns the maximum basic block label for the given CFG.

[getMaxStatementNum(bb::CompilerTools.CFGs.BasicBlock)](CompilerTools.CFGs.md#method__getmaxstatementnum.1)  Get the maximum statement index for a given BasicBlock.

[getMinBB(bl::CompilerTools.CFGs.CFG)](CompilerTools.CFGs.md#method__getminbb.1)  Returns the minimum basic block label for the given CFG.

[insertBefore(bl::CompilerTools.CFGs.CFG,  after::Int64)](CompilerTools.CFGs.md#method__insertbefore.1)  Given a CFG in input parameter "bl" and a basic block label "after" in that CFG,

[insertBefore(bl::CompilerTools.CFGs.CFG,  after::Int64,  excludeBackEdge::Bool)](CompilerTools.CFGs.md#method__insertbefore.2)  Given a CFG in input parameter "bl" and a basic block label "after" in that CFG,

[insertBefore(bl::CompilerTools.CFGs.CFG,  after::Int64,  excludeBackEdge::Bool,  back_edge)](CompilerTools.CFGs.md#method__insertbefore.3)  Given a CFG in input parameter "bl" and a basic block label "after" in that CFG,

[insertBetween(bl::CompilerTools.CFGs.CFG,  before::Int64,  after::Int64)](CompilerTools.CFGs.md#method__insertbetween.1)  Insert a new basic block into the CFG "bl" between the basic blocks whose labels are "before" and "after".

[insertStatementAfter(bl::CompilerTools.CFGs.CFG,  block,  stmt_idx,  new_stmt)](CompilerTools.CFGs.md#method__insertstatementafter.1)  For a given CFG "bl" and a "block" in that CFG, add a new statement "new_stmt" to the basic block

[insertStatementBefore(bl::CompilerTools.CFGs.CFG,  block,  stmt_idx,  new_stmt)](CompilerTools.CFGs.md#method__insertstatementbefore.1)  For a given CFG "bl" and a "block" in that CFG, add a new statement "new_stmt" to the basic block

[insertat!(a,  value,  idx)](CompilerTools.CFGs.md#method__insertat.1)  Insert into an array "a" with a given "value" at the specified index "idx".

[not_handled(a,  b)](CompilerTools.CFGs.md#method__not_handled.1)  A default callback that handles no extra AST node types.

[removeUselessBlocks(bbs::Dict{Int64, CompilerTools.CFGs.BasicBlock})](CompilerTools.CFGs.md#method__removeuselessblocks.1)  This function simplifies the dict of basic blocks "bbs".

[replaceSucc(cur_bb::CompilerTools.CFGs.BasicBlock,  orig_succ::CompilerTools.CFGs.BasicBlock,  new_succ::CompilerTools.CFGs.BasicBlock)](CompilerTools.CFGs.md#method__replacesucc.1)  For a given basic block "cur_bb", replace one of its successors "orig_succ" with a different successor "new_succ".

[uncompressed_ast(l::LambdaStaticData)](CompilerTools.CFGs.md#method__uncompressed_ast.1)  Convert a compressed LambdaStaticData format into the uncompressed AST format.

[update_label(x::Expr,  state::CompilerTools.CFGs.UpdateLabelState,  top_level_number,  is_top_level,  read)](CompilerTools.CFGs.md#method__update_label.1)  An AstWalk callback that pattern matches GotoNode's and :gotoifnot Expr nodes and determines if the

[wrapInConditional(bl::CompilerTools.CFGs.CFG,  cond_gotoifnot::Expr,  first::Int64,  merge::Int64)](CompilerTools.CFGs.md#method__wrapinconditional.1)  Modifies the CFG to create a conditional (i.e., if statement) that wraps a certain region of the CFG whose entry block is

[wrapInConditional(bl::CompilerTools.CFGs.CFG,  cond_gotoifnot::Expr,  first::Int64,  merge::Int64,  back_edge::Union{CompilerTools.CFGs.BasicBlock, Void})](CompilerTools.CFGs.md#method__wrapinconditional.2)  Modifies the CFG to create a conditional (i.e., if statement) that wraps a certain region of the CFG whose entry block is

[CompilerTools.CFGs.BasicBlock](CompilerTools.CFGs.md#type__basicblock.1)  Data structure to hold information about one basic block in the control-flow graph.

[CompilerTools.CFGs.CFG](CompilerTools.CFGs.md#type__cfg.1)  The main data structure to hold information about the control flow graph.

[CompilerTools.CFGs.TopLevelStatement](CompilerTools.CFGs.md#type__toplevelstatement.1)  Data structure to hold the index (relative to the beginning of the body of the function) of a top-level statement

[CompilerTools.CFGs.UpdateLabelState](CompilerTools.CFGs.md#type__updatelabelstate.1)  The opaque callback data type for the update_label callback.

[CompilerTools.CFGs.expr_state](CompilerTools.CFGs.md#type__expr_state.1)  Collects information about the CFG as it is being constructed.

## MODULE: CompilerTools.LambdaHandling

---

## Exported

[addEscapingVariable(s::Symbol,  typ,  desc::Int64,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__addescapingvariable.1)  Adds a new escaping variable with the given Symbol "s", type "typ", descriptor "desc" in LambdaInfo "li".

[addEscapingVariable(vd::CompilerTools.LambdaHandling.VarDef,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__addescapingvariable.2)  Adds a new escaping variable from a VarDef in parameter "vd" into LambdaInfo "li".

[addGenSym(typ,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__addgensym.1)  Add a new GenSym to the LambdaInfo in "li" with the given type in "typ".

[addLocalVariable(s::Symbol,  typ,  desc::Int64,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__addlocalvariable.1)  Adds a new local variable with the given Symbol "s", type "typ", descriptor "desc" in LambdaInfo "li".

[addLocalVariable(vd::CompilerTools.LambdaHandling.VarDef,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__addlocalvariable.2)  Adds a local variable from a VarDef to the given LambdaInfo.

[getBody(lambda::Expr)](CompilerTools.LambdaHandling.md#method__getbody.1)  Returns the body expression part of a lambda expression.

[getDesc(x::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__getdesc.1)  Returns the descriptor for a local variable or input parameter "x" from LambdaInfo in "li".

[getRefParams(lambdaInfo::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__getrefparams.1)  Returns an array of Symbols corresponding to those parameters to the method that are going to be passed by reference.

[getReturnType(li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__getreturntype.1)  Returns the type of the lambda as stored in LambdaInfo "li" and as extracted during lambdaExprToLambdaInfo.

[getType(x::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__gettype.1)  Returns the type of a Symbol or GenSym in "x" from LambdaInfo in "li".

[getVarDef(s::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__getvardef.1)  Returns the VarDef for a Symbol in LambdaInfo in "li"

[isEscapingVariable(s::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__isescapingvariable.1)  Returns true if the Symbol in "s" is an escaping variable in LambdaInfo in "li".

[isInputParameter(s::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__isinputparameter.1)  Returns true if the Symbol in "s" is an input parameter in LambdaInfo in "li".

[isLocalGenSym(s::GenSym,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__islocalgensym.1)  Returns true if the GenSym in "s" is a GenSym in LambdaInfo in "li".

[isLocalVariable(s::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__islocalvariable.1)  Returns true if the Symbol in "s" is a local variable in LambdaInfo in "li".

[lambdaExprToLambdaInfo(lambda::Expr)](CompilerTools.LambdaHandling.md#method__lambdaexprtolambdainfo.1)  Convert a lambda expression into our internal storage format, LambdaInfo.

[lambdaInfoToLambdaExpr(lambdaInfo::CompilerTools.LambdaHandling.LambdaInfo,  body)](CompilerTools.LambdaHandling.md#method__lambdainfotolambdaexpr.1)  Convert our internal storage format, LambdaInfo, back into a lambda expression.

[lambdaTypeinf(lambda::LambdaStaticData,  typs::Tuple)](CompilerTools.LambdaHandling.md#method__lambdatypeinf.1)  Force type inference on a LambdaStaticData object.

[replaceExprWithDict!(expr::ANY,  dict::Dict{Union{GenSym, Symbol}, Any})](CompilerTools.LambdaHandling.md#method__replaceexprwithdict.1)  Replace the symbols in an expression "expr" with those defined in the

[replaceExprWithDict!(expr::ANY,  dict::Dict{Union{GenSym, Symbol}, Any},  AstWalkFunc)](CompilerTools.LambdaHandling.md#method__replaceexprwithdict.2)  Replace the symbols in an expression "expr" with those defined in the

[replaceExprWithDict(expr,  dict::Dict{Union{GenSym, Symbol}, Any})](CompilerTools.LambdaHandling.md#method__replaceexprwithdict.3)  Replace the symbols in an expression "expr" with those defined in the

[updateAssignedDesc(lambdaInfo::CompilerTools.LambdaHandling.LambdaInfo,  symbol_assigns::Dict{Symbol, Int64})](CompilerTools.LambdaHandling.md#method__updateassigneddesc.1)  Update the descriptor part of the VarDef dealing with whether the variable is assigned or not in the function.

[CompilerTools.LambdaHandling.LambdaInfo](CompilerTools.LambdaHandling.md#type__lambdainfo.1)  An internal format for storing a lambda expression's args[1] and args[2].

[CompilerTools.LambdaHandling.VarDef](CompilerTools.LambdaHandling.md#type__vardef.1)  Represents the triple stored in a lambda's args[2][1].

[SymGen](CompilerTools.LambdaHandling.md#typealias__symgen.1)  Type aliases for different unions of Symbol, SymbolNode, and GenSym.

---

## Internal

[addDescFlag(s::Symbol,  desc_flag::Int64,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__adddescflag.1)  Add one or more bitfields in "desc_flag" to the descriptor for a variable.

[addInputParameter(vd::CompilerTools.LambdaHandling.VarDef,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__addinputparameter.1)  Add Symbol "s" as input parameter to LambdaInfo "li".

[addInputParameters(collection,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__addinputparameters.1)  Add all variable in "collection" as input parameters to LambdaInfo "li".

[addLocalVar(name::AbstractString,  typ,  desc::Int64,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__addlocalvar.1)  Add a local variable to the function corresponding to LambdaInfo in "li" with name (as String), type and descriptor.

[addLocalVar(name::Symbol,  typ,  desc::Int64,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__addlocalvar.2)  Add a local variable to the function corresponding to LambdaInfo in "li" with name (as Symbol), type and descriptor.

[addLocalVariables(collection,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__addlocalvariables.1)  Add multiple local variables from some collection type.

[count_symbols(x::Symbol,  state::CompilerTools.LambdaHandling.CountSymbolState,  top_level_number,  is_top_level,  read)](CompilerTools.LambdaHandling.md#method__count_symbols.1)  Adds symbols and gensyms to their corresponding sets in CountSymbolState when they are seen in the AST.

[createMeta(lambdaInfo::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__createmeta.1)  Create the args[2] part of a lambda expression given an object of our internal storage format LambdaInfo.

[createVarDict(x::Array{Any, 1})](CompilerTools.LambdaHandling.md#method__createvardict.1)  Convert the lambda expression's args[2][1] from Array{Array{Any,1},1} to a Dict{Symbol,VarDef}.

[dictToArray(x::Dict{Symbol, CompilerTools.LambdaHandling.VarDef})](CompilerTools.LambdaHandling.md#method__dicttoarray.1)  Convert the Dict{Symbol,VarDef} internal storage format from a dictionary back into an array of Any triples.

[eliminateUnusedLocals!(li::CompilerTools.LambdaHandling.LambdaInfo,  body::Expr)](CompilerTools.LambdaHandling.md#method__eliminateunusedlocals.1)  Eliminates unused symbols from the LambdaInfo var_defs.

[eliminateUnusedLocals!(li::CompilerTools.LambdaHandling.LambdaInfo,  body::Expr,  AstWalkFunc)](CompilerTools.LambdaHandling.md#method__eliminateunusedlocals.2)  Eliminates unused symbols from the LambdaInfo var_defs.

[getLocalVariables(li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__getlocalvariables.1)  Returns an array of Symbols for local variables.

[mergeLambdaInfo(outer::CompilerTools.LambdaHandling.LambdaInfo,  inner::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__mergelambdainfo.1)  Merge "inner" lambdaInfo into "outer", and "outer" is changed as result.  Note

[removeLocalVar(name::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__removelocalvar.1)  Remove a local variable from lambda "li" given the variable's "name".

[show(io::IO,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LambdaHandling.md#method__show.1)  Pretty print a LambdaInfo.

[CompilerTools.LambdaHandling.CountSymbolState](CompilerTools.LambdaHandling.md#type__countsymbolstate.1)  Holds symbols and gensyms that are seen in a given AST when using the specified callback to handle non-standard Julia AST types.

## MODULE: CompilerTools.LivenessAnalysis

---

## Exported

[find_bb_for_statement(top_number::Int64,  bl::CompilerTools.LivenessAnalysis.BlockLiveness)](CompilerTools.LivenessAnalysis.md#method__find_bb_for_statement.1)  Search for a basic block containing a statement with the given top-level number in the liveness information.

[show(io::IO,  bb::CompilerTools.LivenessAnalysis.BasicBlock)](CompilerTools.LivenessAnalysis.md#method__show.1)  Overload of Base.show to pretty-print a LivenessAnalysis.BasicBlock.

[show(io::IO,  bl::CompilerTools.LivenessAnalysis.BlockLiveness)](CompilerTools.LivenessAnalysis.md#method__show.2)  Overload of Base.show to pretty-print BlockLiveness type.

[show(io::IO,  tls::CompilerTools.LivenessAnalysis.TopLevelStatement)](CompilerTools.LivenessAnalysis.md#method__show.3)  Overload of Base.show to pretty-print a LivenessAnalysis.TopLevelStatement.

[CompilerTools.LivenessAnalysis.BlockLiveness](CompilerTools.LivenessAnalysis.md#type__blockliveness.1)  The main return type from LivenessAnalysis.

---

## Internal

[TypedExpr(typ,  rest...)](CompilerTools.LivenessAnalysis.md#method__typedexpr.1)  Convenience function to create an Expr and make sure the type is filled in as well.

[addUnmodifiedParams(func,  signature::Array{DataType, 1},  unmodifieds,  state::CompilerTools.LivenessAnalysis.expr_state)](CompilerTools.LivenessAnalysis.md#method__addunmodifiedparams.1)  Add an entry the dictionary of which arguments can be modified by which functions.

[add_access(bb,  sym,  read)](CompilerTools.LivenessAnalysis.md#method__add_access.1)  Called when AST traversal finds some Symbol "sym" in a basic block "bb".

[compute_live_ranges(state::CompilerTools.LivenessAnalysis.expr_state,  dfn)](CompilerTools.LivenessAnalysis.md#method__compute_live_ranges.1)  Compute the live_in and live_out information for each basic block and statement.

[countSymbolDefs(s,  lives)](CompilerTools.LivenessAnalysis.md#method__countsymboldefs.1)  Count the number of times that the symbol in "s" is defined in all the basic blocks.

[create_unmodified_args_dict()](CompilerTools.LivenessAnalysis.md#method__create_unmodified_args_dict.1)  Convert the function_descriptions table into a dictionary that can be passed to

[def(x,  bl::CompilerTools.LivenessAnalysis.BlockLiveness)](CompilerTools.LivenessAnalysis.md#method__def.1)  Get the def information for "x" where x can be a liveness or CFG basic block or a liveness or CFG statement.

[dump_bb(bl::CompilerTools.LivenessAnalysis.BlockLiveness)](CompilerTools.LivenessAnalysis.md#method__dump_bb.1)  Dump a bunch of debugging information about BlockLiveness.

[find_top_number(top_number::Int64,  bl::CompilerTools.LivenessAnalysis.BlockLiveness)](CompilerTools.LivenessAnalysis.md#method__find_top_number.1)  Search for a statement with the given top-level number in the liveness information.

[fromCFG(live_res,  cfg::CompilerTools.CFGs.CFG,  callback::Function,  cbdata::ANY)](CompilerTools.LivenessAnalysis.md#method__fromcfg.1)  Extract liveness information from the CFG.

[from_assignment(ast::Array{Any, 1},  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY)](CompilerTools.LivenessAnalysis.md#method__from_assignment.1)  Walk through an assignment expression.

[from_call(ast::Array{Any, 1},  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY)](CompilerTools.LivenessAnalysis.md#method__from_call.1)  Walk through a call expression.

[from_expr(ast::Expr)](CompilerTools.LivenessAnalysis.md#method__from_expr.1)  This function gives you the option of calling the ENTRY point from_expr with an ast and several optional named arguments.

[from_expr(ast::Expr,  callback)](CompilerTools.LivenessAnalysis.md#method__from_expr.2)  ENTRY point to liveness analysis.

[from_expr(ast::Expr,  callback,  cbdata::ANY)](CompilerTools.LivenessAnalysis.md#method__from_expr.3)  ENTRY point to liveness analysis.

[from_expr(ast::Expr,  callback,  cbdata::ANY,  no_mod)](CompilerTools.LivenessAnalysis.md#method__from_expr.4)  ENTRY point to liveness analysis.

[from_expr(ast::LambdaStaticData,  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY)](CompilerTools.LivenessAnalysis.md#method__from_expr.5)  Generic routine for how to walk most AST node types.

[from_exprs(ast::Array{Any, 1},  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY)](CompilerTools.LivenessAnalysis.md#method__from_exprs.1)  Walk through an array of expressions.

[from_if(args,  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY)](CompilerTools.LivenessAnalysis.md#method__from_if.1)  Process a gotoifnot which is just a recursive processing of its first arg which is the conditional.

[from_lambda(ast::Expr,  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY)](CompilerTools.LivenessAnalysis.md#method__from_lambda.1)  Walk through a lambda expression.

[from_return(args,  depth::Int64,  state::CompilerTools.LivenessAnalysis.expr_state,  callback::Function,  cbdata::ANY)](CompilerTools.LivenessAnalysis.md#method__from_return.1)  Process a return Expr node which is just a recursive processing of all of its args.

[getUnmodifiedArgs(func::ANY,  args,  arg_type_tuple::Array{DataType, 1},  state::CompilerTools.LivenessAnalysis.expr_state)](CompilerTools.LivenessAnalysis.md#method__getunmodifiedargs.1)  For a given function and signature, return which parameters can be modified by the function.

[get_function_from_string(mod::AbstractString,  func::AbstractString)](CompilerTools.LivenessAnalysis.md#method__get_function_from_string.1)  Takes a module and a function both as Strings. Looks up the specified module as

[get_info_internal(x::Union{CompilerTools.LivenessAnalysis.BasicBlock, CompilerTools.LivenessAnalysis.TopLevelStatement},  bl::CompilerTools.LivenessAnalysis.BlockLiveness,  field)](CompilerTools.LivenessAnalysis.md#method__get_info_internal.1)  The live_in, live_out, def, and use routines are all effectively the same but just extract a different field name.

[isDef(x::Union{GenSym, Symbol},  live_info)](CompilerTools.LivenessAnalysis.md#method__isdef.1)  Query if the symbol in argument "x" is defined in live_info which can be a BasicBlock or TopLevelStatement.

[isPassedByRef(x,  state::CompilerTools.LivenessAnalysis.expr_state)](CompilerTools.LivenessAnalysis.md#method__ispassedbyref.1)  Returns true if a parameter is passed by reference.

[live_in(x,  bl::CompilerTools.LivenessAnalysis.BlockLiveness)](CompilerTools.LivenessAnalysis.md#method__live_in.1)  Get the live_in information for "x" where x can be a liveness or CFG basic block or a liveness or CFG statement.

[live_out(x,  bl::CompilerTools.LivenessAnalysis.BlockLiveness)](CompilerTools.LivenessAnalysis.md#method__live_out.1)  Get the live_out information for "x" where x can be a liveness or CFG basic block or a liveness or CFG statement.

[not_handled(a,  b)](CompilerTools.LivenessAnalysis.md#method__not_handled.1)  The default callback that processes no non-standard Julia AST nodes.

[recompute_live_ranges(state,  dfn)](CompilerTools.LivenessAnalysis.md#method__recompute_live_ranges.1)  Clear the live_in and live_out data corresponding to all basic blocks and statements and then recompute liveness information.

[typeOfOpr(x::ANY,  li::CompilerTools.LambdaHandling.LambdaInfo)](CompilerTools.LivenessAnalysis.md#method__typeofopr.1)  Get the type of some AST node.

[uncompressed_ast(l::LambdaStaticData)](CompilerTools.LivenessAnalysis.md#method__uncompressed_ast.1)  Convert a compressed LambdaStaticData format into the uncompressed AST format.

[use(x,  bl::CompilerTools.LivenessAnalysis.BlockLiveness)](CompilerTools.LivenessAnalysis.md#method__use.1)  Get the use information for "x" where x can be a liveness or CFG basic block or a liveness or CFG statement.

[CompilerTools.LivenessAnalysis.AccessSummary](CompilerTools.LivenessAnalysis.md#type__accesssummary.1)  Sometimes if new AST nodes are introduced then we need to ask for their def and use set as a whole

[CompilerTools.LivenessAnalysis.BasicBlock](CompilerTools.LivenessAnalysis.md#type__basicblock.1)  Liveness information for a BasicBlock.

[CompilerTools.LivenessAnalysis.TopLevelStatement](CompilerTools.LivenessAnalysis.md#type__toplevelstatement.1)  Liveness information for a TopLevelStatement in the CFG.

[CompilerTools.LivenessAnalysis.expr_state](CompilerTools.LivenessAnalysis.md#type__expr_state.1)  Holds the state during the AST traversal.

## MODULE: ParallelAccelerator.Comprehension

---

## Exported

[@comprehend(ast)](ParallelAccelerator.Comprehension.md#macro___comprehend.1)  Translate all comprehension in an AST into equivalent code that uses cartesianarray call.

---

## Internal

[comprehension_to_cartesianarray(ast)](ParallelAccelerator.Comprehension.md#method__comprehension_to_cartesianarray.1)  Translate an ast whose head is :comprehension into equivalent code that uses cartesianarray call.

[process_node(node,  state,  top_level_number,  is_top_level,  read)](ParallelAccelerator.Comprehension.md#method__process_node.1)  This function is a AstWalker callback.

## MODULE: ParallelAccelerator.DistributedIR

---

## Internal

[checkParforsForDistribution(state::ParallelAccelerator.DistributedIR.DistIrState)](ParallelAccelerator.DistributedIR.md#method__checkparforsfordistribution.1)  All arrays of a parfor should distributable for it to be distributable.

[get_arr_dist_info(node::Expr,  state,  top_level_number,  is_top_level,  read)](ParallelAccelerator.DistributedIR.md#method__get_arr_dist_info.1)  mark sequential arrays

## MODULE: ParallelAccelerator.API.Capture

---

## Internal

[process_node(node::Expr,  state,  top_level_number,  is_top_level,  read)](ParallelAccelerator.API.Capture.md#method__process_node.1)  At macro level, we translate function calls and operators that matches operator names

## MODULE: ParallelAccelerator.DomainIR

---

## Internal

[lookupConstDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode})](ParallelAccelerator.DomainIR.md#method__lookupconstdef.1)  Look up a definition of a variable only when it is const or assigned once.

[lookupConstDefForArg(state::ParallelAccelerator.DomainIR.IRState,  s)](ParallelAccelerator.DomainIR.md#method__lookupconstdefforarg.1)  Look up a definition of a variable recursively until the RHS is no-longer just a variable.

[lookupDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode})](ParallelAccelerator.DomainIR.md#method__lookupdef.1)  Look up a definition of a variable.

[lookupDefInAllScopes(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode})](ParallelAccelerator.DomainIR.md#method__lookupdefinallscopes.1)  Look up a definition of a variable throughout nested states until a definition is found.

[updateDef(state::ParallelAccelerator.DomainIR.IRState,  s::Union{GenSym, Symbol, SymbolNode},  rhs)](ParallelAccelerator.DomainIR.md#method__updatedef.1)  Update the definition of a variable.

## MODULE: CompilerTools.DebugMsg

---

## Exported

[init()](CompilerTools.DebugMsg.md#method__init.1)  A module using DebugMsg must call DebugMsg.init(), which expands to several local definitions

---

## Internal

[PROSPECT_DEV_MODE](CompilerTools.DebugMsg.md#global__prospect_dev_mode.1)  When this module is first loaded, we check if PROSPECT_DEV_MODE is set in environment.

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

[addUnknownRange(x::Array{Any, 1},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__addunknownrange.1)  Given an array of RangeExprs describing loop nest ranges, allocate a new equivalence class for this range.

[add_merge_correlations(old_sym::Union{GenSym, Symbol},  new_sym::Union{GenSym, Symbol},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__add_merge_correlations.1)  If we somehow determine that two arrays must be the same length then 

[asArray(x)](ParallelAccelerator.ParallelIR.md#method__asarray.1)  Return one element array with element x.

[augment_sn(dim::Int64,  index_vars,  range_var::Array{Union{GenSym, SymbolNode}, 1},  range::Array{ParallelAccelerator.ParallelIR.RangeData, 1})](ParallelAccelerator.ParallelIR.md#method__augment_sn.1)  Make sure the index parameters to arrayref or arrayset are Int64 or SymbolNode.

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

[createTempForRangeOffset(num_used,  ranges::Array{ParallelAccelerator.ParallelIR.RangeData, 1},  unique_id::Int64,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__createtempforrangeoffset.1)  Create a variable to hold the offset of a range offset from the start of the array.

[createTempForRangedArray(array_sn::Union{GenSym, Symbol, SymbolNode},  range::Array{Union{GenSym, SymbolNode}, 1},  unique_id::Int64,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__createtempforrangedarray.1)  Create a temporary variable that is parfor private to hold the value of an element of an array.

[create_array_access_desc(array::SymbolNode)](ParallelAccelerator.ParallelIR.md#method__create_array_access_desc.1)  Create an array access descriptor for "array".

[create_equivalence_classes(node::Expr,  state::ParallelAccelerator.ParallelIR.expr_state,  top_level_number::Int64,  is_top_level::Bool,  read::Bool)](ParallelAccelerator.ParallelIR.md#method__create_equivalence_classes.1)  AstWalk callback to determine the array equivalence classes.

[dfsVisit(swd::ParallelAccelerator.ParallelIR.StatementWithDeps,  vtime::Int64,  topo_sort::Array{ParallelAccelerator.ParallelIR.StatementWithDeps, N})](ParallelAccelerator.ParallelIR.md#method__dfsvisit.1)  Construct a topological sort of the dependence graph.

[estimateInstrCount(ast::Expr,  state::ParallelAccelerator.ParallelIR.eic_state,  top_level_number,  is_top_level,  read)](ParallelAccelerator.ParallelIR.md#method__estimateinstrcount.1)  AstWalk callback for estimating the instruction count.

[extractArrayEquivalencies(node::Expr,  state)](ParallelAccelerator.ParallelIR.md#method__extractarrayequivalencies.1)  "node" is a domainIR node.  Take the arrays used in this node, create an array equivalence for them if they 

[findSelectedDimensions(inputInfo::Array{ParallelAccelerator.ParallelIR.InputInfo, 1},  state)](ParallelAccelerator.ParallelIR.md#method__findselecteddimensions.1)  Given all the InputInfo for a Domain IR operation being lowered to Parallel IR,

[flattenParfor(new_body,  the_parfor::ParallelAccelerator.ParallelIR.PIRParForAst)](ParallelAccelerator.ParallelIR.md#method__flattenparfor.1)  Takes a new array of body statements in the process of construction in "new_body" and takes a parfor to add to that

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

[getOrAddRangeCorrelation(ranges::Array{ParallelAccelerator.ParallelIR.RangeExprs, 1},  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__getoraddrangecorrelation.1)  Gets (or adds if absent) the range correlation for the given array of RangeExprs.

[getOrAddSymbolCorrelation(array::Union{GenSym, Symbol},  state::ParallelAccelerator.ParallelIR.expr_state,  dims::Array{Union{GenSym, Symbol}, 1})](ParallelAccelerator.ParallelIR.md#method__getoraddsymbolcorrelation.1)  A new array is being created with an explicit size specification in dims.

[getParforCorrelation(parfor,  state)](ParallelAccelerator.ParallelIR.md#method__getparforcorrelation.1)  Get the equivalence class of the first array who length is extracted in the pre-statements of the specified "parfor".

[getParforNode(node)](ParallelAccelerator.ParallelIR.md#method__getparfornode.1)  Get the parfor object from either a bare parfor or one part of an assignment.

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

[mk_arrayref1(num_dim_inputs,  array_name,  index_vars,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state,  range_var::Array{Union{GenSym, SymbolNode}, 1})](ParallelAccelerator.ParallelIR.md#method__mk_arrayref1.2)  Return an expression that corresponds to getting the index_var index from the array array_name.

[mk_arrayref1(num_dim_inputs,  array_name,  index_vars,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state,  range_var::Array{Union{GenSym, SymbolNode}, 1},  range::Array{ParallelAccelerator.ParallelIR.RangeData, 1})](ParallelAccelerator.ParallelIR.md#method__mk_arrayref1.3)  Return an expression that corresponds to getting the index_var index from the array array_name.

[mk_arrayset1(num_dim_inputs,  array_name,  index_vars,  value,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__mk_arrayset1.1)  Return a new AST node that corresponds to setting the index_var index from the array "array_name" with "value".

[mk_arrayset1(num_dim_inputs,  array_name,  index_vars,  value,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state,  range_var::Array{Union{GenSym, SymbolNode}, 1})](ParallelAccelerator.ParallelIR.md#method__mk_arrayset1.2)  Return a new AST node that corresponds to setting the index_var index from the array "array_name" with "value".

[mk_arrayset1(num_dim_inputs,  array_name,  index_vars,  value,  inbounds,  state::ParallelAccelerator.ParallelIR.expr_state,  range_var::Array{Union{GenSym, SymbolNode}, 1},  range::Array{ParallelAccelerator.ParallelIR.RangeData, 1})](ParallelAccelerator.ParallelIR.md#method__mk_arrayset1.3)  Return a new AST node that corresponds to setting the index_var index from the array "array_name" with "value".

[mk_assignment_expr(lhs::Union{GenSym, Symbol, SymbolNode},  rhs,  state::ParallelAccelerator.ParallelIR.expr_state)](ParallelAccelerator.ParallelIR.md#method__mk_assignment_expr.1)  Create an assignment expression AST node given a left and right-hand side.

[mk_colon_expr(start_expr,  skip_expr,  end_expr)](ParallelAccelerator.ParallelIR.md#method__mk_colon_expr.1)  Returns an expression to construct a :colon object that contains the start of a range, the end and the skip expression.

[mk_convert(new_type,  ex)](ParallelAccelerator.ParallelIR.md#method__mk_convert.1)  Returns an expression that convert "ex" into a another type "new_type".

[mk_gotoifnot_expr(cond,  goto_label)](ParallelAccelerator.ParallelIR.md#method__mk_gotoifnot_expr.1)  Returns a :gotoifnot Expr given a condition "cond" and a label "goto_label".

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

[rangeToRangeData(range::Expr,  pre_offsets::Array{Expr, 1},  arr,  range_num::Int64,  state)](ParallelAccelerator.ParallelIR.md#method__rangetorangedata.1)  Convert a :range Expr introduced by Domain IR into a Parallel IR data structure RangeData.

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

## MODULE: CompilerTools.OptFramework

---

## Exported

[addOptPass(func,  level)](CompilerTools.OptFramework.md#method__addoptpass.1)  Same as the other addOptPass but with a pass call back function and pass level as input.

[addOptPass(pass::CompilerTools.OptFramework.OptPass)](CompilerTools.OptFramework.md#method__addoptpass.2)  Add an optimization pass. If this is going to be called multiple times then you need some external way of corrdinating the code/modules that are calling this function so that optimization passes are added in some sane order.

[@acc(ast1, ast2...)](CompilerTools.OptFramework.md#macro___acc.1)  The @acc macro comes in two forms:

[@noacc(ast)](CompilerTools.OptFramework.md#macro___noacc.1)  The macro @noacc can be used at call site to specifically run the non-accelerated copy of an accelerated function. It has no effect and gives a warning when the given function is not found to have been accelerated. We do not support nested @acc or @noacc. 

---

## Internal

[TypedExpr(typ,  rest...)](CompilerTools.OptFramework.md#method__typedexpr.1)  Creates a typed Expr AST node.

[cleanupASTLabels(ast)](CompilerTools.OptFramework.md#method__cleanupastlabels.1)  Clean up the labels in AST by renaming them, and removing duplicates.

[convertCodeToLevel(ast::ANY,  sig::ANY,  old_level,  new_level,  func)](CompilerTools.OptFramework.md#method__convertcodetolevel.1)  convert AST from "old_level" to "new_level". The input "ast" can be either Expr or Function type. In the latter case, the result AST will be obtained from this function using an matching signature "sig". The last "func" is a skeleton function that is used internally to facility such conversion.

[convert_expr(per_site_opt_set,  ast)](CompilerTools.OptFramework.md#method__convert_expr.1)  When @acc is used at a function's callsite, we use AstWalk to search for callsites via the opt_calls_insert_trampoline callback and to then insert trampolines.  That updated expression containing trampoline calls is then returned as the generated code from the @acc macro.

[convert_function(per_site_opt_set,  opt_set,  macros,  ast)](CompilerTools.OptFramework.md#method__convert_function.1)  When @acc is used at a function definition, it creates a trampoline function, when called with a specific set of signature types, will try to optimize the original function, and call it with the real arguments.  The input "ast" should be an AST of the original function at macro level, which will be   replaced by the trampoline. 

[create_label_map(x,  state::CompilerTools.OptFramework.lmstate,  top_level_number,  is_top_level,  read)](CompilerTools.OptFramework.md#method__create_label_map.1)  An AstWalk callback that collects information about labels in an AST.

[dumpLevel(level)](CompilerTools.OptFramework.md#method__dumplevel.1)  pretty print pass level number as string.

[evalPerSiteOptSet(per_site_opt_set)](CompilerTools.OptFramework.md#method__evalpersiteoptset.1)  Statically evaluate per-site optimization passes setting, and return the result.

[findOriginalFunc(mod::Module,  name::Symbol)](CompilerTools.OptFramework.md#method__findoriginalfunc.1)  Find the original (before @acc macro) function for a wrapper function in the given module. 

[findTargetFunc(mod::Module,  name::Symbol)](CompilerTools.OptFramework.md#method__findtargetfunc.1)  Find the optimizing target function (after @acc macro) for a wrapper function in the given module. 

[getCodeAtLevel(func,  sig,  level)](CompilerTools.OptFramework.md#method__getcodeatlevel.1)  Retrieve the AST of the given function "func" and signature "sig" for at the given pass "level".

[identical{T}(t::Type{T},  x::T)](CompilerTools.OptFramework.md#method__identical.1)  A hack to get around Julia's type inference. This is essentially an identity conversion,

[makeWrapperFunc(new_fname::Symbol,  real_fname::Symbol,  call_sig_args::Array{Any, 1},  per_site_opt_set)](CompilerTools.OptFramework.md#method__makewrapperfunc.1)  Define a wrapper function with the name given by "new_func" that when called will try to optimize the "real_func" function, and run it with given parameters in "call_sig_args". The input "per_site_opt_set" can be either nothing, or a quoted Expr that refers to an array of OptPass.

[opt_calls_insert_trampoline(x,  per_site_opt_set,  top_level_number,  is_top_level,  read)](CompilerTools.OptFramework.md#method__opt_calls_insert_trampoline.1)  An AstWalk callback function.

[processFuncCall(func::ANY,  call_sig_arg_tuple::ANY,  per_site_opt_set::ANY)](CompilerTools.OptFramework.md#method__processfunccall.1)  Takes a function, a signature, and a set of optimizations and applies that set of optimizations to the function,

[removeDupLabels(stmts)](CompilerTools.OptFramework.md#method__removeduplabels.1)  Sometimes update_labels creates two label nodes that are the same.

[setOptPasses(passes::Array{CompilerTools.OptFramework.OptPass, 1})](CompilerTools.OptFramework.md#method__setoptpasses.1)  Set the default set of optimization passes to apply with the @acc macro. 

[tfuncPresent(func,  tt)](CompilerTools.OptFramework.md#method__tfuncpresent.1)  Makes sure that a newly created function is correctly present in the internal Julia method table.

[update_labels(x,  state::CompilerTools.OptFramework.lmstate,  top_level_number,  is_top_level,  read)](CompilerTools.OptFramework.md#method__update_labels.1)  An AstWalk callback that applies the label map created during create_label_map AstWalk.

[CompilerTools.OptFramework.OptPass](CompilerTools.OptFramework.md#type__optpass.1)  A data structure that holds information about one high-level optimization pass to run.

[CompilerTools.OptFramework.lmstate](CompilerTools.OptFramework.md#type__lmstate.1)  The callback state variable used by create_label_map and update_labels.

[gOptFrameworkDict](CompilerTools.OptFramework.md#global__goptframeworkdict.1)  A global memo-table that maps both: the triple (function, signature, optPasses) to the trampoline function, and the trampoline function to the real function.

## MODULE: CompilerTools.UDChains

---

## Internal

[getOrCreate(live::Dict{Symbol, Set{T}},  s::Symbol)](CompilerTools.UDChains.md#method__getorcreate.1)  Get the set of definition blocks reaching this block for a given symbol "s".

[getOrCreate(udchains::Dict{CompilerTools.LivenessAnalysis.BasicBlock, CompilerTools.UDChains.UDInfo},  bb::CompilerTools.LivenessAnalysis.BasicBlock)](CompilerTools.UDChains.md#method__getorcreate.2)  Get the UDInfo for a specified basic block "bb" or create one if it doesn't already exist.

[getUDChains(bl::CompilerTools.LivenessAnalysis.BlockLiveness)](CompilerTools.UDChains.md#method__getudchains.1)  Get the Use-Definition chains at a basic block level given LivenessAnalysis.BlockLiveness as input in "bl".

[printLabels(level,  dict)](CompilerTools.UDChains.md#method__printlabels.1)  Print a live in or live out dictionary in a nice way if the debug level is set high enough.

[printSet(level,  s)](CompilerTools.UDChains.md#method__printset.1)  Print the set part of a live in or live out dictiononary in a nice way if the debug level is set high enough.

[printUDInfo(level,  ud)](CompilerTools.UDChains.md#method__printudinfo.1)  Print UDChains in a nice way if the debug level is set high enough.

[CompilerTools.UDChains.UDInfo](CompilerTools.UDChains.md#type__udinfo.1)  Contains the UDchains for one basic block.

## MODULE: CompilerTools.Loops

---

## Exported

[CompilerTools.Loops.DomLoops](CompilerTools.Loops.md#type__domloops.1)  A type that holds information about which basic blocks dominate which other blocks.

[CompilerTools.Loops.Loop](CompilerTools.Loops.md#type__loop.1)  A type to hold information about a loop.

---

## Internal

[compute_dom_loops(bl::CompilerTools.CFGs.CFG)](CompilerTools.Loops.md#method__compute_dom_loops.1)  Find the loops in a CFGs.CFG in "bl".

[findLoopInvariants(l::CompilerTools.Loops.Loop,  udinfo::Dict{CompilerTools.LivenessAnalysis.BasicBlock, CompilerTools.UDChains.UDInfo},  bl::CompilerTools.LivenessAnalysis.BlockLiveness)](CompilerTools.Loops.md#method__findloopinvariants.1)  Finds those computations within a loop that are iteration invariant.

[findLoopMembers(head,  back_edge,  bbs)](CompilerTools.Loops.md#method__findloopmembers.1)  Find all the members of the loop as specified by the "head" basic block and the "back_edge" basic block.

[flm_internal(cur_bb,  members,  bbs)](CompilerTools.Loops.md#method__flm_internal.1)  Add to the "members" of the loop being accumulated given "cur_bb" which is known to be a member of the loop.

[isInLoop(dl::CompilerTools.Loops.DomLoops,  bb::Int64)](CompilerTools.Loops.md#method__isinloop.1)  Takes a DomLoops object containing loop information about the function.

## MODULE: ParallelAccelerator

---

## Internal

[__init__()](ParallelAccelerator.md#method____init__.1)  Called when the package is loaded to do initialization.

[embed()](ParallelAccelerator.md#method__embed.1)  This version of embed tries to use JULIA_HOME to find the root of the source distribution.

[embed(julia_root)](ParallelAccelerator.md#method__embed.2)  Call this function if you want to embed binary-code of ParallelAccelerator into your Julia build to speed-up @acc compilation time.

[getPackageRoot()](ParallelAccelerator.md#method__getpackageroot.1)  Generate a file path to the directory above the one containing this source file.

[getPseMode()](ParallelAccelerator.md#method__getpsemode.1)  Return internal mode number by looking up environment variable "PROSPECT_MODE".

[getTaskMode()](ParallelAccelerator.md#method__gettaskmode.1)  Return internal mode number by looking up environment variable "PROSPECT_TASK_MODE".

