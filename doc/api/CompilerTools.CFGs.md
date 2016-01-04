# CompilerTools.CFGs

## Exported

---

<a id="method__find_bb_for_statement.1" class="lexicon_definition"></a>
#### find_bb_for_statement(top_number::Int64,  bl::CompilerTools.CFGs.CFG) [¶](#method__find_bb_for_statement.1)
Find the basic block that contains a given statement number.
Returns the basic block label of the basic block that contains the given statement number or "nothing" if the statement number is not found.


*source:*
[CompilerTools/src/CFGs.jl:718](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__from_exprs.1" class="lexicon_definition"></a>
#### from_exprs(ast::Array{Any, 1},  depth,  state,  callback,  cbdata) [¶](#method__from_exprs.1)
Process an array of expressions.
We know that the first array of expressions we will find is for the lambda body.
top_level_number starts out 0 and if we find it to be 0 then we know that we're processing the array of expr for the body
and so we keep track of the index into body so that we can number the statements in the basic blocks by this top level number.
Recursively process each element of the array of expressions.


*source:*
[CompilerTools/src/CFGs.jl:841](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__show.1" class="lexicon_definition"></a>
#### show(io::IO,  bb::CompilerTools.CFGs.BasicBlock) [¶](#method__show.1)
Overload of Base.show to pretty-print a CFGS.BasicBlock object.


*source:*
[CompilerTools/src/CFGs.jl:124](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__show.2" class="lexicon_definition"></a>
#### show(io::IO,  bl::CompilerTools.CFGs.CFG) [¶](#method__show.2)
Overload of Base.show to pretty-print a CFG object.


*source:*
[CompilerTools/src/CFGs.jl:204](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__show.3" class="lexicon_definition"></a>
#### show(io::IO,  tls::CompilerTools.CFGs.TopLevelStatement) [¶](#method__show.3)
Overload of Base.show to pretty-print a TopLevelStatement.


*source:*
[CompilerTools/src/CFGs.jl:63](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

## Internal

---

<a id="method__typedexpr.1" class="lexicon_definition"></a>
#### TypedExpr(typ,  rest...) [¶](#method__typedexpr.1)
Creates a typed Expr AST node.
Convenence function that takes a type as first argument and the varargs thereafter.
The varargs are used to form an Expr AST node and the type parameter is used to fill in the "typ" field of the Expr.


*source:*
[CompilerTools/src/CFGs.jl:43](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__addstatement.1" class="lexicon_definition"></a>
#### addStatement(top_level,  state,  ast::ANY) [¶](#method__addstatement.1)
Adds a top-level statement just encountered during a partial walk of the AST.
First argument indicates if this statement is a top-level statement.
Second argument is a object collecting information about the CFG as we go along.
Third argument is some sub-tree of the AST.


*source:*
[CompilerTools/src/CFGs.jl:107](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__addstatementtoendofblock.1" class="lexicon_definition"></a>
#### addStatementToEndOfBlock(bl::CompilerTools.CFGs.CFG,  block,  stmt) [¶](#method__addstatementtoendofblock.1)
Given a CFG "bl" and a basic "block", add statement "stmt" to the end of that block.


*source:*
[CompilerTools/src/CFGs.jl:622](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__changeendinglabel.1" class="lexicon_definition"></a>
#### changeEndingLabel(bb,  after::CompilerTools.CFGs.BasicBlock,  new_bb::CompilerTools.CFGs.BasicBlock) [¶](#method__changeendinglabel.1)
BasicBlock bb currently is known to contain a jump to the BasicBlock after.
This function changes bb so that it no longer jumps to after but to "new_bb" instead.
The jump has to be in the last statement of the BasicBlock.
AstWalk on the last statement of the BasicBlock is used with the update_label callback function.


*source:*
[CompilerTools/src/CFGs.jl:310](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__compute_dfn.1" class="lexicon_definition"></a>
#### compute_dfn(basic_blocks) [¶](#method__compute_dfn.1)
Computes the depth first numbering of the basic block graph.


*source:*
[CompilerTools/src/CFGs.jl:757](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__compute_dfn_internal.1" class="lexicon_definition"></a>
#### compute_dfn_internal(basic_blocks,  cur_bb,  cur_dfn,  visited,  bbs_df_order) [¶](#method__compute_dfn_internal.1)
The recursive heart of depth first numbering.


*source:*
[CompilerTools/src/CFGs.jl:736](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__compute_dominators.1" class="lexicon_definition"></a>
#### compute_dominators(bl::CompilerTools.CFGs.CFG) [¶](#method__compute_dominators.1)
Compute the dominators of the CFG.


*source:*
[CompilerTools/src/CFGs.jl:1182](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__compute_inverse_dominators.1" class="lexicon_definition"></a>
#### compute_inverse_dominators(bl::CompilerTools.CFGs.CFG) [¶](#method__compute_inverse_dominators.1)
Compute the inverse dominators of the CFG.


*source:*
[CompilerTools/src/CFGs.jl:1246](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__connect.1" class="lexicon_definition"></a>
#### connect(from,  to,  fallthrough) [¶](#method__connect.1)
Connect the "from" input argument basic block to the "to" input argument basic block.
If the third argument "fallthrough" is true then the "to" block is also set as the "from" basic block's fallthrough successor.


*source:*
[CompilerTools/src/CFGs.jl:153](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__connect_finish.1" class="lexicon_definition"></a>
#### connect_finish(state) [¶](#method__connect_finish.1)
Connect the current basic block as a fallthrough to the final invisible basic block (-2).


*source:*
[CompilerTools/src/CFGs.jl:775](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__createfunctionbody.1" class="lexicon_definition"></a>
#### createFunctionBody(bl::CompilerTools.CFGs.CFG) [¶](#method__createfunctionbody.1)
Create the array of statements that go in a :body Expr given a CFG "bl".


*source:*
[CompilerTools/src/CFGs.jl:667](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__dump_bb.1" class="lexicon_definition"></a>
#### dump_bb(bl::CompilerTools.CFGs.CFG) [¶](#method__dump_bb.1)
Prints a CFG "bl" with varying degrees of verbosity from debug level 2 up to 4.
Additionally, at debug level 4 and graphviz bbs.dot file is generated that can be used to visualize the basic block structure of the function.


*source:*
[CompilerTools/src/CFGs.jl:784](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__findreachable.1" class="lexicon_definition"></a>
#### findReachable(reachable,  cur::Int64,  bbs::Dict{Int64, CompilerTools.CFGs.BasicBlock}) [¶](#method__findreachable.1)
Process a basic block and add its successors to the set of reachable blocks
if it isn't already there.  If it is freshly added then recurse to adds its successors.


*source:*
[CompilerTools/src/CFGs.jl:877](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__find_top_number.1" class="lexicon_definition"></a>
#### find_top_number(top_number::Int64,  bl::CompilerTools.CFGs.CFG) [¶](#method__find_top_number.1)
Search for a statement with the given number in the CFG "bl".
Returns the statement corresponding to the given number or "nothing" if the statement number is not found.


*source:*
[CompilerTools/src/CFGs.jl:700](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__from_ast.1" class="lexicon_definition"></a>
#### from_ast(ast) [¶](#method__from_ast.1)
The main entry point to construct a control-flow graph.
Typically you would pass in a :lambda Expr here.


*source:*
[CompilerTools/src/CFGs.jl:1004](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__from_expr.1" class="lexicon_definition"></a>
#### from_expr(ast,  callback,  cbdata) [¶](#method__from_expr.1)
Another entry point to construct a control-flow graph but one that allows you to pass a callback and some opaque object
so that non-standard node types can be processed.


*source:*
[CompilerTools/src/CFGs.jl:1013](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__from_expr.2" class="lexicon_definition"></a>
#### from_expr(ast::LambdaStaticData,  depth,  state,  top_level,  callback,  cbdata) [¶](#method__from_expr.2)
The main routine that switches on all the various AST node types.
The internal nodes of the AST are of type Expr with various different Expr.head field values such as :lambda, :body, :block, etc.
The leaf nodes of the AST all have different types.


*source:*
[CompilerTools/src/CFGs.jl:1123](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__from_goto.1" class="lexicon_definition"></a>
#### from_goto(label,  state,  callback,  cbdata) [¶](#method__from_goto.1)
Process a GotoNode for CFG construction.


*source:*
[CompilerTools/src/CFGs.jl:1052](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__from_if.1" class="lexicon_definition"></a>
#### from_if(args,  depth,  state,  callback,  cbdata) [¶](#method__from_if.1)
Process a :gotoifnot Expr not for CFG construction.


*source:*
[CompilerTools/src/CFGs.jl:1082](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__from_label.1" class="lexicon_definition"></a>
#### from_label(label,  state,  callback,  cbdata) [¶](#method__from_label.1)
Process LabelNode for CFG construction.


*source:*
[CompilerTools/src/CFGs.jl:1036](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__from_lambda.1" class="lexicon_definition"></a>
#### from_lambda(ast::Array{Any, 1},  depth,  state,  callback,  cbdata) [¶](#method__from_lambda.1)
To help construct the CFG given a lambda, we recursively process the body of the lambda.


*source:*
[CompilerTools/src/CFGs.jl:820](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__from_return.1" class="lexicon_definition"></a>
#### from_return(args,  depth,  state,  callback,  cbdata) [¶](#method__from_return.1)
Process a :return Expr for CFG construction.


*source:*
[CompilerTools/src/CFGs.jl:1069](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__getbbbodyorder.1" class="lexicon_definition"></a>
#### getBbBodyOrder(bl::CompilerTools.CFGs.CFG) [¶](#method__getbbbodyorder.1)
Determine a valid and reasonable order of basic blocks in which to reconstruct a :body Expr.
Also useful for printing in a reasonable order.


*source:*
[CompilerTools/src/CFGs.jl:639](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__getdistinctstatementnum.1" class="lexicon_definition"></a>
#### getDistinctStatementNum(bl::CompilerTools.CFGs.CFG) [¶](#method__getdistinctstatementnum.1)
Get a possible new statement number by finding the maximum statement value in any BasicBlock in the given CFG and adding 1.


*source:*
[CompilerTools/src/CFGs.jl:407](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__getmaxbb.1" class="lexicon_definition"></a>
#### getMaxBB(bl::CompilerTools.CFGs.CFG) [¶](#method__getmaxbb.1)
Returns the maximum basic block label for the given CFG.


*source:*
[CompilerTools/src/CFGs.jl:217](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__getmaxstatementnum.1" class="lexicon_definition"></a>
#### getMaxStatementNum(bb::CompilerTools.CFGs.BasicBlock) [¶](#method__getmaxstatementnum.1)
Get the maximum statement index for a given BasicBlock.


*source:*
[CompilerTools/src/CFGs.jl:394](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__getminbb.1" class="lexicon_definition"></a>
#### getMinBB(bl::CompilerTools.CFGs.CFG) [¶](#method__getminbb.1)
Returns the minimum basic block label for the given CFG.


*source:*
[CompilerTools/src/CFGs.jl:225](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__insertbefore.1" class="lexicon_definition"></a>
#### insertBefore(bl::CompilerTools.CFGs.CFG,  after::Int64) [¶](#method__insertbefore.1)
Given a CFG in input parameter "bl" and a basic block label "after" in that CFG,
insert a new basic block into the CFG before block "after".  
Returns a tuple of the new basic block created and if needed a GotoNode AST node to be inserted at the end of the new
basic block so that it will jump to the "after" basic block.  The user of this function is expected to insert
at the end of the new basic block once they are done inserting their other code.
If "after" is the head of a loop, you can stop the basic block containing the loop's back edge from being added to 
the new basic block by setting excludeBackEdge to true and setting back_edge to the loop's basic block containing
the back edge.


*source:*
[CompilerTools/src/CFGs.jl:332](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__insertbefore.2" class="lexicon_definition"></a>
#### insertBefore(bl::CompilerTools.CFGs.CFG,  after::Int64,  excludeBackEdge::Bool) [¶](#method__insertbefore.2)
Given a CFG in input parameter "bl" and a basic block label "after" in that CFG,
insert a new basic block into the CFG before block "after".  
Returns a tuple of the new basic block created and if needed a GotoNode AST node to be inserted at the end of the new
basic block so that it will jump to the "after" basic block.  The user of this function is expected to insert
at the end of the new basic block once they are done inserting their other code.
If "after" is the head of a loop, you can stop the basic block containing the loop's back edge from being added to 
the new basic block by setting excludeBackEdge to true and setting back_edge to the loop's basic block containing
the back edge.


*source:*
[CompilerTools/src/CFGs.jl:332](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__insertbefore.3" class="lexicon_definition"></a>
#### insertBefore(bl::CompilerTools.CFGs.CFG,  after::Int64,  excludeBackEdge::Bool,  back_edge) [¶](#method__insertbefore.3)
Given a CFG in input parameter "bl" and a basic block label "after" in that CFG,
insert a new basic block into the CFG before block "after".  
Returns a tuple of the new basic block created and if needed a GotoNode AST node to be inserted at the end of the new
basic block so that it will jump to the "after" basic block.  The user of this function is expected to insert
at the end of the new basic block once they are done inserting their other code.
If "after" is the head of a loop, you can stop the basic block containing the loop's back edge from being added to 
the new basic block by setting excludeBackEdge to true and setting back_edge to the loop's basic block containing
the back edge.


*source:*
[CompilerTools/src/CFGs.jl:332](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__insertbetween.1" class="lexicon_definition"></a>
#### insertBetween(bl::CompilerTools.CFGs.CFG,  before::Int64,  after::Int64) [¶](#method__insertbetween.1)
Insert a new basic block into the CFG "bl" between the basic blocks whose labels are "before" and "after".
Returns a tuple of the new basic block created and if needed a GotoNode AST node to be inserted at the end of the new
basic block so that it will jump to the "after" basic block.  The user of this function is expected to insert
at the end of the new basic block once they are done inserting their other code.


*source:*
[CompilerTools/src/CFGs.jl:515](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__insertstatementafter.1" class="lexicon_definition"></a>
#### insertStatementAfter(bl::CompilerTools.CFGs.CFG,  block,  stmt_idx,  new_stmt) [¶](#method__insertstatementafter.1)
For a given CFG "bl" and a "block" in that CFG, add a new statement "new_stmt" to the basic block
after statement index "stmt_idx".


*source:*
[CompilerTools/src/CFGs.jl:580](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__insertstatementbefore.1" class="lexicon_definition"></a>
#### insertStatementBefore(bl::CompilerTools.CFGs.CFG,  block,  stmt_idx,  new_stmt) [¶](#method__insertstatementbefore.1)
For a given CFG "bl" and a "block" in that CFG, add a new statement "new_stmt" to the basic block
before statement index "stmt_idx".


*source:*
[CompilerTools/src/CFGs.jl:588](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__insertat.1" class="lexicon_definition"></a>
#### insertat!(a,  value,  idx) [¶](#method__insertat.1)
Insert into an array "a" with a given "value" at the specified index "idx".


*source:*
[CompilerTools/src/CFGs.jl:571](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__not_handled.1" class="lexicon_definition"></a>
#### not_handled(a,  b) [¶](#method__not_handled.1)
A default callback that handles no extra AST node types.


*source:*
[CompilerTools/src/CFGs.jl:995](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__removeuselessblocks.1" class="lexicon_definition"></a>
#### removeUselessBlocks(bbs::Dict{Int64, CompilerTools.CFGs.BasicBlock}) [¶](#method__removeuselessblocks.1)
This function simplifies the dict of basic blocks "bbs".
One such simplification that is necessary for depth first numbering not to fail is the removal of dead blocks.
Other simplifications can be seen commented out below and while they may make the graph nicer to look at they
don't really add anything in terms of functionality.


*source:*
[CompilerTools/src/CFGs.jl:896](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__replacesucc.1" class="lexicon_definition"></a>
#### replaceSucc(cur_bb::CompilerTools.CFGs.BasicBlock,  orig_succ::CompilerTools.CFGs.BasicBlock,  new_succ::CompilerTools.CFGs.BasicBlock) [¶](#method__replacesucc.1)
For a given basic block "cur_bb", replace one of its successors "orig_succ" with a different successor "new_succ".


*source:*
[CompilerTools/src/CFGs.jl:858](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__uncompressed_ast.1" class="lexicon_definition"></a>
#### uncompressed_ast(l::LambdaStaticData) [¶](#method__uncompressed_ast.1)
Convert a compressed LambdaStaticData format into the uncompressed AST format.


*source:*
[CompilerTools/src/CFGs.jl:814](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__update_label.1" class="lexicon_definition"></a>
#### update_label(x::Expr,  state::CompilerTools.CFGs.UpdateLabelState,  top_level_number,  is_top_level,  read) [¶](#method__update_label.1)
An AstWalk callback that pattern matches GotoNode's and :gotoifnot Expr nodes and determines if the
label specified in this nodes is equal to the "old_label" in the UpdateLabelState and if so replaces
the "old_label" with "new_label" and sets the "changed" flag to true to indicate that update_label
was successful.


*source:*
[CompilerTools/src/CFGs.jl:255](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__wrapinconditional.1" class="lexicon_definition"></a>
#### wrapInConditional(bl::CompilerTools.CFGs.CFG,  cond_gotoifnot::Expr,  first::Int64,  merge::Int64) [¶](#method__wrapinconditional.1)
Modifies the CFG to create a conditional (i.e., if statement) that wraps a certain region of the CFG whose entry block is
"first" and whose last block is "last".
Takes a parameters:
1) bl - the CFG to modify
2) cond_gotoifnot - a :gotoifnot Expr whose label is equal to "first"
3) first - the existing starting block of the code to be included in the conditional
4) merge - the existing block to be executed after the conditional
To be eligible for wrapping, first and merge must be in the same scope of source code.
This restriction is validated by confirming that "first" dominates "merge" and that "merge" inverse dominates "first".


*source:*
[CompilerTools/src/CFGs.jl:436](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="method__wrapinconditional.2" class="lexicon_definition"></a>
#### wrapInConditional(bl::CompilerTools.CFGs.CFG,  cond_gotoifnot::Expr,  first::Int64,  merge::Int64,  back_edge::Union{CompilerTools.CFGs.BasicBlock, Void}) [¶](#method__wrapinconditional.2)
Modifies the CFG to create a conditional (i.e., if statement) that wraps a certain region of the CFG whose entry block is
"first" and whose last block is "last".
Takes a parameters:
1) bl - the CFG to modify
2) cond_gotoifnot - a :gotoifnot Expr whose label is equal to "first"
3) first - the existing starting block of the code to be included in the conditional
4) merge - the existing block to be executed after the conditional
To be eligible for wrapping, first and merge must be in the same scope of source code.
This restriction is validated by confirming that "first" dominates "merge" and that "merge" inverse dominates "first".


*source:*
[CompilerTools/src/CFGs.jl:436](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="type__basicblock.1" class="lexicon_definition"></a>
#### CompilerTools.CFGs.BasicBlock [¶](#type__basicblock.1)
Data structure to hold information about one basic block in the control-flow graph.
This structure contains the following fields:
1) label - an Int.  If positive, this basic block corresponds to a basic block declared in the AST through a label node.
                    The special label value -1 corresponds to the starting basic blocks.  The special value -2
                    corresponds to the final basic block (to which everything must flow).  Negative values correspond to
                    implicit basic blocks following gotoifnot nodes.  There nodes may goto some (positive) label but if
                    that branch is not taken they fall-through into an unlabelled basic (in the AST at least) but we
                    give such blocks negative labels.
2) preds - a set of basic blocks from which control may reach the current basic block
3) succs - a set of basic blocks to which control may flow from the current basic block
4) fallthrough_succ - if not "nothing", this indicates which of the basic block successors is reached via falling through from
                    the current basic block rather than a jump (goto or gotoifnot)
5) depth_first_number - a depth first numbering of the basic block graph is performed and this basic block's number is stored here
6) statements - an array of the statements in this basic block.


*source:*
[CompilerTools/src/CFGs.jl:100](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="type__cfg.1" class="lexicon_definition"></a>
#### CompilerTools.CFGs.CFG [¶](#type__cfg.1)
The main data structure to hold information about the control flow graph.
The basic_blocks field is a dictionary from basic block label to BasicBlock object.
The depth_first_numbering is an array of length the number of basic blocks.  
   Entry N in this array is the label of the basic block with depth-first numbering N.


*source:*
[CompilerTools/src/CFGs.jl:195](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="type__toplevelstatement.1" class="lexicon_definition"></a>
#### CompilerTools.CFGs.TopLevelStatement [¶](#type__toplevelstatement.1)
Data structure to hold the index (relative to the beginning of the body of the function) of a top-level statement
and the top-level statement itself.


*source:*
[CompilerTools/src/CFGs.jl:54](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="type__updatelabelstate.1" class="lexicon_definition"></a>
#### CompilerTools.CFGs.UpdateLabelState [¶](#type__updatelabelstate.1)
The opaque callback data type for the update_label callback.
It holds the old_label that should be changed to the new_label.
It also holds a "changed" field that starts as false and gets set to true when the callback actually
finds the old label and replaces it with the new one.


*source:*
[CompilerTools/src/CFGs.jl:239](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

---

<a id="type__expr_state.1" class="lexicon_definition"></a>
#### CompilerTools.CFGs.expr_state [¶](#type__expr_state.1)
Collects information about the CFG as it is being constructed.
Contains a dictionary of the currently known basic blocks that maps the label to a BasicBlock object.
cur_bb is the currently active BasicBlock to which the next statement encountered should be added.
next_if contains the next negative label number to be used for the next needed implicit basic block label.
top_level_number is the last top-level statement added.


*source:*
[CompilerTools/src/CFGs.jl:173](file:///home/etotoni/.julia/v0.4/CompilerTools/src/CFGs.jl)

