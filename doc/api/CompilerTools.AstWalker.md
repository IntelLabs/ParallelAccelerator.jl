# CompilerTools.AstWalker

## Exported

---

<a id="method__astwalk.1" class="lexicon_definition"></a>
#### AstWalk(ast::ANY,  callback,  cbdata::ANY) [¶](#method__astwalk.1)
Entry point into the code to perform an AST walk.
You generally pass a lambda expression as the first argument.
The third argument is an object that is opaque to AstWalk but that is passed to every callback.
You can use this object to collect data about the AST as it is walked or to hold information on
how to change the AST as you are walking over it.
The second argument is a callback function.  For each AST node, AstWalk will invoke this callback.
The signature of the callback must be (Any, Any, Int64, Bool, Bool).  The arguments to the callback
are as follows:
    1) The current node of the AST being walked.
    2) The callback data object that you originally passed as the first argument to AstWalk.
    3) Specifies the index of the body's statement that is currently being processed.
    4) True if the current AST node being walked is the root of a top-level statement, false if the AST node is a sub-tree of a top-level statement.
    5) True if the AST node is being read, false if it is being written.
The callback should return an array of items.  It does this because in some cases it makes sense to return multiple things so
all callbacks have to to keep the interface consistent.


*source:*
[CompilerTools/src/ast_walk.jl:188](file:///home/etotoni/.julia/v0.4/CompilerTools/src/ast_walk.jl)

## Internal

---

<a id="method__from_assignment.1" class="lexicon_definition"></a>
#### from_assignment(ast::Array{Any, 1},  depth,  callback,  cbdata::ANY,  top_level_number,  read) [¶](#method__from_assignment.1)
AstWalk through an assignment expression.
Recursively process the left and right hand sides with AstWalk.


*source:*
[CompilerTools/src/ast_walk.jl:128](file:///home/etotoni/.julia/v0.4/CompilerTools/src/ast_walk.jl)

---

<a id="method__from_body.1" class="lexicon_definition"></a>
#### from_body(ast::Array{Any, 1},  depth,  callback,  cbdata::ANY,  top_level_number,  read) [¶](#method__from_body.1)
AstWalk through a function body.


*source:*
[CompilerTools/src/ast_walk.jl:81](file:///home/etotoni/.julia/v0.4/CompilerTools/src/ast_walk.jl)

---

<a id="method__from_call.1" class="lexicon_definition"></a>
#### from_call(ast::Array{Any, 1},  depth,  callback,  cbdata::ANY,  top_level_number,  read) [¶](#method__from_call.1)
AstWalk through a call expression.
Recursively process the name of the function and each of its arguments.


*source:*
[CompilerTools/src/ast_walk.jl:141](file:///home/etotoni/.julia/v0.4/CompilerTools/src/ast_walk.jl)

---

<a id="method__from_expr.1" class="lexicon_definition"></a>
#### from_expr(ast::ANY,  depth,  callback,  cbdata::ANY,  top_level_number,  is_top_level,  read) [¶](#method__from_expr.1)
The main routine that switches on all the various AST node types.
The internal nodes of the AST are of type Expr with various different Expr.head field values such as :lambda, :body, :block, etc.
The leaf nodes of the AST all have different types.
There are some node types we don't currently recurse into.  Maybe this needs to be extended.


*source:*
[CompilerTools/src/ast_walk.jl:187](file:///home/etotoni/.julia/v0.4/CompilerTools/src/ast_walk.jl)

---

<a id="method__from_exprs.1" class="lexicon_definition"></a>
#### from_exprs(ast::Array{Any, 1},  depth,  callback,  cbdata::ANY,  top_level_number,  read) [¶](#method__from_exprs.1)
AstWalk through an array of expressions.


*source:*
[CompilerTools/src/ast_walk.jl:104](file:///home/etotoni/.julia/v0.4/CompilerTools/src/ast_walk.jl)

---

<a id="method__from_lambda.1" class="lexicon_definition"></a>
#### from_lambda(ast::Array{Any, 1},  depth,  callback,  cbdata::ANY,  top_level_number,  read) [¶](#method__from_lambda.1)
AstWalk through a lambda expression.
Walk through each input parameters and the body of the lambda.


*source:*
[CompilerTools/src/ast_walk.jl:50](file:///home/etotoni/.julia/v0.4/CompilerTools/src/ast_walk.jl)

---

<a id="method__uncompressed_ast.1" class="lexicon_definition"></a>
#### uncompressed_ast(l::LambdaStaticData) [¶](#method__uncompressed_ast.1)
Convert a compressed LambdaStaticData format into the uncompressed AST format.


*source:*
[CompilerTools/src/ast_walk.jl:42](file:///home/etotoni/.julia/v0.4/CompilerTools/src/ast_walk.jl)

