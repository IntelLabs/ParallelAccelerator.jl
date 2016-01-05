# CompilerTools.ReadWriteSet

## Exported

---

<a id="method__from_exprs.1" class="lexicon_definition"></a>
#### from_exprs(ast::Array{T, N}) [¶](#method__from_exprs.1)
Walk through an array of expressions.
Just recursively call from_expr for each expression in the array.


*source:*
[CompilerTools/src/read-write-set.jl:129](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__from_exprs.2" class="lexicon_definition"></a>
#### from_exprs(ast::Array{T, N},  callback::Union{Function, Void},  cbdata::ANY) [¶](#method__from_exprs.2)
Walk through an array of expressions.
Just recursively call from_expr for each expression in the array.
Takes a callback and an opaque object so that non-standard Julia AST nodes can be processed via the callback.


*source:*
[CompilerTools/src/read-write-set.jl:139](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__from_exprs.3" class="lexicon_definition"></a>
#### from_exprs(ast::Array{T, N},  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY) [¶](#method__from_exprs.3)
Walk through an array of expressions.
Just recursively call from_expr for each expression in the array.
Takes a callback and an opaque object so that non-standard Julia AST nodes can be processed via the callback.
Takes a ReadWriteSetType in "rws" into which information will be stored.


*source:*
[CompilerTools/src/read-write-set.jl:167](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__isread.1" class="lexicon_definition"></a>
#### isRead(sym::Union{GenSym, Symbol},  rws::CompilerTools.ReadWriteSet.ReadWriteSetType) [¶](#method__isread.1)
Return true if some symbol in "sym" is read either as a scalar or array within the computed ReadWriteSetType.


*source:*
[CompilerTools/src/read-write-set.jl:82](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__iswritten.1" class="lexicon_definition"></a>
#### isWritten(sym::Union{GenSym, Symbol},  rws::CompilerTools.ReadWriteSet.ReadWriteSetType) [¶](#method__iswritten.1)
Return true if some symbol in "sym" is written either as a scalar or array within the computed ReadWriteSetType.


*source:*
[CompilerTools/src/read-write-set.jl:95](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="type__accessset.1" class="lexicon_definition"></a>
#### CompilerTools.ReadWriteSet.AccessSet [¶](#type__accessset.1)
Holds which scalars and which array are accessed and for array which index expressions are used.


*source:*
[CompilerTools/src/read-write-set.jl:40](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="type__readwritesettype.1" class="lexicon_definition"></a>
#### CompilerTools.ReadWriteSet.ReadWriteSetType [¶](#type__readwritesettype.1)
Stores which scalars and arrays are read or written in some code region.


*source:*
[CompilerTools/src/read-write-set.jl:49](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

## Internal

---

<a id="method__addindexexpr.1" class="lexicon_definition"></a>
#### addIndexExpr!(this_dict,  array_name,  index_expr) [¶](#method__addindexexpr.1)
Takes a dictionary of symbol to an array of index expression.
Takes an array in "array_name" being accessed with expression "index_expr".
Makes sure there is an entry in the dictionary for this array and adds the index expression to this array.


*source:*
[CompilerTools/src/read-write-set.jl:229](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__from_assignment.1" class="lexicon_definition"></a>
#### from_assignment(ast::Array{Any, 1},  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY) [¶](#method__from_assignment.1)
Process an assignment AST node.
The left-hand side of the assignment is added to the writeSet.


*source:*
[CompilerTools/src/read-write-set.jl:214](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__from_call.1" class="lexicon_definition"></a>
#### from_call(ast::Array{Any, 1},  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY) [¶](#method__from_call.1)
Process :call Expr nodes to find arrayref and arrayset calls and adding the corresponding index expressions to the read and write sets respectively.


*source:*
[CompilerTools/src/read-write-set.jl:239](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__from_coloncolon.1" class="lexicon_definition"></a>
#### from_coloncolon(ast::Array{T, N},  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY) [¶](#method__from_coloncolon.1)
Process a :(::) AST node.
Just process the symbol part of the :(::) node in ast[1] (which is args of the node passed in).


*source:*
[CompilerTools/src/read-write-set.jl:189](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__from_expr.1" class="lexicon_definition"></a>
#### from_expr(ast::ANY) [¶](#method__from_expr.1)
Walk through one AST node.
Calls the main internal walking function after initializing an empty ReadWriteSetType.


*source:*
[CompilerTools/src/read-write-set.jl:146](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__from_expr.2" class="lexicon_definition"></a>
#### from_expr(ast::ANY,  callback::Union{Function, Void},  cbdata::ANY) [¶](#method__from_expr.2)
Walk through one AST node.
Calls the main internal walking function after initializing an empty ReadWriteSetType.
Takes a callback and an opaque object so that non-standard Julia AST nodes can be processed via the callback.


*source:*
[CompilerTools/src/read-write-set.jl:156](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__from_expr.3" class="lexicon_definition"></a>
#### from_expr(ast::LambdaStaticData,  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY) [¶](#method__from_expr.3)
The main routine that switches on all the various AST node types.
The internal nodes of the AST are of type Expr with various different Expr.head field values such as :lambda, :body, :block, etc.
The leaf nodes of the AST all have different types.


*source:*
[CompilerTools/src/read-write-set.jl:302](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__from_lambda.1" class="lexicon_definition"></a>
#### from_lambda(ast::Expr,  depth,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY) [¶](#method__from_lambda.1)
Walk through a lambda expression.
We just need to recurse through the lambda body and can ignore the rest.


*source:*
[CompilerTools/src/read-write-set.jl:118](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__from_tuple.1" class="lexicon_definition"></a>
#### from_tuple(ast::Array{T, N},  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType,  callback::Union{Function, Void},  cbdata::ANY) [¶](#method__from_tuple.1)
Walk through a tuple.
Just recursively call from_exprs on the internal tuple array to process each part of the tuple.


*source:*
[CompilerTools/src/read-write-set.jl:181](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__tosymgen.1" class="lexicon_definition"></a>
#### toSymGen(x::Union{GenSym, Symbol}) [¶](#method__tosymgen.1)
In various places we need a SymGen type which is the union of Symbol and GenSym.
This function takes a Symbol, SymbolNode, or GenSym and return either a Symbol or GenSym.


*source:*
[CompilerTools/src/read-write-set.jl:198](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__trycallback.1" class="lexicon_definition"></a>
#### tryCallback(ast::ANY,  callback::Union{Function, Void},  cbdata::ANY,  depth::Integer,  rws::CompilerTools.ReadWriteSet.ReadWriteSetType) [¶](#method__trycallback.1)
If an AST node is not recognized then we try the passing the node to the callback to see if 
it was able to process it.  If so, then we process the regular Julia statement returned by
the callback.


*source:*
[CompilerTools/src/read-write-set.jl:284](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

---

<a id="method__uncompressed_ast.1" class="lexicon_definition"></a>
#### uncompressed_ast(l::LambdaStaticData) [¶](#method__uncompressed_ast.1)
Convert a compressed LambdaStaticData format into the uncompressed AST format.


*source:*
[CompilerTools/src/read-write-set.jl:108](file:///home/etotoni/.julia/v0.4/CompilerTools/src/read-write-set.jl)

