# CompilerTools.OptFramework

## Exported

---

<a id="method__addoptpass.1" class="lexicon_definition"></a>
#### addOptPass(func,  level) [¶](#method__addoptpass.1)
Same as the other addOptPass but with a pass call back function and pass level as input.


*source:*
[CompilerTools/src/OptFramework.jl:103](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__addoptpass.2" class="lexicon_definition"></a>
#### addOptPass(pass::CompilerTools.OptFramework.OptPass) [¶](#method__addoptpass.2)
Add an optimization pass. If this is going to be called multiple times then you need some external way of corrdinating the code/modules that are calling this function so that optimization passes are added in some sane order.


*source:*
[CompilerTools/src/OptFramework.jl:88](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="macro___acc.1" class="lexicon_definition"></a>
#### @acc(ast1, ast2...) [¶](#macro___acc.1)
The @acc macro comes in two forms:
1) @acc expression
3) @acc function ... end
In the first form, the set of optimization passes to apply come from the default set of optimization passes as specified with the funciton setOptPasses.  The @acc macro replaces each call in the expression with a call to a trampolines that determines the types of the call and if that combination of function and signature has not previously been optimized then it calls the set of optimization passes to optimize it.  Then, the trampoline calls the optimized function.
The second form is similar, and instead annotating callsite, the @acc macro can be used in front of a function's declaration. Used this way, it will replace the body of the function with the trampoline itself. The programmer can use @acc either at function callsite, or at function delcaration, but not both.
This macro may optionally take an OptPass array, right after @acc and followed by an expression or function.  In this case, the specified set of optPasses are used just for optimizing the following expression. When used with the second form (in front of a function), the value of this OptPass array will be statically evaluated at the macro expansion stage.


*source:*
[CompilerTools/src/OptFramework.jl:580](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="macro___noacc.1" class="lexicon_definition"></a>
#### @noacc(ast) [¶](#macro___noacc.1)
The macro @noacc can be used at call site to specifically run the non-accelerated copy of an accelerated function. It has no effect and gives a warning when the given function is not found to have been accelerated. We do not support nested @acc or @noacc. 


*source:*
[CompilerTools/src/OptFramework.jl:651](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

## Internal

---

<a id="method__typedexpr.1" class="lexicon_definition"></a>
#### TypedExpr(typ,  rest...) [¶](#method__typedexpr.1)
Creates a typed Expr AST node.
Convenence function that takes a type as first argument and the varargs thereafter.
The varargs are used to form an Expr AST node and the type parameter is used to fill in the "typ" field of the Expr.


*source:*
[CompilerTools/src/OptFramework.jl:42](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__cleanupastlabels.1" class="lexicon_definition"></a>
#### cleanupASTLabels(ast) [¶](#method__cleanupastlabels.1)
Clean up the labels in AST by renaming them, and removing duplicates.


*source:*
[CompilerTools/src/OptFramework.jl:265](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__convertcodetolevel.1" class="lexicon_definition"></a>
#### convertCodeToLevel(ast::ANY,  sig::ANY,  old_level,  new_level,  func) [¶](#method__convertcodetolevel.1)
convert AST from "old_level" to "new_level". The input "ast" can be either Expr or Function type. In the latter case, the result AST will be obtained from this function using an matching signature "sig". The last "func" is a skeleton function that is used internally to facility such conversion.


*source:*
[CompilerTools/src/OptFramework.jl:133](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__convert_expr.1" class="lexicon_definition"></a>
#### convert_expr(per_site_opt_set,  ast) [¶](#method__convert_expr.1)
When @acc is used at a function's callsite, we use AstWalk to search for callsites via the opt_calls_insert_trampoline callback and to then insert trampolines.  That updated expression containing trampoline calls is then returned as the generated code from the @acc macro.


*source:*
[CompilerTools/src/OptFramework.jl:471](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__convert_function.1" class="lexicon_definition"></a>
#### convert_function(per_site_opt_set,  opt_set,  macros,  ast) [¶](#method__convert_function.1)
When @acc is used at a function definition, it creates a trampoline function, when called with a specific set of signature types, will try to optimize the original function, and call it with the real arguments.  The input "ast" should be an AST of the original function at macro level, which will be   replaced by the trampoline. 


*source:*
[CompilerTools/src/OptFramework.jl:486](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__create_label_map.1" class="lexicon_definition"></a>
#### create_label_map(x,  state::CompilerTools.OptFramework.lmstate,  top_level_number,  is_top_level,  read) [¶](#method__create_label_map.1)
An AstWalk callback that collects information about labels in an AST.
The labels in AST are generally not sequential but to feed back into a Function Expr
correctly they need to be.  So, we keep a map from the old label in the AST to a new label
that we monotonically increases.
If we have code in the AST like the following:
   1:
   2:
... then one of these labels is redundant.  We set "last_was_label" if the last AST node
we saw was a label.  If we see another LabelNode right after that then we duplicate the rhs
of the label map.  For example, if you had the code:
   5:
   4:
... and the label 5 was the third label in the code then in the label map you would then have:
   5 -> 3, 4 -> 3.
This indicates that uses of both label 5 and label 4 in the code will become label 3 in the modified AST.


*source:*
[CompilerTools/src/OptFramework.jl:238](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__dumplevel.1" class="lexicon_definition"></a>
#### dumpLevel(level) [¶](#method__dumplevel.1)
pretty print pass level number as string.


*source:*
[CompilerTools/src/OptFramework.jl:54](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__evalpersiteoptset.1" class="lexicon_definition"></a>
#### evalPerSiteOptSet(per_site_opt_set) [¶](#method__evalpersiteoptset.1)
Statically evaluate per-site optimization passes setting, and return the result.


*source:*
[CompilerTools/src/OptFramework.jl:558](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__findoriginalfunc.1" class="lexicon_definition"></a>
#### findOriginalFunc(mod::Module,  name::Symbol) [¶](#method__findoriginalfunc.1)
Find the original (before @acc macro) function for a wrapper function in the given module. 
Return the input function if not found. Always return as a GlobalRef.


*source:*
[CompilerTools/src/OptFramework.jl:616](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__findtargetfunc.1" class="lexicon_definition"></a>
#### findTargetFunc(mod::Module,  name::Symbol) [¶](#method__findtargetfunc.1)
Find the optimizing target function (after @acc macro) for a wrapper function in the given module. 
Return the input function if not found. Always return as a GlobalRef.


*source:*
[CompilerTools/src/OptFramework.jl:607](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__getcodeatlevel.1" class="lexicon_definition"></a>
#### getCodeAtLevel(func,  sig,  level) [¶](#method__getcodeatlevel.1)
Retrieve the AST of the given function "func" and signature "sig" for at the given pass "level".


*source:*
[CompilerTools/src/OptFramework.jl:110](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__identical.1" class="lexicon_definition"></a>
#### identical{T}(t::Type{T},  x::T) [¶](#method__identical.1)
A hack to get around Julia's type inference. This is essentially an identity conversion,
but forces inferred return type to be the given type.


*source:*
[CompilerTools/src/OptFramework.jl:370](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__makewrapperfunc.1" class="lexicon_definition"></a>
#### makeWrapperFunc(new_fname::Symbol,  real_fname::Symbol,  call_sig_args::Array{Any, 1},  per_site_opt_set) [¶](#method__makewrapperfunc.1)
Define a wrapper function with the name given by "new_func" that when called will try to optimize the "real_func" function, and run it with given parameters in "call_sig_args". The input "per_site_opt_set" can be either nothing, or a quoted Expr that refers to an array of OptPass.


*source:*
[CompilerTools/src/OptFramework.jl:374](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__opt_calls_insert_trampoline.1" class="lexicon_definition"></a>
#### opt_calls_insert_trampoline(x,  per_site_opt_set,  top_level_number,  is_top_level,  read) [¶](#method__opt_calls_insert_trampoline.1)
An AstWalk callback function.
Finds call sites in the AST and replaces them with calls to newly generated trampoline functions.
These trampolines functions allow us to capture runtime types which in turn enables optimization passes to run on fully typed AST.
If a function/signature combination has not previously been optimized then call processFuncCall to optimize it.


*source:*
[CompilerTools/src/OptFramework.jl:440](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__processfunccall.1" class="lexicon_definition"></a>
#### processFuncCall(func::ANY,  call_sig_arg_tuple::ANY,  per_site_opt_set::ANY) [¶](#method__processfunccall.1)
Takes a function, a signature, and a set of optimizations and applies that set of optimizations to the function,
returns a new optimized function without modifying the input.  Argument explanation follows:
1) func - the function being optimized
2) call_sig_arg_tuple - the signature of the function, i.e., the types of each of its arguments
3) per_site_opt_set - the set of optimization passes to apply to this function.


*source:*
[CompilerTools/src/OptFramework.jl:309](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__removeduplabels.1" class="lexicon_definition"></a>
#### removeDupLabels(stmts) [¶](#method__removeduplabels.1)
Sometimes update_labels creates two label nodes that are the same.
This function removes such duplicate labels.


*source:*
[CompilerTools/src/OptFramework.jl:245](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__setoptpasses.1" class="lexicon_definition"></a>
#### setOptPasses(passes::Array{CompilerTools.OptFramework.OptPass, 1}) [¶](#method__setoptpasses.1)
Set the default set of optimization passes to apply with the @acc macro. 


*source:*
[CompilerTools/src/OptFramework.jl:79](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__tfuncpresent.1" class="lexicon_definition"></a>
#### tfuncPresent(func,  tt) [¶](#method__tfuncpresent.1)
Makes sure that a newly created function is correctly present in the internal Julia method table.


*source:*
[CompilerTools/src/OptFramework.jl:282](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="method__update_labels.1" class="lexicon_definition"></a>
#### update_labels(x,  state::CompilerTools.OptFramework.lmstate,  top_level_number,  is_top_level,  read) [¶](#method__update_labels.1)
An AstWalk callback that applies the label map created during create_label_map AstWalk.
For each label in the code, replace that label with the rhs of the label map.


*source:*
[CompilerTools/src/OptFramework.jl:190](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="type__optpass.1" class="lexicon_definition"></a>
#### CompilerTools.OptFramework.OptPass [¶](#type__optpass.1)
A data structure that holds information about one high-level optimization pass to run.
"func" is the callback function that does the optimization pass and should have the signature (GlobalRef, Expr, Tuple) where the GlobalRef provides the locate of the function to be optimized, Expr is the AST input to this pass, and Tuple is a tuple of all parameter types of the functions. It must return either an optimized Expr, or a Function.
"level" indicates at which level this pass is to be run. 


*source:*
[CompilerTools/src/OptFramework.jl:70](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="type__lmstate.1" class="lexicon_definition"></a>
#### CompilerTools.OptFramework.lmstate [¶](#type__lmstate.1)
The callback state variable used by create_label_map and update_labels.
label_map is a dictionary mapping old label ID's in the old AST with new label ID's in the new AST.
next_block_num is a monotonically increasing integer starting from 0 so label occur sequentially in the new AST.
last_was_label keeps track of whether we see two consecutive LabelNodes in the AST.


*source:*
[CompilerTools/src/OptFramework.jl:178](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

---

<a id="global__goptframeworkdict.1" class="lexicon_definition"></a>
#### gOptFrameworkDict [¶](#global__goptframeworkdict.1)
A global memo-table that maps both: the triple (function, signature, optPasses) to the trampoline function, and the trampoline function to the real function.


*source:*
[CompilerTools/src/OptFramework.jl:167](file:///home/etotoni/.julia/v0.4/CompilerTools/src/OptFramework.jl)

