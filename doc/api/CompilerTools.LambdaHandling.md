# CompilerTools.LambdaHandling

## Exported

---

<a id="method__addescapingvariable.1" class="lexicon_definition"></a>
#### addEscapingVariable(s::Symbol,  typ,  desc::Int64,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__addescapingvariable.1)
Adds a new escaping variable with the given Symbol "s", type "typ", descriptor "desc" in LambdaInfo "li".
Returns true if the variable already existed and its type and descriptor were updated, false otherwise.


*source:*
[CompilerTools/src/lambda.jl:365](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__addescapingvariable.2" class="lexicon_definition"></a>
#### addEscapingVariable(vd::CompilerTools.LambdaHandling.VarDef,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__addescapingvariable.2)
Adds a new escaping variable from a VarDef in parameter "vd" into LambdaInfo "li".


*source:*
[CompilerTools/src/lambda.jl:384](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__addgensym.1" class="lexicon_definition"></a>
#### addGenSym(typ,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__addgensym.1)
Add a new GenSym to the LambdaInfo in "li" with the given type in "typ".
Returns the new GenSym.


*source:*
[CompilerTools/src/lambda.jl:393](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__addlocalvariable.1" class="lexicon_definition"></a>
#### addLocalVariable(s::Symbol,  typ,  desc::Int64,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__addlocalvariable.1)
Adds a new local variable with the given Symbol "s", type "typ", descriptor "desc" in LambdaInfo "li".
Returns true if the variable already existed and its type and descriptor were updated, false otherwise.


*source:*
[CompilerTools/src/lambda.jl:345](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__addlocalvariable.2" class="lexicon_definition"></a>
#### addLocalVariable(vd::CompilerTools.LambdaHandling.VarDef,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__addlocalvariable.2)
Adds a local variable from a VarDef to the given LambdaInfo.


*source:*
[CompilerTools/src/lambda.jl:323](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__getbody.1" class="lexicon_definition"></a>
#### getBody(lambda::Expr) [¶](#method__getbody.1)
Returns the body expression part of a lambda expression.


*source:*
[CompilerTools/src/lambda.jl:705](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__getdesc.1" class="lexicon_definition"></a>
#### getDesc(x::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__getdesc.1)
Returns the descriptor for a local variable or input parameter "x" from LambdaInfo in "li".


*source:*
[CompilerTools/src/lambda.jl:265](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__getrefparams.1" class="lexicon_definition"></a>
#### getRefParams(lambdaInfo::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__getrefparams.1)
Returns an array of Symbols corresponding to those parameters to the method that are going to be passed by reference.
In short, isbits() types are passed by value and !isbits() types are passed by reference.


*source:*
[CompilerTools/src/lambda.jl:715](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__getreturntype.1" class="lexicon_definition"></a>
#### getReturnType(li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__getreturntype.1)
Returns the type of the lambda as stored in LambdaInfo "li" and as extracted during lambdaExprToLambdaInfo.


*source:*
[CompilerTools/src/lambda.jl:626](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__gettype.1" class="lexicon_definition"></a>
#### getType(x::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__gettype.1)
Returns the type of a Symbol or GenSym in "x" from LambdaInfo in "li".


*source:*
[CompilerTools/src/lambda.jl:235](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__getvardef.1" class="lexicon_definition"></a>
#### getVarDef(s::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__getvardef.1)
Returns the VarDef for a Symbol in LambdaInfo in "li"


*source:*
[CompilerTools/src/lambda.jl:272](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__isescapingvariable.1" class="lexicon_definition"></a>
#### isEscapingVariable(s::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__isescapingvariable.1)
Returns true if the Symbol in "s" is an escaping variable in LambdaInfo in "li".


*source:*
[CompilerTools/src/lambda.jl:300](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__isinputparameter.1" class="lexicon_definition"></a>
#### isInputParameter(s::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__isinputparameter.1)
Returns true if the Symbol in "s" is an input parameter in LambdaInfo in "li".


*source:*
[CompilerTools/src/lambda.jl:279](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__islocalgensym.1" class="lexicon_definition"></a>
#### isLocalGenSym(s::GenSym,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__islocalgensym.1)
Returns true if the GenSym in "s" is a GenSym in LambdaInfo in "li".


*source:*
[CompilerTools/src/lambda.jl:307](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__islocalvariable.1" class="lexicon_definition"></a>
#### isLocalVariable(s::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__islocalvariable.1)
Returns true if the Symbol in "s" is a local variable in LambdaInfo in "li".


*source:*
[CompilerTools/src/lambda.jl:286](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__lambdaexprtolambdainfo.1" class="lexicon_definition"></a>
#### lambdaExprToLambdaInfo(lambda::Expr) [¶](#method__lambdaexprtolambdainfo.1)
Convert a lambda expression into our internal storage format, LambdaInfo.
The input is asserted to be an expression whose head is :lambda.


*source:*
[CompilerTools/src/lambda.jl:586](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__lambdainfotolambdaexpr.1" class="lexicon_definition"></a>
#### lambdaInfoToLambdaExpr(lambdaInfo::CompilerTools.LambdaHandling.LambdaInfo,  body) [¶](#method__lambdainfotolambdaexpr.1)
Convert our internal storage format, LambdaInfo, back into a lambda expression.
This takes a LambdaInfo and a body as input parameters.
This body can be a body expression or you can pass "nothing" if you want but then you will probably need to set the body in args[3] manually by yourself.


*source:*
[CompilerTools/src/lambda.jl:673](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__lambdatypeinf.1" class="lexicon_definition"></a>
#### lambdaTypeinf(lambda::LambdaStaticData,  typs::Tuple) [¶](#method__lambdatypeinf.1)
Force type inference on a LambdaStaticData object.
Return both the inferred AST that is to a "code_typed(Function, (type,...))" call, 
and the inferred return type of the input method.


*source:*
[CompilerTools/src/lambda.jl:637](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__replaceexprwithdict.1" class="lexicon_definition"></a>
#### replaceExprWithDict!(expr::ANY,  dict::Dict{Union{GenSym, Symbol}, Any}) [¶](#method__replaceexprwithdict.1)
Replace the symbols in an expression "expr" with those defined in the
dictionary "dict".  Return the result expression, which may share part of the
input expression, and the input "expr" may be modified inplace and shall not be used
after this call. Note that unlike "replaceExprWithDict", the traversal here is
done by ASTWalker, which has the ability to traverse non-Expr data.


*source:*
[CompilerTools/src/lambda.jl:511](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__replaceexprwithdict.2" class="lexicon_definition"></a>
#### replaceExprWithDict!(expr::ANY,  dict::Dict{Union{GenSym, Symbol}, Any},  AstWalkFunc) [¶](#method__replaceexprwithdict.2)
Replace the symbols in an expression "expr" with those defined in the
dictionary "dict".  Return the result expression, which may share part of the
input expression, and the input "expr" may be modified inplace and shall not be used
after this call. Note that unlike "replaceExprWithDict", the traversal here is
done by ASTWalker, which has the ability to traverse non-Expr data.


*source:*
[CompilerTools/src/lambda.jl:511](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__replaceexprwithdict.3" class="lexicon_definition"></a>
#### replaceExprWithDict(expr,  dict::Dict{Union{GenSym, Symbol}, Any}) [¶](#method__replaceexprwithdict.3)
Replace the symbols in an expression "expr" with those defined in the
dictionary "dict".  Return the result expression, which may share part of the
input expression, but the input "expr" remains intact and is not modified.

Note that unlike "replaceExprWithDict!", we do not recurse down nested lambda
expressions (i.e., LambdaStaticData or DomainLambda or any other none Expr
objects are left unchanged). If such lambdas have escaping names that are to be
replaced, then the result will be wrong.


*source:*
[CompilerTools/src/lambda.jl:473](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__updateassigneddesc.1" class="lexicon_definition"></a>
#### updateAssignedDesc(lambdaInfo::CompilerTools.LambdaHandling.LambdaInfo,  symbol_assigns::Dict{Symbol, Int64}) [¶](#method__updateassigneddesc.1)
Update the descriptor part of the VarDef dealing with whether the variable is assigned or not in the function.
Takes the lambdaInfo and a dictionary that maps symbols names to the number of times they are statically assigned in the function.


*source:*
[CompilerTools/src/lambda.jl:682](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="type__lambdainfo.1" class="lexicon_definition"></a>
#### CompilerTools.LambdaHandling.LambdaInfo [¶](#type__lambdainfo.1)
An internal format for storing a lambda expression's args[1] and args[2].
The input parameters are stored as a Set since they must be unique and it makes for faster searching.
The VarDefs are stored as a dictionary from symbol to VarDef since type lookups are reasonably frequent and need to be fast.
The GenSym part (args[2][3]) is stored as an array since GenSym's are indexed.
Captured_outer_vars and static_parameter_names are stored as arrays for now since we don't expect them to be changed much.


*source:*
[CompilerTools/src/lambda.jl:86](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="type__vardef.1" class="lexicon_definition"></a>
#### CompilerTools.LambdaHandling.VarDef [¶](#type__vardef.1)
Represents the triple stored in a lambda's args[2][1].
The triple is 1) the Symbol of an input parameter or local variable, 2) the type of that Symbol, and 3) a descriptor for that symbol.
The descriptor can be 0 if the variable is an input parameter, 1 if it is captured, 2 if it is assigned within the function, 4 if
it is assigned by an inner function, 8 if it is const, and 16 if it is assigned to statically only once by the function.


*source:*
[CompilerTools/src/lambda.jl:68](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="typealias__symgen.1" class="lexicon_definition"></a>
#### SymGen [¶](#typealias__symgen.1)
Type aliases for different unions of Symbol, SymbolNode, and GenSym.


*source:*
[CompilerTools/src/lambda.jl:54](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

## Internal

---

<a id="method__adddescflag.1" class="lexicon_definition"></a>
#### addDescFlag(s::Symbol,  desc_flag::Int64,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__adddescflag.1)
Add one or more bitfields in "desc_flag" to the descriptor for a variable.


*source:*
[CompilerTools/src/lambda.jl:330](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__addinputparameter.1" class="lexicon_definition"></a>
#### addInputParameter(vd::CompilerTools.LambdaHandling.VarDef,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__addinputparameter.1)
Add Symbol "s" as input parameter to LambdaInfo "li".


*source:*
[CompilerTools/src/lambda.jl:218](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__addinputparameters.1" class="lexicon_definition"></a>
#### addInputParameters(collection,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__addinputparameters.1)
Add all variable in "collection" as input parameters to LambdaInfo "li".


*source:*
[CompilerTools/src/lambda.jl:226](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__addlocalvar.1" class="lexicon_definition"></a>
#### addLocalVar(name::AbstractString,  typ,  desc::Int64,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__addlocalvar.1)
Add a local variable to the function corresponding to LambdaInfo in "li" with name (as String), type and descriptor.
Returns true if variable already existed and was updated, false otherwise.


*source:*
[CompilerTools/src/lambda.jl:402](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__addlocalvar.2" class="lexicon_definition"></a>
#### addLocalVar(name::Symbol,  typ,  desc::Int64,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__addlocalvar.2)
Add a local variable to the function corresponding to LambdaInfo in "li" with name (as Symbol), type and descriptor.
Returns true if variable already existed and was updated, false otherwise.


*source:*
[CompilerTools/src/lambda.jl:410](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__addlocalvariables.1" class="lexicon_definition"></a>
#### addLocalVariables(collection,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__addlocalvariables.1)
Add multiple local variables from some collection type.


*source:*
[CompilerTools/src/lambda.jl:314](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__count_symbols.1" class="lexicon_definition"></a>
#### count_symbols(x::Symbol,  state::CompilerTools.LambdaHandling.CountSymbolState,  top_level_number,  is_top_level,  read) [¶](#method__count_symbols.1)
Adds symbols and gensyms to their corresponding sets in CountSymbolState when they are seen in the AST.


*source:*
[CompilerTools/src/lambda.jl:141](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__createmeta.1" class="lexicon_definition"></a>
#### createMeta(lambdaInfo::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__createmeta.1)
Create the args[2] part of a lambda expression given an object of our internal storage format LambdaInfo.


*source:*
[CompilerTools/src/lambda.jl:655](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__createvardict.1" class="lexicon_definition"></a>
#### createVarDict(x::Array{Any, 1}) [¶](#method__createvardict.1)
Convert the lambda expression's args[2][1] from Array{Array{Any,1},1} to a Dict{Symbol,VarDef}.
The internal triples are extracted and asserted that name and desc are of the appropriate type.


*source:*
[CompilerTools/src/lambda.jl:439](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__dicttoarray.1" class="lexicon_definition"></a>
#### dictToArray(x::Dict{Symbol, CompilerTools.LambdaHandling.VarDef}) [¶](#method__dicttoarray.1)
Convert the Dict{Symbol,VarDef} internal storage format from a dictionary back into an array of Any triples.


*source:*
[CompilerTools/src/lambda.jl:644](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__eliminateunusedlocals.1" class="lexicon_definition"></a>
#### eliminateUnusedLocals!(li::CompilerTools.LambdaHandling.LambdaInfo,  body::Expr) [¶](#method__eliminateunusedlocals.1)
Eliminates unused symbols from the LambdaInfo var_defs.
Takes a LambdaInfo to modify, the body to scan using AstWalk and an optional callback to AstWalk for custom AST types.


*source:*
[CompilerTools/src/lambda.jl:181](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__eliminateunusedlocals.2" class="lexicon_definition"></a>
#### eliminateUnusedLocals!(li::CompilerTools.LambdaHandling.LambdaInfo,  body::Expr,  AstWalkFunc) [¶](#method__eliminateunusedlocals.2)
Eliminates unused symbols from the LambdaInfo var_defs.
Takes a LambdaInfo to modify, the body to scan using AstWalk and an optional callback to AstWalk for custom AST types.


*source:*
[CompilerTools/src/lambda.jl:181](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__getlocalvariables.1" class="lexicon_definition"></a>
#### getLocalVariables(li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__getlocalvariables.1)
Returns an array of Symbols for local variables.


*source:*
[CompilerTools/src/lambda.jl:293](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__mergelambdainfo.1" class="lexicon_definition"></a>
#### mergeLambdaInfo(outer::CompilerTools.LambdaHandling.LambdaInfo,  inner::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__mergelambdainfo.1)
Merge "inner" lambdaInfo into "outer", and "outer" is changed as result.  Note
that the input_params, static_parameter_names, and escaping_defs of "outer" do
not change, other fields are merged. The GenSyms in "inner" will need to adjust
their indices as a result of this merge. We return a dictionary that maps from
old GenSym to new GenSym for "inner", which can be used to adjust the body Expr
of "inner" lambda using "replaceExprWithDict" or "replaceExprWithDict!".


*source:*
[CompilerTools/src/lambda.jl:557](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__removelocalvar.1" class="lexicon_definition"></a>
#### removeLocalVar(name::Symbol,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__removelocalvar.1)
Remove a local variable from lambda "li" given the variable's "name".
Returns true if the variable existed and it was removed, false otherwise.


*source:*
[CompilerTools/src/lambda.jl:426](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="method__show.1" class="lexicon_definition"></a>
#### show(io::IO,  li::CompilerTools.LambdaHandling.LambdaInfo) [¶](#method__show.1)
Pretty print a LambdaInfo.


*source:*
[CompilerTools/src/lambda.jl:98](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

---

<a id="type__countsymbolstate.1" class="lexicon_definition"></a>
#### CompilerTools.LambdaHandling.CountSymbolState [¶](#type__countsymbolstate.1)
Holds symbols and gensyms that are seen in a given AST when using the specified callback to handle non-standard Julia AST types.


*source:*
[CompilerTools/src/lambda.jl:129](file:///home/etotoni/.julia/v0.4/CompilerTools/src/lambda.jl)

