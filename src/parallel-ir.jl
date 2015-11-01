#=
Copyright (c) 2015, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
=#

module ParallelIR
export num_threads_mode

import CompilerTools.DebugMsg
DebugMsg.init()

using CompilerTools
using CompilerTools.LambdaHandling
using ..DomainIR
using CompilerTools.AliasAnalysis
import ..ParallelAccelerator
if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
using Base.Threads
end

import Base.show
import CompilerTools.AstWalker
import CompilerTools.ReadWriteSet
import CompilerTools.LivenessAnalysis
import CompilerTools.Loops
import CompilerTools.Loops.DomLoops

# uncomment this line when using Debug.jl
#using Debug

function ns_to_sec(x)
    x / 1000000000.0
end


const ISCAPTURED = 1
const ISASSIGNED = 2
const ISASSIGNEDBYINNERFUNCTION = 4
const ISCONST = 8
const ISASSIGNEDONCE = 16
const ISPRIVATEPARFORLOOP = 32

unique_num = 1

@doc """
This should pretty always be used instead of Expr(...) to form an expression as it forces the typ to be provided.
"""
function TypedExpr(typ, rest...)
    res = Expr(rest...)
    res.typ = typ
    res
end

@doc """
Holds the information about a loop in a parfor node.
"""
type PIRLoopNest
    indexVariable :: SymbolNode
    lower
    upper
    step
end

@doc """
Holds the information about a reduction in a parfor node.
"""
type PIRReduction
    reductionVar  :: SymbolNode
    reductionVarInit
    reductionFunc
end

@doc """
Holds information about domain operations part of a parfor node.
"""
type DomainOperation
    operation
    input_args :: Array{Any,1}
end

@doc """
Holds a dictionary from an array symbol to an integer corresponding to an equivalence class.
All array symbol in the same equivalence class are known to have the same shape.
"""
type EquivalenceClasses
    data :: Dict{Symbol,Int64}

    function EquivalenceClasses()
        new(Dict{Symbol,Int64}())
    end
end

@doc """
At some point we realize that two arrays must have the same dimensions but up until that point
we might not have known that.  In which case they will start in different equivalence classes,
merge_to and merge_from, but need to be combined into one equivalence class.
Go through the equivalence class dictionary and for any symbol belonging to the merge_from
equivalence class, change it to now belong to the merge_to equivalence class.
"""
function EquivalenceClassesMerge(ec :: EquivalenceClasses, merge_to :: Symbol, merge_from :: Symbol)
    to_int   = EquivalenceClassesAdd(ec, merge_to)
    from_int = EquivalenceClassesAdd(ec, merge_from)

    # For each array in the dictionary.
    for i in ec.data
        # If it is in the "merge_from" class...
        if i[2] == merge_from
            # ...move it to the "merge_to" class.
            ec.data[i[1]] = merge_to
        end
    end
    nothing
end

@doc """
Add a symbol as part of a new equivalence class if the symbol wasn't already in an equivalence class.
Return the equivalence class for the symbol.
"""
function EquivalenceClassesAdd(ec :: EquivalenceClasses, sym :: Symbol)
    # If the symbol isn't already in an equivalence class.
    if !haskey(ec.data, sym)
        # Find the maximum equivalence class "m".
        a = collect(values(ec.data))
        m = length(a) == 0 ? 0 : maximum(a)
        # Create a new equivalence class with this symbol with class "m+1"
        ec.data[sym] = m + 1
    end
    ec.data[sym]
end

@doc """
Clear an equivalence class.
"""
function EquivalenceClassesClear(ec :: EquivalenceClasses)
    empty!(ec.data)
end

@doc """
The parfor AST node type.
While we are lowering domain IR to parfors and fusing we use this representation because it
makes it easier to associate related statements before and after the loop to the loop itself.
"""
type PIRParForAst
    body                                      # holds the body of the innermost loop (outer loops can't have anything in them except inner loops)
    preParFor    :: Array{Any,1}              # do these statements before the parfor
    loopNests    :: Array{PIRLoopNest,1}      # holds information about the loop nests
    reductions   :: Array{PIRReduction,1}     # holds information about the reductions
    postParFor   :: Array{Any,1}              # do these statements after the parfor

    original_domain_nodes :: Array{DomainOperation,1}
    top_level_number :: Array{Int,1}
    rws          :: ReadWriteSet.ReadWriteSetType

    unique_id
    array_aliases :: Dict{SymGen, SymGen}

    # instruction count estimate of the body
    # To get the total loop instruction count, multiply this value by (upper_limit - lower_limit)/step for each loop nest
    # This will be "nothing" if we don't know how to estimate.  If not "nothing" then it is an expression which may
    # include calls.
    instruction_count_expr
    simply_indexed :: Bool

    function PIRParForAst(b, pre, nests, red, post, orig, t, unique, si)
        r = CompilerTools.ReadWriteSet.from_exprs(b)
        new(b, pre, nests, red, post, orig, [t], r, unique, Dict{Symbol,Symbol}(), nothing, si)
    end

    function PIRParForAst(b, pre, nests, red, post, orig, t, r, unique, si)
        new(b, pre, nests, red, post, orig, [t], r, unique, Dict{Symbol,Symbol}(), nothing, si)
    end
end

@doc """
Not currently used but might need it at some point.
Search a whole PIRParForAst object and replace one SymAllGen with another.
"""
function replaceParforWithDict(parfor :: PIRParForAst, gensym_map)
    parfor.body = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.body, gensym_map)
    parfor.preParFor = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.preParFor, gensym_map)
    for i = 1:length(parfor.loopNests)
        parfor.loopNests[i].lower = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.loopNests[i].lower, gensym_map)
        parfor.loopNests[i].upper = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.loopNests[i].upper, gensym_map)
        parfor.loopNests[i].step = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.loopNests[i].step, gensym_map)
    end
    for i = 1:length(parfor.reductions)
        parfor.reductions[i].reductionVarInit = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.reductions[i].reductionVarInit, gensym_map)
    end
    parfor.postParFor = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.postParFor, gensym_map)
end

@doc """
After lowering, it is necessary to make the parfor body top-level statements so that basic blocks
can be correctly identified and labels correctly found.  There is a phase in parallel IR where we 
take a PIRParForAst node and split it into a parfor_start node followed by the body as top-level
statements followed by parfor_end (also a top-level statement).
"""
type PIRParForStartEnd
    loopNests  :: Array{PIRLoopNest,1}      # holds information about the loop nests
    reductions :: Array{PIRReduction,1}     # holds information about the reductions
    instruction_count_expr
    private_vars :: Array{SymAllGen,1}
end

@doc """
State passed around while converting an AST from domain to parallel IR.
"""
type expr_state
    block_lives :: CompilerTools.LivenessAnalysis.BlockLiveness    # holds the output of liveness analysis at the block and top-level statement level
    top_level_number :: Int                          # holds the current top-level statement number...used to correlate with stmt liveness info
    # Arrays created from each other are known to have the same size. Store such correlations here.
    # If two arrays have the same dictionary value, they are equal in size.
    array_length_correlation :: Dict{SymGen,Int}
    symbol_array_correlation :: Dict{Array{SymGen,1},Int}
    lambdaInfo :: CompilerTools.LambdaHandling.LambdaInfo
    max_label :: Int # holds the max number of all LabelNodes

    # Initialize the state for parallel IR translation.
    function expr_state(bl, max_label, input_arrays)
        init_corr = Dict{SymGen,Int}()
        # For each input array, insert into the correlations table with a different value.
        for i = 1:length(input_arrays)
            init_corr[input_arrays[i]] = i
        end
        new(bl, 0, init_corr, Dict{Array{SymGen,1},Int}(), CompilerTools.LambdaHandling.LambdaInfo(), max_label)
    end
end

include("parallel-ir-stencil.jl")

@doc """
Overload of Base.show to pretty print for parfor AST nodes.
"""
function show(io::IO, pnode::ParallelAccelerator.ParallelIR.PIRParForAst)
    println(io,"")
    if pnode.instruction_count_expr != nothing
        println(io,"Instruction count estimate: ", pnode.instruction_count_expr)
    end
    if length(pnode.preParFor) > 0
        println(io,"Prestatements: ")
        for i = 1:length(pnode.preParFor)
            println(io,"    ", pnode.preParFor[i])
            if DEBUG_LVL >= 4
                dump(pnode.preParFor[i])
            end
        end
    end
    println(io,"PIR Body: ")
    for i = 1:length(pnode.body)
        println(io,"    ", pnode.body[i])
    end
    if DEBUG_LVL >= 4
        dump(pnode.body)
    end
    if length(pnode.loopNests) > 0
        println(io,"Loop Nests: ")
        for i = 1:length(pnode.loopNests)
            println(io,"    ", pnode.loopNests[i])
            if DEBUG_LVL >= 4
                dump(pnode.loopNests[i])
            end
        end
    end
    if length(pnode.reductions) > 0
        println(io,"Reductions: ")
        for i = 1:length(pnode.reductions)
            println(io,"    ", pnode.reductions[i])
        end
    end
    if length(pnode.postParFor) > 0
        println(io,"Poststatements: ")
        for i = 1:length(pnode.postParFor)
            println(io,"    ", pnode.postParFor[i])
            if DEBUG_LVL >= 4
                dump(pnode.postParFor[i])
            end
        end
    end
    if length(pnode.original_domain_nodes) > 0 && DEBUG_LVL >= 4
        println(io,"Domain nodes: ")
        for i = 1:length(pnode.original_domain_nodes)
            println(io,pnode.original_domain_nodes[i])
        end
    end
    if DEBUG_LVL >= 3
        println(io, pnode.rws)
    end
end

export PIRLoopNest, PIRReduction, from_exprs, PIRParForAst, AstWalk, PIRSetFuseLimit, PIRNumSimplify, PIRInplace, PIRRunAsTasks, PIRLimitTask, PIRReduceTasks, PIRStencilTasks, PIRFlatParfor, PIRNumThreadsMode, PIRShortcutArrayAssignment, PIRTaskGraphMode, PIRPolyhedral

@doc """
Given an array of outputs in "outs", form a return expression.
If there is only one out then the args of :return is just that expression.
If there are multiple outs then form a tuple of them and that tuple goes in :return args.
"""
function mk_return_expr(outs)
    if length(outs) == 1
        return TypedExpr(outs[1].typ, :return, outs[1])
    else
        tt = Expr(:tuple)
        tt.args = map( x -> x.typ, outs)
        temp_type = eval(tt)

        return TypedExpr(temp_type, :return, mk_tuple_expr(outs, temp_type))
    end
end

@doc """
Create an assignment expression AST node given a left and right-hand side.
The left-hand side has to be a symbol node from which we extract the type so as to type the new Expr.
"""
function mk_assignment_expr(lhs::SymAllGen, rhs, state :: expr_state)
    expr_typ = CompilerTools.LambdaHandling.getType(lhs, state.lambdaInfo)    
    dprintln(2,"mk_assignment_expr lhs type = ", typeof(lhs))
    TypedExpr(expr_typ, symbol('='), lhs, rhs)
end

function mk_assignment_expr(lhs::ANY, rhs, state :: expr_state)
    throw(string("mk_assignment_expr lhs is not of type SymAllGen, is of this type instead: ", typeof(lhs)))
end


function mk_assignment_expr(lhs :: SymbolNode, rhs)
    TypedExpr(lhs.typ, symbol('='), lhs, rhs)
end

@doc """
Only used to create fake expression to force lhs to be seen as written rather than read.
"""
function mk_untyped_assignment(lhs, rhs)
    Expr(symbol('='), lhs, rhs)
end

@doc """
Holds the information from one Domain IR :range Expr.
"""
type RangeData
    start
    skip
    last

    function RangeData(s, sk, l)
        new(s, sk, l)
    end
end

@doc """
Type used by mk_parfor_args... functions to hold information about input arrays.
"""
type InputInfo
    array                                # The name of the array.
    select_bitarrays :: Array{SymNodeGen,1}  # Empty if whole array or range, else on BitArray per dimension.
    range :: Array{RangeData,1}          # Empty if whole array, else one RangeData per dimension.
    range_offset :: Array{SymNodeGen,1}  # New temp variables to hold offset from iteration space for each dimension.
    elementTemp                          # New temp variable to hold the value of this array/range at the current point in iteration space.
    pre_offsets :: Array{Expr,1}         # Assignments that go in the pre-statements that hold range offsets for each dimension.
    rangeconds :: Expr                   # if selecting based on bitarrays, conditional for selecting elements

    function InputInfo()
        new(nothing, Array{SymGen,1}[], RangeData[], SymGen[], nothing, Expr[], Expr(:noop))
    end
end

@doc """
Compute size of a range.
"""
function rangeSize(start, skip, last)
    # TODO: do something with skip!
    return last - start + 1
end

@doc """
Create an expression whose value is the length of the input array.
"""
function mk_arraylen_expr(x :: SymAllGen, dim :: Int64)
    TypedExpr(Int64, :call, TopNode(:arraysize), :($x), dim)
end

@doc """
Create an expression whose value is the length of the input array.
"""
function mk_arraylen_expr(x :: InputInfo, dim :: Int64)
    if dim <= length(x.range)
        #return TypedExpr(Int64, :call, mk_parallelir_ref(rangeSize), x.range[dim].start, x.range[dim].skip, x.range[dim].last)
        # TODO: do something with skip!
        return mk_add_int_expr(mk_sub_int_expr(x.range[dim].last,x.range[dim].start), 1)
    else
        return mk_arraylen_expr(x.array, dim)
    end 
end

@doc """
Create an expression that references something inside ParallelIR.
In other words, returns an expression the equivalent of ParallelAccelerator.ParallelIR.sym where sym is an input argument to this function.
"""
function mk_parallelir_ref(sym, ref_type=Function)
    #inner_call = TypedExpr(Module, :call, TopNode(:getfield), :ParallelAccelerator, QuoteNode(:ParallelIR))
    #TypedExpr(ref_type, :call, TopNode(:getfield), inner_call, QuoteNode(sym))
    TypedExpr(ref_type, :call, TopNode(:getfield), GlobalRef(ParallelAccelerator,:ParallelIR), QuoteNode(sym))
end

@doc """
Returns an expression that convert "ex" into a another type "new_type".
"""
function mk_convert(new_type, ex)
    TypedExpr(new_type, :call, TopNode(:convert), new_type, ex)
end

@doc """
Create an expression which returns the index'th element of the tuple whose name is contained in tuple_var.
"""
function mk_tupleref_expr(tuple_var, index, typ)
    TypedExpr(typ, :call, TopNode(:tupleref), tuple_var, index)
end

@doc """
Make a svec expression.
"""
function mk_svec_expr(parts...)
    TypedExpr(SimpleVector, :call, TopNode(:svec), parts...)
end

@doc """
Return an expression that allocates and initializes a 1D Julia array that has an element type specified by
"elem_type", an array type of "atype" and a "length".
"""
function mk_alloc_array_1d_expr(elem_type, atype, length)
    dprintln(2,"mk_alloc_array_1d_expr atype = ", atype, " elem_type = ", elem_type, " length = ", length, " typeof(length) = ", typeof(length))
    ret_type = TypedExpr(Type{atype}, :call1, TopNode(:apply_type), :Array, elem_type, 1)
    new_svec = TypedExpr(SimpleVector, :call, TopNode(:svec), GlobalRef(Base, :Any), GlobalRef(Base, :Int))
    #arg_types = TypedExpr((Type{Any},Type{Int}), :call1, TopNode(:tuple), :Any, :Int)

    if typeof(length) == SymbolNode
        length_expr = length
    elseif typeof(length) == Symbol
        length_expr = SymbolNode(length,Int)
    elseif typeof(length) == Int64
        length_expr = length
    else
        throw(string("Unhandled length type in mk_alloc_array_1d_expr."))
    end

    TypedExpr(
       atype,
       :call,
       TopNode(:ccall),
       QuoteNode(:jl_alloc_array_1d),
       ret_type,
       new_svec,
       #arg_types,
       atype,
       0,
       length_expr,
       0)
end

@doc """
Return an expression that allocates and initializes a 2D Julia array that has an element type specified by
"elem_type", an array type of "atype" and two dimensions of length in "length1" and "length2".
"""
function mk_alloc_array_2d_expr(elem_type, atype, length1, length2)
    dprintln(2,"mk_alloc_array_2d_expr atype = ", atype)
    ret_type  = TypedExpr(Type{atype}, :call1, TopNode(:apply_type), :Array, elem_type, 2)
    new_svec = TypedExpr(SimpleVector, :call, TopNode(:svec), GlobalRef(Base, :Any), GlobalRef(Base, :Int), GlobalRef(Base, :Int))
    #arg_types = TypedExpr((Type{Any},Type{Int},Type{Int}), :call1, TopNode(:tuple), :Any, :Int, :Int)

    TypedExpr(
       atype,
       :call,
       TopNode(:ccall),
       QuoteNode(:jl_alloc_array_2d),
       ret_type,
       new_svec,
       #arg_types,
       atype,
       0,
       SymbolNode(length1,Int),
       0,
       SymbolNode(length2,Int),
       0)
end

@doc """
Return an expression that allocates and initializes a 3D Julia array that has an element type specified by
"elem_type", an array type of "atype" and two dimensions of length in "length1" and "length2" and "length3".
"""
function mk_alloc_array_3d_expr(elem_type, atype, length1, length2, length3)
    dprintln(2,"mk_alloc_array_3d_expr atype = ", atype)
    ret_type  = TypedExpr(Type{atype}, :call1, TopNode(:apply_type), :Array, elem_type, 3)
    new_svec = TypedExpr(SimpleVector, :call, TopNode(:svec), GlobalRef(Base, :Any), GlobalRef(Base, :Int), GlobalRef(Base, :Int), GlobalRef(Base, :Int))

    TypedExpr(
       atype,
       :call,
       TopNode(:ccall),
       QuoteNode(:jl_alloc_array_3d),
       ret_type,
       new_svec,
       atype,
       0,
       SymbolNode(length1,Int),
       0,
       SymbolNode(length2,Int),
       0,
       SymbolNode(length3,Int),
       0)
end

@doc """
Returns true if the incoming type in "typ" is an array type.
"""
function isArrayType(typ)
    return (typ.name == Array.name || typ.name == BitArray.name)
end

@doc """
Returns the element type of an Array.
"""
function getArrayElemType(atyp :: DataType)
    if atyp.name == Array.name
        atyp.parameters[1]
    elseif atyp.name == BitArray.name
        Bool
    else
        assert(false)
    end
end

@doc """
Returns the element type of an Array.
"""
function getArrayElemType(array :: SymbolNode, state :: expr_state)
    return getArrayElemType(array.typ)
end

@doc """
Returns the element type of an Array.
"""
function getArrayElemType(array :: GenSym, state :: expr_state)
    atyp = CompilerTools.LambdaHandling.getType(array, state.lambdaInfo)
    return getArrayElemType(atyp)
end

@doc """
Return the number of dimensions of an Array.
"""
function getArrayNumDims(array :: SymbolNode, state :: expr_state)
    assert(array.typ.name == Array.name)
    array.typ.parameters[2]
end

@doc """
Return the number of dimensions of an Array.
"""
function getArrayNumDims(array :: GenSym, state :: expr_state)
    gstyp = CompilerTools.LambdaHandling.getType(array, state.lambdaInfo)
    assert(gstyp.name == Array.name)
    gstyp.parameters[2]
end

@doc """
Returns a :call expression to add_int for two operands.
"""
function mk_add_int_expr(op1, op2)
    return TypedExpr(Int64, :call, GlobalRef(Base, :add_int), op1, op2)
end

@doc """
Returns a :call expression to sub_int for two operands.
"""
function mk_sub_int_expr(op1, op2)
    return TypedExpr(Int64, :call, GlobalRef(Base, :sub_int), op1, op2)
end

@doc """
Make sure the index parameters to arrayref or arrayset are Int64 or SymbolNode.
"""
function augment_sn(dim :: Int64, index_vars, range_var :: Array{SymNodeGen,1})
    dprintln(3,"augment_sn dim = ", dim, " index_vars = ", index_vars, " range_var = ", range_var)
    xtyp = typeof(index_vars[dim])

    if xtyp == Int64 || xtyp == GenSym
        base = index_vars[dim]
    else
        base = SymbolNode(index_vars[dim],Int64)
    end

    dprintln(3,"pre-base = ", base)

    if dim <= length(range_var)
        base = mk_add_int_expr(base, range_var[dim])
    end

    dprintln(3,"post-base = ", base)

    return base
end

@doc """
Return an expression that corresponds to getting the index_var index from the array array_name.
If "inbounds" is true then use the faster :unsafe_arrayref call that doesn't do a bounds check.
"""
function mk_arrayref1(array_name, index_vars, inbounds, state :: expr_state, range_var :: Array{SymNodeGen,1} = SymNodeGen[])
    dprintln(3,"mk_arrayref1 typeof(index_vars) = ", typeof(index_vars))
    dprintln(3,"mk_arrayref1 array_name = ", array_name, " typeof(array_name) = ", typeof(array_name))
    elem_typ = getArrayElemType(array_name, state)
    dprintln(3,"mk_arrayref1 element type = ", elem_typ)
    dprintln(3,"mk_arrayref1 range_var = ", range_var)

    if inbounds
        fname = :unsafe_arrayref
    else
        fname = :arrayref
    end

    indsyms = [ augment_sn(x,index_vars,range_var) for x = 1:length(index_vars) ]
    dprintln(3,"mk_arrayref1 indsyms = ", indsyms)

    TypedExpr(
    elem_typ,
    :call,
    TopNode(fname),
    :($array_name),
    indsyms...)
end

@doc """
Add a local variable to the current function's lambdaInfo.
Returns a symbol node of the new variable.
"""
function createStateVar(state, name, typ, access)
    new_temp_sym = symbol(name)
    CompilerTools.LambdaHandling.addLocalVar(new_temp_sym, typ, access, state.lambdaInfo)
    return SymbolNode(new_temp_sym, typ)
end

@doc """
Create a temporary variable that is parfor private to hold the value of an element of an array.
"""
function createTempForArray(array_sn :: SymAllGen, unique_id :: Int64, state :: expr_state)
    key = toSymGen(array_sn) 
    temp_type = getArrayElemType(array_sn, state)
    return createStateVar(state, string("parallel_ir_temp_", key, "_", unique_id), temp_type, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP)
end


@doc """
Takes an existing variable whose name is in "var_name" and adds the descriptor flag ISPRIVATEPARFORLOOP to declare the
variable to be parfor loop private and eventually go in an OMP private clause.
"""
function makePrivateParfor(var_name :: Symbol, state)
    res = CompilerTools.LambdaHandling.addDescFlag(var_name, ISPRIVATEPARFORLOOP, state.lambdaInfo)
    assert(res)
end

@doc """
Return a new AST node that corresponds to setting the index_var index from the array "array_name" with "value".
The paramater "inbounds" is true if this access is known to be within the bounds of the array.
"""
function mk_arrayset1(array_name, index_vars, value, inbounds, state :: expr_state, range_var :: Array{SymNodeGen,1} = SymNodeGen[])
    dprintln(3,"mk_arrayset1 typeof(index_vars) = ", typeof(index_vars))
    dprintln(3,"mk_arrayset1 array_name = ", array_name, " typeof(array_name) = ", typeof(array_name))
    elem_typ = getArrayElemType(array_name, state)  # The type of the array reference will be the element type.
    dprintln(3,"mk_arrayset1 element type = ", elem_typ)
    dprintln(3,"mk_arrayset1 range_var = ", range_var)

    # If the access is known to be within the bounds of the array then use unsafe_arrayset to forego the boundscheck.
    if inbounds
        fname = :unsafe_arrayset
    else
        fname = :arrayset
    end

    # For each index expression in "index_vars", if it isn't an Integer literal then convert the symbol to
    # a SymbolNode containing the index expression type "Int".
    indsyms = [ augment_sn(x,index_vars,range_var) for x = 1:length(index_vars) ]
    dprintln(3,"mk_arrayset1 indsyms = ", indsyms)

    TypedExpr(
       elem_typ,
       :call,
       TopNode(fname),
       array_name,
       :($value),
       indsyms...)
end

@doc """
Returns true if all array references use singular index variables and nothing more complicated involving,
for example, addition or subtraction by a constant.
"""
function simpleIndex(dict)
    # Prepare to iterate over all the keys in the dictionary.
    kv = collect(keys(dict))
    # For each key in the dictionary.
    for k in kv
        # Get the corresponding array of seen indexing expressions.
        array_ae = dict[k]
        # For each indexing expression.
        for i = 1:length(array_ae)
            ae = array_ae[i]
            dprintln(3,"typeof(ae) = ", typeof(ae), " ae = ", ae)
            for j = 1:length(ae)
                # If the indexing expression isn't simple then return false.
                if (typeof(ae[j]) != SymbolNode &&
                    (typeof(ae[j]) != Expr ||
                    ae[j].head != :(::)   ||
                    typeof(ae[j].args[1]) != Symbol))
                    return false
                end
            end
        end
    end
    # All indexing expressions must have been fine so return true.
    return true
end

@doc """
In various places we need a SymGen type which is the union of Symbol and GenSym.
This function takes a Symbol, SymbolNode, or GenSym and return either a Symbol or GenSym.
"""
function toSymGen(x :: Symbol)
    return x
end

function toSymGen(x :: SymbolNode)
    return x.name
end

function toSymGen(x :: GenSym)
    return x
end

function toSymGen(x)
    xtyp = typeof(x)
    throw(string("Found object type ", xtyp, " for object ", x, " in toSymGen and don't know what to do with it."))
end

@doc """
Form a SymbolNode with the given typ if possible or a GenSym if that is what is passed in.
"""
function toSymNodeGen(x :: Symbol, typ)
    return SymbolNode(x, typ)
end

function toSymNodeGen(x :: SymbolNode, typ)
    return x
end

function toSymNodeGen(x :: GenSym, typ)
    return x
end

function toSymNodeGen(x, typ)
    xtyp = typeof(x)
    throw(string("Found object type ", xtyp, " for object ", x, " in toSymNodeGen and don't know what to do with it."))
end

@doc """
Returns the next usable label for the current function.
"""
function next_label(state :: expr_state)
    state.max_label = state.max_label + 1
    return state.max_label
end

@doc """
Given an array whose name is in "x", allocate a new equivalence class for this array.
"""
function addUnknownArray(x :: SymGen, state :: expr_state)
    a = collect(values(state.array_length_correlation))
    m = length(a) == 0 ? 0 : maximum(a)
    state.array_length_correlation[x] = m + 1
end

@doc """
If we somehow determine that two sets of correlations are actually the same length then merge one into the other.
"""
function merge_correlations(state, unchanging, eliminate)
    # For each array in the dictionary.
    for i in state.array_length_correlation
        # If it is in the "eliminate" class...
        if i[2] == eliminate
            # ...move it to the "unchanging" class.
            state.array_length_correlation[i[1]] = unchanging
        end
    end
    for i in state.symbol_array_correlation
        if i[2] == eliminate
            state.symbol_array_correlation[i[1]] = unchanging
        end
    end
    nothing
end

@doc """
If we somehow determine that two arrays must be the same length then 
get the equivalence classes for the two arrays and merge those equivalence classes together.
"""
function add_merge_correlations(old_sym :: SymGen, new_sym :: SymGen, state :: expr_state)
    dprintln(3, "add_merge_correlations ", old_sym, " ", new_sym, " ", state.array_length_correlation)
    old_corr = getOrAddArrayCorrelation(old_sym, state)
    new_corr = getOrAddArrayCorrelation(new_sym, state)
    merge_correlations(state, old_corr, new_corr)
    dprintln(3, "add_merge_correlations post ", state.array_length_correlation)
end

@doc """
Return a correlation set for an array.  If the array was not previously added then add it and return it.
"""
function getOrAddArrayCorrelation(x :: SymGen, state :: expr_state)
    if !haskey(state.array_length_correlation, x)
        dprintln(3,"Correlation for array not found = ", x)
        addUnknownArray(x, state)
    end
    state.array_length_correlation[x]
end

@doc """
A new array is being created with an explicit size specification in dims.
"""
function getOrAddSymbolCorrelation(array :: SymGen, state :: expr_state, dims :: Array{SymGen,1})
    if !haskey(state.symbol_array_correlation, dims)
        # We haven't yet seen this combination of dims used to create an array.
        dprintln(3,"Correlation for symbol set not found, dims = ", dims)
        if haskey(state.array_length_correlation, array)
            return state.symbol_array_correlation[dims] = state.array_length_correlation[array]
        else
            # Create a new array correlation number for this array and associate that number with the dim sizes.
            return state.symbol_array_correlation[dims] = addUnknownArray(array, state)
        end
    else
        dprintln(3,"Correlation for symbol set found, dims = ", dims)
        # We have previously seen this combination of dim sizes used to create an array so give the new
        # array the same array length correlation number as the previous one.
        return state.array_length_correlation[array] = state.symbol_array_correlation[dims]
    end
end

@doc """
If we need to generate a name and make sure it is unique then include an monotonically increasing number.
"""
function get_unique_num()
    ret = unique_num
    global unique_num = unique_num + 1
    ret
end

# ===============================================================================================================================

include("parallel-ir-mk-parfor.jl")

@doc """
The AstWalk callback function for getPrivateSet.
For each AST in a parfor body, if the node is an assignment or loop head node then add the written entity to the state.
"""
function getPrivateSetInner(x::Expr, state :: Set{SymAllGen}, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    # If the node is an assignment node or a loop head node.
    if isAssignmentNode(x) || isLoopheadNode(x)
        lhs = x.args[1]
        assert(isa(lhs, SymAllGen))
        if isa(lhs, GenSym)
            push!(state, lhs)
        else
            sname = getSName(lhs)
            red_var_start = "parallel_ir_reduction_output_"
            red_var_len = length(red_var_start)
            sstr = string(sname)
            if length(sstr) >= red_var_len
                if sstr[1:red_var_len] == red_var_start
                    # Skip this symbol if it begins with "parallel_ir_reduction_output_" signifying a reduction variable.
                    return CompilerTools.AstWalker.ASTWALK_RECURSE
                end
            end
            push!(state, sname)
        end
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function getPrivateSetInner(x::ANY, state :: Set{SymAllGen}, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
Go through the body of a parfor and collect those Symbols, GenSyms, etc. that are assigned to within the parfor except reduction variables.
"""
function getPrivateSet(body :: Array{Any,1})
    dprintln(3,"getPrivateSet")
    printBody(3, body)
    private_set = Set{SymAllGen}()
    for i = 1:length(body)
        AstWalk(body[i], getPrivateSetInner, private_set)
    end
    dprintln(3,"private_set = ", private_set)
    return private_set
end

# ===============================================================================================================================

@doc """
Convert a compressed LambdaStaticData format into the uncompressed AST format.
"""
uncompressed_ast(l::LambdaStaticData) =
isa(l.ast,Expr) ? l.ast : ccall(:jl_uncompress_ast, Any, (Any,Any), l, l.ast)

@doc """
AstWalk callback to count the number of static times that a symbol is assigne within a method.
"""
function count_assignments(x, symbol_assigns :: Dict{Symbol, Int}, top_level_number, is_top_level, read)
    if isAssignmentNode(x) || isLoopheadNode(x)
        lhs = x.args[1]
        # GenSyms don't have descriptors so no need to count their assignment.
        if !hasSymbol(lhs)
            return CompilerTools.AstWalker.ASTWALK_RECURSE
        end
        sname = getSName(lhs)
        if !haskey(symbol_assigns, sname)
            symbol_assigns[sname] = 0
        end
        symbol_assigns[sname] = symbol_assigns[sname] + 1
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE 
end

@doc """
Just call the AST walker for symbol for parallel IR nodes with no state.
"""
function pir_live_cb_def(x)
    pir_live_cb(x, nothing)
end

@doc """
Process a :lambda Expr.
"""
function from_lambda(lambda :: Expr, depth, state)
    # :lambda expression
    assert(lambda.head == :lambda)
    dprintln(4,"from_lambda starting")

    # Save the current lambdaInfo away so we can restore it later.
    save_lambdaInfo  = state.lambdaInfo
    state.lambdaInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(lambda)
    body = CompilerTools.LambdaHandling.getBody(lambda)

    # Process the lambda's body.
    dprintln(3,"state.lambdaInfo.var_defs = ", state.lambdaInfo.var_defs)
    body = get_one(from_expr(body, depth, state, false))
    dprintln(4,"from_lambda after from_expr")
    dprintln(3,"After processing lambda body = ", state.lambdaInfo)
    dprintln(3,"from_lambda: after body = ")
    printBody(3, body)

    # Count the number of static assignments per var.
    symbol_assigns = Dict{Symbol, Int}()
    AstWalk(body, count_assignments, symbol_assigns)

    # After counting static assignments, update the lambdaInfo for those vars
    # to say whether the var is assigned once or multiple times.
    CompilerTools.LambdaHandling.updateAssignedDesc(state.lambdaInfo, symbol_assigns)

    body = CompilerTools.LambdaHandling.eliminateUnusedLocals!(state.lambdaInfo, body, ParallelAccelerator.ParallelIR.AstWalk)

    # Write the lambdaInfo back to the lambda AST node.
    lambda = CompilerTools.LambdaHandling.lambdaInfoToLambdaExpr(state.lambdaInfo, body)
    dprintln(3,"new lambda = ", lambda)

    state.lambdaInfo = save_lambdaInfo

    dprintln(4,"from_lambda ending")
    return lambda
end

@doc """
Is a node an assignment expression node.
"""
function isAssignmentNode(node :: Expr)
    return node.head == :(=)
end

function isAssignmentNode(node::Any)
    return false
end

@doc """
Is a node a loophead expression node (a form of assignment).
"""
function isLoopheadNode(node :: Expr)
    return node.head == :loophead
end

function isLoopheadNode(node)
    return false
end

@doc """
Is this a parfor node not part of an assignment statement.
"""
function isBareParfor(node :: Expr)
    return node.head == :parfor
end

function isBareParfor(node)
    return false
end


function isParforAssignmentNodeInner(lhs::SymAllGen, rhs::Expr)
    if rhs.head==:parfor
        dprintln(4,"Found a parfor assignment node.")
        return true
    end
    return false
end

function isParforAssignmentNodeInner(lhs::Any, rhs::Any)
    return false
end

@doc """
Is a node an assignment expression with a parfor node as the right-hand side.
"""
function isParforAssignmentNode(node::Expr)
    dprintln(4,"isParforAssignmentNode")
    dprintln(4,node)

    if isAssignmentNode(node)
        assert(length(node.args) >= 2)
        lhs = node.args[1]
        dprintln(4,lhs)
        rhs = node.args[2]
        dprintln(4,rhs)
        return isParforAssignmentNodeInner(lhs, rhs)
    else
        dprintln(4,"node is not an assignment Expr")
    end

    return false
end

function isParforAssignmentNode(node::Any)
    dprintln(4,"node is not an Expr")
    return false
end

@doc """
Get the parfor object from either a bare parfor or one part of an assignment.
"""
function getParforNode(node)
    if isBareParfor(node)
        return node.args[1]
    else
        return node.args[2].args[1]
    end
end

@doc """
Get the right-hand side of an assignment expression.
"""
function getRhsFromAssignment(assignment)
    assignment.args[2]
end

@doc """
Get the left-hand side of an assignment expression.
"""
function getLhsFromAssignment(assignment)
    assignment.args[1]
end

@doc """
Returns true if the domain operation mapped to this parfor has the property that the iteration space
is identical to the dimenions of the inputs.
"""
function iterations_equals_inputs(node :: ParallelAccelerator.ParallelIR.PIRParForAst)
    assert(length(node.original_domain_nodes) > 0)

    first_domain_node = node.original_domain_nodes[1]
    first_type = first_domain_node.operation
    if first_type == :map   ||
        first_type == :map!  ||
        first_type == :mmap  ||
        first_type == :mmap! ||
        first_type == :reduce
        dprintln(3,"iteration count of node equals length of inputs")
        return true
    else
        dprintln(3,"iteration count of node does not equal length of inputs")
        return false
    end
end

@doc """
Returns a Set with all the arrays read by this parfor.
"""
function getInputSet(node :: ParallelAccelerator.ParallelIR.PIRParForAst)
    ret = Set(collect(keys(node.rws.readSet.arrays)))
    dprintln(3,"Input set = ", ret)
    ret
end

@doc """
Get the real outputs of an assignment statement.
If the assignment expression is normal then the output is just the left-hand side.
If the assignment expression is augmented with a FusionSentinel then the real outputs
are the 4+ arguments to the expression.
"""
function getLhsOutputSet(lhs, assignment)
    ret = Set()

    typ = typeof(lhs)

    # Created by fusion.
    if isFusionAssignment(assignment)
        # For each real output.
        for i = 4:length(assignment.args)
            assert(typeof(assignment.args[i]) == SymbolNode)
            dprintln(3,"getLhsOutputSet FusionSentinal assignment with symbol ", assignment.args[i].name)
            # Add to output set.
            push!(ret,assignment.args[i].name)
        end
    else
        # LHS could be Symbol or SymbolNode.
        if typ == SymbolNode
            push!(ret,lhs.name)
            dprintln(3,"getLhsOutputSet SymbolNode with symbol ", lhs.name)
        elseif typ == Symbol
            push!(ret,lhs)
            dprintln(3,"getLhsOutputSet symbol ", lhs)
        else
            dprintln(0,"Unknown LHS type ", typ, " in getLhsOutputSet.")
        end
    end

    ret
end

@doc """
Return an expression which creates a tuple.
"""
function mk_tuple_expr(tuple_fields, typ)
    # Tuples are formed with a call to :tuple.
    TypedExpr(typ, :call, TopNode(:tuple), tuple_fields...)
end

@doc """
Forms a SymbolNode given a symbol in "name" and get the type of that symbol from the incoming dictionary "sym_to_type".
"""
function nameToSymbolNode(name :: Symbol, sym_to_type)
    return SymbolNode(name, sym_to_type[name])
end

function nameToSymbolNode(name :: GenSym, sym_to_type)
    return name
end

function nameToSymbolNode(name, sym_to_type)
    throw(string("Unknown name type ", typeof(name), " passed to nameToSymbolNode."))
end

function getAliasMap(loweredAliasMap, sym)
    if haskey(loweredAliasMap, sym)
        return loweredAliasMap[sym]
    else
        return sym
    end
end

function create_merged_output_from_map(output_map, unique_id, state, sym_to_type, loweredAliasMap)
    dprintln(3,"create_merged_output_from_map, output_map = ", output_map, " sym_to_type = ", sym_to_type)
    # If there are no outputs then return nothing.
    if length(output_map) == 0
        return (nothing, [], true, nothing, [])    
    end

    # If there is only one output then all we need is the symbol to return.
    if length(output_map) == 1
        for i in output_map
            new_lhs = nameToSymbolNode(i[1], sym_to_type)
            new_rhs = nameToSymbolNode(getAliasMap(loweredAliasMap, i[2]), sym_to_type)
            return (new_lhs, [new_lhs], true, [new_rhs])
        end
    end

    lhs_order = Union{SymbolNode,GenSym}[]
    rhs_order = Union{SymbolNode,GenSym}[]
    for i in output_map
        push!(lhs_order, nameToSymbolNode(i[1], sym_to_type))
        push!(rhs_order, nameToSymbolNode(getAliasMap(loweredAliasMap, i[2]), sym_to_type))
    end
    num_map = length(lhs_order)

    # Multiple outputs.

    # First, form the type of the tuple for those multiple outputs.
    tt = Expr(:tuple)
    for i = 1:num_map
        push!(tt.args, CompilerTools.LambdaHandling.getType(rhs_order[i], state.lambdaInfo))
    end
    temp_type = eval(tt)

    ( createRetTupleType(lhs_order, unique_id, state), lhs_order, false, rhs_order )
end

@doc """
Pull the information from the inner lambda into the outer lambda.
"""
function mergeLambdaIntoOuterState(state, inner_lambda :: Expr)
    inner_lambdaInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(inner_lambda)
    dprintln(3,"mergeLambdaIntoOuterState")
    dprintln(3,"state.lambdaInfo = ", state.lambdaInfo)
    dprintln(3,"inner_lambdaInfo = ", inner_lambdaInfo)
    CompilerTools.LambdaHandling.mergeLambdaInfo(state.lambdaInfo, inner_lambdaInfo)
end

# Create a variable for a left-hand side of an assignment to hold the multi-output tuple of a parfor.
function createRetTupleType(rets :: Array{Union{SymbolNode, GenSym},1}, unique_id :: Int64, state :: expr_state)
    # Form the type of the tuple var.
    tt = Expr(:tuple)
    tt.args = map( x -> CompilerTools.LambdaHandling.getType(x, state.lambdaInfo), rets)
    temp_type = eval(tt)

    new_temp_name  = string("parallel_ir_ret_holder_",unique_id)
    new_temp_snode = SymbolNode(symbol(new_temp_name), temp_type)
    dprintln(3, "Creating variable for multiple return from parfor = ", new_temp_snode)
    CompilerTools.LambdaHandling.addLocalVar(new_temp_name, temp_type, ISASSIGNEDONCE | ISCONST | ISASSIGNED, state.lambdaInfo)

    new_temp_snode
end

# Takes the output of two parfors and merges them while eliminating outputs from
# the previous parfor that have their only use in the current parfor.
function create_arrays_assigned_to_by_either_parfor(arrays_assigned_to_by_either_parfor :: Array{Symbol,1}, allocs_to_eliminate, unique_id, state, sym_to_typ)
    dprintln(3,"create_arrays_assigned_to_by_either_parfor arrays_assigned_to_by_either_parfor = ", arrays_assigned_to_by_either_parfor)
    dprintln(3,"create_arrays_assigned_to_by_either_parfor allocs_to_eliminate = ", allocs_to_eliminate, " typeof(allocs) = ", typeof(allocs_to_eliminate))

    # This is those outputs of the prev parfor which don't die during cur parfor.
    prev_minus_eliminations = Symbol[]
    for i = 1:length(arrays_assigned_to_by_either_parfor)
        if !in(arrays_assigned_to_by_either_parfor[i], allocs_to_eliminate)
            push!(prev_minus_eliminations, arrays_assigned_to_by_either_parfor[i])
        end
    end
    dprintln(3,"create_arrays_assigned_to_by_either_parfor: outputs from previous parfor that continue to live = ", prev_minus_eliminations)

    # Create an array of SymbolNode for real values to assign into.
    all_array = map(x -> SymbolNode(x,sym_to_typ[x]), prev_minus_eliminations)
    dprintln(3,"create_arrays_assigned_to_by_either_parfor: all_array = ", all_array, " typeof(all_array) = ", typeof(all_array))

    # If there is only one such value then the left side is just a simple SymbolNode.
    if length(all_array) == 1
        return (all_array[1], all_array, true)
    end

    # Create a new var to hold multi-output tuple.
    (createRetTupleType(all_array, unique_id, state), all_array, false)
end

function getAllAliases(input :: Set{SymGen}, aliases :: Dict{SymGen, SymGen})
    dprintln(3,"getAllAliases input = ", input, " aliases = ", aliases)
    out = Set()

    for i in input
        dprintln(3, "input = ", i)
        push!(out, i)
        cur = i
        while haskey(aliases, cur)
            cur = aliases[cur]
            dprintln(3, "cur = ", cur)
            push!(out, cur)
        end
    end

    dprintln(3,"getAllAliases out = ", out)
    return out
end

function isAllocation(expr :: Expr)
    return expr.head == :call && 
    expr.args[1] == TopNode(:ccall) && 
    (expr.args[2] == QuoteNode(:jl_alloc_array_1d) || expr.args[2] == QuoteNode(:jl_alloc_array_2d) || expr.args[2] == QuoteNode(:jl_alloc_array_3d))
end

function isAllocation(expr)
    return false
end

# Takes one statement in the preParFor of a parfor and a set of variables that we've determined we can eliminate.
# Returns true if this statement is an allocation of one such variable.
function is_eliminated_allocation_map(x :: Expr, all_aliased_outputs :: Set)
    dprintln(4,"is_eliminated_allocation_map: x = ", x, " typeof(x) = ", typeof(x), " all_aliased_outputs = ", all_aliased_outputs)
    dprintln(4,"is_eliminated_allocation_map: head = ", x.head)
    if x.head == symbol('=')
        assert(typeof(x.args[1]) == SymbolNode)
        lhs = x.args[1]
        rhs = x.args[2]
        if isAllocation(rhs)
            dprintln(4,"is_eliminated_allocation_map: lhs = ", lhs)
            if !in(lhs.name, all_aliased_outputs)
                dprintln(4,"is_eliminated_allocation_map: this will be removed => ", x)
                return true
            end
        end
    end

    return false
end

function is_eliminated_allocation_map(x, all_aliased_outputs :: Set)
    dprintln(4,"is_eliminated_allocation_map: x = ", x, " typeof(x) = ", typeof(x), " all_aliased_outputs = ", all_aliased_outputs)
    return false
end

function is_dead_arrayset(x, all_aliased_outputs :: Set)
    if isArraysetCall(x)
        array_to_set = x.args[2]
        if !in(toSymGen(array_to_set), all_aliased_outputs)
            return true
        end
    end

    return false
end

@doc """
Holds data for modifying arrayset calls.
"""
type sub_arrayset_data
    arrays_set_in_cur_body #remove_arrayset
    output_items_with_aliases
end

@doc """
Is a node an arrayset node?
"""
function isArrayset(x)
    if x == TopNode(:arrayset) || x == TopNode(:unsafe_arrayset)
        return true
    end
    return false
end

@doc """
Is a node an arrayref node?
"""
function isArrayref(x)
    if x == TopNode(:arrayref) || x == TopNode(:unsafe_arrayref)
        return true
    end
    return false
end

@doc """
Is a node a call to arrayset.
"""
function isArraysetCall(x :: Expr)
    return x.head == :call && isArrayset(x.args[1])
end

function isArraysetCall(x)
    return false
end

@doc """
Is a node a call to arrayref.
"""
function isArrayrefCall(x :: Expr)
    return x.head == :call && isArrayref(x.args[1])
end

function isArrayrefCall(x)
    return false
end

@doc """
AstWalk callback that does the work of substitute_arrayset on a node-by-node basis.
"""
function sub_arrayset_walk(x::Expr, cbd, top_level_number, is_top_level, read)
    use_dbg_level = 3
    dprintln(use_dbg_level,"sub_arrayset_walk ", x, " ", cbd.arrays_set_in_cur_body, " ", cbd.output_items_with_aliases)

    dprintln(use_dbg_level,"sub_arrayset_walk is Expr")
    if x.head == :call
        dprintln(use_dbg_level,"sub_arrayset_walk is :call")
        if x.args[1] == TopNode(:arrayset) || x.args[1] == TopNode(:unsafe_arrayset)
            # Here we have a call to arrayset.
            dprintln(use_dbg_level,"sub_arrayset_walk is :arrayset")
            array_name = x.args[2]
            value      = x.args[3]
            index      = x.args[4]
            assert(isa(array_name, SymNodeGen))
            # If the array being assigned to is in temp_map.
            if in(toSymGen(array_name), cbd.arrays_set_in_cur_body)
                return nothing
            elseif !in(toSymGen(array_name), cbd.output_items_with_aliases)
                return nothing
            else
                dprintln(use_dbg_level,"sub_arrayset_walk array_name will not substitute ", array_name)
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function sub_arrayset_walk(x::ANY, cbd, top_level_number, is_top_level, read)
    dprintln(use_dbg_level,"sub_arrayset_walk ", x, " ", cbd.arrays_set_in_cur_body, " ", cbd.output_items_with_aliases)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
Modify the body of a parfor.
temp_map holds a map of array names whose arraysets should be turned into a mapped variable instead of the arrayset. a[i] = b. a=>c. becomes c = b
map_for_non_eliminated holds arrays for which we need to add a variable to save the value but we can't eiminate the arrayset. a[i] = b. a=>c. becomes c = a[i] = b
    map_drop_arrayset drops the arrayset without replacing with a variable.  This is because a variable was previously added here with a map_for_non_eliminated case.
    a[i] = b. becomes b
"""
function substitute_arrayset(x, arrays_set_in_cur_body, output_items_with_aliases)
    dprintln(3,"substitute_arrayset ", x, " ", arrays_set_in_cur_body, " ", output_items_with_aliases)
    # Walk the AST and call sub_arrayset_walk for each node.
    return AstWalk(x, sub_arrayset_walk, sub_arrayset_data(arrays_set_in_cur_body, output_items_with_aliases))
end

@doc """
Get the variable which holds the length of the first input array to a parfor.
"""
function getFirstArrayLens(prestatements, num_dims)
    ret = Any[]

    # Scan the prestatements and find the assignment nodes.
    # If it is an assignment from arraysize.
    for i = 1:length(prestatements)
        x = prestatements[i]
        if (typeof(x) == Expr) && (x.head == symbol('='))
            lhs = x.args[1]
            rhs = x.args[2]
            if (typeof(lhs) == SymbolNode) && (typeof(rhs) == Expr) && (rhs.head == :call) && (rhs.args[1] == TopNode(:arraysize))
                push!(ret, lhs)
            end
        end
    end
    assert(length(ret) == num_dims)
    ret
end

@doc """
Holds the data for substitute_cur_body AST walk.
"""
type cur_body_data
    temp_map  :: Dict{SymGen, SymNodeGen}    # Map of array name to temporary.  Use temporary instead of arrayref of the array name.
    index_map :: Dict{SymGen, SymGen}        # Map index variables from parfor being fused to the index variables of the parfor it is being fused with.
    arrays_set_in_cur_body :: Set{SymGen}    # Used as output.  Collects the arrays set in the current body.
    replace_array_name_in_arrayset :: Dict{SymGen, SymGen}  # Map from one array to another.  Replace first array with second when used in arrayset context.
    state :: expr_state
end

@doc """
AstWalk callback that does the work of substitute_cur_body on a node-by-node basis.
"""
function sub_cur_body_walk(x :: ANY, cbd :: cur_body_data, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    dbglvl = 3
    dprintln(dbglvl,"sub_cur_body_walk ", x)
    xtype = typeof(x)

    if xtype == Expr
        dprintln(dbglvl,"sub_cur_body_walk xtype is Expr")
        if x.head == :call
            dprintln(dbglvl,"sub_cur_body_walk xtype is call")
            # Found a call to arrayref.
            if x.args[1] == TopNode(:arrayref) || x.args[1] == TopNode(:unsafe_arrayref)
                dprintln(dbglvl,"sub_cur_body_walk xtype is arrayref")
                array_name = x.args[2]
                index      = x.args[3]
                assert(isa(array_name, SymNodeGen))
                lowered_array_name = toSymGen(array_name)
                assert(isa(lowered_array_name, SymGen))
                dprintln(dbglvl, "array_name = ", array_name, " index = ", index, " lowered_array_name = ", lowered_array_name)
                # If the array name is in cbd.temp_map then replace the arrayref call with the mapped variable.
                if haskey(cbd.temp_map, lowered_array_name)
                    dprintln(dbglvl,"sub_cur_body_walk IS substituting ", cbd.temp_map[lowered_array_name])
                    return cbd.temp_map[lowered_array_name]
                end
            elseif x.args[1] == TopNode(:arrayset) || x.args[1] == TopNode(:unsafe_arrayset)
                array_name = x.args[2]
                assert(isa(array_name, SymNodeGen))
                push!(cbd.arrays_set_in_cur_body, toSymGen(array_name))
                if haskey(cbd.replace_array_name_in_arrayset, toSymGen(array_name))
                    new_symgen = cbd.replace_array_name_in_arrayset[toSymGen(array_name)]
                    x.args[2]  = toSymNodeGen(new_symgen, CompilerTools.LambdaHandling.getType(new_symgen, cbd.state.lambdaInfo))
                end
            end
        end
    elseif xtype == Symbol
        dprintln(dbglvl,"sub_cur_body_walk xtype is Symbol")
        if haskey(cbd.index_map, x)
            # Detected the use of an index variable.  Change it to the first parfor's index variable.
            dprintln(dbglvl,"sub_cur_body_walk IS substituting ", cbd.index_map[x])
            return cbd.index_map[x]
        end
    elseif xtype == SymbolNode
        dprintln(dbglvl,"sub_cur_body_walk xtype is SymbolNode")
        if haskey(cbd.index_map, x.name)
            # Detected the use of an index variable.  Change it to the first parfor's index variable.
            dprintln(dbglvl,"sub_cur_body_walk IS substituting ", cbd.index_map[x.name])
            x.name = cbd.index_map[x.name]
            return x
        end
    end
    dprintln(dbglvl,"sub_cur_body_walk not substituting")

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
Make changes to the second parfor body in the process of parfor fusion.
temp_map holds array names for which arrayrefs should be converted to a variable.  a[i].  a=>b. becomes b
    index_map holds maps between index variables.  The second parfor is modified to use the index variable of the first parfor.
    arrays_set_in_cur_body           # Used as output.  Collects the arrays set in the current body.
    replace_array_name_in_arrayset   # Map from one array to another.  Replace first array with second when used in arrayset context.
"""
function substitute_cur_body(x, 
    temp_map :: Dict{SymGen, SymNodeGen}, 
    index_map :: Dict{SymGen, SymGen}, 
    arrays_set_in_cur_body :: Set{SymGen}, 
    replace_array_name_in_arrayset :: Dict{SymGen, SymGen},
    state :: expr_state)
    dprintln(3,"substitute_cur_body ", x)
    dprintln(3,"temp_map = ", temp_map)
    dprintln(3,"index_map = ", index_map)
    dprintln(3,"arrays_set_in_cur_body = ", arrays_set_in_cur_body)
    dprintln(3,"replace_array_name_in_array_set = ", replace_array_name_in_arrayset)
    # Walk the AST and call sub_cur_body_walk for each node.
    return DomainIR.AstWalk(x, sub_cur_body_walk, cur_body_data(temp_map, index_map, arrays_set_in_cur_body, replace_array_name_in_arrayset, state))
end

@doc """
Returns true if the input node is an assignment node where the right-hand side is a call to arraysize.
"""
function is_eliminated_arraylen(x)
    dprintln(3,"is_eliminated_arraylen ", x)
    if typeof(x) == Expr
        dprintln(3,"is_eliminated_arraylen is Expr")
        if x.head == symbol('=')
            assert(typeof(x.args[1]) == SymbolNode)
            rhs = x.args[2]
            if isa(rhs, Expr) && rhs.head == :call
                dprintln(3,"is_eliminated_arraylen is :call")
                if rhs.args[1] == TopNode(:arraysize)
                    dprintln(3,"is_eliminated_arraylen is :arraysize")
                    return true
                end
            end
        end
    end
    return false
end

@doc """
AstWalk callback that does the work of substitute_arraylen on a node-by-node basis.
replacement is an array containing the length of the dimensions of the arrays a part of this parfor.
If we see a call to create an array, replace the length params with those in the common set in "replacement".
"""
function sub_arraylen_walk(x, replacement, top_level_number, is_top_level, read)
    dprintln(4,"sub_arraylen_walk ", x)
    if typeof(x) == Expr
        if x.head == symbol('=')
            rhs = x.args[2]
            if isa(rhs, Expr) && rhs.head == :call
                if rhs.args[1] == TopNode(:ccall)
                    if rhs.args[2] == QuoteNode(:jl_alloc_array_1d)
                        rhs.args[7] = replacement[1]
                    elseif rhs.args[2] == QuoteNode(:jl_alloc_array_2d)
                        rhs.args[7] = replacement[1]
                        rhs.args[9] = replacement[2]
                    end
                end
            end
        end
    end
    dprintln(4,"sub_arraylen_walk not substituting")

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
replacement is an array containing the length of the dimensions of the arrays a part of this parfor.
If we see a call to create an array, replace the length params with those in the common set in "replacement".
"""
function substitute_arraylen(x, replacement)
    dprintln(3,"substitute_arraylen ", x, " ", replacement)
    # Walk the AST and call sub_arraylen_walk for each node.
    return DomainIR.AstWalk(x, sub_arraylen_walk, replacement)
end

fuse_limit = -1
@doc """
Control how many parfor can be fused for testing purposes.
    -1 means fuse all possible parfors.
    0  means don't fuse any parfors.
    1+ means fuse the specified number of parfors but then stop fusing beyond that.
"""
function PIRSetFuseLimit(x)
    global fuse_limit = x
end

rearrange_passes = 2
@doc """
Specify the number of passes over the AST that do things like hoisting and other rearranging to maximize fusion.
"""
function PIRNumSimplify(x)
    global rearrange_passes = x
end

@doc """
Add to the map of symbol names to types.
"""
function rememberTypeForSym(sym_to_type :: Dict{SymGen, DataType}, sym :: SymGen, typ :: DataType)
    if typ == Any
        dprintln(0, "rememberTypeForSym: sym = ", sym, " typ = ", typ)
    end
    assert(typ != Any)
    sym_to_type[sym] = typ
end

@doc """
Just used to hold a spot in an array to indicate the this is a special assignment expression with embedded real array output names from a fusion.
"""
type FusionSentinel
end

@doc """
Check if an assignement is a fusion assignment.
    In regular assignments, there are only two args, the left and right hand sides.
    In fusion assignments, we introduce a third arg that is marked by an object of FusionSentinel type.
"""
function isFusionAssignment(x :: Expr)
    if x.head != symbol('=')
        return false
    elseif length(x.args) <= 2
        return false
    else
        assert(typeof(x.args[3]) == FusionSentinel)
        return true
    end
end

@doc """
Returns true if any variable in the collection "vars" is used in any statement whose top level number is in "top_level_numbers".
    We use expr_state "state" to get the block liveness information from which we use "def" and "use" to determine if a variable
        usage is present.
"""
function isSymbolsUsed(vars, top_level_numbers :: Array{Int,1}, state)
    dprintln(3,"isSymbolsUsed: vars = ", vars, " typeof(vars) = ", typeof(vars), " top_level_numbers = ", top_level_numbers)
    bl = state.block_lives

    for i in top_level_numbers
        tls = CompilerTools.LivenessAnalysis.find_top_number(i, bl)
        assert(tls != nothing)

        for v in vars
            if in(v, tls.def)
                dprintln(3, "isSymbolsUsed: ", v, " defined in statement ", i)
                return true
            elseif in(v, tls.use)
                dprintln(3, "isSymbolsUsed: ", v, " used in statement ", i)
                return true
            end
        end
    end

    dprintln(3, "isSymbolsUsed: ", vars, " not used in statements ", top_level_numbers)
    return false
end

@doc """
Get the equivalence class of the first array who length is extracted in the pre-statements of the specified "parfor".
"""
function getParforCorrelation(parfor, state)
    if length(parfor.preParFor) == 0
        return nothing
    end
    # FIXME: is this reliable?? -- PaulLiu
    for i in 1:length(parfor.preParFor)
        first_stmt = parfor.preParFor[i]
        if (typeof(first_stmt) == Expr) && (first_stmt.head == symbol('='))
            rhs = first_stmt.args[2]
            if (typeof(rhs) == Expr) && (rhs.head == :call)
                if (rhs.args[1] == TopNode(:arraysize)) && (isa(rhs.args[2], SymNodeGen))
                    dprintln(3,"Getting parfor array correlation for array = ", rhs.args[2])
                    return getOrAddArrayCorrelation(toSymGen(rhs.args[2]), state) 
                end
            end
        end
    end
    return nothing
end

@doc """
Creates a mapping between variables on the left-hand side of an assignment where the right-hand side is a parfor
and the arrays or scalars in that parfor that get assigned to the corresponding parts of the left-hand side.
Returns a tuple where the first element is a map for arrays between left-hand side and parfor and the second
element is a map for reduction scalars between left-hand side and parfor.
is_multi is true if the assignment is a fusion assignment.
parfor_assignment is the AST of the whole expression.
the_parfor is the PIRParForAst type part of the incoming assignment.
sym_to_type is an out parameter that maps symbols in the output mapping to their types.
"""
function createMapLhsToParfor(parfor_assignment, the_parfor, is_multi :: Bool, sym_to_type :: Dict{SymGen, DataType}, state :: expr_state)
    map_lhs_post_array     = Dict{SymGen, SymGen}()
    map_lhs_post_reduction = Dict{SymGen, SymGen}()

    if is_multi
        last_post = the_parfor.postParFor[end]
        assert(isa(last_post, Array)) 
        dprintln(3,"multi postParFor = ", the_parfor.postParFor, " last_post = ", last_post)

        # In our special AST node format for assignment to make fusion easier, args[3] is a FusionSentinel node
        # and additional args elements are the real symbol to be assigned to in the left-hand side.
        for i = 4:length(parfor_assignment.args)
            corresponding_elem = last_post[i-3]

            assert(isa(parfor_assignment.args[i], SymNodeGen))
            rememberTypeForSym(sym_to_type, toSymGen(parfor_assignment.args[i]), CompilerTools.LambdaHandling.getType(parfor_assignment.args[i], state.lambdaInfo))
            rememberTypeForSym(sym_to_type, toSymGen(corresponding_elem), CompilerTools.LambdaHandling.getType(corresponding_elem, state.lambdaInfo))
            if isArrayType(CompilerTools.LambdaHandling.getType(parfor_assignment.args[i], state.lambdaInfo))
                # For fused parfors, the last post statement is a tuple variable.
                # That tuple variable is declared in the previous statement (end-1).
                # The statement is an Expr with head == :call and top(:tuple) as the first arg.
                # So, the first member of the tuple is at offset 2 which corresponds to index 4 of this loop, ergo the "i-2".
                map_lhs_post_array[toSymGen(parfor_assignment.args[i])]     = toSymGen(corresponding_elem)
            else
                map_lhs_post_reduction[toSymGen(parfor_assignment.args[i])] = toSymGen(corresponding_elem)
            end
        end
    else
        # There is no mapping if this isn't actually an assignment statement but really a bare parfor.
        if !isBareParfor(parfor_assignment)
            lhs_pa = getLhsFromAssignment(parfor_assignment)
            ast_lhs_pa_typ = typeof(lhs_pa)
            lhs_pa_typ = CompilerTools.LambdaHandling.getType(lhs_pa, state.lambdaInfo)
            if isa(lhs_pa, SymNodeGen)
                ppftyp = typeof(the_parfor.postParFor[end]) 
                assert(isa(the_parfor.postParFor[end], SymNodeGen))
                rememberTypeForSym(sym_to_type, toSymGen(lhs_pa), lhs_pa_typ)
                rhs = the_parfor.postParFor[end]
                rememberTypeForSym(sym_to_type, toSymGen(rhs), CompilerTools.LambdaHandling.getType(rhs, state.lambdaInfo))

                if isArrayType(lhs_pa_typ)
                    map_lhs_post_array[toSymGen(lhs_pa)]     = toSymGen(the_parfor.postParFor[end])
                else
                    map_lhs_post_reduction[toSymGen(lhs_pa)] = toSymGen(the_parfor.postParFor[end])
                end
            elseif typeof(lhs_pa) == Symbol
                throw(string("lhs_pa as a symbol no longer supported"))
            else
                dprintln(3,"typeof(lhs_pa) = ", typeof(lhs_pa))
                assert(false)
            end
        end
    end

    map_lhs_post_array, map_lhs_post_reduction
end

@doc """
Given an "input" Symbol, use that Symbol as key to a dictionary.  While such a Symbol is present
in the dictionary replace it with the corresponding value from the dict.
"""
function fullyLowerAlias(dict :: Dict{SymGen, SymGen}, input :: SymGen)
    while haskey(dict, input)
        input = dict[input]
    end
    input
end

@doc """
Take a single-step alias map, e.g., a=>b, b=>c, and create a lowered dictionary, a=>c, b=>c, that
maps each array to the transitively lowered array.
"""
function createLoweredAliasMap(dict1)
    ret = Dict{SymGen, SymGen}()

    for i in dict1
        ret[i[1]] = fullyLowerAlias(dict1, i[2])
    end

    ret
end

run_as_tasks = 0
@doc """
Debugging feature to specify the number of tasks to create and to stop thereafter.
"""
function PIRRunAsTasks(x)
    global run_as_tasks = x
end

@doc """
Returns a single element of an array if there is only one or the array otherwise.
"""
function oneIfOnly(x)
    if isa(x,Array) && length(x) == 1
        return x[1]
    else
        return x
    end
end

@doc """
Test whether we can fuse the two most recent parfor statements and if so to perform that fusion.
"""
function fuse(body, body_index, cur, state)
    global fuse_limit
    prev = body[body_index]

    # Handle the debugging case where we want to limit the amount of parfor fusion to a certain number.
    if fuse_limit == 0
        return false
    end
    if fuse_limit > 0
        global fuse_limit = fuse_limit - 1
    end

    dprintln(2, "Testing if fusion is possible.")
    prev_parfor = getParforNode(prev)
    cur_parfor  = getParforNode(cur)

    sym_to_type   = Dict{SymGen, DataType}()

    dprintln(2, "prev = ", prev)
    dprintln(2, "cur = ", cur)
    dprintln(2, "prev.typ = ", prev.typ)
    dprintln(2, "cur.typ = ", cur.typ)

    prev_assignment = isAssignmentNode(prev)
    cur_assignment  = isAssignmentNode(cur)

    cur_input_set = getInputSet(cur_parfor)
    dprintln(2, "cur_input_set = ", cur_input_set)
    first_in = first(cur_input_set)
    out_correlation = getParforCorrelation(prev_parfor, state)
    if out_correlation == nothing
        return false
    end
    in_correlation  = state.array_length_correlation[first_in]
    if in_correlation == nothing
        return false
    end
    dprintln(3,"first_in = ", first_in)
    dprintln(3,"Fusion correlations ", out_correlation, " ", in_correlation)

    is_prev_multi = isFusionAssignment(prev)
    is_cur_multi  = isFusionAssignment(cur)

    prev_num_dims = length(prev_parfor.loopNests)
    cur_num_dims  = length(cur_parfor.loopNests)

    map_prev_lhs_post, map_prev_lhs_reduction = createMapLhsToParfor(prev, prev_parfor, is_prev_multi, sym_to_type, state)
    map_prev_lhs_all = merge(map_prev_lhs_post, map_prev_lhs_reduction)
    map_cur_lhs_post,  map_cur_lhs_reduction  = createMapLhsToParfor(cur,  cur_parfor,  is_cur_multi,  sym_to_type, state)
    map_cur_lhs_all  = merge(map_cur_lhs_post, map_cur_lhs_reduction)
    prev_output_arrays = collect(values(map_prev_lhs_post))
    prev_output_reduce = collect(values(map_prev_lhs_reduction))
    cur_output_arrays  = collect(values(map_cur_lhs_post))
    cur_output_reduce  = collect(values(map_cur_lhs_reduction))

    merge!(prev_parfor.array_aliases, map_prev_lhs_post)
    assert(length(cur_parfor.array_aliases) == 0)

    loweredAliasMap = createLoweredAliasMap(prev_parfor.array_aliases)

    dprintln(3, "map_prev_lhs_post = ", map_prev_lhs_post)
    dprintln(3, "map_prev_lhs_reduction = ", map_prev_lhs_reduction)
    dprintln(3, "map_cur_lhs_post = ", map_cur_lhs_post)
    dprintln(3, "map_cur_lhs_reduction = ", map_cur_lhs_reduction)
    dprintln(3, "sym_to_type = ", sym_to_type)
    reduction_var_used = isSymbolsUsed(map_prev_lhs_reduction, cur_parfor.top_level_number, state)
    dprintln(3, "reduction_var_used = ", reduction_var_used)
    prev_iei = iterations_equals_inputs(prev_parfor)
    cur_iei  = iterations_equals_inputs(cur_parfor)
    dprintln(3, "iterations_equals_inputs prev and cur = ", prev_iei, " ", cur_iei)
    dprintln(3, "loweredAliasMap = ", loweredAliasMap)

    if prev_iei &&
        cur_iei  &&
        out_correlation == in_correlation &&
        !reduction_var_used &&
        prev_parfor.simply_indexed &&
        cur_parfor.simply_indexed   
        assert(prev_num_dims == cur_num_dims)

        dprintln(3, "Fusion will happen here.")

        # Get the top-level statement for the previous parfor.
        prev_stmt_live_first = CompilerTools.LivenessAnalysis.find_top_number(prev_parfor.top_level_number[1], state.block_lives)
        assert(prev_stmt_live_first != nothing)
        dprintln(2,"Prev parfor first = ", prev_stmt_live_first)
        prev_stmt_live_last = CompilerTools.LivenessAnalysis.find_top_number(prev_parfor.top_level_number[end], state.block_lives)
        assert(prev_stmt_live_last != nothing)
        dprintln(2,"Prev parfor last = ", prev_stmt_live_last)

        # Get the top-level statement for the current parfor.
        assert(length(cur_parfor.top_level_number) == 1)
        cur_stmt_live  = CompilerTools.LivenessAnalysis.find_top_number(cur_parfor.top_level_number[1],  state.block_lives)
        assert(cur_stmt_live != nothing)
        dprintln(2,"Cur parfor = ", cur_stmt_live)

        # Get the variables live after the previous parfor.
        live_in_prev = prev_stmt_live_first.live_in
        def_prev     = prev_stmt_live_first.def
        dprintln(2,"live_in_prev = ", live_in_prev, " def_prev = ", def_prev)

        # Get the variables live after the previous parfor.
        live_out_prev = prev_stmt_live_last.live_out
        dprintln(2,"live_out_prev = ", live_out_prev)

        # Get the live variables into the current parfor.
        live_in_cur   = cur_stmt_live.live_in
        def_cur       = cur_stmt_live.def
        dprintln(2,"live_in_cur = ", live_in_cur, " def_cur = ", def_cur)

        # Get the variables live after the current parfor.
        live_out_cur  = cur_stmt_live.live_out
        dprintln(2,"live_out_cur = ", live_out_cur)

        surviving_def = intersect(union(def_prev, def_cur), live_out_cur)

        new_in_prev = setdiff(live_out_prev, live_in_prev)
        new_in_cur  = setdiff(live_out_cur,  live_in_cur)
        dprintln(3,"new_in_prev = ", new_in_prev)
        dprintln(3,"new_in_cur = ", new_in_cur)
        dprintln(3,"surviving_def = ", surviving_def)

        # The things that come in live to cur but leave it dead.
        not_used_after_cur = setdiff(live_out_prev, live_out_cur)
        dprintln(2,"not_used_after_cur = ", not_used_after_cur)

        live_out_prev_aliases = getAllAliases(live_out_prev, prev_parfor.array_aliases)
        live_out_cur_aliases  = getAllAliases(live_out_cur, prev_parfor.array_aliases)
        dprintln(2, "live_out_prev_aliases = ", live_out_prev_aliases)
        dprintln(2, "live_out_cur_aliases  = ", live_out_cur_aliases)
        not_used_after_cur_with_aliases = setdiff(live_out_prev_aliases, live_out_cur_aliases)
        dprintln(2,"not_used_after_cur_with_aliases = ", not_used_after_cur_with_aliases)

        unique_id = prev_parfor.unique_id

        # Output of this parfor are the new things in the current parfor plus the new things in the previous parfor
        # that don't die during the current parfor.
        output_map = Dict{SymGen, SymGen}()
        for i in map_prev_lhs_all
            if !in(i[1], not_used_after_cur)
                output_map[i[1]] = i[2]
            end
        end
        for i in map_cur_lhs_all
            output_map[i[1]] = i[2]
        end

        new_aliases = Dict{SymGen, SymGen}()
        for i in map_prev_lhs_post
            if !in(i[1], not_used_after_cur)
                new_aliases[i[1]] = i[2]
            end
        end

        dprintln(3,"output_map = ", output_map)
        dprintln(3,"new_aliases = ", new_aliases)

        # return code 2 if there is no output in the fused parfor
        # this means the parfor is dead and should be removed
        if length(surviving_def)==0
            dprintln(1,"No output for the fused parfor so the parfor is dead and is being removed.")
            return 2;
        end

        first_arraylen = getFirstArrayLens(prev_parfor.preParFor, prev_num_dims)

        # Merge each part of the two parfor nodes.

        # loopNests - nothing needs to be done to the loopNests
        # But we use them to establish a mapping between corresponding indices in the two parfors.
        # Then, we use that map to convert indices in the second parfor to the corresponding ones in the first parfor.
        index_map = Dict{SymGen, SymGen}()
        assert(length(prev_parfor.loopNests) == length(cur_parfor.loopNests))
        for i = 1:length(prev_parfor.loopNests)
            index_map[cur_parfor.loopNests[i].indexVariable.name] = prev_parfor.loopNests[i].indexVariable.name
        end

        dprintln(3,"array_aliases before merge ", prev_parfor.array_aliases)
        for i in map_cur_lhs_post
            from = i[1]
            to   = i[2]
            prev_parfor.array_aliases[from] = to
        end
        dprintln(3,"array_aliases after merge ", prev_parfor.array_aliases)

        # postParFor - can merge everything but the last entry in the postParFor's.
        # The last entries themselves need to be merged extending the tuple if the prev parfor's output stays live and
        # just keeping the cur parfor output if the prev parfor output dies.
        (new_lhs, all_rets, single, output_items) = create_merged_output_from_map(output_map, unique_id, state, sym_to_type, loweredAliasMap)
        dprintln(3,"new_lhs = ", new_lhs)
        dprintln(3,"all_rets = ", all_rets)
        dprintln(3,"single = ", single)
        dprintln(3,"output_items = ", output_items)
        output_items_set = live_out_cur
        output_items_with_aliases = getAllAliases(output_items_set, prev_parfor.array_aliases)

        dprintln(3,"output_items_set = ", output_items_set)
        dprintln(3,"output_items_with_aliases = ", output_items_with_aliases)

        # Create a dictionary of arrays to the last variable containing the array's value at the current index space.
        save_body = prev_parfor.body
        arrayset_dict = Dict{SymGen, SymNodeGen}()
        for i = 1:length(save_body)
            x = save_body[i]
            if isArraysetCall(x)
                # Here we have a call to arrayset.
                array_name = x.args[2]
                value      = x.args[3]
                assert(isa(array_name, SymNodeGen))
                assert(isa(value, SymNodeGen))
                arrayset_dict[toSymGen(array_name)] = value
            elseif typeof(x) == Expr && x.head == :(=)
                lhs = x.args[1]
                rhs = x.args[2]
                assert(isa(lhs, SymNodeGen))
                if isArrayrefCall(rhs)
                    array_name = rhs.args[2]
                    assert(isa(array_name, SymNodeGen))
                    arrayset_dict[toSymGen(array_name)] = lhs
                end
            end
        end
        dprintln(3,"arrayset_dict = ", arrayset_dict)

        # Extend the arrayset_dict to include the lhs of the prev parfor.
        for i in map_prev_lhs_post
            lhs_sym = i[1]
            rhs_sym = i[2]
            arrayset_dict[lhs_sym] = arrayset_dict[rhs_sym]
        end
        dprintln(3,"arrayset_dict = ", arrayset_dict)

        # body - Append cur body to prev body but replace arrayset's in prev with a temp variable
        # and replace arrayref's in cur with the same temp.
        arrays_set_in_cur_body = Set{SymGen}()
        # Convert the cur_parfor body.
        new_cur_body = map(x -> substitute_cur_body(x, arrayset_dict, index_map, arrays_set_in_cur_body, loweredAliasMap, state), cur_parfor.body)
        arrays_set_in_cur_body_with_aliases = getAllAliases(arrays_set_in_cur_body, prev_parfor.array_aliases)
        dprintln(3,"arrays_set_in_cur_body = ", arrays_set_in_cur_body)
        dprintln(3,"arrays_set_in_cur_body_with_aliases = ", arrays_set_in_cur_body_with_aliases)
        combined = union(arrays_set_in_cur_body_with_aliases, not_used_after_cur_with_aliases)
        dprintln(2,"combined = ", combined)

        prev_parfor.body = Any[]
        for i = 1:length(save_body)
            new_body_line = substitute_arrayset(save_body[i], combined, output_items_with_aliases)
            dprintln(3,"new_body_line = ", new_body_line)
            if new_body_line != nothing
                push!(prev_parfor.body, new_body_line)
            end
        end
        append!(prev_parfor.body, new_cur_body)
        dprintln(2,"New body = ")
        printBody(2, prev_parfor.body)

        # preParFor - Append cur preParFor to prev parParFor but eliminate array creation from
        # prevParFor where the array is in allocs_to_eliminate.
        prev_parfor.preParFor = [ filter(x -> !is_eliminated_allocation_map(x, output_items_with_aliases), prev_parfor.preParFor);
        map(x -> substitute_arraylen(x,first_arraylen) , filter(x -> !is_eliminated_arraylen(x), cur_parfor.preParFor)) ]
        dprintln(2,"New preParFor = ", prev_parfor.preParFor)

        # if allocation of an array is removed, arrayset should be removed as well since the array doesn't exist anymore
        dprintln(4,"prev_parfor.body before removing dead arrayset: ", prev_parfor.body)
        filter!( x -> !is_dead_arrayset(x, output_items_with_aliases), prev_parfor.body)
        dprintln(4,"prev_parfor.body after_ removing dead arrayset: ", prev_parfor.body)

        # reductions - a simple append with the caveat that you can't fuse parfor where the first has a reduction that the second one uses
        # need to check this caveat above.
        append!(prev_parfor.reductions, cur_parfor.reductions)
        dprintln(2,"New reductions = ", prev_parfor.reductions)

        prev_parfor.postParFor = [ prev_parfor.postParFor[1:end-1]; cur_parfor.postParFor[1:end-1]]
        push!(prev_parfor.postParFor, oneIfOnly(output_items))
        dprintln(2,"New postParFor = ", prev_parfor.postParFor, " typeof(postParFor) = ", typeof(prev_parfor.postParFor), " ", typeof(prev_parfor.postParFor[end]))

        # original_domain_nodes - simple append
        append!(prev_parfor.original_domain_nodes, cur_parfor.original_domain_nodes)
        dprintln(2,"New domain nodes = ", prev_parfor.original_domain_nodes)

        # top_level_number - what to do here? is this right?
        push!(prev_parfor.top_level_number, cur_parfor.top_level_number[1])

        # rws
        prev_parfor.rws = CompilerTools.ReadWriteSet.from_exprs(prev_parfor.body, pir_live_cb, state.lambdaInfo)

        dprintln(3,"New lhs = ", new_lhs)
        if prev_assignment
            # The prev parfor was of the form "var = parfor(...)".
            if new_lhs != nothing
                dprintln(2,"prev was assignment and is staying an assignment")
                # The new lhs is not empty and so this is the normal case where "prev" stays an assignment expression and we update types here and if necessary FusionSentinel.
                prev.args[1] = new_lhs
                prev.typ = getType(new_lhs, state.lambdaInfo)
                prev.args[2].typ = prev.typ
                # Strip off a previous FusionSentinel() if it exists in the expression.
                prev.args = prev.args[1:2]
                if !single
                    push!(prev.args, FusionSentinel())
                    append!(prev.args, all_rets)
                    dprintln(3,"New multiple ret prev args is ", prev.args)
                end
            else
                dprintln(2,"prev was assignment and is becoming bare")
                # The new lhs is empty and so we need to transform "prev" into an assignment expression.
                body[body_index] = TypedExpr(nothing, :parfor, prev_parfor)
            end
        else
            # The prev parfor was a bare-parfor (not in the context of an assignment).
            if new_lhs != nothing
                dprintln(2,"prev was bare and is becoming an assignment")
                # The new lhs is not empty so the fused parfor will not be bare and "prev" needs to become an assignment expression.
                body[body_index] = mk_assignment_expr(new_lhs, prev, state)
                prev = body[body_index]
                prev.args[2].typ = CompilerTools.LambdaHandling.getType(new_lhs, state.lambdaInfo)
                if !single
                    push!(prev.args, FusionSentinel())
                    append!(prev.args, all_rets)
                    dprintln(3,"New multiple ret prev args is ", prev.args)
                end
            else
                dprintln(2,"prev was bare and is staying bare")
            end
        end

        dprintln(2,"New parfor = ", prev_parfor)

        #throw(string("not finished"))

        return 1
    else
        dprintln(3, "Fusion could not happen here.")
    end

    return 0

    false
end

@doc """
Returns true if the incoming AST node can be interpreted as a Symbol.
"""
function hasSymbol(ssn :: Symbol)
    return true
end

function hasSymbol(ssn :: SymbolNode)
    return true
end

function hasSymbol(ssn :: Expr)
    return ssn.head == :(::)
end

function hasSymbol(ssn)
    return false
end

@doc """
Get the name of a symbol whether the input is a Symbol or SymbolNode or :(::) Expr.
"""
function getSName(ssn :: Symbol)
    return ssn
end

function getSName(ssn :: SymbolNode)
    return ssn.name
end

function getSName(ssn :: Expr)
    assert(ssn.head == :(::))
    return ssn.args[1]
end

function getSName(ssn :: GenSym)
    return ssn
end

function getSName(ssn)
    stype = typeof(ssn)

    dprintln(0, "getSName ssn = ", ssn, " stype = ", stype)
    if stype == Expr
        dprintln(0, "ssn.head = ", ssn.head)
    end
    throw(string("getSName called with something of type ", stype))
end

#@doc """
#Store information about a section of a body that will be translated into a task.
#"""
#type TaskGraphSection
#  start_body_index :: Int
#  end_body_index   :: Int
#  exprs            :: Array{Any,1}
#end

@doc """
Process an array of expressions.
Differentiate between top-level arrays of statements and arrays of expression that may occur elsewhere than the :body Expr.
"""
function from_exprs(ast::Array{Any,1}, depth, state)
    # sequence of expressions
    # ast = [ expr, ... ]
    # Is this the first node in the AST with an array of expressions, i.e., is it the top-level?
    top_level = (state.top_level_number == 0)
    if top_level
        return top_level_from_exprs(ast, depth, state)
    else
        return intermediate_from_exprs(ast, depth, state)
    end
end

@doc """
Process an array of expressions that aren't from a :body Expr.
"""
function intermediate_from_exprs(ast::Array{Any,1}, depth, state)
    # sequence of expressions
    # ast = [ expr, ... ]
    len  = length(ast)
    res = Any[]

    # For each expression in the array, process that expression recursively.
    for i = 1:len
        dprintln(2,"Processing ast #",i," depth=",depth)

        # Convert the current expression.
        new_exprs = from_expr(ast[i], depth, state, false)
        assert(isa(new_exprs,Array))

        append!(res, new_exprs)  # Take the result of the recursive processing and add it to the result.
    end

    return res 
end

@doc """
Find the basic block before the entry to a loop.
"""
function getNonBlock(head_preds, back_edge)
    assert(length(head_preds) == 2)

    # Scan the predecessors of the head of the loop and the non-back edge must be the block prior to the loop.
    for i in head_preds
        if i.label != back_edge
            return i
        end
    end

    throw(string("All entries in head preds list were back edges."))
end

@doc """
Store information about a section of a body that will be translated into a task.
"""
type ReplacedRegion
    start_index
    end_index
    bb
    tasks
end

type EntityType
    name :: SymAllGen
    typ
end

@doc """
Structure for storing information about task formation.
"""
type TaskInfo
    task_func       :: Function                  # The Julia task function that we generated for a task.
    function_sym
    join_func                                    # The name of the C join function that we constructed and forced into the C file.
    input_symbols   :: Array{EntityType,1}       # Variables that are need as input to the task.
    modified_inputs :: Array{EntityType,1} 
    io_symbols      :: Array{EntityType,1}
    reduction_vars  :: Array{EntityType,1}
    code
    loopNests       :: Array{PIRLoopNest,1}      # holds information about the loop nests
end

@doc """
Translated to pert_range_Nd_t in the task runtime.
This represents an iteration space.
dim is the number of dimensions in the iteration space.
lower_bounds contains the lower bound of the iteration space in each dimension.
upper_bounds contains the upper bound of the iteration space in each dimension.
lower_bounds and upper_bounds can be expressions.
"""
type pir_range
    dim :: Int
    lower_bounds :: Array{Any,1}
    upper_bounds :: Array{Any,1}
    function pir_range()
        new(0, Any[], Any[])
    end
end

@doc """
Similar to pir_range but used in circumstances where the expressions must have already been evaluated.
Therefore the arrays are typed as Int64.
Up to 3 dimensional iteration space constructors are supported to make it easier to do code generation later.
"""
type pir_range_actual
    dim :: Int
    lower_bounds :: Array{Int64, 1}
    upper_bounds :: Array{Int64, 1}
    function pir_range_actual()
        new(0, Int64[], Int64[])
    end
    function pir_range_actual(l1, u1)
        new(1, Int64[l1], Int64[u1])
    end
    function pir_range_actual(l1, u1, l2, u2)
        new(2, Int64[l1; l2], Int64[u1; u2])
    end
    function pir_range_actual(l1, u1, l2, u2, l3, u3)
        new(3, Int64[l1; l2; l3], Int64[u1; u2; u3])
    end
end

ARG_OPT_IN = 1
ARG_OPT_OUT = 2
ARG_OPT_INOUT = 3
#ARG_OPT_VALUE = 4
ARG_OPT_ACCUMULATOR = 5

type pir_aad_dim
    len
    a1
    a2
    l_b
    u_b
end

@doc """
Describes an array.
row_major is true if the array is stored in row major format.
dim_info describes which portion of the array is accessed for a given point in the iteration space.
"""
type pir_array_access_desc
    dim_info :: Array{pir_aad_dim, 1}
    row_major :: Bool

    function pir_array_access_desc()
        new(pir_aad_dim[],false)
    end
end

@doc """
Create an array access descriptor for "array".
Presumes that for point "i" in the iteration space that only index "i" is accessed.
"""
function create1D_array_access_desc(array :: SymbolNode)
    ret = pir_array_access_desc()
    push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 1), 1, 0, 0, 0 ))
    ret
end

@doc """
Create an array access descriptor for "array".
Presumes that for points "(i,j)" in the iteration space that only indices "(i,j)" is accessed.
"""
function create2D_array_access_desc(array :: SymbolNode)
    ret = pir_array_access_desc()
    push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 1), 1, 0, 0, 0 ))
    push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 2), 1, 0, 0, 0 ))
    ret
end

@doc """
Create an array access descriptor for "array".
"""
function create_array_access_desc(array :: SymbolNode)
    if array.typ.parameters[2] == 1
        return create1D_array_access_desc(array)
    elseif array.typ.parameters[2] == 2
        return create2D_array_access_desc(array)
    else
        throw(string("Greater than 2D arrays not supported in create_array_access_desc."))
    end
end

@doc """
A Julia representation of the argument metadata that will be passed to the runtime.
"""
type pir_arg_metadata
    value   :: SymbolNode
    options :: Int
    access  # nothing OR pir_array_access_desc

    function pir_arg_metadata()
        new(nothing, 0, 0, nothing)
    end

    function pir_arg_metadata(v, o)
        new(v, o, nothing)
    end

    function pir_arg_metadata(v, o, a)
        new(v, o, a)
    end
end

@doc """
A Julia representation of the grain size that will be passed to the runtime.
"""
type pir_grain_size
    dim   :: Int
    sizes :: Array{Int,1}

    function pir_grain_size()
        new(0, Int[])
    end
end

TASK_FINISH = 1 << 16           # 0x10000
TASK_PRIORITY  = 1 << 15        # 0x08000
TASK_SEQUENTIAL = 1 << 14       # 0x04000
TASK_AFFINITY_XEON = 1 << 13    # 0x02000
TASK_AFFINITY_PHI = 1 << 12     # 0x01000
TASK_STATIC_SCHEDULER = 1 << 11 # 0x00800

@doc """
A data type containing the information that cgen uses to generate a call to pert_insert_divisible_task.
"""
type InsertTaskNode
    ranges :: pir_range
    args   :: Array{pir_arg_metadata,1}
    task_func :: Any
    join_func :: AbstractString  # empty string for no join function
    task_options :: Int
    host_grain_size :: pir_grain_size
    phi_grain_size :: pir_grain_size

    function InsertTaskNode()
        new(pir_range(), pir_arg_metadata[], nothing, string(""), 0, pir_grain_size(), pir_grain_size())
    end
end

@doc """
If run_as_tasks is positive then convert this parfor to a task and decrement the count so that only the
original number run_as_tasks if the number of tasks created.
"""
function run_as_task_decrement()
    if run_as_tasks == 0
        return false
    end
    if run_as_tasks == -1
        return true
    end
    global run_as_tasks = run_as_tasks - 1
    return true
end

@doc """
Return true if run_as_task_decrement would return true but don't update the run_as_tasks count.
"""
function run_as_task()
    if run_as_tasks == 0
        return false
    end
    return true
end

put_loops_in_task_graph = false

limit_task = -1
function PIRLimitTask(x)
    global limit_task = x
end

# These two functions are just to maintain the interface with the old PSE system for the moment.
function PIRFlatParfor(x)
end
function PIRPreEq(x)
end

pir_stop = 0
function PIRStop(x)
    global pir_stop = x
end

polyhedral = 0
function PIRPolyhedral(x)
    global polyhedral = x
end

num_threads_mode = 0
function PIRNumThreadsMode(x)
    global num_threads_mode = x
end

stencil_tasks = 1
function PIRStencilTasks(x)
    global stencil_tasks = x
end

reduce_tasks = 0 
function PIRReduceTasks(x)
    global reduce_tasks = x
end

@doc """
Returns true if the "node" is a parfor and the task limit hasn't been exceeded.
Also controls whether stencils or reduction can become tasks.
"""
function taskableParfor(node)
    dprintln(3,"taskableParfor for: ", node)
    if limit_task == 0
        dprintln(3,"task limit exceeded so won't convert parfor to task")
        return false
    end
    if isParforAssignmentNode(node) || isBareParfor(node)
        dprintln(3,"Found parfor node, stencil: ", stencil_tasks, " reductions: ", reduce_tasks)
        if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
            return true
        end

        the_parfor = getParforNode(node)

        for i = 1:length(the_parfor.original_domain_nodes)
            if (the_parfor.original_domain_nodes[i].operation == :stencil! && stencil_tasks == 0) ||
                (the_parfor.original_domain_nodes[i].operation == :reduce && reduce_tasks == 0)
                return false
            end
        end
        if limit_task > 0
            global limit_task = limit_task - 1
        end
        return true
    end
    false
end

type eic_state
    non_calls      :: Float64    # estimated instruction count for non-calls
    fully_analyzed :: Bool
    lambdaInfo     :: Union{CompilerTools.LambdaHandling.LambdaInfo, Void}
end

ASSIGNMENT_COST = 1.0
RETURN_COST = 1.0
ARRAYSET_COST = 4.0
UNSAFE_ARRAYSET_COST = 2.0
ARRAYREF_COST = 4.0
UNSAFE_ARRAYREF_COST = 2.0
FLOAT_ARITH = 1.0

call_costs = Dict{Any,Any}()
call_costs[(TopNode(:mul_float),(Float64,Float64))] = 1.0
call_costs[(TopNode(:div_float),(Float64,Float64))] = 1.0
call_costs[(TopNode(:add_float),(Float64,Float64))] = 1.0
call_costs[(TopNode(:sub_float),(Float64,Float64))] = 1.0
call_costs[(TopNode(:neg_float),(Float64,))] = 1.0
call_costs[(TopNode(:mul_float),(Float32,Float32))] = 1.0
call_costs[(TopNode(:div_float),(Float32,Float32))] = 1.0
call_costs[(TopNode(:add_float),(Float32,Float32))] = 1.0
call_costs[(TopNode(:sub_float),(Float32,Float32))] = 1.0
call_costs[(TopNode(:neg_float),(Float32,))] = 1.0
call_costs[(TopNode(:mul_int),(Int64,Int64))] = 1.0
call_costs[(TopNode(:div_int),(Int64,Int64))] = 1.0
call_costs[(TopNode(:add_int),(Int64,Int64))] = 1.0
call_costs[(TopNode(:sub_int),(Int64,Int64))] = 1.0
call_costs[(TopNode(:sle_int),(Int64,Int64))] = 1.0
call_costs[(TopNode(:sitofp),(DataType,Int64))] = 1.0
call_costs[(:log10,(Float64,))] = 160.0
call_costs[(:erf,(Float64,))] = 75.0

@doc """
A sentinel in the instruction count estimation process.
Before recursively processing a call, we add a sentinel for that function so that if we see that
sentinel later we know we've tried to recursively process it and so can bail out by setting
fully_analyzed to false.
"""
type InProgress
end

@doc """
Generate an instruction count estimate for a call instruction.
"""
function call_instruction_count(args, state :: eic_state, debug_level)
    func  = args[1]
    fargs = args[2:end]

    dprintln(3,"call_instruction_count: func = ", func, " fargs = ", fargs)
    sig_expr = Expr(:tuple)
    sig_expr.args = map(x -> CompilerTools.LivenessAnalysis.typeOfOpr(x, state.lambdaInfo), fargs)
    signature = eval(sig_expr)
    fs = (func, signature)

    # If we've previously cached an instruction count estimate then use it.
    if haskey(call_costs, fs)
        res = call_costs[fs]
        if res == nothing
            dprintln(debug_level, "Didn't process call to function ", func, " ", signature, " ", args)
            state.fully_analyzed = false
            return nothing
        end
        # See the comment on the InProgress type above for how this prevents infinite recursive analysis.
        if typeof(res) == InProgress
            dprintln(debug_level, "Got recursive call to function ", func, " ", signature, " ", args)
            state.fully_analyzed = false
            return nothing
        end
    else
        # See the comment on the InProgress type above for how this prevents infinite recursive analysis.
        call_costs[fs] = InProgress()
        # If not then try to generate an instruction count estimated.
        res = generate_instr_count(func, signature)
        call_costs[fs] = res
        if res == nothing
            # If we couldn't do it then set fully_analyzed to false.
            dprintln(debug_level, "Didn't process call to function ", func, " ", signature, " ", args)
            state.fully_analyzed = false
            return nothing
        end
    end

    assert(typeof(res) == Float64)
    state.non_calls = state.non_calls + res
    return nothing
end

@doc """
Try to figure out the instruction count for a given call.
"""
function generate_instr_count(function_name, signature)
    # Estimate instructions for some well-known functions.
    if function_name == TopNode(:arrayset) || function_name == TopNode(:arrayref)
        call_costs[(function_name, signature)] = 4.0
        return call_costs[(function_name, signature)]
    elseif function_name == TopNode(:unsafe_arrayset) || function_name == TopNode(:unsafe_arrayref)
        call_costs[(function_name, signature)] = 2.0
        return call_costs[(function_name, signature)]
    elseif function_name == TopNode(:safe_arrayref)
        call_costs[(function_name, signature)] = 6.0
        return call_costs[(function_name, signature)]
    elseif function_name == TopNode(:box)
        call_costs[(function_name, signature)] = 20.0
        return call_costs[(function_name, signature)]
    elseif function_name == TopNode(:lt_float) || 
        function_name == TopNode(:le_float) ||
        function_name == TopNode(:not_int)
        call_costs[(function_name, signature)] = 1.0
        return call_costs[(function_name, signature)]
    end

    ftyp = typeof(function_name)

    if ftyp != Function
        dprintln(3,"generate_instr_count: instead of Function, got ", ftyp, " ", function_name)
    end

    if ftyp == Expr
        dprintln(3,"eval'ing Expr to Function")
        function_name = eval(function_name)
    elseif ftyp == GlobalRef
        #dprintln(3,"Calling getfield")
        function_name = eval(function_name)
        #function_name = getfield(function_name.mod, function_name.name)
    elseif ftyp == IntrinsicFunction
        dprintln(3, "generate_instr_count: found IntrinsicFunction = ", function_name)
        call_costs[(function_name, signature)] = nothing
        return call_costs[(function_name, signature)]
    end

    if typeof(function_name) != Function || !isgeneric(function_name)
        dprintln(3, "generate_instr_count: function_name is not a Function = ", function_name)
        call_costs[(function_name, signature)] = nothing
        return call_costs[(function_name, signature)]
    end

    m = methods(function_name, signature)
    if length(m) < 1
        return nothing
        #    error("Method for ", function_name, " with signature ", signature, " is not found")
    end

    ct = ParallelAccelerator.Driver.code_typed(function_name, signature)      # get information about code for the given function and signature

    dprintln(2,"generate_instr_count ", function_name, " ", signature)
    state = eic_state(0, true, nothing)
    # Try to estimate the instruction count for the other function.
    AstWalk(ct[1], estimateInstrCount, state)
    dprintln(2,"instruction count estimate for parfor = ", state)
    # If so then cache the result.
    if state.fully_analyzed
        call_costs[(function_name, signature)] = state.non_calls
    else
        call_costs[(function_name, signature)] = nothing
    end
    return call_costs[(function_name, signature)]
end

@doc """
AstWalk callback for estimating the instruction count.
"""
function estimateInstrCount(ast, state :: eic_state, top_level_number, is_top_level, read)
    debug_level = 2

    asttyp = typeof(ast)
    if asttyp == Expr
        if is_top_level
            dprint(debug_level,"instruction count estimator: Expr ")
        end
        head = ast.head
        args = ast.args
        typ  = ast.typ
        if is_top_level
            dprintln(debug_level,head, " ", args)
        end
        if head == :lambda
            state.lambdaInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(ast)
        elseif head == :body
            # skip
        elseif head == :block
            # skip
        elseif head == :(.)
            #        args = from_exprs(args, depth+1, callback, cbdata, top_level_number, read)
        elseif head == :(=)
            state.non_calls = state.non_calls + ASSIGNMENT_COST
        elseif head == :(::)
            # skip
        elseif head == :return
            state.non_calls = state.non_calls + RETURN_COST
        elseif head == :call
            call_instruction_count(args, state, debug_level)
        elseif head == :call1
            dprintln(debug_level,head, " not handled in instruction counter")
            #        args = from_call(args, depth, callback, cbdata, top_level_number, read)
            # TODO?: tuple
        elseif head == :line
            # skip
        elseif head == :copy
            dprintln(debug_level,head, " not handled in instruction counter")
            # turn array copy back to plain Julia call
            #        head = :call
            #        args = vcat(:copy, args)
        elseif head == :copyast
            #        dprintln(2,"copyast type")
            # skip
        elseif head == :gotoifnot
            #        dprintln(debug_level,head, " not handled in instruction counter")
            #        assert(length(args) == 2)
            #        args[1] = get_one(from_expr(args[1], depth, callback, cbdata, top_level_number, false, read))
            #      state.fully_analyzed = false
            state.non_calls = state.non_calls + 1
        elseif head == :new
            dprintln(debug_level,head, " not handled in instruction counter")
            #        args = from_exprs(args,depth, callback, cbdata, top_level_number, read)
        elseif head == :arraysize
            dprintln(debug_level,head, " not handled in instruction counter")
            #        assert(length(args) == 2)
            #        args[1] = get_one(from_expr(args[1], depth, callback, cbdata, top_level_number, false, read))
            #        args[2] = get_one(from_expr(args[2], depth, callback, cbdata, top_level_number, false, read))
        elseif head == :alloc
            dprintln(debug_level,head, " not handled in instruction counter")
            #        assert(length(args) == 2)
            #        args[2] = from_exprs(args[2], depth, callback, cbdata, top_level_number, read)
        elseif head == :boundscheck
            dprintln(debug_level,head, " not handled in instruction counter")
            # skip
        elseif head == :type_goto
            dprintln(debug_level,head, " not handled in instruction counter")
            #        assert(length(args) == 2)
            #        args[1] = get_one(from_expr(args[1], depth, callback, cbdata, top_level_number, false, read))
            #        args[2] = get_one(from_expr(args[2], depth, callback, cbdata, top_level_number, false, read))
            state.fully_analyzed = false
        elseif head == :enter
            dprintln(debug_level,head, " not handled in instruction counter")
            # skip
            state.fully_analyzed = false
        elseif head == :leave
            dprintln(debug_level,head, " not handled in instruction counter")
            # skip
            state.fully_analyzed = false
        elseif head == :the_exception
            # skip
            state.fully_analyzed = false
        elseif head == :&
            # skip
        else
            dprintln(1,"instruction count estimator: unknown Expr head :", head, " ", ast)
        end
    elseif asttyp == Symbol
        #skip
    elseif asttyp == SymbolNode # name, typ
        #skip
    elseif asttyp == TopNode    # name
        #skip
    elseif asttyp == GlobalRef
        #dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #    mod = ast.mod
        #    name = ast.name
        #    typ = typeof(mod)
        state.non_calls = state.non_calls + 1
    elseif asttyp == QuoteNode
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #    value = ast.value
        #TODO: fields: value
    elseif asttyp == LineNumberNode
        #skip
    elseif asttyp == LabelNode
        #dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
    elseif asttyp == GotoNode
        #dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
        #state.fully_analyzed = false
        state.non_calls = state.non_calls + 1
    elseif asttyp == DataType
        #skip
    elseif asttyp == ()
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
    elseif asttyp == ASCIIString
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
    elseif asttyp == NewvarNode
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
    elseif asttyp == Void
        #skip
        #elseif asttyp == Int64 || asttyp == Int32 || asttyp == Float64 || asttyp == Float32
    elseif isbits(asttyp)
        #skip
    elseif isa(ast,Tuple)
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #    new_tt = Expr(:tuple)
        #    for i = 1:length(ast)
        #      push!(new_tt.args, get_one(from_expr(ast[i], depth, callback, cbdata, top_level_number, false, read)))
        #    end
        #    new_tt.typ = asttyp
        #    ast = eval(new_tt)
    elseif asttyp == Module
        #skip
    elseif asttyp == NewvarNode
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
    else
        dprintln(1,"instruction count estimator: unknown AST (", typeof(ast), ",", ast, ")")
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
Takes a parfor and walks the body of the parfor and estimates the number of instruction needed for one instance of that body.
"""
function createInstructionCountEstimate(the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, state :: expr_state)
    if num_threads_mode == 1 || num_threads_mode == 2 || num_threads_mode == 3
        dprintln(2,"instruction count estimate for parfor = ", the_parfor)
        new_state = eic_state(0, true, state.lambdaInfo)
        for i = 1:length(the_parfor.body)
            AstWalk(the_parfor.body[i], estimateInstrCount, new_state)
        end
        # If fully_analyzed is true then there's nothing that couldn't be analyzed so store the instruction count estimate in the parfor.
        if new_state.fully_analyzed
            the_parfor.instruction_count_expr = new_state.non_calls
        else
            the_parfor.instruction_count_expr = nothing
        end
        dprintln(2,"instruction count estimate for parfor = ", the_parfor.instruction_count_expr)
    end
end

@doc """
Marks an assignment statement where the left-hand side can take over the storage from the right-hand side.
"""
type RhsDead
end

# Task Graph Modes
SEQUENTIAL_TASKS = 1    # Take the first parfor in the block and the last parfor and form tasks for all parallel and sequential parts inbetween.
ONE_AT_A_TIME = 2       # Just forms tasks out of one parfor at a time.
MULTI_PARFOR_SEQ_NO = 3 # Forms tasks from multiple parfor in sequence but not sequential tasks.

task_graph_mode = ONE_AT_A_TIME
@doc """
Control how blocks of code are made into tasks.
"""
function PIRTaskGraphMode(x)
    global task_graph_mode = x
end

include("parallel-ir-top-exprs.jl")

if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE

#println_mutex = Mutex()

function test1(full_iteration_space :: ParallelAccelerator.ParallelIR.pir_range_actual)
  println("test1 space = ", full_iteration_space)
  nothing
end

function tprintln(args...)
  for a in args
    ccall(:puts, Cint, (Cstring,), string(a))
  end
end

@doc """
An intermediate scheduling function for passing to jl_threading_run.
It takes the task function to run, the full iteration space to run and the normal argument to the task function in "rest..."
"""
function isf(t :: Function, 
             full_iteration_space :: ParallelAccelerator.ParallelIR.pir_range_actual,
             rest...)
    tid = Base.Threads.threadid()
#    ta = [typeof(x) for x in rest]
#    Base.Threads.lock!(println_mutex)
#    tprintln("Starting isf. tid = ", tid, " space = ", full_iteration_space, " ta = ", ta)
#    Base.Threads.unlock!(println_mutex)

    if full_iteration_space.dim == 1
        # Compute how many iterations to run.
        num_iters = full_iteration_space.upper_bounds[1] - full_iteration_space.lower_bounds[1] + 1

#        Base.Threads.lock!(println_mutex)
#        tprintln("tid = ", tid, " num_iters = ", num_iters)
#        Base.Threads.unlock!(println_mutex)

        # Handle the case where iterations is less than the core count.
        if num_iters <= nthreads()
            if false
                if tid == 1
                    #println("ISF tid = ", tid, " t = ", t, " fis = ", full_iteration_space, " args = ", rest...)
                    return t(ParallelAccelerator.ParallelIR.pir_range_actual(1,2), rest...)
                    #return t(full_iteration_space, rest...)
                else
                    return nothing
                end
            else
                if tid <= num_iters
                    #return t(ParallelAccelerator.ParallelIR.pir_range_actual(1,2), rest...)
                    return t(ParallelAccelerator.ParallelIR.pir_range_actual(tid,tid), rest...)
                else
                    #return t(pir_range_actual([0],[-1]), rest...)
                    return nothing
                end
            end
        else
            # one dimensional scheduling
            len, rem = divrem(num_iters, nthreads())
            ls = (len * (tid-1)) + 1
            if tid == nthreads()
                le = full_iteration_space.upper_bounds[1]
            else
                le = (len * tid)
            end
#            Base.Threads.lock!(println_mutex)
#            tprintln("tid = ", tid, " ls = ", ls, " le = ", le, " func = ", Base.function_name(t))
#            Base.Threads.unlock!(println_mutex)
            try 
              tres = t(ParallelAccelerator.ParallelIR.pir_range_actual(ls,le), rest...)
#             range_actual = ParallelAccelerator.ParallelIR.pir_range_actual(ls,le)
#             tres = test1(range_actual)
            catch something
             # println("Call to t created exception ", something)
              ccall(:puts, Cint, (Cstring,), "caught some exception")
            end 
#            tprintln("After t call tid = ", tid)
            return tres
        end
    elseif full_iteration_space.dim == 2
        assert(0)
    else
        assert(0)
    end
end

end

@doc """
Returns true if a given SymbolNode "x" is an Array type.
"""
function isArrayType(x :: SymbolNode)
    the_type = x.typ
    if typeof(the_type) == DataType
        return isArrayType(x.typ)
    end
    return false
end

@doc """
For a given start and stop index in some body and liveness information, form a set of tasks.
"""
function makeTasks(start_index, stop_index, body, bb_live_info, state, task_graph_mode)
    task_list = Any[]
    seq_accum = Any[]
    dprintln(3,"makeTasks starting")

    # ONE_AT_A_TIME mode should only have parfors in the list of things to replace so we assert if we see a non-parfor in this mode.
    # SEQUENTIAL_TASKS mode bunches up all non-parfors into one sequential task.  Maybe it would be better to do one sequential task per non-parfor stmt?
    # MULTI_PARFOR_SEQ_NO mode converts parfors to tasks but leaves non-parfors as non-tasks.  This implies that the code calling this function has ensured
    #     that none of the non-parfor stmts depend on the completion of a parfor in the batch.

    if ParallelAccelerator.getPseMode() != ParallelAccelerator.THREADS_MODE
        if task_graph_mode == SEQUENTIAL_TASKS
            task_finish = false
        elseif task_graph_mode == ONE_AT_A_TIME
            task_finish = true
        elseif task_graph_mode == MULTI_PARFOR_SEQ_NO
            task_finish = false
        else
            throw(string("Unknown task_graph_mode in makeTasks"))
        end
    end

    for j = start_index:stop_index
        if taskableParfor(body[j])
            # is a parfor node
            if length(seq_accum) > 0
                assert(task_graph_mode == SEQUENTIAL_TASKS)
                st = seqTask(seq_accum, bb_live_info.statements, body, state)
                dprintln(3,"Adding sequential task to task_list. ", st)
                push!(task_list, st)
                seq_accum = Any[]
            end
            ptt = parforToTask(j, bb_live_info.statements, body, state)
            dprintln(3,"Adding parfor task to task_list. ", ptt)
            push!(task_list, ptt)
        else
            # is not a parfor node
            assert(task_graph_mode != ONE_AT_A_TIME)
            if task_graph_mode == SEQUENTIAL_TASKS
                # Collect the non-parfor stmts in a batch to be turned into one sequential task.
                push!(seq_accum, body[j])
            else
                dprintln(3,"Adding non-parfor node to task_list. ", body[j])
                # MULTI_PARFOR_SEQ_NO mode.
                # Just add the non-parfor stmts directly to the output statements.
                push!(task_list, body[j])
            end
        end
    end

    #if ParallelAccelerator.getPseMode() != ParallelAccelerator.THREADS_MODE
        # If each task doesn't wait to finish then add a call to pert_wait_all_task to wait for the batch to finish.
     #   if !task_finish
            #julia_root      = ParallelAccelerator.getJuliaRoot()
            #runtime_libpath = string(julia_root, "/intel-runtime/lib/libintel-runtime.so")
            #runtime_libpath = ParallelAccelerator.runtime_libpath

            #call_wait = Expr(:ccall, Expr(:tuple, QuoteNode(:pert_wait_all_task), runtime_libpath), :Void, Expr(:tuple))
            #push!(task_list, call_wait) 

      #      call_wait = TypedExpr(Void, 
      #      :call, 
      #      TopNode(:ccall), 
      #      Expr(:call1, TopNode(:tuple), QuoteNode(:pert_wait_all_task), runtime_libpath), 
      #      :Void, 
      #      Expr(:call1, TopNode(:tuple)))
      #      push!(task_list, call_wait)

            #    call_wait = quote ccall((:pert_wait_all_task, $runtime_libpath), Void, ()) end
            #    assert(typeof(call_wait) == Expr && call_wait.head == :block)
            #    append!(task_list, call_wait.args) 
       # end
    #end

    task_list
end

@doc """
Given a set of statement IDs and liveness information for the statements of the function, determine
which symbols are needed at input and which symbols are purely local to the functio.
"""
function getIO(stmt_ids, bb_statements)
    assert(length(stmt_ids) > 0)

    # Get the statements out of the basic block statement array such that those statement's ID's are in the stmt_ids ID array.
    stmts_for_ids = filter(x -> in(x.tls.index, stmt_ids) , bb_statements)
    # Make sure that we found a statement for every statement ID.
    if length(stmt_ids) != length(stmts_for_ids)
        dprintln(0,"length(stmt_ids) = ", length(stmt_ids))
        dprintln(0,"length(stmts_for_ids) = ", length(stmts_for_ids))
        dprintln(0,"stmt_ids = ", stmt_ids)
        dprintln(0,"stmts_for_ids = ", stmts_for_ids)
        assert(length(stmt_ids) == length(stmts_for_ids))
    end
    # The initial set of inputs is those variables "use"d by the first statement.
    # The inputs to the task are those variables used in the set of statements before they are defined in any of those statements.
    cur_inputs = stmts_for_ids[1].use
    # Keep track of variables defined in the set of statements processed thus far.
    cur_defs   = stmts_for_ids[1].def
    for i = 2:length(stmts_for_ids)
        # For each additional statement, the new set of inputs is the previous set plus uses in the current statement except for those symbols already defined in the function.
        cur_inputs = union(cur_inputs, setdiff(stmts_for_ids[i].use, cur_defs))
        # For each additional statement, the defs are just union with the def for the current statement.
        cur_defs   = union(cur_defs, stmts_for_ids[i].def)
    end
    IntrinsicSet = Set()
    # We will ignore the :Intrinsics symbol as it isn't something you need to pass as a param.
    push!(IntrinsicSet, :Intrinsics)
    # Task functions don't return anything.  They must return via an input parameter so outputs should be empty.
    outputs = setdiff(intersect(cur_defs, stmts_for_ids[end].live_out), IntrinsicSet)
    cur_defs = setdiff(cur_defs, IntrinsicSet)
    cur_inputs = setdiff(filter(x -> !(is(x, :Int64) || is(x, :Float32)), cur_inputs), IntrinsicSet)
    # The locals are those things defined that aren't inputs or outputs of the function.
    cur_inputs, outputs, setdiff(cur_defs, union(cur_inputs, outputs))
end

@doc """
Returns an expression to construct a :colon object that contains the start of a range, the end and the skip expression.
"""
function mk_colon_expr(start_expr, skip_expr, end_expr)
    TypedExpr(Any, :call, :colon, start_expr, skip_expr, end_expr)
end

@doc """
Returns an expression to get the start of an iteration range from a :colon object.
"""
function mk_start_expr(colon_sym)
    TypedExpr(Any, :call, TopNode(:start), colon_sym)
end

@doc """
Returns a :next call Expr that gets the next element of an iteration range from a :colon object.
"""
function mk_next_expr(colon_sym, start_sym)
    TypedExpr(Any, :call, TopNode(:next), colon_sym, start_sym)
end

@doc """
Returns a :gotoifnot Expr given a condition "cond" and a label "goto_label".
"""
function mk_gotoifnot_expr(cond, goto_label)
    TypedExpr(Any, :gotoifnot, cond, goto_label)
end

@doc """
Just to hold the "found" Bool that says whether a unsafe variant was replaced with a regular version.
"""
type cuw_state
    found
    function cuw_state()
        new(false)
    end
end

@doc """
The AstWalk callback to find unsafe arrayset and arrayref variants and
replace them with the regular Julia versions.  Sets the "found" flag
in the state when such a replacement is performed.
"""
function convertUnsafeWalk(x, state, top_level_number, is_top_level, read)
    use_dbg_level = 3
    dprintln(use_dbg_level,"convertUnsafeWalk ", x)

    if typeof(x) == Expr
        dprintln(use_dbg_level,"convertUnsafeWalk is Expr")
        if x.head == :call
            dprintln(use_dbg_level,"convertUnsafeWalk is :call")
            if x.args[1] == TopNode(:unsafe_arrayset)
                x.args[1] = TopNode(:arrayset)
                state.found = true
                return x
            elseif x.args[1] == TopNode(:unsafe_arrayref)
                x.args[1] = TopNode(:arrayref)
                state.found = true
                return x
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
Remove unsafe array access Symbols from the incoming "stmt".
Returns the updated statement if something was modifed, else returns "nothing".
"""
function convertUnsafe(stmt)
    dprintln(3,"convertUnsafe: ", stmt)
    state = cuw_state() 
    # Uses AstWalk to do the pattern match and replace.
    res = AstWalk(stmt, convertUnsafeWalk, state)
    # state.found is set if the callback convertUnsafeWalk found and replaced an unsafe variant.
    if state.found
        dprintln(3, "state.found ", state, " ", res)
        dprintln(3,"Replaced unsafe: ", res)
        return res
    else
        return nothing
    end
end

@doc """
Try to remove unsafe array access Symbols from the incoming "stmt".  If successful, then return the updated
statement, else return the unmodified statement.
"""
function convertUnsafeOrElse(stmt)
    res = convertUnsafe(stmt)
    if res == nothing
        res = stmt
    end
    return res
end

function first_unless(gs0 :: StepRange{Int64,Int64}, pound :: Int64)
    res = 
     !(
       (
         (!(gs0.start == gs0.stop)) && 
         (!((0 < gs0.step) == (gs0.start < gs0.stop)))
       ) || 
       (pound == (gs0.stop + gs0.step))
      )
    dprintln(4,"first_unless res = ", res)
    return res
end

function assign_gs4(gs0 :: StepRange{Int64,Int64}, pound :: Int64)
    pound + gs0.step
end

function second_unless(gs0 :: StepRange{Int64,Int64}, pound :: Int64)
    res = 
     !(
       !(
         (
           (!(gs0.start == gs0.stop)) && 
           (!((0 < gs0.step) == (gs0.start < gs0.stop)))
         ) || 
         (pound == (gs0.stop + gs0.step))
        )
      ) 
    dprintln(4,"second_unless res = ", res)
    return res
end

precompile(first_unless, (StepRange{Int64,Int64}, Int64))
precompile(assign_gs4, (StepRange{Int64,Int64}, Int64))
precompile(second_unless, (StepRange{Int64,Int64}, Int64))

@doc """
This is a recursive routine to reconstruct a regular Julia loop nest from the loop nests described in PIRParForAst.
One call of this routine handles one level of the loop nest.
If the incoming loop nest level is more than the number of loops nests in the parfor then that is the spot to
insert the body of the parfor into the new function body in "new_body".
"""
function recreateLoopsInternal(new_body, the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, loop_nest_level, next_available_label, state, newLambdaInfo)
    dprintln(3,"recreateLoopsInternal ", loop_nest_level, " ", next_available_label)
    if loop_nest_level > length(the_parfor.loopNests)
        # A loop nest level greater than number of nests in the parfor means we can insert the body of the parfor here.
        # For each statement in the parfor body.
        for i = 1:length(the_parfor.body)
            dprintln(3, "Body index ", i)
            # Convert any unsafe_arrayref or sets in this statements to regular arrayref or arrayset.
            # But if it was labeled as "unsafe" then output :boundscheck false Expr so that Julia won't generate a boundscheck on the array access.
            cu_res = convertUnsafe(the_parfor.body[i])
            dprintln(3, "cu_res = ", cu_res)
            if cu_res != nothing
                push!(new_body, Expr(:boundscheck, false)) 
                push!(new_body, cu_res)
                push!(new_body, Expr(:boundscheck, Expr(:call, TopNode(:getfield), Base, QuoteNode(:pop))))
            else
                push!(new_body, the_parfor.body[i])
            end
        end
    else
        # See the following example from the REPL for how Julia structures loop nests into labels and gotos.
        # Our code below generates the same structure for each loop in the parfor.

        # function f1(x,y,z)
        #   for i = x:z:y
        #     println(i)
        #   end
        # end
        
        #  (Base.not_int)(
        #     (Base.or_int)(
        #        (Base.and_int)(
        #           (Base.not_int)(
        #              (top(getfield))(GenSym(0),:start)::Int64 === (top(getfield))(GenSym(0),:stop)::Int64::Bool
        #           ),
        #           (Base.not_int)(
        #              (Base.slt_int)(0,(top(getfield))(GenSym(0),:step)::Int64)::Bool === (Base.slt_int)((top(getfield))(GenSym(0),:start)::Int64,(top(getfield))(GenSym(0),:stop)::Int64)::Bool::Bool
        #           )
        #        ),
        #        #s1::Int64 === (Base.add_int)(
        #                         (top(getfield))(GenSym(0),:stop)::Int64,(top(getfield))(GenSym(0),:step)::Int64
        #                       )
        #     )
        #  )

        #  (Base.not_int)(
        #     (Base.not_int)(
        #        (Base.or_int)(
        #           (Base.and_int)(
        #              (Base.not_int)(
        #                 (top(getfield))(GenSym(0),:start)::Int64 === (top(getfield))(GenSym(0),:stop)::Int64::Bool
        #              ),
        #              (Base.not_int)(
        #                 (Base.slt_int)(0,(top(getfield))(GenSym(0),:step)::Int64)::Bool === (Base.slt_int)((top(getfield))(GenSym(0),:start)::Int64,(top(getfield))(GenSym(0),:stop)::Int64)::Bool::Bool
        #              )
        #           ),
        #           #s1::Int64 === (Base.box)(Base.Int,(Base.add_int)((top(getfield))(GenSym(0),:stop)::Int64,(top(getfield))(GenSym(0),:step)::Int64))::Int64::Bool
        #        )
        #     )
        #  )

        #   :($(Expr(:lambda, Any[:x,:y,:z], Any[Any[Any[:x,Int64,0],Any[:y,Int64,0],Any[:z,Int64,0],Any[symbol("#s1"),Int64,2],Any[:i,Int64,18],Any[symbol("##xs#7035"),Tuple{Int64},0]],Any[],Any[StepRange{Int64,Int64},Tuple{Int64,Int64},Int64,Int64,Int64],Any[]], :(begin  # none, line 2:
        #          GenSym(2) = (Base.steprange_last)(x::Int64,z::Int64,y::Int64)::Int64
        #          GenSym(0) = $(Expr(:new, StepRange{Int64,Int64}, :(x::Int64), :(z::Int64), GenSym(2)))
        #          #s1 = (top(getfield))(GenSym(0),:start)::Int64
        #          unless (Base.box)(Base.Bool,(Base.not_int)((Base.box)(Base.Bool,(Base.or_int)((Base.box)(Base.Bool,(Base.and_int)((Base.box)(Base.Bool,(Base.not_int)((top(getfield))(GenSym(0),:start)::Int64 === (top(getfield))(GenSym(0),:stop)::Int64::Bool))::Bool,(Base.box)(Base.Bool,(Base.not_int)((Base.slt_int)(0,(top(getfield))(GenSym(0),:step)::Int64)::Bool === (Base.slt_int)((top(getfield))(GenSym(0),:start)::Int64,(top(getfield))(GenSym(0),:stop)::Int64)::Bool::Bool))::Bool))::Bool,#s1::Int64 === (Base.box)(Base.Int,(Base.add_int)((top(getfield))(GenSym(0),:stop)::Int64,(top(getfield))(GenSym(0),:step)::Int64))::Int64::Bool))::Bool))::Bool goto 1
        #          2: 
        #          GenSym(3) = #s1::Int64
        #          GenSym(4) = (Base.box)(Base.Int,(Base.add_int)(#s1::Int64,(top(getfield))(GenSym(0),:step)::Int64))::Int64
        #          i = GenSym(3)
        #          #s1 = GenSym(4) # none, line 3:
        #          (Base.println)(Base.STDOUT,i::Int64)
        #          3: 
        #          unless (Base.box)(Base.Bool,(Base.not_int)((Base.box)(Base.Bool,(Base.not_int)((Base.box)(Base.Bool,(Base.or_int)((Base.box)(Base.Bool,(Base.and_int)((Base.box)(Base.Bool,(Base.not_int)((top(getfield))(GenSym(0),:start)::Int64 === (top(getfield))(GenSym(0),:stop)::Int64::Bool))::Bool,(Base.box)(Base.Bool,(Base.not_int)((Base.slt_int)(0,(top(getfield))(GenSym(0),:step)::Int64)::Bool === (Base.slt_int)((top(getfield))(GenSym(0),:start)::Int64,(top(getfield))(GenSym(0),:stop)::Int64)::Bool::Bool))::Bool))::Bool,#s1::Int64 === (Base.box)(Base.Int,(Base.add_int)((top(getfield))(GenSym(0),:stop)::Int64,(top(getfield))(GenSym(0),:step)::Int64))::Int64::Bool))::Bool))::Bool))::Bool goto 2
        #          1: 
        #          0: 
        #          return
        #      end::Void))))

        this_nest = the_parfor.loopNests[loop_nest_level]

        label_after_first_unless   = next_available_label
        label_before_second_unless = next_available_label + 1
        label_after_second_unless  = next_available_label + 2
        label_last                 = next_available_label + 3

        num_vars = 5

        gensym2_var = string("#recreate_gensym2_", (loop_nest_level-1) * num_vars + 0)
        gensym2_sym = symbol(gensym2_var)
        gensym0_var = string("#recreate_gensym0_", (loop_nest_level-1) * num_vars + 1)
        gensym0_sym = symbol(gensym0_var)
        pound_s1_var = string("#recreate_pound_s1_", (loop_nest_level-1) * num_vars + 2)
        pound_s1_sym = symbol(pound_s1_var)
        gensym3_var = string("#recreate_gensym3_", (loop_nest_level-1) * num_vars + 3)
        gensym3_sym = symbol(gensym3_var)
        gensym4_var = string("#recreate_gensym4_", (loop_nest_level-1) * num_vars + 4)
        gensym4_sym = symbol(gensym4_var)
        CompilerTools.LambdaHandling.addLocalVar(gensym2_sym, Int64, ISASSIGNED, newLambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(gensym0_sym, StepRange{Int64,Int64}, ISASSIGNED, newLambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(pound_s1_sym, Int64, ISASSIGNED, newLambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(gensym3_sym, Int64, ISASSIGNED, newLambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(gensym4_sym, Int64, ISASSIGNED, newLambdaInfo)

        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "ranges = ", SymbolNode(:ranges, pir_range_actual)))
        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.lower = ", this_nest.lower))
        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.step  = ", this_nest.step))
        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.upper = ", this_nest.upper))

        if true
           push!(new_body, mk_assignment_expr(SymbolNode(gensym2_sym,Int64), Expr(:call, GlobalRef(Base,:steprange_last), convertUnsafeOrElse(this_nest.lower), convertUnsafeOrElse(this_nest.step), convertUnsafeOrElse(this_nest.upper)), state))
           push!(new_body, mk_assignment_expr(SymbolNode(gensym0_sym,StepRange{Int64,Int64}), Expr(:new, StepRange{Int64,Int64}, convertUnsafeOrElse(this_nest.lower), convertUnsafeOrElse(this_nest.step), SymbolNode(gensym2_sym,Int64)), state))
           push!(new_body, mk_assignment_expr(SymbolNode(pound_s1_sym,Int64), Expr(:call, TopNode(:getfield), SymbolNode(gensym0_sym,StepRange{Int64,Int64}), QuoteNode(:start)), state))
           push!(new_body, mk_gotoifnot_expr(TypedExpr(Bool, :call, mk_parallelir_ref(:first_unless), SymbolNode(gensym0_sym,StepRange{Int64,Int64}), SymbolNode(pound_s1_sym,Int64)), label_after_second_unless))
           push!(new_body, LabelNode(label_after_first_unless))

#           push!(new_body, Expr(:call, GlobalRef(Base,:println), GlobalRef(Base,:STDOUT), " in label_after_first_unless section"))

           push!(new_body, mk_assignment_expr(SymbolNode(gensym3_sym,Int64), SymbolNode(pound_s1_sym,Int64), state))
           push!(new_body, mk_assignment_expr(SymbolNode(gensym4_sym,Int64), Expr(:call, mk_parallelir_ref(:assign_gs4), SymbolNode(gensym0_sym,StepRange{Int64,Int64}), SymbolNode(pound_s1_sym,Int64)), state))
           push!(new_body, mk_assignment_expr(this_nest.indexVariable, SymbolNode(gensym3_sym,Int64), state))
           push!(new_body, mk_assignment_expr(SymbolNode(pound_s1_sym,Int64), SymbolNode(gensym4_sym,Int64), state))

           recreateLoopsInternal(new_body, the_parfor, loop_nest_level + 1, next_available_label + 4, state, newLambdaInfo)

           push!(new_body, LabelNode(label_before_second_unless))
           push!(new_body, mk_gotoifnot_expr(TypedExpr(Bool, :call, mk_parallelir_ref(:second_unless), SymbolNode(gensym0_sym,StepRange{Int64,Int64}), SymbolNode(pound_s1_sym,Int64)), label_after_first_unless))
           push!(new_body, LabelNode(label_after_second_unless))
           push!(new_body, LabelNode(label_last))
        else
           # OLD CODE
           push!(new_body, mk_assignment_expr(SymbolNode(colon_sym,Any), mk_colon_expr(convertUnsafeOrElse(this_nest.lower), convertUnsafeOrElse(this_nest.step), convertUnsafeOrElse(this_nest.upper)), state))
           push!(new_body, mk_assignment_expr(SymbolNode(start_sym,Any), mk_start_expr(colon_sym), state))
           push!(new_body, mk_gotoifnot_expr(TypedExpr(Any, :call, TopNode(:(!)), TypedExpr(Any, :call, TopNode(:done), colon_sym, start_sym) ), label_after_second_unless))
           push!(new_body, LabelNode(label_after_first_unless))

           push!(new_body, mk_assignment_expr(SymbolNode(next_sym,Any),  mk_next_expr(colon_sym, start_sym), state))
           push!(new_body, mk_assignment_expr(this_nest.indexVariable,   mk_tupleref_expr(next_sym, 1, Any), state))
           push!(new_body, mk_assignment_expr(SymbolNode(start_sym,Any), mk_tupleref_expr(next_sym, 2, Any), state))

           #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "loopIndex = ", this_nest.indexVariable))
           #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), colon_sym, " ", start_sym))
           recreateLoopsInternal(new_body, the_parfor, loop_nest_level + 1, next_available_label + 4, state, newLambdaInfo)

           push!(new_body, LabelNode(label_before_second_unless))
           push!(new_body, mk_gotoifnot_expr(TypedExpr(Any, :call, TopNode(:(!)), TypedExpr(Any, :call, TopNode(:(!)), TypedExpr(Any, :call, TopNode(:done), colon_sym, start_sym))), label_after_first_unless))
           push!(new_body, LabelNode(label_after_second_unless))
           push!(new_body, LabelNode(label_last))
        end
    end
end

@doc """
In threads mode, we can't have parfor_start and parfor_end in the code since Julia has to compile the code itself and so
we have to reconstruct a loop infrastructure based on the parfor's loop nest information.  This function takes a parfor
and outputs that parfor to the new function body as regular Julia loops.
"""
function recreateLoops(new_body, the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, state, newLambdaInfo)
    max_label = getMaxLabel(0, the_parfor.body)
    dprintln(2,"recreateLoops ", the_parfor, " max_label = ", max_label)
    # Call the internal loop re-construction code after initializing which loop nest we are working with and the next usable label ID (max_label+1).
    recreateLoopsInternal(new_body, the_parfor, 1, max_label + 1, state, newLambdaInfo)
    nothing
end

@doc """
Takes a new array of body statements in the process of construction in "new_body" and takes a parfor to add to that
body.  This parfor is in the nested (parfor code is in the parfor node itself) temporary form we use for fusion although 
pre-statements and post-statements are already elevated by this point.  We replace this nested form with a non-nested
form where we have a parfor_start and parfor_end to delineate the parfor code.
"""
function flattenParfor(new_body, the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst)
    dprintln(2,"Flattening ", the_parfor)

    private_set = getPrivateSet(the_parfor.body)
    private_array = collect(private_set)

    # Output to the new body that this is the start of a parfor.
    push!(new_body, TypedExpr(Int64, :parfor_start, PIRParForStartEnd(the_parfor.loopNests, the_parfor.reductions, the_parfor.instruction_count_expr, private_array)))
    # Output the body of the parfor as top-level statements in the new function body.
    append!(new_body, the_parfor.body)
    # Output to the new body that this is the end of a parfor.
    push!(new_body, TypedExpr(Int64, :parfor_end, PIRParForStartEnd(deepcopy(the_parfor.loopNests), deepcopy(the_parfor.reductions), deepcopy(the_parfor.instruction_count_expr), deepcopy(private_array))))
    nothing
end

function toTaskArgName(x :: Symbol, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    x
end
function toTaskArgName(x :: SymbolNode, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    x.name
end
function toTaskArgName(x :: GenSym, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    newstr = string("parforToTask_gensym_", x.id)
    ret    = symbol(newstr)
    gsmap[x] = CompilerTools.LambdaHandling.VarDef(ret, CompilerTools.LambdaHandling.getType(x, lambdaInfo), 0)
    return ret
end

function toTaskArgVarDef(x :: Symbol, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    CompilerTools.LambdaHandling.getVarDef(x, lambdaInfo)
end
function toTaskArgVarDef(x :: SymbolNode, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    CompilerTools.LambdaHandling.getVarDef(x.name, lambdaInfo)
end
function toTaskArgVarDef(x :: GenSym, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    gsmap[x]
end

@doc """
Given a parfor statement index in "parfor_index" in the "body"'s statements, create a TaskInfo node for this parfor.
"""
function parforToTask(parfor_index, bb_statements, body, state)
    assert(typeof(body[parfor_index]) == Expr)
    assert(body[parfor_index].head == :parfor)  # Make sure we got a parfor node to convert.
    the_parfor = body[parfor_index].args[1]     # Get the PIRParForAst object from the :parfor Expr.
    dprintln(3,"parforToTask = ", the_parfor)

    # Create an array of the reduction vars used in this parfor.
    reduction_vars = Symbol[]
    for i in the_parfor.reductions
        push!(reduction_vars, i.reductionVar.name)
    end

    # The call to getIO determines which variables are live at input to this parfor, live at output from this parfor
    # or just used exclusively in the parfor.  These latter become local variables.
    in_vars , out, locals = getIO([parfor_index], bb_statements)
    dprintln(3,"in_vars = ", in_vars)
    dprintln(3,"out_vars = ", out)
    dprintln(3,"local_vars = ", locals)

    # Convert Set to Array
    in_array_names   = Any[]
    modified_symbols = Any[]
    io_symbols       = Any[]
    for i in in_vars
        # Determine for each input var whether the variable is just read, just written, or both.
        swritten = CompilerTools.ReadWriteSet.isWritten(i, the_parfor.rws)
        sread    = CompilerTools.ReadWriteSet.isRead(i, the_parfor.rws)
        sio      = swritten & sread
        if in(i, reduction_vars)
            # reduction_vars must be initialized before the parfor and updated during the parfor 
            # so must be io
            assert(sio)   
        elseif sio
            # If input is both read and written then remember this symbol is input and output.
            push!(io_symbols, i)       
        elseif swritten
            push!(modified_symbols, i)
        else
            push!(in_array_names, i)
        end
    end
    if length(out) != 0
        throw(string("out variable of parfor task not supported right now."))
    end

    # Start to form the lambda VarDef array for the locals to the task function.
    locals_array = CompilerTools.LambdaHandling.VarDef[]
    gensyms = Any[]
    gensyms_table = Dict{SymGen, Any}()
    for i in locals
        if isa(i, Symbol) 
            push!(locals_array, CompilerTools.LambdaHandling.getVarDef(i,state.lambdaInfo))
        elseif isa(i, GenSym) 
            push!(gensyms, CompilerTools.LambdaHandling.getType(i,state.lambdaInfo))
            gensyms_table[i] = GenSym(length(gensyms) - 1)
        else
            assert(false)
        end
    end

    # Form an array of argument access flags for each argument.
    arg_types = Cint[]
    push!(arg_types, ARG_OPT_IN) # for ranges parameter
    for i = 1:length(in_array_names)
        push!(arg_types, ARG_OPT_IN)
    end
    for i = 1:length(modified_symbols)
        push!(arg_types, ARG_OPT_OUT)
    end
    for i = 1:length(io_symbols)
        push!(arg_types, ARG_OPT_INOUT)
    end
    for i in 1:length(reduction_vars)
        push!(arg_types, ARG_OPT_ACCUMULATOR)
    end

    dprintln(3,"in_array_names = ", in_array_names)
    dprintln(3,"modified_symbols = ", modified_symbols)
    dprintln(3,"io_symbols = ", io_symbols)
    dprintln(3,"reduction_vars = ", reduction_vars)
    dprintln(3,"locals_array = ", locals_array)
    dprintln(3,"gensyms = ", gensyms)
    dprintln(3,"gensyms_table = ", gensyms_table)
    dprintln(3,"arg_types = ", arg_types)

    # Will hold GenSym's that would become parameters but can't so need to be made
    # into symbols.
    gsmap = Dict{GenSym,CompilerTools.LambdaHandling.VarDef}()

    # Form an array including symbols for all the in and output parameters plus the additional iteration control parameter "ranges".
    # If we detect a GenSym in the parameter list we replace it with a symbol derived from
    # the GenSym number and add it to gsmap.
    all_arg_names = [:ranges;
                     map(x -> toTaskArgName(x, gsmap, state.lambdaInfo), in_array_names);
                     map(x -> toTaskArgName(x, gsmap, state.lambdaInfo), modified_symbols);
                     map(x -> toTaskArgName(x, gsmap, state.lambdaInfo), io_symbols);
                     map(x -> toTaskArgName(x, gsmap, state.lambdaInfo), reduction_vars)]

    dprintln(3,"gsmap = ", gsmap)
    # Add gsmap to gensyms_table so that the GenSyms in the body can be translated to
    # the corresponding new input parameter name.
    for gsentry in gsmap
      gensyms_table[gsentry[1]] = gsentry[2].name
    end 

    # Form a tuple that contains the type of each parameter.
    all_arg_types_tuple = Expr(:tuple)
    all_arg_types_tuple.args = [
        pir_range_actual;
        map(x -> CompilerTools.LambdaHandling.getType(x, state.lambdaInfo), in_array_names);
        map(x -> CompilerTools.LambdaHandling.getType(x, state.lambdaInfo), modified_symbols);
        map(x -> CompilerTools.LambdaHandling.getType(x, state.lambdaInfo), io_symbols);
        map(x -> CompilerTools.LambdaHandling.getType(x, state.lambdaInfo), reduction_vars)]
    all_arg_type = eval(all_arg_types_tuple)
    # Forms VarDef's for the local variables to the task function.
    args_var = CompilerTools.LambdaHandling.VarDef[]
    push!(args_var, CompilerTools.LambdaHandling.VarDef(:ranges, pir_range_actual, 0))
    append!(args_var, 
      [ map(x -> toTaskArgVarDef(x, gsmap, state.lambdaInfo), in_array_names);
        map(x -> toTaskArgVarDef(x, gsmap, state.lambdaInfo), modified_symbols);
        map(x -> toTaskArgVarDef(x, gsmap, state.lambdaInfo), io_symbols);
        map(x -> toTaskArgVarDef(x, gsmap, state.lambdaInfo), reduction_vars)])
    dprintln(3,"all_arg_names = ", all_arg_names)
    dprintln(3,"all_arg_type = ", all_arg_type)
    dprintln(3,"args_var = ", args_var)

    unique_node_id = get_unique_num()

    # The name of the new task function.
    task_func_name = string("task_func_",unique_node_id)
    task_func_sym  = symbol(task_func_name)

    # Just stub out the new task function...the body and lambda will be replaced below.
    task_func = @eval function ($task_func_sym)($(all_arg_names...))
        throw(string("Some task function's body was not replaced."))
    end
    dprintln(3,"task_func = ", task_func)

    # DON'T DELETE.  Forces function into existence.
    unused_ct = ParallelAccelerator.Driver.code_typed(task_func, all_arg_type)
    dprintln(3, "unused_ct = ", unused_ct)

    newLambdaInfo = CompilerTools.LambdaHandling.LambdaInfo()
    CompilerTools.LambdaHandling.addInputParameters(deepcopy(args_var), newLambdaInfo)
    CompilerTools.LambdaHandling.addLocalVariables(deepcopy(locals_array), newLambdaInfo)
    # Change all variables in the task function to have ASSIGNED desc.
    for vd in newLambdaInfo.var_defs
        var_def = vd[2]
        var_def.desc = CompilerTools.LambdaHandling.ISASSIGNED
    end
    newLambdaInfo.gen_sym_typs = gensyms

    # Creating the new body for the task function.
    task_body = TypedExpr(Int, :body)
    saved_loopNests = deepcopy(the_parfor.loopNests)

    #  for i in all_arg_names
    #    push!(task_body.args, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), string(i), " = ", Symbol(i)))
    #  end

    if ParallelAccelerator.getTaskMode() != ParallelAccelerator.NO_TASK_MODE
        for i = 1:length(the_parfor.loopNests)
            # Put outerloop first in the loopNest
            j = length(the_parfor.loopNests) - i + 1
            the_parfor.loopNests[j].lower = TypedExpr(Int64, :call, TopNode(:add_int),
            TypedExpr(Int64, :call, TopNode(:unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, TopNode(:getfield), :ranges, QuoteNode(:lower_bounds)), i),
            1)
            the_parfor.loopNests[j].upper = TypedExpr(Int64, :call, TopNode(:add_int), 
            TypedExpr(Int64, :call, TopNode(:unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, TopNode(:getfield), :ranges, QuoteNode(:upper_bounds)), i),
            1)
        end
    elseif ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
        for i = 1:length(the_parfor.loopNests)
            # Put outerloop first in the loopNest
            j = length(the_parfor.loopNests) - i + 1
            the_parfor.loopNests[j].lower = TypedExpr(Int64, :call, TopNode(:unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, TopNode(:getfield), :ranges, QuoteNode(:lower_bounds)), i)
            the_parfor.loopNests[j].upper = TypedExpr(Int64, :call, TopNode(:unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, TopNode(:getfield), :ranges, QuoteNode(:upper_bounds)), i)
        end
    end

    dprintln(3, "Before recreation or flattening")

    # Add the parfor stmt to the task function body.
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
        #push!(task_body.args, Expr(:call, GlobalRef(Base,:println), GlobalRef(Base,:STDOUT), "in task func"))
        recreateLoops(task_body.args, the_parfor, state, newLambdaInfo)
    else
        flattenParfor(task_body.args, the_parfor)
    end

    # Add the return statement to the end of the task function.
    # If this is not a reduction parfor then return "nothing".
    # If it is a reduction in threading mode, return a tuple of the reduction variables.
    # The threading infrastructure will then call a user-specified reduction function.
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE &&
       length(the_parfor.reductions) > 0
        ret_tt = Expr(:tuple)
        ret_tt.args = map(x -> x.reductionVar.typ, the_parfor.reductions)
        ret_types = eval(ret_tt)
        dprintln(3, "ret_types = ", ret_types)

        ret_names = map(x -> x.reductionVar.name, the_parfor.reductions)
        dprintln(3, "ret_names = ", ret_names)

        push!(task_body.args, TypedExpr(ret_types, :return, mk_tuple_expr(ret_names, ret_types)))
    else
        #push!(task_body.args, TypedExpr(Void, :return, 0))
        push!(task_body.args, TypedExpr(Void, :return, nothing))
    end

    task_body = CompilerTools.LambdaHandling.replaceExprWithDict!(task_body, gensyms_table)
    # Create the new :lambda Expr for the task function.
    code = CompilerTools.LambdaHandling.lambdaInfoToLambdaExpr(newLambdaInfo, task_body)

    dprintln(3, "New task = ", code)

    m = methods(task_func, all_arg_type)
    if length(m) < 1
        error("Method for ", task_func_name, " is not found")
    else
        dprintln(3,"New ", task_func_name, " = ", m)
    end
    def = m[1].func.code
    dprintln(3, "def = ", def, " type = ", typeof(def))
    dprintln(3, "tfunc = ", def.tfunc)
    def.tfunc[2] = ccall(:jl_compress_ast, Any, (Any,Any), def, code)

    m = methods(task_func, all_arg_type)
    def = m[1].func.code
    if ParallelAccelerator.getPseMode() != ParallelAccelerator.THREADS_MODE
        def.j2cflag = convert(Int32,6)
        ccall(:set_j2c_task_arg_types, Void, (Ptr{UInt8}, Cint, Ptr{Cint}), task_func_name, length(arg_types), arg_types)
    end
    precompile(task_func, all_arg_type)
    dprintln(3, "def post = ", def, " type = ", typeof(def))

    if DEBUG_LVL >= 3
        task_func_ct = ParallelAccelerator.Driver.code_typed(task_func, all_arg_type)
        if length(task_func_ct) == 0
            println("Error getting task func code.\n")
        else
            task_func_ct = task_func_ct[1]
            println("Task func code for ", task_func)
            println(task_func_ct)    
        end
    end

    # If this task has reductions, then we create a Julia buffer that holds a C function that we build up in the section below.
    # We then force this C code into the rest of the C code generated by cgen with a special call.
    reduction_func_name = string("")
    if length(the_parfor.reductions) > 0
        if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
            if true
            reduction_func_name = string("reduction_func_",unique_node_id)
            fstring = string("function ", reduction_func_name, "(in1, in2)\n")
            fstring = string(fstring, "return (")

            for i = 1:length(the_parfor.reductions)
                fstring = string(fstring, "in1[", i, "]")
                if the_parfor.reductions[i].reductionFunc == :+
                    fstring = string(fstring, "+")
                elseif the_parfor.reductions[i].reductionFunc == :*
                    fstring = string(fstring, "*")
                else
                    throw(string("Unsupported reduction function ", the_parfor.reductions[i].reductionFunc, " during join function generation."))
                end

                fstring = string(fstring, "in2[", i, "]")

                # Add comma between tuple elements if there is at least one more tuple element.
                if i != length(the_parfor.reductions)
                    fstring = string(fstring, " , ")
                end
            end
 
            fstring = string(fstring, ")\nend\n")
            dprintln(3, "Reduction function for threads mode = ", fstring)
            fparse  = parse(fstring)
            feval   = eval(fparse)
            dprintln(3, "Reduction function for threads done.")
            end
        else
            # The name of the new reduction function.
            reduction_func_name = string("reduction_func_",unique_node_id)

            the_types = AbstractString[]
            for i = 1:length(the_parfor.reductions)
                if the_parfor.reductions[i].reductionVar.typ == Float64
                    push!(the_types, "double")
                elseif the_parfor.reductions[i].reductionVar.typ == Float32
                    push!(the_types, "float")
                elseif the_parfor.reductions[i].reductionVar.typ == Int64
                    push!(the_types, "int64_t")
                elseif the_parfor.reductions[i].reductionVar.typ == Int32
                    push!(the_types, "int32_t")
                else
                    throw(string("Unsupported reduction var type ", the_parfor.reductions[i].reductionVar.typ))
                end
            end

            # The reduction function doesn't return anything.
            c_reduction_func = string("void _")
            c_reduction_func = string(c_reduction_func, reduction_func_name)
            # All reduction functions have the same signature so that the runtime can call back into them.
            # The accumulator is effectively a pointer to Any[].  This accumulator is initialized by the runtime with the initial reduction value for the given type.
            # The new_reduction_vars are also a pointer to Any[] and contains new value to merge into the accumulating reduction var in accumulator.
            c_reduction_func = string(c_reduction_func, "(void **accumulator, void **new_reduction_vars) {\n")
            for i = 1:length(the_parfor.reductions)
                # For each reduction, put one statement in the C function.  Figure out here if it is a + or * reduction.
                if the_parfor.reductions[i].reductionFunc == :+
                    this_op = "+"
                elseif the_parfor.reductions[i].reductionFunc == :*
                    this_op = "*"
                else
                    throw(string("Unsupported reduction function ", the_parfor.reductions[i].reductionFunc, " during join function generation."))
                end
                # Add the statement to the function basically of this form *(accumulator[i]) = *(accumulator[i]) + *(new_reduction_vars[i]).
                # but instead of "+" use the operation type determined above stored in "this_op" and before you can dereference each element, you have to cast
                # the pointer to the appropriate type for this reduction element as stored in the_types[i].
                c_reduction_func = string(c_reduction_func, "*((", the_types[i], "*)accumulator[", i-1, "]) = *((", the_types[i], "*)accumulator[", i-1, "]) ", this_op, " *((", the_types[i], "*)new_reduction_vars[", i-1, "]);\n")
            end
            c_reduction_func = string(c_reduction_func, "}\n")
            dprintln(3,"Created reduction function is:")
            dprintln(3,c_reduction_func)

            # Tell cgen to put this reduction function directly into the C code.
            ccall(:set_j2c_add_c_code, Void, (Ptr{UInt8},), c_reduction_func)
        end
    end

    dprintln(3,"End of parforToTask")

    ret = TaskInfo(task_func,     # The task function that we just generated of type Function.
                    task_func_sym, # The task function's Symbol name.
                    reduction_func_name, # The name of the C reduction function created for this task.
                    map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.lambdaInfo)), in_array_names),
                    map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.lambdaInfo)), modified_symbols),
                    map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.lambdaInfo)), io_symbols),
                    map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.lambdaInfo)), reduction_vars),
                    code,          # The AST for the task function.
                    saved_loopNests)
    return ret
end

@doc """
Form a task out of a range of sequential statements.
This is not currently implemented.
"""
function seqTask(body_indices, bb_statements, body, state)
    getIO(body_indices, bb_statements)  
    throw(string("seqTask construction not implemented yet."))
    TaskInfo(:FIXFIXFIX, :FIXFIXFIX, Any[], Any[], nothing, PIRLoopNest[])
end

@doc """
Pretty print the args part of the "body" of a :lambda Expr at a given debug level in "dlvl".
"""
function printBody(dlvl, body :: Array{Any,1})
    for i = 1:length(body)
        dprintln(dlvl, "    ", body[i])
    end
end

function printBody(dlvl, body :: Expr)
    printBody(dlvl, body.args)
end

@doc """
Pretty print a :lambda Expr in "node" at a given debug level in "dlvl".
"""
function printLambda(dlvl, node :: Expr)
    assert(node.head == :lambda)
    dprintln(dlvl, "Lambda:")
    dprintln(dlvl, "Input parameters: ", node.args[1])
    dprintln(dlvl, "Metadata: ", node.args[2])
    body = node.args[3]
    if typeof(body) != Expr
        dprintln(0, "printLambda got ", typeof(body), " for a body, len = ", length(node.args))
        dprintln(0, node)
    end
    assert(body.head == :body)
    dprintln(dlvl, "typeof(body): ", body.typ)
    printBody(dlvl, body.args)
    if body.typ == Any
        dprintln(1,"Body type is Any.")
    end
end

@doc """
A LivenessAnalysis callback that handles ParallelIR introduced AST node types.
For each ParallelIR specific node type, form an array of expressions that liveness
can analysis to reflect the read/write set of the given AST node.
If we read a symbol it is sufficient to just return that symbol as one of the expressions.
If we write a symbol, then form a fake mk_assignment_expr just to get liveness to realize the symbol is written.
"""
function pir_live_cb(ast :: ANY, cbdata :: ANY)
    dprintln(4,"pir_live_cb")
    asttyp = typeof(ast)
    if asttyp == Expr
        head = ast.head
        args = ast.args
        if head == :parfor
            dprintln(3,"pir_live_cb for :parfor")
            expr_to_process = Any[]

            assert(typeof(args[1]) == ParallelAccelerator.ParallelIR.PIRParForAst)
            this_parfor = args[1]

            append!(expr_to_process, this_parfor.preParFor)
            for i = 1:length(this_parfor.loopNests)
                # force the indexVariable to be treated as an rvalue
                push!(expr_to_process, mk_untyped_assignment(this_parfor.loopNests[i].indexVariable, 1))
                push!(expr_to_process, this_parfor.loopNests[i].lower)
                push!(expr_to_process, this_parfor.loopNests[i].upper)
                push!(expr_to_process, this_parfor.loopNests[i].step)
            end
            #emptyLambdaInfo = CompilerTools.LambdaHandling.LambdaInfo()
            #fake_body = CompilerTools.LambdaHandling.lambdaInfoToLambdaExpr(emptyLambdaInfo, TypedExpr(nothing, :body, this_parfor.body...))
            assert(typeof(cbdata) == CompilerTools.LambdaHandling.LambdaInfo)
            fake_body = CompilerTools.LambdaHandling.lambdaInfoToLambdaExpr(cbdata, TypedExpr(nothing, :body, this_parfor.body...))

            body_lives = CompilerTools.LivenessAnalysis.from_expr(fake_body, pir_live_cb, cbdata)
            live_in_to_start_block = body_lives.basic_blocks[body_lives.cfg.basic_blocks[-1]].live_in
            all_defs = Set()
            for bb in body_lives.basic_blocks
                all_defs = union(all_defs, bb[2].def)
            end 
            as = CompilerTools.LivenessAnalysis.AccessSummary(setdiff(all_defs, live_in_to_start_block), live_in_to_start_block)

            push!(expr_to_process, as)

            append!(expr_to_process, this_parfor.postParFor)

            return expr_to_process
        elseif head == :parfor_start
            dprintln(3,"pir_live_cb for :parfor_start")
            expr_to_process = Any[]

            assert(typeof(args[1]) == PIRParForStartEnd)
            this_parfor = args[1]

            for i = 1:length(this_parfor.loopNests)
                # Force the indexVariable to be treated as an rvalue
                push!(expr_to_process, mk_untyped_assignment(this_parfor.loopNests[i].indexVariable, 1))
                push!(expr_to_process, this_parfor.loopNests[i].lower)
                push!(expr_to_process, this_parfor.loopNests[i].upper)
                push!(expr_to_process, this_parfor.loopNests[i].step)
            end

            return expr_to_process
        elseif head == :parfor_end
            # Intentionally do nothing
            return Any[]
        elseif head == :insert_divisible_task
            # Is this right?  Do I need pir_range stuff here too?
            dprintln(3,"pir_live_cb for :insert_divisible_task")
            expr_to_process = Any[]

            cur_task = args[1]
            assert(typeof(cur_task) == InsertTaskNode)

            for i = 1:length(cur_task.args)
                if cur_task.args[i].options == ARG_OPT_IN
                    push!(expr_to_process, cur_task.args[i].value)
                else
                    push!(expr_to_process, mk_untyped_assignment(cur_task.args[i].value, 1))
                end
            end

            return expr_to_process
        elseif head == :loophead
            dprintln(3,"pir_live_cb for :loophead")
            assert(length(args) == 3)

            expr_to_process = Any[]
            push!(expr_to_process, mk_untyped_assignment(SymbolNode(args[1], Int64), 1))  # force args[1] to be seen as an rvalue
            push!(expr_to_process, args[2])
            push!(expr_to_process, args[3])

            return expr_to_process
        elseif head == :loopend
            # There is nothing really interesting in the loopend node to signify something being read or written.
            assert(length(args) == 1)
            return Any[]
        elseif head == :call
            if args[1] == TopNode(:unsafe_arrayref)
                expr_to_process = Any[]
                new_expr = deepcopy(ast)
                new_expr.args[1] = TopNode(:arrayref)
                push!(expr_to_process, new_expr)
                return expr_to_process
            elseif args[1] == TopNode(:safe_arrayref)
                expr_to_process = Any[]
                new_expr = deepcopy(ast)
                new_expr.args[1] = TopNode(:arrayref)
                push!(expr_to_process, new_expr)
                return expr_to_process
            elseif args[1] == TopNode(:unsafe_arrayset)
                expr_to_process = Any[]
                new_expr = deepcopy(ast)
                new_expr.args[1] = TopNode(:arrayset)
                push!(expr_to_process, new_expr)
                return expr_to_process
            end
        elseif head == :(=)
            dprintln(3,"pir_live_cb for :(=)")
            if length(args) > 2
                expr_to_process = Any[]
                push!(expr_to_process, args[1])
                push!(expr_to_process, args[2])
                for i = 4:length(args)
                    push!(expr_to_process, args[i])
                end
                return expr_to_process
            end
        end
    end
    return DomainIR.dir_live_cb(ast, cbdata)
end

@doc """
Sometimes statements we exist in the AST of the form a=Expr where a is a Symbol that isn't live past the assignment
and we'd like to eliminate the whole assignment statement but we have to know that the right-hand side has no
side effects before we can do that.  This function says whether the right-hand side passed into it has side effects
or not.  Several common function calls that otherwise we wouldn't know are safe are explicitly checked for.
"""
function hasNoSideEffects(node :: Union{Symbol, SymbolNode, GenSym, LambdaStaticData, Number})
    return true
end

function hasNoSideEffects(node)
    return false
end

function hasNoSideEffects(node :: Expr)
    if node.head == :ccall
        func = node.args[1]
        if func == QuoteNode(:jl_alloc_array_1d) ||
            func == QuoteNode(:jl_alloc_array_2d)
            return true
        end
    elseif node.head == :call1
        func = node.args[1]
        if func == TopNode(:apply_type) ||
            func == TopNode(:tuple)
            return true
        end
    elseif node.head == :lambda
        return true
    elseif node.head == :new
        if node.args[1] <: Range
            return true
        end
    elseif node.head == :call
        func = node.args[1]
        if func == TopNode(:box) ||
            func == TopNode(:tuple) ||
            func == TopNode(:getindex_bool_1d) ||
            func == :getindex
            return true
        end
    end

    return false
end

@doc """
Implements one of the main ParallelIR passes to remove assertEqShape AST nodes from the body if they are statically known to be in the same equivalence class.
"""
function removeAssertEqShape(args :: Array{Any,1}, state)
    newBody = Any[]
    for i = 1:length(args)
        # Add the current statement to the new body unless the statement is an assertEqShape Expr and the array in the assertEqShape are known to be the same size.
        if !(typeof(args[i]) == Expr && args[i].head == :assertEqShape && from_assertEqShape(args[i], state))
            push!(newBody, args[i])
        end
    end
    return newBody
end

@doc """
Create array equivalences from an assertEqShape AST node.
There are two arrays in the args to assertEqShape.
"""
function from_assertEqShape(node::Expr, state)
    dprintln(3,"from_assertEqShape ", node)
    a1 = node.args[1]    # first array to compare
    a2 = node.args[2]    # second array to compare
    a1_corr = getOrAddArrayCorrelation(toSymGen(a1), state)  # get the length set of the first array
    a2_corr = getOrAddArrayCorrelation(toSymGen(a2), state)  # get the length set of the second array
    if a1_corr == a2_corr
        # If they are the same then return an empty array so that the statement is eliminated.
        dprintln(3,"assertEqShape statically verified and eliminated for ", a1, " and ", a2)
        return true
    else
        dprintln(3,"a1 = ", a1, " ", a1_corr, " a2 = ", a2, " ", a2_corr, " correlations = ", state.array_length_correlation)
        # If assertEqShape is called on e.g., inputs, then we can't statically eliminate the assignment
        # but if the assert doesn't fire then we do thereafter know that the arrays are in the same length set.
        merge_correlations(state, a1_corr, a2_corr)
        dprintln(3,"assertEqShape NOT statically verified.  Merge correlations = ", state.array_length_correlation)
        return false
    end
end

function from_assignment_fusion(args::Array{Any,1}, depth, state)
    lhs = args[1]
    rhs = args[2]
    dprintln(3,"from_assignment lhs = ", lhs)
    dprintln(3,"from_assignment rhs = ", rhs)
    if isa(rhs, Expr) && rhs.head == :lambda
        # skip handling rhs lambdas
        rhs = [rhs]
    else
        rhs = from_expr(rhs, depth, state, false)
    end
    dprintln(3,"from_assignment rhs after = ", rhs)
    assert(isa(rhs,Array))
    assert(length(rhs) == 1)
    rhs = rhs[1]

    # Eliminate assignments to variables which are immediately dead.
    # The variable name.
    lhsName = toSymGen(lhs)
    # Get liveness information for the current statement.
    statement_live_info = CompilerTools.LivenessAnalysis.find_top_number(state.top_level_number, state.block_lives)
    @assert statement_live_info!=nothing "$(state.top_level_number) $(state.block_lives)"

    dprintln(3,statement_live_info)
    dprintln(3,"def = ", statement_live_info.def)

    # Make sure this variable is listed as a "def" for this statement.
    assert(CompilerTools.LivenessAnalysis.isDef(lhsName, statement_live_info))

    # If the lhs symbol is not in the live out information for this statement then it is dead.
    if !in(lhsName, statement_live_info.live_out) && hasNoSideEffects(rhs)
        dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
        # Eliminate the statement.
        return [], nothing
    end

    @assert typeof(rhs)==Expr && rhs.head==:parfor "Expected :parfor assignment"
    out_typ = rhs.typ
    #dprintln(3, "from_assignment rhs is Expr, type = ", out_typ, " rhs.head = ", rhs.head, " rhs = ", rhs)
    # If we have "a = parfor(...)" then record that array "a" has the same length as the output array of the parfor.
    the_parfor = rhs.args[1]
    for i = 4:length(args)
        rhs_entry = the_parfor.postParFor[end][i-3]
        assert(typeof(args[i]) == SymbolNode)
        assert(typeof(rhs_entry) == SymbolNode)
        if rhs_entry.typ.name == Array.name
            add_merge_correlations(toSymGen(rhs_entry), toSymGen(args[i]), state)
        end
    end

    return [toSNGen(lhs, out_typ); rhs], out_typ
end

@doc """
Process an assignment expression.
Starts by recurisvely processing the right-hand side of the assignment.
Eliminates the assignment of a=b if a is dead afterwards and b has no side effects.
    Does some array equivalence class work which may be redundant given that we now run a separate equivalence class pass so consider removing that part of this code.
"""
function from_assignment(lhs, rhs, depth, state)
    # :(=) assignment
    # ast = [ ... ]
    dprintln(3,"from_assignment lhs = ", lhs)
    dprintln(3,"from_assignment rhs = ", rhs)
    if isa(rhs, Expr) && rhs.head == :lambda
        # skip handling rhs lambdas
        rhs = [rhs]
    else
        rhs = from_expr(rhs, depth, state, false)
    end
    dprintln(3,"from_assignment rhs after = ", rhs)
    assert(isa(rhs,Array))
    assert(length(rhs) == 1)
    rhs = rhs[1]

    # Eliminate assignments to variables which are immediately dead.
    # The variable name.
    lhsName = toSymGen(lhs)
    # Get liveness information for the current statement.
    statement_live_info = CompilerTools.LivenessAnalysis.find_top_number(state.top_level_number, state.block_lives)
    @assert statement_live_info!=nothing "$(state.top_level_number) $(state.block_lives)"

    dprintln(3,statement_live_info)
    dprintln(3,"def = ", statement_live_info.def)

    # Make sure this variable is listed as a "def" for this statement.
    assert(CompilerTools.LivenessAnalysis.isDef(lhsName, statement_live_info))

    # If the lhs symbol is not in the live out information for this statement then it is dead.
    if !in(lhsName, statement_live_info.live_out) && hasNoSideEffects(rhs)
        dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
        # Eliminate the statement.
        return [], nothing
    end

    if typeof(rhs) == Expr
        out_typ = rhs.typ
        #dprintln(3, "from_assignment rhs is Expr, type = ", out_typ, " rhs.head = ", rhs.head, " rhs = ", rhs)

        # If we have "a = parfor(...)" then record that array "a" has the same length as the output array of the parfor.
        if rhs.head == :parfor
            the_parfor = rhs.args[1]
            if !(isa(out_typ, Tuple)) && out_typ.name == Array.name # both lhs and out_typ could be a tuple
                dprintln(3,"Adding parfor array length correlation ", lhs, " to ", rhs.args[1].postParFor[end])
                add_merge_correlations(toSymGen(the_parfor.postParFor[end]), lhsName, state)
            end
            # assertEqShape nodes can prevent fusion and slow things down regardless so we can try to remove them
            # statically if our array length correlations indicate they are in the same length set.
        elseif rhs.head == :assertEqShape
            if from_assertEqShape(rhs, state)
                return [], nothing
            end
        elseif rhs.head == :call
            dprintln(3, "Detected call rhs in from_assignment.")
            dprintln(3, "from_assignment call, arg1 = ", rhs.args[1])
            if length(rhs.args) > 1
                dprintln(3, " arg2 = ", rhs.args[2])
            end
            if rhs.args[1] == TopNode(:ccall)
                if rhs.args[2] == QuoteNode(:jl_alloc_array_1d)
                    dim1 = rhs.args[7]
                    dprintln(3, "Detected 1D array allocation. dim1 = ", dim1, " type = ", typeof(dim1))
                    if typeof(dim1) == SymbolNode
                        si1 = CompilerTools.LambdaHandling.getDesc(dim1.name, state.lambdaInfo)
                        if si1 & ISASSIGNEDONCE == ISASSIGNEDONCE
                            dprintln(3, "Will establish array length correlation for const size ", dim1)
                            getOrAddSymbolCorrelation(lhsName, state, SymGen[dim1.name])
                        end
                    end
                elseif rhs.args[2] == QuoteNode(:jl_alloc_array_2d)
                    dim1 = rhs.args[7]
                    dim2 = rhs.args[9]
                    dprintln(3, "Detected 2D array allocation. dim1 = ", dim1, " dim2 = ", dim2)
                    if typeof(dim1) == SymbolNode && typeof(dim2) == SymbolNode
                        si1 = CompilerTools.LambdaHandling.getDesc(dim1.name, state.lambdaInfo)
                        si2 = CompilerTools.LambdaHandling.getDesc(dim2.name, state.lambdaInfo)
                        if (si1 & ISASSIGNEDONCE == ISASSIGNEDONCE) && (si2 & ISASSIGNEDONCE == ISASSIGNEDONCE)
                            dprintln(3, "Will establish array length correlation for const size ", dim1, " ", dim2)
                            getOrAddSymbolCorrelation(lhsName, state, SymGen[dim1.name, dim2.name])
                            dprintln(3, "correlations = ", state.array_length_correlation)
                        end
                    end
                end
            end
        end
    elseif typeof(rhs) == SymbolNode
        out_typ = rhs.typ
        if DomainIR.isarray(out_typ)
            # Add a length correlation of the form "a = b".
            dprintln(3,"Adding array length correlation ", lhs, " to ", rhs.name)
            add_merge_correlations(toSymGen(rhs), lhsName, state)
        end
    else
        # Get the type of the lhs from its metadata declaration.
        out_typ = CompilerTools.LambdaHandling.getType(lhs, state.lambdaInfo)
    end

    return [toSNGen(lhs, out_typ); rhs], out_typ
end

@doc """
If we have the type, convert a Symbol to SymbolNode.
If we have a GenSym then we have to keep it.
"""
function toSNGen(x :: Symbol, typ)
    return SymbolNode(x, typ)
end

function toSNGen(x :: SymbolNode, typ)
    return x
end

function toSNGen(x :: GenSym, typ)
    return x
end

function toSNGen(x, typ)
    xtyp = typeof(x)
    throw(string("Found object type ", xtyp, " for object ", x, " in toSNGen and don't know what to do with it."))
end

@doc """
Process a call AST node.
"""
function from_call(ast::Array{Any,1}, depth, state)
    assert(length(ast) >= 1)
    fun  = ast[1]
    args = ast[2:end]
    dprintln(2,"from_call fun = ", fun, " typeof fun = ", typeof(fun))
    if length(args) > 0
        dprintln(2,"first arg = ",args[1], " type = ", typeof(args[1]))
    end
    # We don't need to translate Function Symbols but potentially other call targets we do.
    if typeof(fun) != Symbol
        fun = from_expr(fun, depth, state, false)
        assert(isa(fun,Array))
        assert(length(fun) == 1)
        fun = fun[1]
    end
    # Recursively process the arguments to the call.  
    args = from_exprs(args, depth+1, state)

    return [fun; args]
end

@doc """
State to aide in the copy propagation phase.
"""
type CopyPropagateState
    lives  :: CompilerTools.LivenessAnalysis.BlockLiveness
    copies :: Dict{SymGen, SymGen}

    function CopyPropagateState(l, c)
        new(l,c)
    end
end

@doc """
In each basic block, if there is a "copy" (i.e., something of the form "a = b") then put
that in copies as copies[a] = b.  Then, later in the basic block if you see the symbol
"a" then replace it with "b".  Note that this is not SSA so "a" may be written again
and if it is then it must be removed from copies.
"""
function copy_propagate(node :: ANY, data :: CopyPropagateState, top_level_number, is_top_level, read)
    dprintln(3,"copy_propagate starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    dprintln(3,"copy_propagate node = ", node, " type = ", typeof(node))
    if typeof(node) == Expr
        dprintln(3,"node.head = ", node.head)
    end
    ntype = typeof(node)

    if is_top_level
        dprintln(3,"copy_propagate is_top_level")
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)

        if live_info != nothing
            # Remove elements from data.copies if the original RHS is modified by this statement.
            # For each symbol modified by this statement...
            for def in live_info.def
                dprintln(4,"Symbol ", def, " is modifed by current statement.")
                # For each copy we currently have recorded.
                for copy in data.copies
                    dprintln(4,"Current entry in data.copies = ", copy)
                    # If the rhs of the copy is modified by the statement.
                    if def == copy[2]
                        dprintln(3,"RHS of data.copies is modified so removing ", copy," from data.copies.")
                        # Then remove the lhs = rhs entry from copies.
                        delete!(data.copies, copy[1])
                    elseif def == copy[1]
                        # LHS is def.  We can maintain the mapping if RHS is dead.
                        if in(copy[2], live_info.live_out)
                            dprintln(3,"LHS of data.copies is modified and RHS is live so removing ", copy," from data.copies.")
                            # Then remove the lhs = rhs entry from copies.
                            delete!(data.copies, copy[1])
                        end
                    end
                end
            end
        end

        if isa(node, LabelNode) || isa(node, GotoNode) || (isa(node, Expr) && is(node.head, :gotoifnot))
            # Only copy propagate within a basic block.  this is now a new basic block.
            data.copies = Dict{SymGen, SymGen}() 
        elseif isAssignmentNode(node)
            dprintln(3,"Is an assignment node.")
            lhs = node.args[1] = AstWalk(node.args[1], copy_propagate, data)
            dprintln(4,lhs)
            rhs = node.args[2] = AstWalk(node.args[2], copy_propagate, data)
            dprintln(4,rhs)

            if isa(rhs, SymAllGen)
                dprintln(3,"Creating copy, lhs = ", lhs, " rhs = ", rhs)
                # Record that the left-hand side is a copy of the right-hand side.
                data.copies[toSymGen(lhs)] = toSymGen(rhs)
            end
            return node
        end
    end

    if isa(node, Symbol)
        if haskey(data.copies, node)
            dprintln(3,"Replacing ", node, " with ", data.copies[node])
            return data.copies[node]
        end
    elseif isa(node, SymbolNode)
        if haskey(data.copies, node.name)
            dprintln(3,"Replacing ", node.name, " with ", data.copies[node.name])
            tmp_node = data.copies[node.name]
            return isa(tmp_node, Symbol) ? SymbolNode(tmp_node, node.typ) : tmp_node
        end
    elseif isa(node, GenSym)
        if haskey(data.copies, node)
            dprintln(3,"Replacing ", node, " with ", data.copies[node])
            return data.copies[node]
        end
    elseif isa(node, DomainLambda)
        dprintln(3,"Found DomainLambda in copy_propagate, dl = ", node)
        intersection_dict = Dict{SymGen,Any}()
        for copy in data.copies
            if haskey(node.linfo.escaping_defs, copy[1])
                ed = node.linfo.escaping_defs[copy[1]]
                intersection_dict[copy[1]] = SymbolNode(copy[2], ed.typ)
                delete!(node.linfo.escaping_defs, copy[1])
                node.linfo.escaping_defs[copy[2]] = ed
            end
        end 
        dprintln(3,"Intersection dict = ", intersection_dict)
        if !isempty(intersection_dict)
            origBody      = node.genBody
            newBody(linfo, args) = CompilerTools.LambdaHandling.replaceExprWithDict(origBody(linfo, args), intersection_dict)
            node.genBody  = newBody
            return node
        end 
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
Holds liveness information for the remove_dead AstWalk phase.
"""
type RemoveDeadState
    lives :: CompilerTools.LivenessAnalysis.BlockLiveness
end

@doc """
An AstWalk callback that uses liveness information in "data" to remove dead stores.
"""
function remove_dead(node, data :: RemoveDeadState, top_level_number, is_top_level, read)
    dprintln(3,"remove_dead starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    dprintln(3,"remove_dead node = ", node, " type = ", typeof(node))
    if typeof(node) == Expr
        dprintln(3,"node.head = ", node.head)
    end
    ntype = typeof(node)

    if is_top_level
        dprintln(3,"remove_dead is_top_level")
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        if live_info != nothing
            dprintln(3,"remove_dead live_info = ", live_info)
            dprintln(3,"remove_dead live_info.use = ", live_info.use)

            if isAssignmentNode(node)
                dprintln(3,"Is an assignment node.")
                lhs = node.args[1]
                dprintln(4,lhs)
                rhs = node.args[2]
                dprintln(4,rhs)

                if typeof(lhs) == SymbolNode || typeof(lhs) == Symbol
                    lhs_sym = getSName(lhs)
                    dprintln(3,"remove_dead found assignment with lhs symbol ", lhs, " ", rhs, " typeof(rhs) = ", typeof(rhs))
                    # Remove a dead store
                    if !in(lhs_sym, live_info.live_out)
                        dprintln(3,"remove_dead lhs is NOT live out")
                        if hasNoSideEffects(rhs)
                            dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
                            return CompilerTools.AstWalker.ASTWALK_REMOVE
                        else
                            # Just eliminate the assignment but keep the rhs
                            dprintln(3,"Eliminating dead variable but keeping rhs, dead = ", lhs_sym, " rhs = ", rhs)
                            return rhs
                        end
                    end
                end
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

type DictInfo
    live_info
    expr
end

@doc """
State for the remove_no_deps and insert_no_deps_beginning phases.
"""
type RemoveNoDepsState
    lives             :: CompilerTools.LivenessAnalysis.BlockLiveness
    top_level_no_deps :: Array{Any,1}
    hoistable_scalars :: Set{SymGen}
    dict_sym          :: Dict{SymGen, DictInfo}

    function RemoveNoDepsState(l, hs)
        new(l, Any[], hs, Dict{SymGen, DictInfo}())
    end
end

@doc """
Works with remove_no_deps below to move statements with no dependencies to the beginning of the AST.
"""
function insert_no_deps_beginning(node, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    if is_top_level && top_level_number == 1
        return [data.top_level_no_deps; node]
    end
    nothing
end

@doc """
# This routine gathers up nodes that do not use
# any variable and removes them from the AST into top_level_no_deps.  This works in conjunction with
# insert_no_deps_beginning above to move these statements with no dependencies to the beginning of the AST
# where they can't prevent fusion.
"""
function remove_no_deps(node :: ANY, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    dprintln(3,"remove_no_deps starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    dprintln(3,"remove_no_deps node = ", node, " type = ", typeof(node))
    if typeof(node) == Expr
        dprintln(3,"node.head = ", node.head)
    end
    ntype = typeof(node)

    if is_top_level
        dprintln(3,"remove_no_deps is_top_level")

        if isa(node, LabelNode) || isa(node, GotoNode) || (isa(node, Expr) && is(node.head, :gotoifnot))
            # Empty the state at the end or begining of a basic block
            data.dict_sym = Dict{SymGen,DictInfo}()
        end

        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        # Remove line number statements.
        if ntype == LineNumberNode || (ntype == Expr && node.head == :line)
            return CompilerTools.AstWalker.ASTWALK_REMOVE
        end
        if live_info == nothing
            dprintln(3,"remove_no_deps no live_info")
        else
            dprintln(3,"remove_no_deps live_info = ", live_info)
            dprintln(3,"remove_no_deps live_info.use = ", live_info.use)

            if isAssignmentNode(node)
                dprintln(3,"Is an assignment node.")
                lhs = node.args[1]
                dprintln(4,lhs)
                rhs = node.args[2]
                dprintln(4,rhs)

                if isa(rhs, Expr) && (is(rhs.head, :parfor) || is(rhs.head, :mmap!))
                    # Always keep parfor assignment in order to work with fusion
                    dprintln(3, "keep assignment due to parfor or mmap! node")
                    return node
                end
                if isa(lhs, SymAllGen)
                    lhs_sym = toSymGen(lhs)
                    dprintln(3,"remove_no_deps found assignment with lhs symbol ", lhs, " ", rhs, " typeof(rhs) = ", typeof(rhs))
                    # Remove a dead store
                    if !in(lhs_sym, live_info.live_out)
                        dprintln(3,"remove_no_deps lhs is NOT live out")
                        if hasNoSideEffects(rhs)
                            dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
                            return CompilerTools.AstWalker.ASTWALK_REMOVE
                        else
                            # Just eliminate the assignment but keep the rhs
                            dprintln(3,"Eliminating dead variable but keeping rhs, dead = ", lhs_sym)
                            return rhs
                        end
                    else
                        dprintln(3,"remove_no_deps lhs is live out")
                        if isa(rhs, SymAllGen)
                            rhs_sym = toSymGen(rhs)
                            dprintln(3,"remove_no_deps rhs is symbol ", rhs_sym)
                            if !in(rhs_sym, live_info.live_out)
                                dprintln(3,"remove_no_deps rhs is NOT live out")
                                if haskey(data.dict_sym, rhs_sym)
                                    di = data.dict_sym[rhs_sym]
                                    di_live = di.live_info
                                    prev_expr = di.expr

                                    if !in(lhs_sym, di_live.live_out)
                                        prev_expr.args[1] = lhs_sym
                                        delete!(data.dict_sym, rhs_sym)
                                        data.dict_sym[lhs_sym] = DictInfo(di_live, prev_expr)
                                        dprintln(3,"Lhs is live but rhs is not so substituting rhs for lhs ", lhs_sym, " => ", rhs_sym)
                                        dprintln(3,"New expr = ", prev_expr)
                                        return CompilerTools.AstWalker.ASTWALK_REMOVE
                                    else
                                        delete!(data.dict_sym, rhs_sym)
                                        dprintln(3,"Lhs is live but rhs is not.  However, lhs is read between def of rhs and current statement so not substituting.")
                                    end
                                end
                            else
                                dprintln(3,"Lhs and rhs are live so forgetting assignment ", lhs_sym, " ", rhs_sym)
                                delete!(data.dict_sym, rhs_sym)
                            end
                        else
                            data.dict_sym[lhs_sym] = DictInfo(live_info, node)
                            dprintln(3,"Remembering assignment for symbol ", lhs_sym, " ", rhs)
                        end
                    end
                end
            else
                dprintln(3,"Not an assignment node.")
            end

            for j = live_info.use
                delete!(data.dict_sym, j)
            end

            # Here we try to determine which scalar assigns can be hoisted to the beginning of the function.
            #
            # If this statement defines some variable.
            if !isempty(live_info.def)
                # Assume that hoisting is safe until proven otherwise.
                dep_only_on_parameter = true
                # Look at all the variables on which this statement depends.
                # If any of them are not a hoistable scalar then we can't hoist the current scalar definition.
                for i in live_info.use
                    if !in(i, data.hoistable_scalars)
                        dep_only_on_parameter = false
                        break
                    end
                end
                if dep_only_on_parameter 
                    # If this statement is defined in more than one place then it isn't hoistable.
                    for i in live_info.def 
                        if CompilerTools.LivenessAnalysis.countSymbolDefs(i, data.lives) > 1
                            dep_only_on_parameter = false
                            dprintln(3,"variable ", i, " assigned more than once")
                            break
                        end
                    end
                    if dep_only_on_parameter 
                        dprintln(3,"remove_no_deps removing ", node, " because it only depends on hoistable scalars.")
                        push!(data.top_level_no_deps, node)
                        # If the defs in this statement are hoistable then other statements which depend on them may also be hoistable.
                        for i in live_info.def
                            push!(data.hoistable_scalars, i)
                        end
                        return CompilerTools.AstWalker.ASTWALK_REMOVE
                    end
                end
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
"node" is a domainIR node.  Take the arrays used in this node, create an array equivalence for them if they 
don't already have one and make sure they all share one equivalence class.
"""
function extractArrayEquivalencies(node :: Expr, state)
    input_args = node.args

    # Make sure we get what we expect from domain IR.
    # There should be two entries in the array, another array of input array symbols and a DomainLambda type
    if(length(input_args) < 2)
        throw(string("extractArrayEquivalencies input_args length should be at least 2 but is ", length(input_args)))
    end

    # First arg is an array of input arrays to the mmap!
    input_arrays = input_args[1]
    len_input_arrays = length(input_arrays)
    dprintln(2,"Number of input arrays: ", len_input_arrays)
    dprintln(3,"input_arrays =  ", input_arrays)
    assert(len_input_arrays > 0)

    # Second arg is a DomainLambda
    ftype = typeof(input_args[2])
    dprintln(2,"extractArrayEquivalencies function = ",input_args[2])
    if(ftype != DomainLambda)
        throw(string("extractArrayEquivalencies second input_args should be a DomainLambda but is of type ", typeof(input_args[2])))
    end

    if !isa(input_arrays[1], SymAllGen)
        dprintln(1, "extractArrayEquivalencies input_arrays[1] is not SymAllGen")
        return nothing
    end

    # Get the correlation set of the first input array.
    main_length_correlation = getOrAddArrayCorrelation(toSymGen(input_arrays[1]), state)

    # Make sure each input array is a SymbolNode
    # Also, create indexed versions of those symbols for the loop body
    for i = 2:len_input_arrays
        dprintln(3,"extractArrayEquivalencies input_array[i] = ", input_arrays[i], " type = ", typeof(input_arrays[i]))
        this_correlation = getOrAddArrayCorrelation(toSymGen(input_arrays[i]), state)
        # Verify that all the inputs are the same size by verifying they are in the same correlation set.
        if this_correlation != main_length_correlation
            merge_correlations(state, main_length_correlation, this_correlation)
        end
    end

    dprintln(3,"extractArrayEq result = ", state.array_length_correlation)
    return main_length_correlation
end

@doc """
Make sure all the dimensions are SymbolNodes.
Make sure each dimension variable is assigned to only once in the function.
Extract just the dimension variables names into dim_names and then register the correlation from lhs to those dimension names.
"""
function checkAndAddSymbolCorrelation(lhs :: SymGen, state, dim_array)
    dim_names = SymGen[]
    for i = 1:length(dim_array)
        if typeof(dim_array[i]) != SymbolNode
            return false
        end
        if CompilerTools.LambdaHandling.getDesc(dim_array[i].name, state.lambdaInfo) & ISASSIGNEDONCE != ISASSIGNEDONCE
            return false
        end
        push!(dim_names, dim_array[i].name)
    end

    dprintln(3, "Will establish array length correlation for const size lhs = ", lhs, " dims = ", dim_names)
    getOrAddSymbolCorrelation(lhs, state, dim_names)
    return true
end

@doc """
Apply a function "f" that takes the :body from the :lambda and returns a new :body that is stored back into the :lambda.
"""
function processAndUpdateBody(lambda :: Expr, f :: Function, state)
    assert(lambda.head == :lambda) 
    lambda.args[3].args = f(lambda.args[3].args, state)
    return lambda
end

@doc """
Empty statements can be added to the AST by some passes in ParallelIR.
This pass over the statements of the :body excludes such "nothing" statements from the new :body.
"""
function removeNothingStmts(args :: Array{Any,1}, state)
    newBody = Any[]
    for i = 1:length(args)
        if args[i] != nothing
            push!(newBody, args[i])
        end
    end
    return newBody
end



function create_equivalence_classes_assignment(lhs, rhs::Expr, state)

    dprintln(4,lhs)
    dprintln(4,rhs)

    if rhs.head == :assertEqShape
        # assertEqShape lets us know that the array mentioned in the assertEqShape node must have the same shape.
        dprintln(3,"Creating array length assignment from assertEqShape")
        from_assertEqShape(rhs, state)
    elseif rhs.head == :alloc
        # Here an array on the left-hand side is being created from size specification on the right-hand side.
        # Map those array sizes to the corresponding array equivalence class.
        sizes = rhs.args[2]
        n = length(sizes)
        assert(n >= 1 && n <= 3)
        dprintln(3, "Detected :alloc array allocation. dims = ", sizes)
        checkAndAddSymbolCorrelation(lhs, state, sizes)            
    elseif rhs.head == :call
        dprintln(3, "Detected call rhs in from_assignment.")
        dprintln(3, "from_assignment call, arg1 = ", rhs.args[1])
        if length(rhs.args) > 1
            dprintln(3, " arg2 = ", rhs.args[2])
        end
        if rhs.args[1] == TopNode(:ccall)
            # Same as :alloc above.  Detect an array allocation call and map the specified array sizes to an array equivalence class.
            if rhs.args[2] == QuoteNode(:jl_alloc_array_1d)
                dim1 = rhs.args[7]
                dprintln(3, "Detected 1D array allocation. dim1 = ", dim1, " type = ", typeof(dim1))
                checkAndAddSymbolCorrelation(lhs, state, [dim1])            
            elseif rhs.args[2] == QuoteNode(:jl_alloc_array_2d)
                dim1 = rhs.args[7]
                dim2 = rhs.args[9]
                dprintln(3, "Detected 2D array allocation. dim1 = ", dim1, " dim2 = ", dim2)
                checkAndAddSymbolCorrelation(lhs, state, [dim1, dim2])            
            elseif rhs.args[2] == QuoteNode(:jl_alloc_array_3d)
                dim1 = rhs.args[7]
                dim2 = rhs.args[9]
                dim3 = rhs.args[11]
                dprintln(3, "Detected 2D array allocation. dim1 = ", dim1, " dim2 = ", dim2, " dim3 = ", dim3)
                checkAndAddSymbolCorrelation(lhs, state, [dim1, dim2, dim3])            
            end
        elseif rhs.args[1] == TopNode(:arraylen)
            # This is the other direction.  Takes an array and extract dimensional information that maps to the array's equivalence class.
            array_param = rhs.args[2]                  # length takes one param, which is the array
            assert(typeof(array_param) == SymbolNode)  # should be a SymbolNode
            array_param_type = array_param.typ         # get its type
            if ndims(array_param_type) == 1            # can only associate when number of dimensions is 1
                dim_symbols = [getSName(lhs)]
                dprintln(3,"Adding symbol correlation from arraylen, name = ", rhs.args[2].name, " dims = ", dim_symbols)
                checkAndAddSymbolCorrelation(rhs.args[2].name, state, dim_symbols)
            end
        elseif rhs.args[1] == TopNode(:arraysize)
            # This is the other direction.  Takes an array and extract dimensional information that maps to the array's equivalence class.
            if length(rhs.args) == 2
                array_param = rhs.args[2]                  # length takes one param, which is the array
                assert(typeof(array_param) == SymbolNode)  # should be a SymbolNode
                array_param_type = array_param.typ         # get its type
                array_dims = ndims(array_param_type)
                dim_symbols = Symbol[]
                for dim_i = 1:array_dims
                    push!(dim_symbols, lhs[dim_i])
                end
                dprintln(3,"Adding symbol correlation from arraysize, name = ", rhs.args[2].name, " dims = ", dim_symbols)
                checkAndAddSymbolCorrelation(rhs.args[2].name, state, dim_symbols)
            elseif length(rhs.args) == 3
                dprintln(1,"Can't establish symbol to array length correlations yet in the case where dimensions are extracted individually.")
            else
                throw(string("arraysize AST node didn't have 2 or 3 arguments."))
            end
        end
    elseif rhs.head == :mmap! || rhs.head == :mmap || rhs.head == :map! || rhs.head == :map 
        # Arguments to these domain operations implicit assert that equality of sizes so add/merge equivalence classes for the arrays to this operation.
        rhs_corr = extractArrayEquivalencies(rhs, state)
        dprintln(3,"lhs = ", lhs, " type = ", typeof(lhs))
        if rhs_corr != nothing && isa(lhs, SymAllGen)
            lhs_corr = getOrAddArrayCorrelation(toSymGen(lhs), state) 
            merge_correlations(state, lhs_corr, rhs_corr)
            dprintln(3,"Correlations after map merge into lhs = ", state.array_length_correlation)
        end
    end
end

function create_equivalence_classes_assignment(lhs, rhs::ANY, state)
end

@doc """
AstWalk callback to determine the array equivalence classes.
"""
function create_equivalence_classes(node :: Expr, state :: expr_state, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    dprintln(3,"create_equivalence_classes starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    dprintln(3,"create_equivalence_classes node = ", node, " type = ", typeof(node))
    dprintln(3,"node.head = ", node.head)

    if node.head == :lambda
        save_lambdaInfo  = state.lambdaInfo
        state.lambdaInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(node)
        body = CompilerTools.LambdaHandling.getBody(node)
        AstWalk(body, create_equivalence_classes, state)
        state.lambdaInfo = save_lambdaInfo
        return node
    end

    # We can only extract array equivalences from top-level statements.
    if is_top_level
        dprintln(3,"create_equivalence_classes is_top_level")

        if isAssignmentNode(node)
            # Here the node is an assignment.
            dprintln(3,"Is an assignment node.")
            create_equivalence_classes_assignment(node.args[1], node.args[2], state)
        else
            if node.head == :mmap! || node.head == :mmap || node.head == :map! || node.head == :map
                extractArrayEquivalencies(node, state)
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function create_equivalence_classes(node :: ANY, state :: expr_state, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    dprintln(3,"create_equivalence_classes starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    dprintln(3,"create_equivalence_classes node = ", node, " type = ", typeof(node))
    dprintln(3,"Not an assignment or expr node.")
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end


# mmapInline() helper function
function modify!(dict, lhs, i)
    if haskey(dict, lhs)
        push!(dict[lhs], i)
    else
        dict[lhs] = Int[i]
    end
end

# function that inlines from src mmap into dst mmap
function inline!(src, dst, lhs)
    args = dst.args[1]
    pos = 0
    for k = 1:length(args)
        s = args[k]
        if isa(s, SymbolNode) s = s.name end
        if s == lhs
            pos = k
            break
        end
    end
    @assert pos>0 "mmapInline(): position of mmap output not found"
    args = vcat(args[1:pos-1], args[pos+1:end], src.args[1])
    f = src.args[2]
    assert(length(f.outputs)==1)
    dst.args[1] = args
    g = dst.args[2]
    inputs = g.inputs
    g_inps = length(inputs) 
    inputs = vcat(inputs[1:pos-1], inputs[pos+1:end], f.inputs)
    linfo = g.linfo
    # add escaping variables from f into g, since mergeLambdaInfo only deals
    # with nesting lambda, but not parallel ones.
    for (v, d) in f.linfo.escaping_defs
        if !CompilerTools.LambdaHandling.isEscapingVariable(v, linfo)
            CompilerTools.LambdaHandling.addEscapingVariable(d, linfo)
        end
    end
    # gensym_map = CompilerTools.LambdaHandling.mergeLambdaInfo(linfo, f.linfo)
    tmp_t = f.outputs[1]
    dst.args[2] = DomainLambda(inputs, g.outputs, 
    (linfo, args) -> begin
        fb = f.genBody(linfo, args[g_inps:end])
        tmp_v = CompilerTools.LambdaHandling.addGenSym(tmp_t, linfo)
        expr = TypedExpr(tmp_t, :(=), tmp_v, fb[end].args[1])
        gb = g.genBody(linfo, vcat(args[1:pos-1], [tmp_v], args[pos:g_inps-1]))
        return [fb[1:end-1]; [expr]; gb]
    end, linfo)
    DomainIR.mmapRemoveDupArg!(dst)
end

# mmapInline() helper function
function eliminateShapeAssert(dict, lhs, body)
    if haskey(dict, lhs)
        for k in dict[lhs]
            dprintln(3, "MI: eliminate shape assert at line ", k)
            body.args[k] = nothing
        end
    end
end

# mmapInline() helper function
function check_used(defs, usedAt, shapeAssertAt, expr,i)
    if isa(expr, Expr)
        if is(expr.head, :assertEqShape)
            # handle assertEqShape separately, do not consider them
            # as valid references
            for arg in expr.args
                s = isa(arg, SymbolNode) ? arg.name : arg 
                if isa(s, Symbol) || isa(s, GenSym)
                    modify!(shapeAssertAt, s, i)
                end
            end
        else
            for arg in expr.args
                check_used(defs, usedAt, shapeAssertAt, arg,i)
            end
        end
    elseif isa(expr, Symbol) || isa(expr, GenSym)
        if haskey(usedAt, expr) # already used? remove from defs
            delete!(defs, expr)
        else
            usedAt[expr] = i
        end 
    elseif isa(expr, SymbolNode)
        if haskey(usedAt, expr.name) # already used? remove from defs
            dprintln(3, "MI: def ", expr.name, " removed due to multi-use")
            delete!(defs, expr.name)
        else
            usedAt[expr.name] = i
        end 
    elseif isa(expr, Array) || isa(expr, Tuple)
        for e in expr
            check_used(defs, usedAt, shapeAssertAt, e, i)
        end
    end
end


# mmapInline() helper function
function mmapInline_refs(expr, i, uniqSet, defs::Dict{Union{Symbol, GenSym}, Int}, usedAt::Dict{Union{Symbol, GenSym}, Int}, 
                                modifiedAt::Dict{Union{Symbol, GenSym}, Array{Int}}, shapeAssertAt::Dict{Union{Symbol, GenSym}, Array{Int}})
        head = isa(expr, Expr) ? expr.head : nothing
        # record usedAt, and reject those used more than once
        # record definition
        if is(head, :(=))
            lhs = expr.args[1]
            rhs = expr.args[2]
            check_used(defs, usedAt, shapeAssertAt, rhs,i)
            assert(isa(lhs, Symbol) || isa(lhs, GenSym))
            modify!(modifiedAt, lhs, i)
            if isa(rhs, Expr) && is(rhs.head, :mmap) && in(lhs, uniqSet)
                ok = true
                for j in rhs.args[1]
                    if isa(j, SymbolNode) j = j.name end
                    if !in(j, uniqSet) # being conservative by demanding arguments not being aliased
                        ok=false
                        break
                    end
                end
                if (ok) defs[lhs] = i end
                dprintln(3, "MI: def for ", lhs, " ok=", ok, " defs=", defs)
            end
        else
            check_used(defs, usedAt, shapeAssertAt, expr,i)
        end
        # check if args may be modified in place
        if is(head, :mmap!)
            for j in 1:length(expr.args[2].outputs)
                v = expr.args[1][j]
                if isa(v, SymbolNode)
                    v = v.name
                end
                if isa(v, Symbol) || isa(v, GenSym)
                    modify!(modifiedAt, v, i)
                end
            end
        elseif is(head, :stencil!)
            krnStat = expr.args[1]
            iterations = expr.args[2]
            bufs = expr.args[3]
            for k in krnStat.modified
                s = bufs[k]
                if isa(s, SymbolNode) s = s.name end
                modify!(modifiedAt, s, i)
            end
            if !((isa(iterations, Number) && iterations == 1) || krnStat.rotateNum == 0)
                for j in 1:min(krnStat.rotateNum, length(bufs))
                    s = bufs[j]
                    if isa(s, SymbolNode) s = s.name end
                    modify!(modifiedAt, s, i)
                end
            end
        end
end


@doc """
# If a definition of a mmap is only used once and not aliased, it can be inlined into its
# use side as long as its dependencies have not been changed.
# FIXME: is the implementation still correct when branches are present?
"""
function mmapInline(ast::Expr, lives, uniqSet)
    body = ast.args[3]
    defs = Dict{Union{Symbol, GenSym}, Int}()
    usedAt = Dict{Union{Symbol, GenSym}, Int}()
    modifiedAt = Dict{Union{Symbol, GenSym}, Array{Int}}()
    shapeAssertAt = Dict{Union{Symbol, GenSym}, Array{Int}}()
    assert(isa(body, Expr) && is(body.head, :body))

    # first do a loop to see which def is only referenced once
    for i =1:length(body.args)
        mmapInline_refs(body.args[i], i, uniqSet, defs, usedAt, modifiedAt, shapeAssertAt)
    end
    dprintln(3, "MI: defs = ", defs)
    # print those that are used once

    revdefs = Dict()
    for (lhs, i) in defs
        revdefs[i] = lhs
    end
    for i in sort!([keys(revdefs)...])
        lhs = revdefs[i]
        expr = body.args[i] # must be an assignment
        src = expr.args[2]  # must be a mmap
        # dprintln(3, "MI: def of ", lhs)
        if haskey(usedAt, lhs)
            j = usedAt[lhs]
            ok = true
            # dprintln(3, "MI: def of ", lhs, " at line ", i, " used by line ", j)
            for v in src.args[1]
                @assert isa(v,Symbol) || isa(v,GenSym) || isa(v,SymbolNode) "mmapInline(): Arguments of mmap should be Symbol or GenSym or SymbolNode."
                if isa(v, SymbolNode) v = v.name end
                if haskey(modifiedAt, v)
                    for k in modifiedAt[v]
                        if k >= i && k <= j
                            ok = false
                            break
                        end
                    end
                end
                if (!ok) break end
            end 
            if (!ok) continue end
            dprintln(3, "MI: found mmap: ", lhs, " can be inlined into defintion of line ", j)
            dst = body.args[j]
            if isa(dst, Expr) && is(dst.head, :(=))
                dst = dst.args[2]
            end
            if isa(dst, Expr) && is(dst.head, :mmap) && in(lhs, dst.args[1])
                # inline mmap into mmap
                inline!(src, dst, lhs)
                body.args[i] = nothing
                eliminateShapeAssert(shapeAssertAt, lhs, body)
                dprintln(3, "MI: result: ", body.args[j])
            elseif isa(dst, Expr) && is(dst.head, :mmap!) && in(lhs, dst.args[1])
                s = dst.args[1][1]
                if isa(s, SymbolNode) s = s.name end
                if s == lhs 
                    # when lhs is the inplace array that dst operates on
                    # change dst to mmap
                    inline!(src, dst, lhs)
                    dst.head = :mmap
                else
                    # otherwise just normal inline
                    inline!(src, dst, lhs)
                end
                body.args[i] = nothing
                eliminateShapeAssert(shapeAssertAt, lhs, body)
                # inline mmap into mmap!
                dprintln(3, "MI: result: ", body.args[j])
            else
                # otherwise ignore, e.g., when dst is some simple assignments.
            end
        end
    end
end

@doc """
Try to hoist allocations outside the loop if possible.
"""
function hoistAllocation(ast::Array{Any,1}, lives, domLoop::DomLoops, state :: expr_state)
    for l in domLoop.loops
        dprintln(3, "HA: loop from block ", l.head, " to ", l.back_edge)
        headBlk = lives.cfg.basic_blocks[ l.head ]
        tailBlk = lives.cfg.basic_blocks[ l.back_edge ]
        if length(headBlk.preds) != 2
            continue
        end
        preBlk = nothing
        for blk in headBlk.preds
            if blk.label != tailBlk.label
                preBlk = blk
                break
            end
        end
        if (is(preBlk, nothing) || length(preBlk.statements) == 0) continue end
        tls = lives.basic_blocks[ preBlk ]
        preHead = preBlk.statements[end].index
        head = headBlk.statements[1].index
        tail = tailBlk.statements[1].index
        dprintln(3, "HA: line before head is ", ast[preHead-1])
        # Is iterating through statement indices this way safe?
        for i = head:tail
            if isAssignmentNode(ast[i]) && isAllocation(ast[i].args[2])
                dprintln(3, "HA: found allocation at line ", i, ": ", ast[i])
                lhs = ast[i].args[1]
                rhs = ast[i].args[2]
                if isa(lhs, SymbolNode) lhs = lhs.name end
                if (haskey(state.array_length_correlation, lhs))
                    c = state.array_length_correlation[lhs]
                    for (d, v) in state.symbol_array_correlation
                        if v == c
                            ok = true
                            for j = 1:length(d)
                                if !in(d[j], tls.live_out)
                                    ok = false
                                    break
                                end
                            end
                            dprintln(3, "HA: found correlation dimension ", d, " ", ok, " ", length(rhs.args)-6)
                            if ok && length(rhs.args) - 6 == 2 * length(d) # dimension must match
                                rhs.args = rhs.args[1:6]
                                for s in d
                                    push!(rhs.args, SymbolNode(s, Int))
                                    push!(rhs.args, 0)
                                end
                                dprintln(3, "HA: hoist ", ast[i], " out of loop before line ", head)
                                ast = [ ast[1:preHead-1], ast[i], ast[preHead:i-1], ast[i+1:end] ]
                                break
                            end
                        end
                    end
                end
            end
        end
    end
    return ast
end

@doc """
Performs the mmap to mmap! phase.
If the arguments of a mmap dies aftewards, and is not aliased, then
we can safely change the mmap to mmap!.
"""
function mmapToMmap!(ast, lives, uniqSet)
    lambdaInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(ast)
    body = ast.args[3]
    assert(isa(body, Expr) && is(body.head, :body))
    # For each statement in the body.
    for i =1:length(body.args)
        expr = body.args[i]
        # If the statement is an assignment.
        if isa(expr, Expr) && is(expr.head, :(=))
            lhs = expr.args[1]
            rhs = expr.args[2]
            # right now assume all
            assert(isa(lhs, SymAllGen))
            lhsTyp = CompilerTools.LambdaHandling.getType(lhs, lambdaInfo) 
            # If the right-hand side is an mmap.
            if isa(rhs, Expr) && is(rhs.head, :mmap)
                args = rhs.args[1]
                tls = CompilerTools.LivenessAnalysis.find_top_number(i, lives)
                assert(tls != nothing)
                assert(CompilerTools.LivenessAnalysis.isDef(lhs, tls))
                dprintln(4, "mmap lhs=", lhs, " args=", args, " live_out = ", tls.live_out)
                reuse = nothing
                j = 0
                # Find some input array to the mmap that is dead after this statement.
                while j < length(args)
                    j = j + 1
                    v = args[j]
                    v = isa(v, SymbolNode) ? v.name : v
                    if (isa(v, Symbol) || isa(v, GenSym)) && !in(v, tls.live_out) && in(v, uniqSet) &&
                        CompilerTools.LambdaHandling.getType(v, lambdaInfo) == lhsTyp
                        reuse = v  # Found a dying symbol.
                        break
                    end
                end
                # If we found a dying array whose space we can reuse.
                if !is(reuse, nothing)
                    rhs.head = :mmap!   # Change to mmap!
                    dprintln(2, "mmapToMMap!: successfully reuse ", reuse, " for ", lhs)
                    if j != 1  # The array to reuse has to be first.  If it isn't already then reorder the args to make it so.
                        # swap j-th and 1st argument
                        rhs.args[1] = DomainIR.arraySwap(rhs.args[1], 1, j)
                        rhs.args[2] = DomainIR.lambdaSwapArg(rhs.args[2], 1, j)
                        dprintln(3, "mmapToMMap!: after swap, ", lhs, " = ", rhs)
                    end
                end
            end
        end
    end
end

mmap_to_mmap! = 1
@doc """
If set to non-zero, perform the phase where non-inplace maps are converted to inplace maps to reduce allocations.
"""
function PIRInplace(x)
    global mmap_to_mmap! = x
end

hoist_allocation = 1
@doc """
If set to non-zero, perform the rearrangement phase that tries to moves alllocations outside of loops.
"""
function PIRHoistAllocation(x)
    global hoist_allocation = x
end

bb_reorder = 0
@doc """
If set to non-zero, perform the bubble-sort like reordering phase to coalesce more parfor nodes together for fusion.
"""
function PIRBbReorder(x)
    global bb_reorder = x
end 

shortcut_array_assignment = 0
@doc """
Enables an experimental mode where if there is a statement a = b and they are arrays and b is not live-out then 
use a special assignment node like a move assignment in C++.
"""
function PIRShortcutArrayAssignment(x)
    global shortcut_array_assignment = x
end

@doc """
Type for dependence graph creation and topological sorting.
"""
type StatementWithDeps
    stmt :: CompilerTools.LivenessAnalysis.TopLevelStatement
    deps :: Set{StatementWithDeps}
    dfs_color :: Int64 # 0 = white, 1 = gray, 2 = black
    discovery :: Int64
    finished  :: Int64

    function StatementWithDeps(s)
        new(s, Set{StatementWithDeps}(), 0, 0, 0)
    end
end

@doc """
Construct a topological sort of the dependence graph.
"""
function dfsVisit(swd :: StatementWithDeps, vtime :: Int64, topo_sort :: Array{StatementWithDeps})
    swd.dfs_color = 1 # color gray
    swd.discovery = vtime
    vtime += 1
    for dep in swd.deps
        if dep.dfs_color == 0
            vtime = dfsVisit(dep, vtime, topo_sort)
        end
    end
    swd.dfs_color = 2 # color black
    swd.finished  = vtime
    vtime += 1
    unshift!(topo_sort, swd)
    return vtime
end

@doc """
Returns true if the given "ast" node is a DomainIR operation.
"""

function isDomainNode(ast :: Expr)
    head = ast.head
    args = ast.args

    if head == :mmap || head == :mmap! || head == :reduce || head == :stencil!
        return true
    end

    for i = 1:length(args)
        if isDomainNode(args[i])
            return true
        end
    end

    return false
end

function isDomainNode(ast)
    return false
end


@doc """
Returns true if the given AST "node" must remain the last statement in a basic block.
This is true if the node is a GotoNode or a :gotoifnot Expr.
"""
function mustRemainLastStatementInBlock(node :: GotoNode)
    return true
end

function mustRemainLastStatementInBlock(node)
    return false
end

function mustRemainLastStatementInBlock(node :: Expr)
    return node.head == :gotoifnot  ||  node.head == :return
end

@doc """
For every basic block, try to push domain IR statements down and non-domain IR statements up so that domain nodes
are next to each other and can be fused.
"""
function maxFusion(bl :: CompilerTools.LivenessAnalysis.BlockLiveness)
    # We will try to optimize the order in each basic block.
    for bb in collect(values(bl.basic_blocks))
        if false
            # One approach to this problem is to create a dependency graph and then use that to calculate valid reorderings
            # that maximize fusion.  We may still want to switch to this at some point but it is more complicated and the 
            # simpler approach works for now.

            # Start with a pseudo-statement corresponding to the variables live-in to this basic block.
            livein     = StatementWithDeps(nothing)
            # The variable "last_write" is a dictionary mapping a symbol to the last statement to have written that variable.
            last_write = Dict{Symbol, StatementWithDeps}()
            # For each symbol live-in to this basic block, add it to "last_write".
            for def in bb.live_in
                assert(typeof(def) == Symbol)
                last_write[def] = livein
            end

            # Create the dependency graph.
            for i = 1:length(bb.statements)
                # Get the i'th statement in this basic block.
                stmt    = bb.statements[i]
                # Create an object to associate this statement with its dependencies.
                new_swd = StatementWithDeps(stmt)

                # Establish dependency links from prior statements to this one based on the last statement to have written each symbol used in this statement.
                for use in stmt.use
                    assert(typeof(use) == Symbol)
                    assert(haskey(last_write, use))
                    push!(last_write[use].deps, new_swd)
                end
                # Update last_write with any symbols def in this statement.
                for def in stmt.def
                    assert(typeof(def) == Symbol)
                    last_write[def] = new_swd
                end
            end

            topo_sort = StatementWithDeps[]
            dfsVisit(livein, 1, topo_sort)
        else
            dprintln(3, "Doing maxFusion in block ", bb)
            # A bubble-sort style of coalescing domain nodes together in the AST.
            earliest_parfor = 1
            found_change = true

            # While the lastest pass over the AST created some change, keep searching for more interchanges that can coalesce domain nodes.
            while found_change
                found_change = false

                # earliest_parfor is an optimization that we don't have to scan every statement in the block but only those statements from
                # the first parfor to the last statement in the block.
                i = earliest_parfor
                # dprintln(3,"bb.statements = ", bb.statements)
                earliest_parfor = length(bb.statements)

                while i < length(bb.statements)
                    cur  = bb.statements[i]
                    next = bb.statements[i+1]
                    cannot_move_next = mustRemainLastStatementInBlock(next.tls.expr)
                    dprintln(3,"maxFusion cur = ", cur.tls.expr)
                    dprintln(3,"maxFusion next = ", next.tls.expr)
                    cur_domain_node  = isDomainNode(cur.tls.expr)  
                    next_domain_node = isDomainNode(next.tls.expr) 
                    intersection     = intersect(cur.def, next.use)
                    dprintln(3,"cur_domain_node = ", cur_domain_node, " next_domain_node = ", next_domain_node, " intersection = ", intersection)
                    if cur_domain_node && !cannot_move_next
                        if !next_domain_node && isempty(intersection)
                            # If the current statement is a domain node and the next staterment isn't and we are allowed to move the next node
                            # in the block and the next statement doesn't use anything produced by this statement then we can switch the order of
                            # the current and next statement.
                            dprintln(3,"bubbling domain node down")
                            (bb.statements[i], bb.statements[i+1]) = (bb.statements[i+1], bb.statements[i])
                            (bb.cfgbb.statements[i], bb.cfgbb.statements[i+1]) = (bb.cfgbb.statements[i+1], bb.cfgbb.statements[i])
                            (bb.cfgbb.statements[i].index, bb.cfgbb.statements[i+1].index) = (bb.cfgbb.statements[i+1].index, bb.cfgbb.statements[i].index)
                            found_change = true
                        else
                            if i < earliest_parfor
                                earliest_parfor = i
                            end
                        end
                    end
                    i += 1
                end
            end 
        end
    end
end

@doc """
Debug print the parts of a DomainLambda.
"""
function pirPrintDl(dbg_level, dl)
    dprintln(dbg_level, "inputs = ", dl.inputs)
    dprintln(dbg_level, "output = ", dl.outputs)
    dprintln(dbg_level, "linfo  = ", dl.linfo)
end

@doc """
Scan the body of a function in "stmts" and return the max label in a LabelNode AST seen in the body.
"""
function getMaxLabel(max_label, stmts :: Array{Any, 1})
    for i =1:length(stmts)
        if isa(stmts[i], LabelNode)
            max_label = max(max_label, stmts[i].label)
        end
    end
    return max_label
end

@doc """
Form a Julia :lambda Expr from a DomainLambda.
"""
function lambdaFromDomainLambda(domain_lambda, dl_inputs)
    #  inputs_as_symbols = map(x -> CompilerTools.LambdaHandling.VarDef(x.name, x.typ, 0), dl_inputs)
    type_data = CompilerTools.LambdaHandling.VarDef[]
    input_arrays = Symbol[]
    for di in dl_inputs
        push!(type_data, CompilerTools.LambdaHandling.VarDef(di.name, di.typ, 0))
        if isArrayType(di.typ)
            push!(input_arrays, di.name)
        end
    end
    #  dprintln(3,"inputs = ", inputs_as_symbols)
    dprintln(3,"types = ", type_data)
    dprintln(3,"DomainLambda is:")
    pirPrintDl(3, domain_lambda)
    newLambdaInfo = CompilerTools.LambdaHandling.LambdaInfo()
    CompilerTools.LambdaHandling.addInputParameters(type_data, newLambdaInfo)
    stmts = domain_lambda.genBody(newLambdaInfo, dl_inputs)
    newLambdaInfo.escaping_defs = copy(domain_lambda.linfo.escaping_defs)
    ast = CompilerTools.LambdaHandling.lambdaInfoToLambdaExpr(newLambdaInfo, Expr(:body, stmts...))
    # copy escaping defs from domain lambda since mergeDomainLambda doesn't do it (for good reasons)
    return (ast, input_arrays) 
end

@doc """
A routine similar to the main parallel IR entry put but designed to process the lambda part of
domain IR AST nodes.
"""
function nested_function_exprs(max_label, domain_lambda, dl_inputs)
    dprintln(2,"nested_function_exprs max_label = ", max_label)
    dprintln(2,"domain_lambda = ", domain_lambda, " dl_inputs = ", dl_inputs)
    (ast, input_arrays) = lambdaFromDomainLambda(domain_lambda, dl_inputs)
    dprintln(1,"Starting nested_function_exprs. ast = ", ast, " input_arrays = ", input_arrays)

    start_time = time_ns()

    dprintln(1,"Starting liveness analysis.")
    lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
    dprintln(1,"Finished liveness analysis.")

    dprintln(1,"Liveness Analysis time = ", ns_to_sec(time_ns() - start_time))

    mtm_start = time_ns()

    if mmap_to_mmap! != 0
        dprintln(1, "starting mmap to mmap! transformation.")
        uniqSet = AliasAnalysis.analyze_lambda(ast, lives, pir_alias_cb, nothing)
        dprintln(3, "uniqSet = ", uniqSet)
        mmapInline(ast, lives, uniqSet)
        lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
        uniqSet = AliasAnalysis.analyze_lambda(ast, lives, pir_alias_cb, nothing)
        mmapToMmap!(ast, lives, uniqSet)
        dprintln(1, "Finished mmap to mmap! transformation.")
        dprintln(3, "AST = ", ast)
    end

    dprintln(1,"mmap_to_mmap! time = ", ns_to_sec(time_ns() - mtm_start))

    # We pass only the non-array params to the rearrangement code because if we pass array params then
    # the code will detect statements that depend only on array params and move them to the top which
    # leaves other non-array operations after that and so prevents fusion.
    dprintln(3,"All params = ", ast.args[1])
    non_array_params = Set{SymGen}()
    for param in ast.args[1]
        if !in(param, input_arrays) && CompilerTools.LivenessAnalysis.countSymbolDefs(param, lives) == 0
            push!(non_array_params, param)
        end
    end
    dprintln(3,"Non-array params = ", non_array_params)

    # Find out max_label.
    body = ast.args[3]
    assert(isa(body, Expr) && is(body.head, :body))
    max_label = getMaxLabel(max_label, body.args)

    eq_start = time_ns()

    new_vars = expr_state(lives, max_label, input_arrays)
    dprintln(3,"Creating equivalence classes.")
    AstWalk(ast, create_equivalence_classes, new_vars)
    dprintln(3,"Done creating equivalence classes.")

    dprintln(1,"Creating equivalence classes time = ", ns_to_sec(time_ns() - eq_start))

    rep_start = time_ns()

    for i = 1:rearrange_passes
        dprintln(1,"Removing statement with no dependencies from the AST with parameters = ", ast.args[1])
        rnd_state = RemoveNoDepsState(lives, non_array_params)
        ast = AstWalk(ast, remove_no_deps, rnd_state)
        dprintln(3,"ast after no dep stmts removed = ", ast)

        dprintln(3,"top_level_no_deps = ", rnd_state.top_level_no_deps)

        dprintln(1,"Adding statements with no dependencies to the start of the AST.")
        ast = addStatementsToBeginning(ast, rnd_state.top_level_no_deps)
        dprintln(3,"ast after no dep stmts re-inserted = ", ast)

        dprintln(1,"Re-starting liveness analysis.")
        lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
        dprintln(1,"Finished liveness analysis.")
    end

    dprintln(1,"Rearranging passes time = ", ns_to_sec(time_ns() - rep_start))

    dprintln(1,"Doing conversion to parallel IR.")

    new_vars.block_lives = lives

    # Do the main work of Parallel IR.
    ast = from_expr(ast, 1, new_vars, false)
    assert(isa(ast,Array))
    assert(length(ast) == 1)
    ast = ast[1]

    dprintln(3,"Final ParallelIR = ", ast)

    #throw(string("STOPPING AFTER PARALLEL IR CONVERSION"))
    (new_vars.max_label, ast, ast.args[3].args)
end

function addStatementsToBeginning(lambda :: Expr, stmts :: Array{Any,1})
    assert(lambda.head == :lambda)
    assert(typeof(lambda.args[3]) == Expr)
    assert(lambda.args[3].head == :body)
    lambda.args[3].args = [stmts; lambda.args[3].args]
    return lambda
end

doRemoveAssertEqShape = true
generalSimplification = true

function get_input_arrays(linfo::LambdaInfo)
    ret = Symbol[]
    input_vars = linfo.input_params
    dprintln(3,"input_vars = ", input_vars)

    for iv in input_vars
        it = getType(iv, linfo)
        dprintln(3,"iv = ", iv, " type = ", it)
        if it.name == Array.name
            dprintln(3,"Parameter is an Array.")
            push!(ret, iv)
        end
    end

    ret
end

@doc """
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
"""
function from_expr(function_name, ast :: Expr)
    assert(ast.head == :lambda)
    dprintln(1,"Starting main ParallelIR.from_expr.  function = ", function_name, " ast = ", ast)

    start_time = time_ns()

    # Create CFG from AST.  This will automatically filter out dead basic blocks.
    cfg = CompilerTools.CFGs.from_ast(ast)
    lambdaInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(ast)
    input_arrays = get_input_arrays(lambdaInfo)
    body = CompilerTools.LambdaHandling.getBody(ast)
    # Re-create the body minus any dead basic blocks.
    body.args = CompilerTools.CFGs.createFunctionBody(cfg)
    # Re-create the lambda minus any dead basic blocks.
    ast = CompilerTools.LambdaHandling.lambdaInfoToLambdaExpr(lambdaInfo, body)
    dprintln(1,"ast after dead blocks removed function = ", function_name, " ast = ", ast)

    #CompilerTools.LivenessAnalysis.set_debug_level(3)

    dprintln(1,"Starting liveness analysis. function = ", function_name)
    lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)

    #  udinfo = CompilerTools.UDChains.getUDChains(lives)
    dprintln(3,"lives = ", lives)
    #  dprintln(3,"udinfo = ", udinfo)
    dprintln(1,"Finished liveness analysis. function = ", function_name)

    dprintln(1,"Liveness Analysis time = ", ns_to_sec(time_ns() - start_time))

    mtm_start = time_ns()

    if mmap_to_mmap! != 0
        dprintln(1, "starting mmap to mmap! transformation.")
        uniqSet = AliasAnalysis.analyze_lambda(ast, lives, pir_alias_cb, nothing)
        dprintln(3, "uniqSet = ", uniqSet)
        mmapInline(ast, lives, uniqSet)
        lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
        uniqSet = AliasAnalysis.analyze_lambda(ast, lives, pir_alias_cb, nothing)
        mmapToMmap!(ast, lives, uniqSet)
        dprintln(1, "Finished mmap to mmap! transformation. function = ", function_name)
        printLambda(3, ast)
    end

    dprintln(1,"mmap_to_mmap! time = ", ns_to_sec(time_ns() - mtm_start))

    # We pass only the non-array params to the rearrangement code because if we pass array params then
    # the code will detect statements that depend only on array params and move them to the top which
    # leaves other non-array operations after that and so prevents fusion.
    dprintln(3,"All params = ", ast.args[1])
    non_array_params = Set{SymGen}()
    for param in ast.args[1]
        if !in(param, input_arrays) && CompilerTools.LivenessAnalysis.countSymbolDefs(param, lives) == 0
            push!(non_array_params, param)
        end
    end
    dprintln(3,"Non-array params = ", non_array_params, " function = ", function_name)

    # Find out max_label
    body = ast.args[3]
    assert(isa(body, Expr) && is(body.head, :body))
    max_label = getMaxLabel(0, body.args)
    dprintln(3,"maxLabel = ", max_label, " body type = ", body.typ)

    rep_start = time_ns()

    for i = 1:rearrange_passes
        dprintln(1,"Removing statement with no dependencies from the AST with parameters = ", ast.args[1], " function = ", function_name)
        rnd_state = RemoveNoDepsState(lives, non_array_params)
        ast = AstWalk(ast, remove_no_deps, rnd_state)
        dprintln(3,"ast after no dep stmts removed = ", " function = ", function_name)
        printLambda(3, ast)

        dprintln(3,"top_level_no_deps = ", rnd_state.top_level_no_deps)

        dprintln(1,"Adding statements with no dependencies to the start of the AST.", " function = ", function_name)
        ast = addStatementsToBeginning(ast, rnd_state.top_level_no_deps)
        dprintln(3,"ast after no dep stmts re-inserted = ", " function = ", function_name)
        printLambda(3, ast)

        dprintln(1,"Re-starting liveness analysis.", " function = ", function_name)
        lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
        dprintln(1,"Finished liveness analysis.", " function = ", function_name)
        dprintln(3,"lives = ", lives)
    end

    dprintln(1,"Rearranging passes time = ", ns_to_sec(time_ns() - rep_start))

    processAndUpdateBody(ast, removeNothingStmts, nothing)
    dprintln(3,"ast after removing nothing stmts = ", " function = ", function_name)
    printLambda(3, ast)
    lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)

    if generalSimplification
        ast   = AstWalk(ast, copy_propagate, CopyPropagateState(lives, Dict{Symbol,Symbol}()))
        lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
        dprintln(3,"ast after copy_propagate = ", " function = ", function_name)
        printLambda(3, ast)
    end

    ast   = AstWalk(ast, remove_dead, RemoveDeadState(lives))
    lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
    dprintln(3,"ast after remove_dead = ", " function = ", function_name)
    printLambda(3, ast)

    eq_start = time_ns()

    new_vars = expr_state(lives, max_label, input_arrays)
    dprintln(3,"Creating equivalence classes.", " function = ", function_name)
    AstWalk(ast, create_equivalence_classes, new_vars)
    dprintln(3,"Done creating equivalence classes.", " function = ", function_name)
    dprintln(3,"symbol_correlations = ", new_vars.symbol_array_correlation)
    dprintln(3,"array_correlations  = ", new_vars.array_length_correlation)

    dprintln(1,"Creating equivalence classes time = ", ns_to_sec(time_ns() - eq_start))

    if doRemoveAssertEqShape
        processAndUpdateBody(ast, removeAssertEqShape, new_vars)
        dprintln(3,"ast after removing assertEqShape = ", " function = ", function_name)
        printLambda(3, ast)
        lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
    end

    if bb_reorder != 0
        maxFusion(lives)
        # Set the array of statements in the Lambda body to a new array constructed from the updated basic blocks.
        ast.args[3].args = CompilerTools.CFGs.createFunctionBody(lives.cfg)
        dprintln(3,"ast after maxFusion = ", " function = ", function_name)
        printLambda(3, ast)
        lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
    end

    dprintln(1,"Doing conversion to parallel IR.", " function = ", function_name)

    new_vars.block_lives = lives
    dprintln(3,"Lives before main Parallel IR = ")
    dprintln(3,lives)

    # Do the main work of Parallel IR.
    ast = from_expr(ast, 1, new_vars, false)
    assert(isa(ast,Array))
    assert(length(ast) == 1)
    ast = ast[1]

    dprintln(3,"Final ParallelIR function = ", function_name, " ast = ")
    printLambda(3, ast)
    if pir_stop != 0
        throw(string("STOPPING AFTER PARALLEL IR CONVERSION"))
    end
    ast
end

@doc """
Returns true if input "a" is a tuple and each element of the tuple of isbits type.
"""
function isbitstuple(a::Tuple)
    for i in a
        if !isbits(i)
            return false
        end
    end
    return true
end

function isbitstuple(a::Any)
    return false
end

function from_expr(ast ::LambdaStaticData, depth, state :: expr_state, top_level)
    ast = uncompressed_ast(ast)
    return from_expr(ast, depth, state, top_level)
end

function from_expr(ast::Union{SymAllGen,TopNode,LineNumberNode,LabelNode,Char,
    GotoNode,DataType,ASCIIString,NewvarNode,Void,Module}, depth, state :: expr_state, top_level)
    #skip
    return [ast]
end

function from_expr(ast::GlobalRef, depth, state :: expr_state, top_level)
    mod = ast.mod
    name = ast.name
    # typ = ast.typ  # FIXME: is this type needed?
    typ = typeof(mod)
    dprintln(2,"GlobalRef type ",typeof(mod))
    return [ast]
end


function from_expr(ast::QuoteNode, depth, state :: expr_state, top_level)
    value = ast.value
    #TODO: fields: value
    dprintln(2,"QuoteNode type ",typeof(value))
    return [ast] 
end

function from_expr(ast::Tuple, depth, state :: expr_state, top_level)
    @assert isbitstuple(ast) "Only bits type tuples allowed in from_expr()"
    return [ast] 
end

function from_expr(ast::Number, depth, state :: expr_state, top_level)
    @assert isbits(ast) "only bits (plain) types supported in from_expr()"
    return [ast] 
end

@doc """
The main ParallelIR function for processing some node in the AST.
"""
function from_expr(ast ::Expr, depth, state :: expr_state, top_level)
    if is(ast, nothing)
        return [nothing]
    end
    dprintln(2,"from_expr depth=",depth," ")
    dprint(2,"Expr ")
    head = ast.head
    args = ast.args
    typ  = ast.typ
    dprintln(2,head, " ", args)
    if head == :lambda
        ast = from_lambda(ast, depth, state)
        dprintln(3,"After from_lambda = ", ast)
        return [ast]
    elseif head == :body
        dprintln(3,"Processing body start")
        args = from_exprs(args,depth+1,state)
        dprintln(3,"Processing body end")
    elseif head == :(=)
        dprintln(3,"Before from_assignment typ is ", typ)
        if length(args)>=3
            @assert isa(args[3], FusionSentinel) "Parallel-IR invalid fusion assignment"
            args, new_typ = from_assignment_fusion(args, depth, state)
        else
            @assert length(args)==2 "Parallel-IR invalid assignment"
            args, new_typ = from_assignment(args[1], args[2], depth, state)
        end
        if length(args) == 0
            return []
        end
        if new_typ != nothing
            typ = new_typ
        end
    elseif head == :return
        args = from_exprs(args,depth,state)
    elseif head == :call
        args = from_call(args,depth,state)
        # TODO: catch domain IR result here
    elseif head == :call1
        args = from_call(args, depth, state)
        # TODO?: tuple
    elseif head == :line
        # remove line numbers
        return []
        # skip
    elseif head == :mmap
        head = :parfor
        # Make sure we get what we expect from domain IR.
        # There should be two entries in the array, another array of input array symbols and a DomainLambda type
        if(length(args) < 2)
            throw(string("mk_parfor_args_from_mmap! input_args length should be at least 2 but is ", length(args)))
        end
        # first arg is input arrays, second arg is DomainLambda
        domain_oprs = [DomainOperation(:mmap, args)]
        args = mk_parfor_args_from_mmap(args[1], args[2], domain_oprs, state)
        dprintln(1,"switching to parfor node for mmap, got ", args, " something wrong!")
    elseif head == :mmap!
        head = :parfor
        # Make sure we get what we expect from domain IR.
        # There should be two entries in the array, another array of input array symbols and a DomainLambda type
        if(length(args) < 2)
            throw(string("mk_parfor_args_from_mmap! input_args length should be at least 2 but is ", length(args)))
        end
        # third arg is withIndices
        with_indices = length(args) >= 3 ? args[3] : false
        # first arg is input arrays, second arg is DomainLambda
        domain_oprs = [DomainOperation(:mmap!, args)]
        args = mk_parfor_args_from_mmap!(args[1], args[2], with_indices, domain_oprs, state)
        dprintln(1,"switching to parfor node for mmap!")
    elseif head == :reduce
        head = :parfor
        args = mk_parfor_args_from_reduce(args, state)
        dprintln(1,"switching to parfor node for reduce")
    elseif head == :parallel_for
        head = :parfor
        args = mk_parfor_args_from_parallel_for(args, state)
        dprintln(1,"switching to parfor node for parallel_for")
    elseif head == :copy
        # turn array copy back to plain Julia call
        head = :call
        args = vcat(:copy, args)
    elseif head == :arraysize
        # turn array size back to plain Julia call
        head = :call
        args = vcat(TopNode(:arraysize), args)
    elseif head == :alloc
        # turn array alloc back to plain Julia ccall
        head = :call
        args = from_alloc(args)
    elseif head == :stencil!
        head = :parfor
        ast = mk_parfor_args_from_stencil(typ, head, args, state)
        dprintln(1,"switching to parfor node for stencil")
        return ast
    elseif head == :copyast
        dprintln(2,"copyast type")
        # skip
    elseif head == :assertEqShape
        if top_level && from_assertEqShape(ast, state)
            return []
        end
    elseif head == :gotoifnot
        assert(length(args) == 2)
        args[1] = get_one(from_expr(args[1], depth, state, false))
    elseif head == :new
        args = from_exprs(args,depth,state)
    elseif head == :tuple
        for i = 1:length(args)
            args[i] = get_one(from_expr(args[i], depth, state, false))
        end
    elseif head == :getindex
        args = from_exprs(args,depth,state)
    elseif head == :assert
        args = from_exprs(args,depth,state)
    elseif head == :boundscheck
        # skip
    elseif head == :meta
        # skip
    elseif head == :type_goto
        # skip
    else
        throw(string("ParallelAccelerator.ParallelIR.from_expr: unknown Expr head :", head))
    end
    ast = Expr(head, args...)
    dprintln(3,"New expr type = ", typ, " ast = ", ast)
    ast.typ = typ
    return [ast]
end

function from_alloc(args::Array{Any,1})
    elemTyp = args[1]
    sizes = args[2]
    n = length(sizes)
    assert(n >= 1 && n <= 3)
    name = symbol(string("jl_alloc_array_", n, "d"))
    appTypExpr = TypedExpr(Type{Array{elemTyp,n}}, :call, TopNode(:apply_type), GlobalRef(Base,:Array), elemTyp, n)
    #tupExpr = Expr(:call1, TopNode(:tuple), :Any, [ :Int for i=1:n ]...)
    #tupExpr.typ = ntuple(i -> (i==1) ? Type{Any} : Type{Int}, n+1)
    new_svec = TypedExpr(SimpleVector, :call, TopNode(:svec), GlobalRef(Base, :Any), [ GlobalRef(Base, :Int) for i=1:n ]...)
    realArgs = Any[QuoteNode(name), appTypExpr, new_svec, Array{elemTyp,n}, 0]
    #realArgs = Any[QuoteNode(name), appTypExpr, tupExpr, Array{elemTyp,n}, 0]
    for i=1:n
        push!(realArgs, sizes[i])
        push!(realArgs, 0)
    end
    return vcat(TopNode(:ccall), realArgs)
end


@doc """
Take something returned from AstWalk and assert it should be an array but in this
context that the array should also be of length 1 and then return that single element.
"""
function get_one(ast::Array)
    assert(length(ast) == 1)
    ast[1]
end

@doc """
Wraps the callback and opaque data passed from the user of ParallelIR's AstWalk.
"""
type DirWalk
    callback
    cbdata
end

@doc """
Return one element array with element x.
"""
function asArray(x)
    ret = Any[]
    push!(ret, x)
    return ret
end

@doc """
AstWalk callback that handles ParallelIR AST node types.
"""
function AstWalkCallback(x :: Expr, dw :: DirWalk, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    dprintln(3,"PIR AstWalkCallback starting")
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    dprintln(3,"PIR AstWalkCallback ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end

    head = x.head
    args = x.args
    #    typ  = x.typ
    if head == :parfor
        cur_parfor = args[1]
        for i = 1:length(cur_parfor.preParFor)
            x.args[1].preParFor[i] = AstWalk(cur_parfor.preParFor[i], dw.callback, dw.cbdata)
        end
        for i = 1:length(cur_parfor.loopNests)
            x.args[1].loopNests[i].indexVariable = AstWalk(cur_parfor.loopNests[i].indexVariable, dw.callback, dw.cbdata)
            # There must be some reason that I was faking an assignment expression although this really shouldn't happen in an AstWalk. In liveness callback yes, but not here.
            AstWalk(mk_assignment_expr(cur_parfor.loopNests[i].indexVariable, 1), dw.callback, dw.cbdata)
            x.args[1].loopNests[i].lower = AstWalk(cur_parfor.loopNests[i].lower, dw.callback, dw.cbdata)
            x.args[1].loopNests[i].upper = AstWalk(cur_parfor.loopNests[i].upper, dw.callback, dw.cbdata)
            x.args[1].loopNests[i].step  = AstWalk(cur_parfor.loopNests[i].step, dw.callback, dw.cbdata)
        end
        for i = 1:length(cur_parfor.reductions)
            x.args[1].reductions[i].reductionVar     = AstWalk(cur_parfor.reductions[i].reductionVar, dw.callback, dw.cbdata)
            x.args[1].reductions[i].reductionVarInit = AstWalk(cur_parfor.reductions[i].reductionVarInit, dw.callback, dw.cbdata)
            x.args[1].reductions[i].reductionFunc    = AstWalk(cur_parfor.reductions[i].reductionFunc, dw.callback, dw.cbdata)
        end
        for i = 1:length(cur_parfor.body)
            x.args[1].body[i] = AstWalk(cur_parfor.body[i], dw.callback, dw.cbdata)
        end
        for i = 1:length(cur_parfor.postParFor)-1
            x.args[1].postParFor[i] = AstWalk(cur_parfor.postParFor[i], dw.callback, dw.cbdata)
        end
        return x
    elseif head == :parfor_start || head == :parfor_end
        dprintln(3, "parfor_start or parfor_end walking, dw = ", dw)
        dprintln(3, "pre x = ", x)
        cur_parfor = args[1]
        for i = 1:length(cur_parfor.loopNests)
            x.args[1].loopNests[i].indexVariable = AstWalk(cur_parfor.loopNests[i].indexVariable, dw.callback, dw.cbdata)
            AstWalk(mk_assignment_expr(cur_parfor.loopNests[i].indexVariable, 1), dw.callback, dw.cbdata)
            x.args[1].loopNests[i].lower = AstWalk(cur_parfor.loopNests[i].lower, dw.callback, dw.cbdata)
            x.args[1].loopNests[i].upper = AstWalk(cur_parfor.loopNests[i].upper, dw.callback, dw.cbdata)
            x.args[1].loopNests[i].step  = AstWalk(cur_parfor.loopNests[i].step, dw.callback, dw.cbdata)
        end
        for i = 1:length(cur_parfor.reductions)
            x.args[1].reductions[i].reductionVar     = AstWalk(cur_parfor.reductions[i].reductionVar, dw.callback, dw.cbdata)
            x.args[1].reductions[i].reductionVarInit = AstWalk(cur_parfor.reductions[i].reductionVarInit, dw.callback, dw.cbdata)
            x.args[1].reductions[i].reductionFunc    = AstWalk(cur_parfor.reductions[i].reductionFunc, dw.callback, dw.cbdata)
        end
        for i = 1:length(cur_parfor.private_vars)
            x.args[1].private_vars[i] = AstWalk(cur_parfor.private_vars[i], dw.callback, dw.cbdata)
        end
        dprintln(3, "post x = ", x)
        return x
    elseif head == :insert_divisible_task
        cur_task = args[1]
        for i = 1:length(cur_task.args)
            x.args[1].value = AstWalk(cur_task.args[i].value, dw.callback, dw.cbdata)
        end
        return x
    elseif head == :loophead
        for i = 1:length(args)
            x.args[i] = AstWalk(x.args[i], dw.callback, dw.cbdata)
        end
        return x
    elseif head == :loopend
        for i = 1:length(args)
            x.args[i] = AstWalk(x.args[i], dw.callback, dw.cbdata)
        end
        return x
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end


function AstWalkCallback(x :: pir_range_actual, dw :: DirWalk, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    dprintln(3,"PIR AstWalkCallback starting")
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    dprintln(3,"PIR AstWalkCallback ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end

    for i = 1:length(x.dim)
        x.lower_bounds[i] = AstWalk(x.lower_bounds[i], dw.callback, dw.cbdata)
        x.upper_bounds[i] = AstWalk(x.upper_bounds[i], dw.callback, dw.cbdata)
    end
    return x
end

function AstWalkCallback(x :: ANY, dw :: DirWalk, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    dprintln(3,"PIR AstWalkCallback starting")
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    dprintln(3,"PIR AstWalkCallback ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
ParallelIR version of AstWalk.
Invokes the DomainIR version of AstWalk and provides the parallel IR AstWalk callback AstWalkCallback.

Parallel IR AstWalk calls Domain IR AstWalk which in turn calls CompilerTools.AstWalker.AstWalk.
For each AST node, CompilerTools.AstWalker.AstWalk calls Domain IR callback to give it a chance to handle the node if it is a Domain IR node.
Likewise, Domain IR callback first calls Parallel IR callback to give it a chance to handle Parallel IR nodes.
The Parallel IR callback similarly first calls the user-level callback to give it a chance to process the node.
If a callback returns "nothing" it means it didn't modify that node and that the previous code should process it.
The Parallel IR callback will return "nothing" if the node isn't a Parallel IR node.
The Domain IR callback will return "nothing" if the node isn't a Domain IR node.
"""
function AstWalk(ast::Any, callback, cbdata)
    dw = DirWalk(callback, cbdata)
    DomainIR.AstWalk(ast, AstWalkCallback, dw)
end

@doc """
An AliasAnalysis callback (similar to LivenessAnalysis callback) that handles ParallelIR introduced AST node types.
For each ParallelIR specific node type, form an array of expressions that AliasAnalysis
    can analyze to reflect the aliases of the given AST node.
    If we read a symbol it is sufficient to just return that symbol as one of the expressions.
    If we write a symbol, then form a fake mk_assignment_expr just to get liveness to realize the symbol is written.
"""
function pir_alias_cb(ast::Expr, state, cbdata)
    dprintln(4,"pir_alias_cb")

    head = ast.head
    args = ast.args
    if head == :parfor
        dprintln(3,"pir_alias_cb for :parfor")
        expr_to_process = Any[]

        assert(typeof(args[1]) == ParallelAccelerator.ParallelIR.PIRParForAst)
        this_parfor = args[1]

        AliasAnalysis.increaseNestLevel(state);
        AliasAnalysis.from_exprs(state, this_parfor.preParFor, pir_alias_cb, cbdata)
        AliasAnalysis.from_exprs(state, this_parfor.body, pir_alias_cb, cbdata)
        ret = AliasAnalysis.from_exprs(state, this_parfor.postParFor, pir_alias_cb, cbdata)
        AliasAnalysis.decreaseNestLevel(state);

        return ret[end]

    elseif head == :call
        if args[1] == TopNode(:unsafe_arrayref)
            return AliasAnalysis.NotArray 
        elseif args[1] == TopNode(:unsafe_arrayset)
            return AliasAnalysis.NotArray 
        end
    end

    return DomainIR.dir_alias_cb(ast, state, cbdata)
end

function pir_alias_cb(ast::ANY, state, cbdata)
    dprintln(4,"pir_alias_cb")
    return DomainIR.dir_alias_cb(ast, state, cbdata)
end

end
