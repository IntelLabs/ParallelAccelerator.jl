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
using CompilerTools.Helper
using ..DomainIR
using CompilerTools.AliasAnalysis
import ..ParallelAccelerator
#if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
if VERSION >= v"0.5"
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

num_threads_mode = 0
function PIRNumThreadsMode(x)
    global num_threads_mode = x
end

print_times = true
function PIRPrintTimes(x)
    global print_times = x
end

const ISCAPTURED = 1
const ISASSIGNED = 2
const ISASSIGNEDBYINNERFUNCTION = 4
const ISCONST = 8
const ISASSIGNEDONCE = 16
const ISPRIVATEPARFORLOOP = 32


function getLoopPrivateFlags()
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
        return 0
    else
        return ISPRIVATEPARFORLOOP
    end
end

"""
Ad-hoc support to mimic closures when we want the arguments to be processed during AstWalk.
"""
type DelayedFunc
  func :: Function
  args :: Array{Any, 1}
end

function callDelayedFuncWith(f::DelayedFunc, args...)
    @dprintln(3,"callDelayedFuncWith f = ", f, " args = ", args...)
    full_args = vcat(f.args, Any[args...])
    f.func(full_args...)
end


"""
Holds the information about a loop in a parfor node.
"""
type PIRLoopNest
    indexVariable :: RHSVar
    lower
    upper
    step
end

"""
Holds the information about a reduction in a parfor node.
"""
type PIRReduction
    reductionVar  :: RHSVar
    reductionVarInit
    reductionFunc
end

"""
Holds information about domain operations part of a parfor node.
"""
type DomainOperation
    operation
    input_args :: Array{Any,1}
end

"""
Holds a dictionary from an array symbol to an integer corresponding to an equivalence class.
All array symbol in the same equivalence class are known to have the same shape.
"""
type EquivalenceClasses
    data :: Dict{Symbol,Int64}

    function EquivalenceClasses()
        new(Dict{Symbol,Int64}())
    end
end

"""
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

"""
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

mk_call(fun,args) = Expr(:call, fun, args...)

function boxOrNot(typ, expr)
if VERSION >= v"0.6.0-pre"
    return expr
else
    return mk_call(GlobalRef(Base, :box), [typ, expr]) 
end
end

"""
Clear an equivalence class.
"""
function EquivalenceClassesClear(ec :: EquivalenceClasses)
    empty!(ec.data)
end

function mk_mult_int_expr(args::Array)
    if length(args)==0
        return 1
    elseif length(args)==1
        return args[1]
    end
    next = 2
    prev_expr = args[1]

    while next<=length(args)
        m_call = mk_call(GlobalRef(Base,:mul_int),[prev_expr,args[next]])
        prev_expr  = boxOrNot(Int64, m_call)
        next += 1
    end
    return prev_expr
end

import Base.hash
import Base.isequal

type RangeExprs
    start_val
    skip_val
    last_val
end

"""
Holds the information from one Domain IR :range Expr.
"""
type RangeData
    start
    skip
    last
    exprs :: RangeExprs
    offset_temp_var :: Union{Symbol, RHSVar}        # New temp variables to hold offset from iteration space for each dimension.

    function RangeData(s, sk, l, sv, skv, lv, temp_var)
        new(s, sk, l, RangeExprs(sv, skv, lv), temp_var)
    end
    function RangeData(re :: RangeExprs)
        new(nothing, nothing, nothing, re, :you_should_never_see_this_used)
    end
end

function hash(x :: RangeData)
    @dprintln(4, "hash of RangeData ", x)
    hash(x.exprs.last_val)
end
function isequal(x :: RangeData, y :: RangeData)
    @dprintln(4, "isequal of RangeData ", x, " ", y)
    isequal(x.exprs, y.exprs)
end
function isequal(x :: RangeExprs, y :: RangeExprs)
    isequal(x.start_val, y.start_val) &&
    isequal(x.skip_val, y.skip_val ) &&
    isequal(x.last_val, y.last_val)
end

function isStartOneRange(re :: RangeExprs)
    return re.start_val == 1
end

type MaskSelector
    value :: RHSVar
end

function hash(x :: MaskSelector)
    ret = hash(x.value)
    @dprintln(4, "hash of MaskSelector ", x, " = ", ret)
    return ret
end
function isequal(x :: MaskSelector, y :: MaskSelector)
    @dprintln(4, "isequal of MaskSelector ", x, " ", y)
    isequal(x.value, y.value)
end

type SingularSelector
    value :: Union{RHSVar,Number}
    offset_temp_var :: RHSVar        # New temp variables to hold offset from iteration space for each dimension.
end

function hash(x :: SingularSelector)
    @dprintln(4, "hash of SingularSelector ", x)
    hash(x.value)
end
function isequal(x :: SingularSelector, y :: SingularSelector)
    @dprintln(4, "isequal of SingularSelector ", x, " ", y)
    isequal(x.value, y.value)
end

const DimensionSelector = Union{RangeData, MaskSelector, SingularSelector}

function hash(x :: Array{DimensionSelector,1})
    @dprintln(4, "Array{DimensionSelector,1} hash")
    sum([hash(i) for i in x])
end
function isequal(x :: Array{DimensionSelector,1}, y :: Array{DimensionSelector,1})
    @dprintln(4, "Array{DimensionSelector,1} isequal")
    if length(x) != length(y)
        return false
    end
    for i = 1:length(x)
        if !isequal(x[i], y[i])
             return false
        end
    end
    return true
end

function hash(x :: Expr)
    @dprintln(4, "hash of Expr")
    return hash(x.head) + hash(x.args)
end
function isequal(x :: Expr, y :: Expr)
    return isequal(x.head, y.head) && isequal(x.args, y.args)
end

#function hash(x :: Array{Any,1})
#@dprintln(4, "hash array ", x)
#    return sum([hash(y) for y in x])
#end
function isequal(x :: Array{Any,1}, y :: Array{Any,1})
    if length(x) != length(y)
       return false
    end
    for i = 1:length(x)
        if !isequal(x[i], y[i])
            return false
        end
    end
    return true
end

"""
Type used by mk_parfor_args... functions to hold information about input arrays.
"""
type InputInfo
    array                                # The name of the array.
    dim                                  # The number of dimensions.
    out_dim                              # The number of indexed (non-const) dimensions.
    indexed_dims :: Array{Bool,1}        # Length of dim where true means we index that dimension and false means we don't (it is singular).
    range :: Array{DimensionSelector,1}  # Empty if whole array, else one RangeData or BitArray mask per dimension.
    elementTemp                          # New temp variable to hold the value of this array/range at the current point in iteration space.
    pre_offsets :: Array{Expr,1}         # Assignments that go in the pre-statements that hold range offsets for each dimension.
    rangeconds :: Array{Expr,1}          # If selecting based on bitarrays, conditional for selecting elements

    function InputInfo()
        new(nothing, 0, 0, Bool[], DimensionSelector[], nothing, Expr[], Expr[])
    end
    function InputInfo(arr)
        new(arr, 0, 0, Bool[], DimensionSelector[], nothing, Expr[], Expr[])
    end
end

function show(io::IO, ii :: ParallelAccelerator.ParallelIR.InputInfo)
    println(io,"")
    println(io,"array   = ", ii.array)
    println(io,"dim     = ", ii.dim)
    println(io,"out_dim = ", ii.out_dim)
    println(io,"indexed_dims = ", ii.indexed_dims)
    println(io,"range   = ", length(ii.range), " ", ii.range)
    println(io,"eltemp  = ", ii.elementTemp)
    println(io,"pre     = ", ii.pre_offsets)
    println(io,"conds   = ", ii.rangeconds)
end

"""
The parfor AST node type.
While we are lowering domain IR to parfors and fusing we use this representation because it
makes it easier to associate related statements before and after the loop to the loop itself.
"""
type PIRParForAst
    first_input  :: InputInfo
    body                                      # holds the body of the innermost loop (outer loops can't have anything in them except inner loops)
    preParFor    :: Array{Any,1}              # do these statements before the parfor
    hoisted      :: Array{Any,1}              # statements hoisted from inside the body (do these statements before the parfor)
    loopNests    :: Array{PIRLoopNest,1}      # holds information about the loop nests
    reductions   :: Array{PIRReduction,1}     # holds information about the reductions
    postParFor   :: Array{Any,1}              # do these statements after the parfor

    original_domain_nodes :: Array{DomainOperation,1}
    top_level_number :: Array{Int,1}

    unique_id
    array_aliases :: Dict{LHSVar, LHSVar}

    # instruction count estimate of the body
    # To get the total loop instruction count, multiply this value by (upper_limit - lower_limit)/step for each loop nest
    # This will be "nothing" if we don't know how to estimate.  If not "nothing" then it is an expression which may
    # include calls.
    instruction_count_expr
    arrays_written_past_index :: Set{LHSVar}
    arrays_read_past_index :: Set{LHSVar}

    force_simd::Bool # generate pragma simd in backend
    function PIRParForAst(fi, b, pre, hoisted, nests, red, post, orig, t, unique, wrote_past_index, read_past_index)
        new(fi, b, pre, hoisted, nests, red, post, orig, [t], unique, Dict{Symbol,Symbol}(), nothing, wrote_past_index, read_past_index, false)
    end
end

function parforArrayInput(parfor :: PIRParForAst)
    return !parforRangeInput(parfor)
#    return isa(parfor.first_input, RHSVar)
end
function parforRangeInput(parfor :: PIRParForAst)
    return isRange(parfor.first_input)
#    return isa(parfor.first_input, Array{DimensionSelector,1})
end

"""
Not currently used but might need it at some point.
Search a whole PIRParForAst object and replace one RHSVar with another.
"""
function replaceParforWithDict(parfor :: PIRParForAst, gensym_map, linfo :: LambdaVarInfo)
    parfor.body = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.body, gensym_map, linfo)
    parfor.preParFor = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.preParFor, gensym_map, linfo)
    parfor.hoisted = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.hoisted, gensym_map, linfo)
    for i = 1:length(parfor.loopNests)
        parfor.loopNests[i].lower = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.loopNests[i].lower, gensym_map, linfo)
        parfor.loopNests[i].upper = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.loopNests[i].upper, gensym_map, linfo)
        parfor.loopNests[i].step = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.loopNests[i].step, gensym_map, linfo)
    end
    for i = 1:length(parfor.reductions)
        parfor.reductions[i].reductionVarInit = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.reductions[i].reductionVarInit, gensym_map, linfo)
    end
    parfor.postParFor = CompilerTools.LambdaHandling.replaceExprWithDict!(parfor.postParFor, gensym_map, linfo)
end

"""
After lowering, it is necessary to make the parfor body top-level statements so that basic blocks
can be correctly identified and labels correctly found.  There is a phase in parallel IR where we
take a PIRParForAst node and split it into a parfor_start node followed by the body as top-level
statements followed by parfor_end (also a top-level statement).
"""
type PIRParForStartEnd
    loopNests  :: Array{PIRLoopNest,1}      # holds information about the loop nests
    reductions :: Array{PIRReduction,1}     # holds information about the reductions
    instruction_count_expr
    private_vars :: Array{RHSVar,1}
    force_simd::Bool
end

"""
State passed around while converting an AST from domain to parallel IR.
"""
type expr_state
    function_name
    block_lives :: CompilerTools.LivenessAnalysis.BlockLiveness    # holds the output of liveness analysis at the block and top-level statement level
    top_level_number :: Int                          # holds the current top-level statement number...used to correlate with stmt liveness info
    # Arrays created from each other are known to have the same size. Store such correlations here.
    # If two arrays have the same dictionary value, they are equal in size.
    next_eq_class            :: Int
    array_length_correlation :: Dict{LHSVar,Int} # map array var -> size group
    symbol_array_correlation :: Dict{Array{Union{RHSVar,Int},1},Int} # map size -> size group
    # keep values for constant tuples. They are often used for allocating and reshaping arrays.
    tuple_table              :: Dict{RHSVar,Array{Union{RHSVar,Int},1}}
    range_correlation        :: Dict{Array{DimensionSelector,1},Int}
    LambdaVarInfo            :: CompilerTools.LambdaHandling.LambdaVarInfo
    max_label :: Int # holds the max number of all LabelNodes
    multi_correlation::Int # correlation number for arrays with multiple assignment
    in_nested :: Bool
    tuple_assigns :: Dict{LHSVar,Array{Any,1}}

    # Initialize the state for parallel IR translation.
    function expr_state(function_name, bl, max_label, input_arrays)
        init_corr = Dict{LHSVar,Int}()
        init_sym_corr = Dict{Array{Union{RHSVar,Int},1},Int}()
        init_tup_table =  Dict{RHSVar,Array{Union{RHSVar,Int},1}}()
        # For each input array, insert into the correlations table with a different value.
        for i = 1:length(input_arrays)
            init_corr[input_arrays[i]] = i
        end
        new(function_name, bl, 0, length(input_arrays)+1, init_corr, init_sym_corr, init_tup_table, Dict{Array{DimensionSelector,1},Int}(), CompilerTools.LambdaHandling.LambdaVarInfo(), max_label, 0, false, Dict{LHSVar,Array{Any,1}}())
    end
end

include("parallel-ir-stencil.jl")

"""
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
            if DEBUG_LVL >= 5
                dump(pnode.preParFor[i])
            end
        end
    end
    if length(pnode.hoisted) > 0
        println(io,"Hoisted: ")
        for i = 1:length(pnode.hoisted)
            println(io,"    ", pnode.hoisted[i])
            if DEBUG_LVL >= 5
                dump(pnode.hoisted[i])
            end
        end
    end
    println(io,"PIR Body: ")
    for i = 1:length(pnode.body)
        println(io,"    ", pnode.body[i])
    end
    if DEBUG_LVL >= 5
        dump(pnode.body)
    end
    if length(pnode.loopNests) > 0
        println(io,"Loop Nests: ")
        for i = 1:length(pnode.loopNests)
            println(io,"    ", pnode.loopNests[i])
            if DEBUG_LVL >= 5
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
            if DEBUG_LVL >= 5
                dump(pnode.postParFor[i])
            end
        end
    end
    if length(pnode.original_domain_nodes) > 0 && DEBUG_LVL >= 5
        println(io,"Domain nodes: ")
        for i = 1:length(pnode.original_domain_nodes)
            println(io,pnode.original_domain_nodes[i])
        end
    end
end

export PIRLoopNest, PIRReduction, from_exprs, PIRParForAst, AstWalk, PIRSetFuseLimit,
       PIRNumSimplify, PIRInplace, PIRRunAsTasks, PIRLimitTask, PIRReduceTasks,
       PIRStencilTasks, PIRFlatParfor, PIRNumThreadsMode, PIRShortcutArrayAssignment,
       PIRTaskGraphMode, PIRPolyhedral, PIRHoistParfors, PIRLateSimplify

late_simplify = true
"""
Controls whether copy propagation and other simplifications are performed after Parallel-IR translation.
"""
function PIRLateSimplify(x :: Bool)
   global late_simplify = x
end

unroll_small_parfors = false
"""
Controls whether copy propagation and other simplifications are performed after Parallel-IR translation.
"""
function PIRUnrollSmallParfors(x :: Bool)
   global unroll_small_parfors = x
end

"""
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

"""
Create an assignment expression AST node given a left and right-hand side.
The left-hand side has to be a symbol node from which we extract the type so as to type the new Expr.
"""
function mk_assignment_expr(lhs::RHSVar, rhs, linfo :: LambdaVarInfo)
    expr_typ = CompilerTools.LambdaHandling.getType(lhs, linfo)
    @dprintln(2,"mk_assignment_expr lhs = ", lhs, " type = ", typeof(lhs), " expr_typ = ", expr_typ, " rhs = ", rhs)
    TypedExpr(expr_typ, :(=), toLHSVar(lhs), rhs)
end

mk_assignment_expr(lhs::RHSVar, rhs, state :: expr_state) = mk_assignment_expr(lhs, rhs, state.LambdaVarInfo)

function mk_assignment_expr(lhs::ANY, rhs, state :: expr_state)
    throw(string("mk_assignment_expr lhs is not of type RHSVar, is of this type instead: ", typeof(lhs)))
end

mk_assignment_expr(lhs :: TypedVar, rhs) = TypedExpr(lhs.typ, :(=), toLHSVar(lhs), rhs)
mk_assignment_expr(lhs :: LHSVar, rhs, typ :: DataType) = TypedExpr(typ, :(=), lhs, rhs)

"""
Only used to create fake expression to force lhs to be seen as written rather than read.
"""
function mk_untyped_assignment(lhs, rhs)
    Expr(:(=), lhs, rhs)
end

function isWholeArray(inputInfo :: InputInfo)
    return length(inputInfo.range) == 0
end

function isRange(inputInfo :: InputInfo)
    return length(inputInfo.range) > 0
end

"""
Compute size of a range.
"""
function rangeSize(start, skip, last)
    # TODO: do something with skip!
    return last - start + 1
end

"""
Create an expression whose value is the length of the input array.
"""
function mk_arraylen_expr(x :: RHSVar, dim :: Int64)
    TypedExpr(Int64, :call, GlobalRef(Base, :arraysize), deepcopy(x), dim)
    #TypedExpr(Int64, :call, GlobalRef(Base, :arraysize), :($x), dim)
    #TypedExpr(Int64, :call, GlobalRef(Base, :arraysize), deepcopy(:($x)), dim)
end

"""
Create an expression whose value is the length of the input array.
"""
function mk_arraylen_expr(x :: InputInfo, dim :: Int64)
    if dim <= length(x.range)
        r = x.range[dim]
        if isa(x.range[dim], RangeData)
            # TODO: do something with skip!
            last  = isa(r.exprs.last_val, Expr)  ? r.last  : r.exprs.last_val
            start = isa(r.exprs.start_val, Expr) ? r.start : r.exprs.start_val
            ret = DomainIR.add(DomainIR.sub(last, start), 1)
            @dprintln(3, "mk_arraylen_expr for range = ", r, " last = ", last, " start = ", start, " ret = ", ret)
            return deepcopy(ret)
        elseif isa(x.range[dim], SingularSelector)
            return 1
        end
    end

    return mk_arraylen_expr(x.array, dim)
end

"""
Create an expression that references something inside ParallelIR.
In other words, returns an expression the equivalent of ParallelAccelerator.ParallelIR.sym where sym is an input argument to this function.
"""
function mk_parallelir_ref(sym :: Symbol, ref_type=Function)
    #inner_call = TypedExpr(Module, :call, TopNode(:getfield), :ParallelAccelerator, QuoteNode(:ParallelIR))
    #TypedExpr(ref_type, :call, TopNode(:getfield), inner_call, QuoteNode(sym))
    #TypedExpr(ref_type, :call, GlobalRef(Base, :getfield), GlobalRef(ParallelAccelerator,:ParallelIR), QuoteNode(sym))
    TypedExpr(ref_type, GlobalRef(ParallelAccelerator.ParallelIR, sym))
end

"""
Returns an expression that convert "ex" into a another type "new_type".
"""
function mk_convert(new_type, ex)
    TypedExpr(new_type, :call, GlobalRef(Base, :convert), new_type, ex)
end

"""
Create an expression which returns the index'th element of the tuple whose name is contained in tuple_var.
"""
function mk_tupleref_expr(tuple_var, index, typ)
    TypedExpr(typ, :call, GlobalRef(Base, :tupleref), tuple_var, index)
end

"""
Make a svec expression.
"""
function mk_svec_expr(parts...)
    TypedExpr(SimpleVector, :call, GlobalRef(Core, :svec), parts...)
end

"""
Return an expression that allocates and initializes a 1D Julia array that has an element type specified by
"elem_type", an array type of "atype" and a "length".
"""
function mk_alloc_array_expr(elem_type, atype, length)
    @dprintln(2,"mk_alloc_array_1d_expr atype = ", atype, " elem_type = ", elem_type, " length = ", length, " typeof(length) = ", typeof(length))
    ret_type = TypedExpr(Type{atype},  :call, GlobalRef(Core, :apply_type), GlobalRef(Core, :Array), elem_type, 1)
    new_svec = TypedExpr(SimpleVector, :call, GlobalRef(Core, :svec), GlobalRef(Base, :Any), GlobalRef(Base, :Int))

    length_expr = get_length_expr(length)

if VERSION >= v"0.6.0-pre"
    return TypedExpr(
       atype,
       :foreigncall,
       QuoteNode(:jl_alloc_array_1d),
       atype,
       eval(new_svec),
       atype,
       0,
       length_expr,
       0)
else
    return TypedExpr(
       atype,
       :call,
       GlobalRef(Core,:ccall),
       QuoteNode(:jl_alloc_array_1d),
       ret_type,
       new_svec,
       atype,
       0,
       length_expr,
       0)
end
end

function get_length_expr(length::Union{RHSVar,Int64})
    return length
end

function get_length_expr(length::Expr)
    return length
end

function get_length_expr(length::Any)
    throw(string("Unhandled length type ", typeof(length), " in mk_alloc_array_1d_expr."))
end

"""
Return an expression that allocates and initializes a 2D Julia array that has an element type specified by
"elem_type", an array type of "atype" and two dimensions of length in "length1" and "length2".
"""
function mk_alloc_array_expr(elem_type, atype, length1, length2)
    @dprintln(2,"mk_alloc_array_2d_expr atype = ", atype)

    ret_type = TypedExpr(Type{atype},  :call, GlobalRef(Core, :apply_type), GlobalRef(Core, :Array), elem_type, 2)
    new_svec = TypedExpr(SimpleVector, :call, GlobalRef(Core,:svec), GlobalRef(Base, :Any), GlobalRef(Base, :Int), GlobalRef(Base, :Int))

if VERSION >= v"0.6.0-pre"
    TypedExpr(
       atype,
       :foreigncall,
       QuoteNode(:jl_alloc_array_2d),
       atype,
       eval(new_svec),
       atype,
       0,
       get_length_expr(length1),
       0,
       get_length_expr(length2),
       0
       )
else
    TypedExpr(
       atype,
       :call,
       GlobalRef(Core,:ccall),
       QuoteNode(:jl_alloc_array_2d),
       ret_type,
       new_svec,
       atype,
       0,
       get_length_expr(length1),
       0,
       get_length_expr(length2),
       0
       )
end
end

"""
Return an expression that allocates and initializes a 3D Julia array that has an element type specified by
"elem_type", an array type of "atype" and two dimensions of length in "length1" and "length2" and "length3".
"""
function mk_alloc_array_expr(elem_type, atype, length1, length2, length3)
    @dprintln(2,"mk_alloc_array_3d_expr atype = ", atype)
    ret_type = TypedExpr(Type{atype},  :call, GlobalRef(Core, :apply_type), GlobalRef(Core, :Array), elem_type, 3)
    new_svec = TypedExpr(SimpleVector, :call, GlobalRef(Core, :svec), GlobalRef(Base, :Any), GlobalRef(Base, :Int), GlobalRef(Base, :Int), GlobalRef(Base, :Int))

if VERSION >= v"0.6.0-pre"
    TypedExpr(
       atype,
       :foreigncall,
       QuoteNode(:jl_alloc_array_3d),
       atype,
       eval(new_svec),
       atype,
       0,
       get_length_expr(length1),
       0,
       get_length_expr(length2),
       0,
       get_length_expr(length3),
       0)
else
    TypedExpr(
       atype,
       :call,
       GlobalRef(Core, :ccall),
       QuoteNode(:jl_alloc_array_3d),
       ret_type,
       new_svec,
       atype,
       0,
       get_length_expr(length1),
       0,
       get_length_expr(length2),
       0,
       get_length_expr(length3),
       0)
end
end

"""
Returns the element type of an Array.
"""
function getArrayElemType(array :: RHSVar, state :: expr_state)
    atyp = CompilerTools.LambdaHandling.getType(array, state.LambdaVarInfo)
    return eltype(atyp)
end

"""
Return the number of dimensions of an Array.
"""
function getArrayNumDims(array :: RHSVar, state :: expr_state)
    gstyp = CompilerTools.LambdaHandling.getType(array, state.LambdaVarInfo)
    @dprintln(3, "getArrayNumDims ", array, " gstyp = ", gstyp, " state = ", state)
    @assert isArrayType(gstyp) "Array expected, but got " * string(gstyp)
    ndims(gstyp)
end

"""
Add a local variable to the current function's LambdaVarInfo.
Returns a symbol node of the new variable.
"""
function createStateVar(state :: expr_state, name, typ, access)
    new_temp_sym = Symbol(name)
    CompilerTools.LambdaHandling.addLocalVariable(new_temp_sym, typ, access, state.LambdaVarInfo)
    return toRHSVar(new_temp_sym, typ, state.LambdaVarInfo)
end

"""
Create a temporary variable that is parfor private to hold the value of an element of an array.
"""
function createTempForArray(array_sn :: RHSVar, unique_id :: Int64, state :: expr_state, temp_type = nothing)
    key = toLHSVar(array_sn)
    if temp_type == nothing
        temp_type = getArrayElemType(array_sn, state)
    end
    return createStateVar(state, string("parallel_ir_array_temp_", key, "_", get_unique_num(), "_", unique_id), temp_type, ISASSIGNED | getLoopPrivateFlags())
end


"""
Takes an existing variable whose name is in "var_name" and adds the descriptor flag ISPRIVATEPARFORLOOP to declare the
variable to be parfor loop private and eventually go in an OMP private clause.
"""
function makePrivateParfor(var_name :: Symbol, state)
    res = CompilerTools.LambdaHandling.addDescFlag(var_name, getLoopPrivateFlags(), state.LambdaVarInfo)
    assert(res)
end

"""
Returns true if all array references use singular index variables and nothing more complicated involving,
for example, addition or subtraction by a constant.
"""
function simpleIndex(dict)
    # For each entry in the dictionary.
    for k in dict
        # Get the corresponding array of seen indexing expressions.
        array_ae = k[2]
        # For each indexing expression.
        for i = 1:length(array_ae)
            ae = array_ae[i]
            @dprintln(3,"typeof(ae) = ", typeof(ae), " ae = ", ae)
            for j = 1:length(ae)
                # If the indexing expression isn't simple then return false.
                if (!isa(ae[j], Number) &&
                    !isa(ae[j], RHSVar) &&
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

"""
Returns the next usable label for the current function.
"""
function next_label(state :: expr_state)
    state.max_label = state.max_label + 1
    return state.max_label
end



function simplify_internal(x :: ANY, state, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Do some simplification to expressions that are part of ranges.
For example, the range 2:s-1 becomes a length (s-1)-2 which this function in turn transforms to s-3.
"""
function simplify_internal(x :: Expr, state, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    is_sub = DomainIR.isSubExpr(x)
    is_add = DomainIR.isAddExpr(x)
    @dprintln(3, "simplify_internal ", x, " is_sub = ", is_sub, " is_add = ", is_add)
    # We only do any simpilfication to addition or subtraction statements at the moment.
    if is_sub || is_add
        # Recursively translate the ops to this operator first.
        x.args[2] = AstWalk(x.args[2], simplify_internal, nothing)
        x.args[3] = AstWalk(x.args[3], simplify_internal, nothing)
        # Extract the two operands to this operation.
        op1 = x.args[2]
        op2 = x.args[3]
        # We only support simplification when operand1 is itself an addition or subtraction operator.
        op1_sub = DomainIR.isSubExpr(op1)
        op1_add = DomainIR.isAddExpr(op1)
        @dprintln(3, "op1 = ", op1, " op2 = ", op2)
        # If operand1 is an addition or subtraction operator and operand2 is a number then keep checking if we can simplify.
        if (op1_sub || op1_add) && isa(op2, Number)
            # Get the two operands to the operand1.
            op1_op1 = op1.args[2]
            op1_op2 = op1.args[3]
            @dprintln(3, "op1_op1 = ", op1_op1, " op1_op2 = ", op1_op2)
            # We can do some simplification if the second operand2 here is also a number.
            if isa(op1_op2, Number)
                @dprintln(3, "simplify will modify")
                # If we have like operations then we can combine the second operands by addition.
                if is_sub == op1_sub
                    new_number = op1_op2 + op2
                    @dprintln(3, "same ops so added to get ", new_number)
                else
                    # Consider, (s-1)+2 and (s+1)-2, where the operations are different.
                    # In both case, we can do 1-2 (op2 is 2, op1_op2 is 1).
                    # This would become (s-(-1)) and (s+(-1)) respectively.
                    new_number = op1_op2 - op2
                    @dprintln(3, "diff ops so subtracted to get ", new_number)
                end

                # If we happen to get a zero then we can eliminate both operations.
                if new_number == 0
                    @dprintln(3, "new_number is 0 so eliminating both operations")
                    return op1_op1
                elseif new_number < 0
                    # Canonicalize so that op2 is always positive by switching the operation from add to sub or vice versa if necessary.
                    @dprintln(3, "new_number < 0 so switching op1 from add to sub or vice versa")
                    op1_sub = !op1_sub
                    new_number = abs(new_number)
                end

                # Form a sub or add expression to replace the current node.
                if op1_sub
                    ret = DomainIR.sub_expr(op1_op1, new_number)
                else
                    ret = DomainIR.add_expr(op1_op1, new_number)
                end
                @dprintln(3,"new simplified expr is ", ret)
                return ret
            end
        end
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Convert one RangeData to some length expression and then simplify it.
"""
function form_and_simplify(rd :: RangeData)
    re = rd.exprs
    @dprintln(3, "form_and_simplify ", re)
    # Number of iteration is (last-start)/skip.  This is only approximate due to the non-linear effects of integer div.
    # We don't attempt to equate different ranges with different skips.
    last_minus_start = DomainIR.sub_expr(re.last_val, re.start_val)
    if re.skip_val != 1
        with_skip = DomainIR.sdiv_int_expr(last_minus_start, re.skip_val)
    else
        with_skip = last_minus_start
    end
    @dprintln(3, "before simplify = ", with_skip)
    ret = AstWalk(with_skip, simplify_internal, nothing)
    @dprintln(3, "after simplify = ", ret)
    return ret
end

function form_and_simplify(x :: ANY)
    return x
end

"""
For each entry in ranges, form a range length expression and simplify them.
"""
function form_and_simplify(ranges :: Array{DimensionSelector,1})
    return [form_and_simplify(x) for x in ranges]
end

"""
We can only do exact matches in the range correlation dict but there can still be non-exact matches
where the ranges are different but equivalent in length.  In this function, we can the dictionary
and look for equivalent ranges.
"""
function nonExactRangeSearch(ranges :: Array{DimensionSelector,1}, range_correlations)
    # Get the simplified form of the range we are looking for.
    simplified = form_and_simplify(ranges)
    @dprintln(3, "searching for simplified expr ", simplified)
    # For each range correlation in the dictionary.
    for kv in range_correlations
        key = kv[1]
        correlation = kv[2]

        @dprintln(3, "Before form_and_simplify(key)")
        # Simplify the current dictionary entry to enable comparison.
        simplified_key = form_and_simplify(key)
        @dprintln(3, "comparing ", simplified, " against simplified_key ", simplified_key)
        # If the simplified form of the incoming range and the dictionary entry are equal now then the ranges are equivalent.
        if isequal(simplified, simplified_key)
            @dprintln(3, "simplified and simplified key are equal")
            return correlation
        else
            @dprintln(3, "simplified and simplified key are not equal")
        end
    end
    # No equivalent range entry in the dictionary.
    return nothing
end


if VERSION >= v"0.5"
    unique_num = Atomic{Int}(1)
    """
    If we need to generate a name and make sure it is unique then include an monotonically increasing number.
    """
    function get_unique_num()
        atomic_add!(unique_num, 1)
    end
else
    unique_num = 1
    """
    If we need to generate a name and make sure it is unique then include an monotonically increasing number.
    """
    function get_unique_num()
        ret = unique_num
        global unique_num = unique_num + 1
        ret
    end
end


# ===============================================================================================================================

include("parallel-ir-mk-parfor.jl")

type PrivateSetData
    privates :: Set{RHSVar}
    linfo    :: LambdaVarInfo
end

"""
The AstWalk callback function for getPrivateSet.
For each AST in a parfor body, if the node is an assignment or loop head node then add the written entity to the state.
"""
function getPrivateSetInner(x::Expr, state :: PrivateSetData, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    # If the node is an assignment node or a loop head node.
    if isAssignmentNode(x) || isLoopheadNode(x)
        lhs = x.args[1]
        assert(isa(lhs, RHSVar))
        if isa(lhs, GenSym)
            push!(state.privates, lhs)
        else
            if LambdaHandling.getDesc(lhs, state.linfo) & ISPRIVATEPARFORLOOP != 0
              push!(state.privates, lhs)
            end
            #sname = CompilerTools.LambdaHandling.lookupVariableName(lhs, state.linfo)
            #red_var_start = "parallel_ir_reduction_output_"
            #red_var_len = length(red_var_start)
            #sstr = string(sname)
            #if length(sstr) >= red_var_len
            #    if sstr[1:red_var_len] == red_var_start
            #        # Skip this symbol if it begins with "parallel_ir_reduction_output_" signifying a reduction variable.
            #        return CompilerTools.AstWalker.ASTWALK_RECURSE
            #    end
            #end
        end
    elseif isBareParfor(x)
        for ln = x.args[1].loopNests
            push!(state.privates, ln.indexVariable)
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function getPrivateSetInner(x::ANY, state :: PrivateSetData, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Go through the body of a parfor and collect those Symbols, GenSyms, etc. that are assigned to within the parfor except reduction variables.
"""
function getPrivateSet(body :: Array{Any,1}, linfo :: LambdaVarInfo)
    @dprintln(3,"getPrivateSet")
    printBody(3, body)
    state = PrivateSetData(Set{RHSVar}(), linfo)
    for i = 1:length(body)
        AstWalk(body[i], getPrivateSetInner, state)
    end
    @dprintln(3,"private_set = ", state.privates)
    return state.privates
end

# ===============================================================================================================================

type CountAssignmentsState
    symbol_assigns :: Dict{Symbol, Int}
    linfo          :: LambdaVarInfo
end

"""
AstWalk callback to count the number of static times that a symbol is assigned within a method.
"""
function count_assignments(x, state :: CountAssignmentsState, top_level_number, is_top_level, read)
    if isAssignmentNode(x) || isLoopheadNode(x)
        lhs = x.args[1]
        # GenSyms don't have descriptors so no need to count their assignment.
        if !hasSymbol(lhs)
            return CompilerTools.AstWalker.ASTWALK_RECURSE
        end
        sname = CompilerTools.LambdaHandling.lookupVariableName(lhs, state.linfo)
        if !haskey(state.symbol_assigns, sname)
            state.symbol_assigns[sname] = 0
        end
        state.symbol_assigns[sname] = state.symbol_assigns[sname] + 1
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Just call the AST walker for symbol for parallel IR nodes with no state.
"""
function pir_live_cb_def(x)
    pir_live_cb(x, nothing)
end

function from_lambda_body(body :: Expr, depth, state)
    # Process the lambda's body.
    @dprintln(3,"state.LambdaVarInfo.var_defs = ", state.LambdaVarInfo.var_defs)
    body = get_one(from_expr(body, depth, state, false))
    @dprintln(4,"from_lambda after from_expr")
    @dprintln(3,"After processing lambda body = ", state.LambdaVarInfo)
    @dprintln(3,"from_lambda: after body = ")
    printBody(3, body)
    body = update_lambda_vars(state.LambdaVarInfo, body)
    return body
end

function update_lambda_vars(LambdaVarInfo, body)
    # Count the number of static assignments per var.
    cas = CountAssignmentsState(Dict{Symbol, Int}(), LambdaVarInfo)
    AstWalk(body, count_assignments, cas)

    # After counting static assignments, update the LambdaVarInfo for those vars
    # to say whether the var is assigned once or multiple times.
    CompilerTools.LambdaHandling.updateAssignedDesc(LambdaVarInfo, cas.symbol_assigns)

    body = CompilerTools.LambdaHandling.eliminateUnusedLocals!(LambdaVarInfo, body, ParallelAccelerator.ParallelIR.AstWalk)
    @dprintln(3,"After eliminating unused locals = ", LambdaVarInfo)
    CompilerTools.LambdaHandling.stripCaptureFlag(LambdaVarInfo)
    return body
end

"""
Process a :lambda Expr.
"""
function from_lambda(lambda :: Expr, depth, state)
    # :lambda expression
    assert(lambda.head == :lambda)
    @dprintln(3,"from_lambda starting")

    # Save the current LambdaVarInfo away so we can restore it later.
    save_LambdaVarInfo  = state.LambdaVarInfo
    state.LambdaVarInfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(lambda)
    body = from_lambda_body(body, depth, state)

if false
    # Process the lambda's body.
    @dprintln(3,"state.LambdaVarInfo.var_defs = ", state.LambdaVarInfo.var_defs)
    body = get_one(from_expr(body, depth, state, false))
    @dprintln(4,"from_lambda after from_expr")
    @dprintln(3,"After processing lambda body = ", state.LambdaVarInfo)
    @dprintln(3,"from_lambda: after body = ")
    printBody(3, body)

    # Count the number of static assignments per var.
    cas = CountAssignmentsState(Dict{Symbol, Int}(), state.LambdaVarInfo)
    AstWalk(body, count_assignments, cas)

    # After counting static assignments, update the LambdaVarInfo for those vars
    # to say whether the var is assigned once or multiple times.
    CompilerTools.LambdaHandling.updateAssignedDesc(state.LambdaVarInfo, cas.symbol_assigns)

    body = CompilerTools.LambdaHandling.eliminateUnusedLocals!(state.LambdaVarInfo, body, ParallelAccelerator.ParallelIR.AstWalk)
end

    @dprintln(3,"LambdaVarInfo = ", state.LambdaVarInfo)
    @dprintln(3,"new body = ", body)
    # Write the LambdaVarInfo back to the lambda AST node.
    lambda = CompilerTools.LambdaHandling.LambdaVarInfoToLambda(state.LambdaVarInfo, body, ParallelAccelerator.ParallelIR.AstWalk)

    state.LambdaVarInfo = save_LambdaVarInfo

    @dprintln(4,"from_lambda ending")
    return lambda
end

"""
Is a node a loophead expression node (a form of assignment).
"""
function isLoopheadNode(node :: Expr)
    return node.head == :loophead
end

function isLoopheadNode(node)
    return false
end

"""
Is this a parfor node not part of an assignment statement.
"""
function isBareParfor(node :: Expr)
    return node.head == :parfor
end

function isBareParfor(node)
    return false
end


function isParforAssignmentNodeInner(lhs::RHSVar, rhs::Expr)
    if rhs.head==:parfor
        @dprintln(4,"Found a parfor assignment node.")
        return true
    end
    return false
end

function isParforAssignmentNodeInner(lhs::Any, rhs::Any)
    return false
end

"""
Is a node an assignment expression with a parfor node as the right-hand side.
"""
function isParforAssignmentNode(node::Expr)
    @dprintln(4,"isParforAssignmentNode")
    @dprintln(4,node)

    if isAssignmentNode(node)
        assert(length(node.args) >= 2)
        lhs = node.args[1]
        @dprintln(4,lhs)
        rhs = node.args[2]
        @dprintln(4,rhs)
        return isParforAssignmentNodeInner(lhs, rhs)
    else
        @dprintln(4,"node is not an assignment Expr")
    end

    return false
end

function isParforAssignmentNode(node::Any)
    @dprintln(4,"node is not an Expr")
    return false
end

"""
Get the parfor object from either a bare parfor or one part of an assignment.
"""
function getParforNode(node)
    if isBareParfor(node)
        return node.args[1]
    else
        return node.args[2].args[1]
    end
end

"""
Get the right-hand side of an assignment expression.
"""
function getRhsFromAssignment(assignment)
    assignment.args[2]
end

"""
Get the left-hand side of an assignment expression.
"""
function getLhsFromAssignment(assignment)
    assignment.args[1]
end

"""
Returns true if the domain operation mapped to this parfor has the property that the iteration space
is identical to the dimenions of the inputs.
"""
function iterations_equals_inputs(node :: ParallelAccelerator.ParallelIR.PIRParForAst)
    @assert length(node.original_domain_nodes)>0 "parfor original_domain_nodes is empty"

    first_domain_node = node.original_domain_nodes[1]
    first_type = first_domain_node.operation
    if first_type == :map   ||
        first_type == :map!  ||
        first_type == :mmap  ||
        first_type == :mmap! ||
        first_type == :reduce
        @dprintln(3,"iteration count of node equals length of inputs")
        return true
    else
        @dprintln(3,"iteration count of node does not equal length of inputs")
        return false
    end
end

"""
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
            assert(isa(assignment.args[i], TypedVar))
            @dprintln(3,"getLhsOutputSet FusionSentinal assignment with symbol ", assignment.args[i].name)
            # Add to output set.
            push!(ret,assignment.args[i].name)
        end
    else
        lhsVar = toLHSVar(lhs)
        push!(ret,lhsVar)
        @dprintln(3,"getLhsOutputSet lhsVar = ", lhsVar)
    end

    ret
end

"""
Return an expression which creates a tuple.
"""
function mk_tuple_expr(tuple_fields, typ)
    # Tuples are formed with a call to :tuple.
    TypedExpr(typ, :call, GlobalRef(Base, :tuple), tuple_fields...)
end

function getAliasMap(loweredAliasMap, sym)
    if haskey(loweredAliasMap, sym)
        return loweredAliasMap[sym]
    else
        return sym
    end
end

"""
Pull the information from the inner domain lambda into the outer lambda after applying it to a set of arguments.
Return the body (as an array) after application.
"""
function mergeLambdaIntoOuterState(state, dl :: DomainLambda, args :: Array{Any, 1})
    @dprintln(3,"mergeLambdaIntoOuterState")
    @dprintln(3,"state.LambdaVarInfo = ", state.LambdaVarInfo)
    @dprintln(3,"DomainLambda = ", dl)
    @dprintln(3,"arguments = ", args)
    params = getInputParameters(dl.linfo)
    @dprintln(3,"parameters = ", params)
    @assert (length(params) == length(args)) "Parameters and arguments do not match: params = " * string(params)
    repl_dict = CompilerTools.LambdaHandling.mergeLambdaVarInfo(state.LambdaVarInfo, dl.linfo, getLoopPrivateFlags())
    for i = 1:length(params)
        repl_dict[lookupLHSVarByName(params[i], dl.linfo)] = args[i]
    end
    @dprintln(3, "repl_dict = ", repl_dict)
    body = CompilerTools.LambdaHandling.replaceExprWithDict!(deepcopy(dl.body.args), repl_dict, dl.linfo, AstWalk)
    @dprintln(3, "after replacement, body = ")
    printBody(3, body)
    return body
end

# Create a variable for a left-hand side of an assignment to hold the multi-output tuple of a parfor.
function createRetTupleType(rets :: Array{RHSVar,1}, unique_id :: Int64, state :: expr_state)
    # Form the type of the tuple var.
    tt_args = [ CompilerTools.LambdaHandling.getType(x, state.LambdaVarInfo) for x in rets]
    temp_type = Tuple{tt_args...}

    new_temp_name  = Symbol(string("parallel_ir_ret_holder_",unique_id))
    CompilerTools.LambdaHandling.addLocalVariable(new_temp_name, temp_type, ISASSIGNEDONCE | ISCONST | ISASSIGNED, state.LambdaVarInfo)
    new_temp_snode = toRHSVar(new_temp_name, temp_type, state.LambdaVarInfo)
    @dprintln(3, "Creating variable for multiple return from parfor = ", new_temp_snode)

    new_temp_snode
end

# Takes the output of two parfors and merges them while eliminating outputs from
# the previous parfor that have their only use in the current parfor.
function create_arrays_assigned_to_by_either_parfor(arrays_assigned_to_by_either_parfor :: Array{Symbol,1}, allocs_to_eliminate, unique_id, state, sym_to_typ)
    @dprintln(3,"create_arrays_assigned_to_by_either_parfor arrays_assigned_to_by_either_parfor = ", arrays_assigned_to_by_either_parfor)
    @dprintln(3,"create_arrays_assigned_to_by_either_parfor allocs_to_eliminate = ", allocs_to_eliminate, " typeof(allocs) = ", typeof(allocs_to_eliminate))

    # This is those outputs of the prev parfor which don't die during cur parfor.
    prev_minus_eliminations = Symbol[]
    for i = 1:length(arrays_assigned_to_by_either_parfor)
        if !in(arrays_assigned_to_by_either_parfor[i], allocs_to_eliminate)
            push!(prev_minus_eliminations, arrays_assigned_to_by_either_parfor[i])
        end
    end
    @dprintln(3,"create_arrays_assigned_to_by_either_parfor: outputs from previous parfor that continue to live = ", prev_minus_eliminations)

    # Create an array of TypedVar for real values to assign into.
    all_array = map(x -> toRHSVar(x,sym_to_typ[x], state.LambdaVarInfo), prev_minus_eliminations)
    @dprintln(3,"create_arrays_assigned_to_by_either_parfor: all_array = ", all_array, " typeof(all_array) = ", typeof(all_array))

    # If there is only one such value then the left side is just a simple TypedVar.
    if length(all_array) == 1
        return (all_array[1], all_array, true)
    end

    # Create a new var to hold multi-output tuple.
    (createRetTupleType(all_array, unique_id, state), all_array, false)
end

function getAllAliases(input :: LHSVar, aliases :: Dict{LHSVar, LHSVar})
    return getAllAliases(Set{LHSVar}([input]), aliases)
end

function getAllAliases(input :: Set{LHSVar}, aliases :: Dict{LHSVar, LHSVar})
    @dprintln(3,"getAllAliases input = ", input, " aliases = ", aliases)
    out = Set()

    for i in input
        @dprintln(3, "input = ", i)
        push!(out, i)
        cur = i
        while haskey(aliases, cur)
            cur = aliases[cur]
            @dprintln(3, "cur = ", cur)
            push!(out, cur)
        end
    end

    @dprintln(3,"getAllAliases out = ", out)
    return out
end

function isAllocation(expr :: Expr)
    if (expr.head == :call && isBaseFunc(expr.args[1], :ccall)) || expr.head == :foreigncall
        call_offset = getCallOffset(expr)        
        if expr.args[call_offset] == QuoteNode(:jl_alloc_array_1d) || 
           expr.args[call_offset] == QuoteNode(:jl_alloc_array_2d) || 
           expr.args[call_offset] == QuoteNode(:jl_alloc_array_3d) || 
           expr.args[call_offset] == QuoteNode(:jl_new_array)
            return true
        end
    end
    return false
end

function isAllocation(expr)
    return false
end

function getCallOffset(expr :: Expr)
    if expr.head == :call
        assert(isBaseFunc(expr.args[1], :ccall))
        return 2
    elseif expr.head == :foreigncall
        return 1
    else
        throw(string("getFirstAllocSize called for a non-allocation."))
    end
end

function getFirstAllocSize(expr :: Expr)
    call_offset = getCallOffset(expr)
    size_offset = call_offset + 5

    if expr.args[call_offset] == QuoteNode(:jl_alloc_array_1d)
        num_dim = 1
    elseif expr.args[call_offset] == QuoteNode(:jl_alloc_array_2d)
        num_dim = 2
    elseif expr.args[call_offset] == QuoteNode(:jl_alloc_array_3d)
        num_dim = 3
    end
    return num_dim, size_offset
end

# Takes one statement in the preParFor of a parfor and a set of variables that we've determined we can eliminate.
# Returns true if this statement is an allocation of one such variable.
function is_eliminated_allocation_map(x :: Expr, all_aliased_outputs :: Set, removed_allocs :: Set)
    @dprintln(4,"is_eliminated_allocation_map: x = ", x, " typeof(x) = ", typeof(x), " all_aliased_outputs = ", all_aliased_outputs)
    @dprintln(4,"is_eliminated_allocation_map: head = ", x.head)
    if x.head == :(=)
        lhs = toLHSVar(x.args[1])
        rhs = x.args[2]
        if isAllocation(rhs)
            @dprintln(4,"is_eliminated_allocation_map: lhs = ", lhs)
            if !in(lhs, all_aliased_outputs)
                @dprintln(4,"is_eliminated_allocation_map: this will be removed => ", x)
                push!(removed_allocs, lhs)
                return true
            end
        end
    end

    return false
end

function is_eliminated_allocation_map(x, all_aliased_outputs :: Set)
    @dprintln(4,"is_eliminated_allocation_map: x = ", x, " typeof(x) = ", typeof(x), " all_aliased_outputs = ", all_aliased_outputs)
    return false
end

function is_dead_arrayset(x, removed_allocs :: Set)
    if isArraysetCall(x)
        array_to_set = x.args[2]
        if in(toLHSVar(array_to_set), removed_allocs)
            return true
        end
    end

    return false
end

"""
Holds data for modifying arrayset calls.
"""
type sub_arrayset_data
    arrays_set_in_cur_body #remove_arrayset
    output_items_with_aliases
    escaping_sets
end

"""
Is a node an arrayset node?
"""
function isArrayset(x)
    isBaseFunc(x, :arrayset) || isBaseFunc(x, :unsafe_arrayset)
end

"""
Is a node an arrayref node?
"""
function isArrayref(x)
    isBaseFunc(x, :arrayref) || isBaseFunc(x, :unsafe_arrayref)
end

"""
Is a node a call to arrayset.
"""
function isArraysetCall(x :: Expr)
    return x.head == :call && isArrayset(x.args[1])
end

function isArraysetCall(x)
    return false
end

"""
Is a node a call to arrayref.
"""
function isArrayrefCall(x :: Expr)
    return x.head == :call && isArrayref(x.args[1])
end

function isArrayrefCall(x)
    return false
end

"""
AstWalk callback that does the work of substitute_arrayset on a node-by-node basis.
"""
function sub_arrayset_walk(x::Expr, cbd, top_level_number, is_top_level, read)
    use_dbg_level = 4
    dprintln(use_dbg_level,"sub_arrayset_walk ", x, " ", cbd.arrays_set_in_cur_body, " ", cbd.output_items_with_aliases)

    dprintln(use_dbg_level,"sub_arrayset_walk is Expr")
    if x.head == :call
        dprintln(use_dbg_level,"sub_arrayset_walk is :call")
        if isArrayset(x.args[1])
            # Here we have a call to arrayset.
            dprintln(use_dbg_level,"sub_arrayset_walk is :arrayset")
            array_name = x.args[2]
            value      = x.args[3]
            index      = x.args[4]
            assert(isa(array_name, RHSVar))
            # If the array being assigned to is in temp_map.
            lhs_var = toLHSVar(array_name)
            dprintln(use_dbg_level,"lhs_var = ", lhs_var)
            # do not eliminate those not in escaping_sets, because they could be used later locally in the parfor body
            if in(lhs_var, cbd.arrays_set_in_cur_body) && in(lhs_var, cbd.escaping_sets)
                return nothing
            elseif !in(lhs_var, cbd.output_items_with_aliases) && in(lhs_var, cbd.escaping_sets)
                return nothing
            else
                dprintln(use_dbg_level,"sub_arrayset_walk array_name will not substitute ", array_name)
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function sub_arrayset_walk(x::ANY, cbd, top_level_number, is_top_level, read)
    use_dbg_level = 4
    dprintln(use_dbg_level,"sub_arrayset_walk ", x, " ", cbd.arrays_set_in_cur_body, " ", cbd.output_items_with_aliases)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Modify the body of a parfor.
temp_map holds a map of array names whose arraysets should be turned into a mapped variable instead of the arrayset. a[i] = b. a=>c. becomes c = b
map_for_non_eliminated holds arrays for which we need to add a variable to save the value but we can't eiminate the arrayset. a[i] = b. a=>c. becomes c = a[i] = b
    map_drop_arrayset drops the arrayset without replacing with a variable.  This is because a variable was previously added here with a map_for_non_eliminated case.
    a[i] = b. becomes b
"""
function substitute_arrayset(x, arrays_set_in_cur_body, output_items_with_aliases, escaping_sets)
    @dprintln(3,"substitute_arrayset ", x, " ", arrays_set_in_cur_body, " ", output_items_with_aliases, " ", escaping_sets)
    # Walk the AST and call sub_arrayset_walk for each node.
    return AstWalk(x, sub_arrayset_walk, sub_arrayset_data(arrays_set_in_cur_body, output_items_with_aliases, escaping_sets))
end

"""
Get the variable which holds the length of the first input array to a parfor.
"""
function getFirstArrayLens(parfor, num_dims, state)
    ret = Any[]
    prestatements = parfor.preParFor

    if parforArrayInput(parfor)
        @dprintln(3, "getFirstArrayLens using prestatements")
        # Scan the prestatements and find the assignment nodes.
        # If it is an assignment from arraysize.
        for i = 1:length(prestatements)
            x = prestatements[i]
            if (typeof(x) == Expr) && (x.head == :(=))
                lhs = x.args[1]
                rhs = x.args[2]
                if isa(rhs, Expr) && (rhs.head == :call) && isBaseFunc(rhs.args[1],:arraysize)
                    push!(ret, toRHSVar(lhs, state.LambdaVarInfo))
                end
                if length(ret) == num_dims
                    return ret
                end
            end
        end
        # if arraysize calls were replaced for constant size array
        if length(ret)==0
            # assuming first_input is valid at this point since it is before fusion of this parfor
            arr = toLHSVar(parfor.first_input.array)
            arr_class = state.array_length_correlation[arr]
            for (d, v) in state.symbol_array_correlation
                if v==arr_class
                    #
                    ret = d
                    break
                end
            end
        end
    else
        @dprintln(3, "getFirstArrayLens using loopNests")
        for ln in parfor.loopNests
            push!(ret, ln.upper)
        end
    end
    assert(length(ret) == num_dims)
    ret
end

"""
Holds the data for substitute_cur_body AST walk.
"""
type cur_body_data
    temp_map  :: Dict{LHSVar, RHSVar}    # Map of array name to temporary.  Use temporary instead of arrayref of the array name.
    index_map :: Dict{LHSVar, LHSVar}        # Map index variables from parfor being fused to the index variables of the parfor it is being fused with.
    arrays_set_in_cur_body :: Set{LHSVar}    # Used as output.  Collects the arrays set in the current body.
    replace_array_name_in_arrayset :: Dict{LHSVar, LHSVar}  # Map from one array to another.  Replace first array with second when used in arrayset context.
    state :: expr_state
end

"""
AstWalk callback that does the work of substitute_cur_body on a node-by-node basis.
"""
function sub_cur_body_walk(x::Expr,
                           cbd::cur_body_data,
                           top_level_number::Int64,
                           is_top_level::Bool,
                           read::Bool)
    dbglvl = 4
    dprintln(dbglvl,"sub_cur_body_walk ", x)

    dprintln(dbglvl,"sub_cur_body_walk xtype is Expr")
    if x.head == :call
        dprintln(dbglvl,"sub_cur_body_walk xtype is call x.args[1] = ", x.args[1], " type = ", typeof(x.args[1]))
        # Found a call to arrayref.
        if isArrayref(x.args[1]) ||
           x.args[1] == GlobalRef(ParallelAccelerator.API, :getindex)
            dprintln(dbglvl,"sub_cur_body_walk xtype is arrayref, unsafe_arrayref, or getindex")
            array_name = x.args[2]
            index      = x.args[3]
            assert(isa(array_name, RHSVar))
            lowered_array_name = toLHSVar(array_name)
            assert(isa(lowered_array_name, LHSVar))
            dprintln(dbglvl, "array_name = ", array_name, " index = ", index, " lowered_array_name = ", lowered_array_name)
            # If the array name is in cbd.temp_map then replace the arrayref call with the mapped variable.
            if haskey(cbd.temp_map, lowered_array_name)
                dprintln(dbglvl,"sub_cur_body_walk IS substituting ", cbd.temp_map[lowered_array_name])
                return cbd.temp_map[lowered_array_name]
            end
        elseif isArrayset(x.args[1])
            array_name = x.args[2]
            assert(isa(array_name, RHSVar))
            push!(cbd.arrays_set_in_cur_body, toLHSVar(array_name))
            if haskey(cbd.replace_array_name_in_arrayset, toLHSVar(array_name))
                new_symgen = cbd.replace_array_name_in_arrayset[toLHSVar(array_name)]
                x.args[2]  = toRHSVar(new_symgen, CompilerTools.LambdaHandling.getType(new_symgen, cbd.state.LambdaVarInfo), cbd.state.LambdaVarInfo)
            end
        end
    end

    dprintln(dbglvl,"sub_cur_body_walk not substituting")
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

#=
function sub_cur_body_walk(x::Symbol,
                           cbd::cur_body_data,
                           top_level_number::Int64,
                           is_top_level::Bool,
                           read::Bool)
    dbglvl = 3
    dprintln(dbglvl,"sub_cur_body_walk ", x)

    dprintln(dbglvl,"sub_cur_body_walk xtype is Symbol")
    if haskey(cbd.index_map, x)
        # Detected the use of an index variable.  Change it to the first parfor's index variable.
        dprintln(dbglvl,"sub_cur_body_walk IS substituting ", cbd.index_map[x])
        return cbd.index_map[x]
    end

    dprintln(dbglvl,"sub_cur_body_walk not substituting")
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end
=#

function sub_cur_body_walk(x::RHSVar,
                           cbd::cur_body_data,
                           top_level_number::Int64,
                           is_top_level::Bool,
                           read::Bool)
    dbglvl = 4
    dprintln(dbglvl,"sub_cur_body_walk ", x)
    lhsVar = toLHSVar(x)

    dprintln(dbglvl,"sub_cur_body_walk xtype is TypedVar")
    if haskey(cbd.index_map, lhsVar)
        # Detected the use of an index variable.  Change it to the first parfor's index variable.
        dprintln(dbglvl,"sub_cur_body_walk IS substituting ", cbd.index_map[lhsVar])
        x = toRHSVar(cbd.index_map[lhsVar], cbd.state.LambdaVarInfo)
        return x
    end

    dprintln(dbglvl,"sub_cur_body_walk not substituting")
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end


function sub_cur_body_walk(x::ANY,
                           cbd::cur_body_data,
                           top_level_number::Int64,
                           is_top_level::Bool,
                           read::Bool)

    dbglvl = 4
    dprintln(dbglvl,"sub_cur_body_walk ", x)

    dprintln(dbglvl,"sub_cur_body_walk not substituting")
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Make changes to the second parfor body in the process of parfor fusion.
temp_map holds array names for which arrayrefs should be converted to a variable.  a[i].  a=>b. becomes b
    index_map holds maps between index variables.  The second parfor is modified to use the index variable of the first parfor.
    arrays_set_in_cur_body           # Used as output.  Collects the arrays set in the current body.
    replace_array_name_in_arrayset   # Map from one array to another.  Replace first array with second when used in arrayset context.
"""
function substitute_cur_body(x,
    temp_map :: Dict{LHSVar, RHSVar},
    index_map :: Dict{LHSVar, LHSVar},
    arrays_set_in_cur_body :: Set{LHSVar},
    replace_array_name_in_arrayset :: Dict{LHSVar, LHSVar},
    state :: expr_state)
    @dprintln(3,"substitute_cur_body ", x)
    @dprintln(3,"temp_map = ", temp_map)
    @dprintln(3,"index_map = ", index_map)
    @dprintln(3,"arrays_set_in_cur_body = ", arrays_set_in_cur_body)
    @dprintln(3,"replace_array_name_in_array_set = ", replace_array_name_in_arrayset)
    # Walk the AST and call sub_cur_body_walk for each node.
    return AstWalk(x, sub_cur_body_walk, cur_body_data(temp_map, index_map, arrays_set_in_cur_body, replace_array_name_in_arrayset, state))
end

function is_eliminated_arraysize(x::Expr, removed_allocs :: Set, aliases)
    @dprintln(3,"is_eliminated_arraylen ", x)

    @dprintln(3,"is_eliminated_arraylen is Expr")
    if x.head == :(=)
        rhs = x.args[2]
        if isa(rhs, Expr) && rhs.head == :call
            @dprintln(5,"is_eliminated_arraylen is :call")
            if isBaseFunc(rhs.args[1], :arraysize)
                array_used_lhsvar = toLHSVar(rhs.args[2])
                array_used_aliases = getAllAliases(array_used_lhsvar, aliases)
                intersection = intersect(array_used_aliases, removed_allocs)
                @dprintln(3,"is_eliminated_arraylen is :arraysize lhsvar = ", array_used_lhsvar, " with_aliases = ", array_used_aliases, " intersection = ", intersection, " removed_allocs = ", removed_allocs)
                if !isempty(intersection)
                    @dprintln(3,"eliminated ", array_used_lhsvar)
                    return true
                end
            end
        end
    end

    return false
end

function is_eliminated_arraysize(x::ANY, removed_allocs :: Set, aliases)
   return false
end

"""
Returns true if the input node is an assignment node where the right-hand side is a call to arraysize.
"""
function is_eliminated_arraylen(x::Expr)
    @dprintln(3,"is_eliminated_arraylen ", x)

    @dprintln(3,"is_eliminated_arraylen is Expr")
    if x.head == :(=)
        rhs = x.args[2]
        if isa(rhs, Expr) && rhs.head == :call
            @dprintln(5,"is_eliminated_arraylen is :call")
            if isBaseFunc(rhs.args[1], :arraysize)
                @dprintln(3,"is_eliminated_arraylen is :arraysize")
                return true
            end
        end
    end

    return false
end

function is_eliminated_arraylen(x::ANY)
    @dprintln(3,"is_eliminated_arraylen ", x)
    return false
end

"""
AstWalk callback that does the work of substitute_arraylen on a node-by-node basis.
replacement is an array containing the length of the dimensions of the arrays a part of this parfor.
If we see a call to create an array, replace the length params with those in the common set in "replacement".
"""
function sub_arraylen_walk(x::Expr, replacement, top_level_number, is_top_level, read)
    @dprintln(4,"sub_arraylen_walk ", x)

    if x.head == :(=)
        rhs = x.args[2]
        if isAllocation(rhs)
            num_dim, size_offset = getFirstAllocSize(rhs)
            if num_dim == 1
                rhs.args[size_offset] = replacement[1]
            elseif num_dim == 2
                rhs.args[size_offset] = replacement[1]
                rhs.args[size_offset+2] = replacement[2]
            elseif num_dim == 3
                rhs.args[size_offset] = replacement[1]
                rhs.args[size_offset+2] = replacement[2]
                rhs.args[size_offset+4] = replacement[3]
            end
        end
    end

    @dprintln(4,"sub_arraylen_walk not substituting")

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function sub_arraylen_walk(x::ANY, replacement, top_level_number, is_top_level, read)
    @dprintln(4,"sub_arraylen_walk ", x)
    @dprintln(4,"sub_arraylen_walk not substituting")

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
replacement is an array containing the length of the dimensions of the arrays a part of this parfor.
If we see a call to create an array, replace the length params with those in the common set in "replacement".
"""
function substitute_arraylen(x, replacement)
    @dprintln(3,"substitute_arraylen ", x, " ", replacement)
    # Walk the AST and call sub_arraylen_walk for each node.
    return DomainIR.AstWalk(x, sub_arraylen_walk, replacement)
end

fuse_limit = -1
"""
Control how many parfor can be fused for testing purposes.
    -1 means fuse all possible parfors.
    0  means don't fuse any parfors.
    1+ means fuse the specified number of parfors but then stop fusing beyond that.
"""
function PIRSetFuseLimit(x)
    global fuse_limit = x
end

"""
Specify the number of passes over the AST that do things like hoisting and other rearranging to maximize fusion.
DEPRECATED.
"""
function PIRNumSimplify(x)
    println("PIRNumSimplify is deprecated.")
end

"""
Add to the map of symbol names to types.
"""
function rememberTypeForSym(sym_to_type :: Dict{LHSVar, DataType}, sym :: LHSVar, typ :: DataType)
    if typ == Any
        @dprintln(0, "rememberTypeForSym: sym = ", sym, " typ = ", typ)
    end
    assert(typ != Any)
    sym_to_type[sym] = typ
end

"""
Just used to hold a spot in an array to indicate the this is a special assignment expression with embedded real array output names from a fusion.
"""
type FusionSentinel
end

"""
Check if an assignement is a fusion assignment.
    In regular assignments, there are only two args, the left and right hand sides.
    In fusion assignments, we introduce a third arg that is marked by an object of FusionSentinel type.
"""
function isFusionAssignment(x :: Expr)
    if x.head != :(=)
        return false
    elseif length(x.args) <= 2
        return false
    else
        assert(typeof(x.args[3]) == FusionSentinel)
        return true
    end
end

"""
Get the equivalence class of the first array who length is extracted in the pre-statements of the specified "parfor".
"""
function getParforCorrelation(parfor, state)
    return getCorrelation(parfor.first_input, state)
end

"""
Get the equivalence class of a domain IR input in inputInfo.
"""
function getCorrelation(sng :: RHSVar, state :: expr_state)
    @dprintln(3, "getCorrelation for RHSVar = ", sng)
    return getOrAddArrayCorrelation(toLHSVar(sng), state)
end

function getCorrelation(array :: RHSVar, are :: Array{DimensionSelector,1}, state :: expr_state)
    @dprintln(3, "getCorrelation for Array{DimensionSelector,1} = ", are)
    return getOrAddRangeCorrelation(array, are, state)
end

function getCorrelation(inputInfo :: InputInfo, state :: expr_state)
    @dprintln(3, "getCorrelation for inputInfo = ", inputInfo)
    num_dim_inputs = findSelectedDimensions([inputInfo], state)
    @dprintln(3, "num_dim_inputs = ", num_dim_inputs)
    if num_dim_inputs == 0 return nothing end
    if isRange(inputInfo)
        assert(length(inputInfo.indexed_dims) == length(inputInfo.range))
        canonical_range = DimensionSelector[
            if inputInfo.indexed_dims[i]
                inputInfo.range[i]
            else
                SingularSelector(0,SlotNumber(0))
            end
            for i = 1:length(inputInfo.indexed_dims)]
        return getCorrelation(inputInfo.array, canonical_range, state)
        #return getCorrelation(inputInfo.array, inputInfo.range[inputInfo.indexed_dims], state)
        #return getCorrelation(inputInfo.array, inputInfo.range[1:num_dim_inputs], state)
    else
        return getCorrelation(inputInfo.array, state)
    end
end

"""
Creates a mapping between variables on the left-hand side of an assignment where the right-hand side is a parfor
and the arrays or scalars in that parfor that get assigned to the corresponding parts of the left-hand side.
Returns a tuple where the first element is a map for arrays between left-hand side and parfor and the second
element is a map for reduction scalars between left-hand side and parfor.
is_multi is true if the assignment is a fusion assignment.
parfor_assignment is the AST of the whole expression.
the_parfor is the PIRParForAst type part of the incoming assignment.
sym_to_type is an out parameter that maps symbols in the output mapping to their types.
"""
function createMapLhsToParfor(parfor_assignment, the_parfor, is_multi :: Bool, sym_to_type :: Dict{LHSVar, DataType}, state :: expr_state)
    map_lhs_post_array     = Dict{LHSVar, LHSVar}()
    map_lhs_post_reduction = Dict{LHSVar, LHSVar}()

    if is_multi
        last_post = the_parfor.postParFor[end]
        assert(isa(last_post, Array))
        @dprintln(3,"multi postParFor = ", the_parfor.postParFor, " last_post = ", last_post)

        # In our special AST node format for assignment to make fusion easier, args[3] is a FusionSentinel node
        # and additional args elements are the real symbol to be assigned to in the left-hand side.
        for i = 4:length(parfor_assignment.args)
            corresponding_elem = last_post[i-3]

            assert(isa(parfor_assignment.args[i], RHSVar))
            rememberTypeForSym(sym_to_type, toLHSVar(parfor_assignment.args[i]), CompilerTools.LambdaHandling.getType(parfor_assignment.args[i], state.LambdaVarInfo))
            rememberTypeForSym(sym_to_type, toLHSVar(corresponding_elem), CompilerTools.LambdaHandling.getType(corresponding_elem, state.LambdaVarInfo))
            if isArrayType(CompilerTools.LambdaHandling.getType(parfor_assignment.args[i], state.LambdaVarInfo))
                # For fused parfors, the last post statement is a tuple variable.
                # That tuple variable is declared in the previous statement (end-1).
                # The statement is an Expr with head == :call and top(:tuple) as the first arg.
                # So, the first member of the tuple is at offset 2 which corresponds to index 4 of this loop, ergo the "i-2".
                map_lhs_post_array[toLHSVar(parfor_assignment.args[i])]     = toLHSVar(corresponding_elem)
            else
                map_lhs_post_reduction[toLHSVar(parfor_assignment.args[i])] = toLHSVar(corresponding_elem)
            end
        end
    else
        # There is no mapping if this isn't actually an assignment statement but really a bare parfor.
        if !isBareParfor(parfor_assignment)
            lhs_pa = getLhsFromAssignment(parfor_assignment)
            ast_lhs_pa_typ = typeof(lhs_pa)
            lhs_pa_typ = CompilerTools.LambdaHandling.getType(lhs_pa, state.LambdaVarInfo)
            if isa(lhs_pa, RHSVar)
                ppftyp = typeof(the_parfor.postParFor[end])
                assert(ppftyp <: RHSVar)
                rememberTypeForSym(sym_to_type, toLHSVar(lhs_pa), lhs_pa_typ)
                rhs = the_parfor.postParFor[end]
                rememberTypeForSym(sym_to_type, toLHSVar(rhs), CompilerTools.LambdaHandling.getType(rhs, state.LambdaVarInfo))

                if isArrayType(lhs_pa_typ)
                    map_lhs_post_array[toLHSVar(lhs_pa)]     = toLHSVar(the_parfor.postParFor[end])
                else
                    map_lhs_post_reduction[toLHSVar(lhs_pa)] = toLHSVar(the_parfor.postParFor[end])
                end
            elseif typeof(lhs_pa) == Symbol
                throw(string("lhs_pa as a symbol no longer supported"))
            else
                @dprintln(3,"typeof(lhs_pa) = ", typeof(lhs_pa))
                assert(false)
            end
        end
    end

    map_lhs_post_array, map_lhs_post_reduction
end

"""
Given an "input" Symbol, use that Symbol as key to a dictionary.  While such a Symbol is present
in the dictionary replace it with the corresponding value from the dict.
"""
function fullyLowerAlias(dict :: Dict{LHSVar, LHSVar}, input :: LHSVar)
    while haskey(dict, input)
        input = dict[input]
    end
    input
end

"""
Take a single-step alias map, e.g., a=>b, b=>c, and create a lowered dictionary, a=>c, b=>c, that
maps each array to the transitively lowered array.
"""
function createLoweredAliasMap(dict1)
    ret = Dict{LHSVar, LHSVar}()

    for i in dict1
        ret[i[1]] = fullyLowerAlias(dict1, i[2])
    end

    ret
end

run_as_tasks = 0
"""
Debugging feature to specify the number of tasks to create and to stop thereafter.
"""
function PIRRunAsTasks(x)
    global run_as_tasks = x
end

"""
Returns a single element of an array if there is only one or the array otherwise.
"""
function oneIfOnly(x)
    if isa(x,Array) && length(x) == 1
        return x[1]
    else
        return x
    end
end


#"""
#Store information about a section of a body that will be translated into a task.
#"""
#type TaskGraphSection
#  start_body_index :: Int
#  end_body_index   :: Int
#  exprs            :: Array{Any,1}
#end

"""
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

"""
Process an array of expressions that aren't from a :body Expr.
"""
function intermediate_from_exprs(ast::Array{Any,1}, depth, state)
    # sequence of expressions
    # ast = [ expr, ... ]
    len  = length(ast)
    res = Any[]

    # For each expression in the array, process that expression recursively.
    for i = 1:len
        @dprintln(2,"Processing ast #",i," depth=",depth)

        # Convert the current expression.
        new_exprs = from_expr(ast[i], depth, state, false)
        assert(isa(new_exprs,Array))

        append!(res, new_exprs)  # Take the result of the recursive processing and add it to the result.
    end

    return res
end

include("parallel-ir-task.jl")
include("parallel-ir-top-exprs.jl")
include("parallel-ir-flatten.jl")


"""
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

"""
Pretty print a :lambda Expr in "node" at a given debug level in "dlvl".
"""
function printLambda(dlvl, linfo, body)
    dprintln(dlvl, "Lambda:")
    dprintln(dlvl, linfo)
    dprintln(dlvl, "typeof(body): ", body.typ)
    printBody(dlvl, body.args)
    if body.typ == Any
        @dprintln(1,"Body type is Any.")
    end
end

function pir_rws_cb(x :: DelayedFunc, cbdata :: ANY)
    @dprintln(4,"pir_rws_cb for DelayedFunc")

    expr_to_process = Any[]
    for i = 1:length(x.args)
        y = x.args[i]
        if isa(y, Array)
            for j=1:length(y)
                push!(expr_to_process, y[j])
            end
        else
            push!(expr_to_process, x.args[i])
        end
    end

    return expr_to_process
end

function pir_rws_cb(ast :: Expr, cbdata :: ANY)
    @dprintln(4,"pir_rws_cb for Expr")

    head = ast.head
    args = ast.args
    expr_to_process = Any[]

    if head == :parfor
        @dprintln(4,"pir_rws_cb for :parfor")
        @dprintln(4,"ast = ", ast)

        assert(typeof(args[1]) == ParallelAccelerator.ParallelIR.PIRParForAst)
        this_parfor = args[1]

        append!(expr_to_process, this_parfor.preParFor)
        append!(expr_to_process, this_parfor.hoisted)
        for i = 1:length(this_parfor.loopNests)
            # force the indexVariable to be treated as an rvalue
            push!(expr_to_process, mk_untyped_assignment(this_parfor.loopNests[i].indexVariable, 1))
            push!(expr_to_process, this_parfor.loopNests[i].lower)
            push!(expr_to_process, this_parfor.loopNests[i].upper)
            push!(expr_to_process, this_parfor.loopNests[i].step)
        end
        assert(typeof(cbdata) == CompilerTools.LambdaHandling.LambdaVarInfo)
        body = CompilerTools.LambdaHandling.getBody(this_parfor.body, CompilerTools.LambdaHandling.getReturnType(cbdata))
        body_rws = CompilerTools.ReadWriteSet.from_expr(body, pir_rws_cb, cbdata, cbdata)
        push!(expr_to_process, body_rws)
        append!(expr_to_process, this_parfor.postParFor)
        return expr_to_process
    elseif head == :call || head == :call1
        cfun  = getCallFunction(ast)
        cargs = getCallArguments(ast)

        if cfun == GlobalRef(ParallelAccelerator.API, :getindex)
            @dprintln(3,"pir_rws_cb for :getindex call")
            @dprintln(3,"ast = ", ast)

            tcopy = deepcopy(ast)
            tcopy.args[1] = GlobalRef(Base, :arrayref)
            push!(expr_to_process, tcopy)
            return expr_to_process
        elseif cfun == GlobalRef(ParallelAccelerator.API, :setindex)
            @dprintln(3,"pir_rws_cb for :setindex call")
            @dprintln(3,"ast = ", ast)

            tcopy = deepcopy(ast)
            tcopy.args[1] = GlobalRef(Base, :arrayset)
            push!(expr_to_process, tcopy)
            return expr_to_process
        elseif cfun == GlobalRef(ParallelAccelerator.API, :SubArrayLastDimRead)
            @dprintln(3,"pir_rws_cb for :SubArrayLastDimRead call")
            @dprintln(3,"ast = ", ast)
            tcopy = deepcopy(ast)
            tcopy.args[1] = GlobalRef(Base, :arrayref)
            tcopy.args[3] = Colon()
            push!(tcopy.args, deepcopy(cargs[2]))
            push!(expr_to_process, tcopy)
            return expr_to_process
        elseif cfun == GlobalRef(ParallelAccelerator.API, :SubArrayLastDimWrite)
            @dprintln(3,"pir_rws_cb for :SubArrayLastDimWrite call")
            @dprintln(3,"ast = ", ast)
            tcopy = deepcopy(ast)
            tcopy.args[1] = GlobalRef(Base, :arrayset)
            tcopy.args[3] = 1
            push!(tcopy.args, Colon())
            push!(tcopy.args, deepcopy(cargs[2]))
            push!(expr_to_process, tcopy)
            return expr_to_process
        end
    end

    # Aside from parfor nodes, the ReadWriteSet callback is the same as the liveness callback.
    return pir_live_cb(ast, cbdata)
end

function pir_rws_cb(ast :: ANY, cbdata :: ANY)
    @dprintln(4,"pir_live_cb")

    # Aside from parfor nodes, the ReadWriteSet callback is the same as the liveness callback.
    return pir_live_cb(ast, cbdata)
end

"""
A LivenessAnalysis callback that handles ParallelIR introduced AST node types.
For each ParallelIR specific node type, form an array of expressions that liveness
can analysis to reflect the read/write set of the given AST node.
If we read a symbol it is sufficient to just return that symbol as one of the expressions.
If we write a symbol, then form a fake mk_assignment_expr just to get liveness to realize the symbol is written.
"""
function pir_live_cb(ast :: Expr, cbdata :: ANY)
    @dprintln(4,"pir_live_cb")

    head = ast.head
    args = ast.args
    if head == :parfor
        @dprintln(4,"pir_live_cb for :parfor")
        expr_to_process = Any[]

        assert(typeof(args[1]) == ParallelAccelerator.ParallelIR.PIRParForAst)
        this_parfor = args[1]

        append!(expr_to_process, this_parfor.preParFor)
        append!(expr_to_process, this_parfor.hoisted)
        for i = 1:length(this_parfor.loopNests)
            # force the indexVariable to be treated as an rvalue
            push!(expr_to_process, mk_untyped_assignment(this_parfor.loopNests[i].indexVariable, 1))
            push!(expr_to_process, this_parfor.loopNests[i].lower)
            push!(expr_to_process, this_parfor.loopNests[i].upper)
            push!(expr_to_process, this_parfor.loopNests[i].step)
        end
        #emptyLambdaVarInfo = CompilerTools.LambdaHandling.LambdaVarInfo()
        #fake_body = CompilerTools.LambdaHandling.LambdaVarInfoToLambda(emptyLambdaVarInfo, TypedExpr(nothing, :body, this_parfor.body...))
        @dprintln(3,"typeof(cbdata) = ", typeof(cbdata))
        assert(typeof(cbdata) == CompilerTools.LambdaHandling.LambdaVarInfo)
        body_lives = CompilerTools.LivenessAnalysis.from_lambda(cbdata, this_parfor.body, pir_live_cb, cbdata)
        @dprintln(3, "body_lives = ", body_lives)
        # This is some old code that looks wrong and should be removed once confirmed that the "all_uses" approach is better.
        live_in_to_start_block = body_lives.basic_blocks[body_lives.cfg.basic_blocks[-1]].live_in
        all_defs = Set()
        all_uses = Set()
        for bb in body_lives.basic_blocks
            all_defs = union(all_defs, bb[2].def)
            all_uses = union(all_uses, bb[2].use)
        end
        # as = CompilerTools.LivenessAnalysis.AccessSummary(setdiff(all_defs, live_in_to_start_block), live_in_to_start_block)
        # FIXME: is this correct?
        as = CompilerTools.LivenessAnalysis.AccessSummary(all_defs, intersect(all_uses, live_in_to_start_block))
        #as = CompilerTools.LivenessAnalysis.AccessSummary(all_defs, live_in_to_start_block)
        @dprintln(3, "as = ", as)
        push!(expr_to_process, as)

        append!(expr_to_process, this_parfor.postParFor)

        return expr_to_process
    elseif head == :parfor_start
        @dprintln(4,"pir_live_cb for :parfor_start")
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
    # task mode commented out
    #=
    elseif head == :insert_divisible_task
        # Is this right?  Do I need pir_range stuff here too?
        @dprintln(4,"pir_live_cb for :insert_divisible_task")
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
    =#
    elseif head == :loophead
        @dprintln(4,"pir_live_cb for :loophead")
        assert(length(args) == 3)

        expr_to_process = Any[]
        assert(typeof(cbdata) == CompilerTools.LambdaHandling.LambdaVarInfo)
        push!(expr_to_process, mk_untyped_assignment(toRHSVar(args[1], Int64, cbdata), 1))  # force args[1] to be seen as an rvalue
        push!(expr_to_process, args[2])
        push!(expr_to_process, args[3])

        return expr_to_process
    elseif head == :loopend
        # There is nothing really interesting in the loopend node to signify something being read or written.
        assert(length(args) == 1)
        return Any[]
    elseif head == :call
        if isBaseFunc(args[1], :unsafe_arrayref) || isBaseFunc(args[1], :safe_arrayref)
            expr_to_process = Any[]
            new_expr = deepcopy(ast)
            new_expr.args[1] = GlobalRef(Base, :arrayref)
            push!(expr_to_process, new_expr)
            return expr_to_process
        elseif isBaseFunc(args[1], :unsafe_arrayset)
            expr_to_process = Any[]
            new_expr = deepcopy(ast)
            new_expr.args[1] = GlobalRef(Base, :arrayset)
            push!(expr_to_process, new_expr)
            return expr_to_process
        end
    elseif head == :(=)
        @dprintln(4,"pir_live_cb for :(=)")
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

    return DomainIR.dir_live_cb(ast, cbdata)
end

function pir_live_cb(ast :: ANY, cbdata :: ANY)
    @dprintln(4,"pir_live_cb")
    return DomainIR.dir_live_cb(ast, cbdata)
end

function isSideEffectFreeAPI(node :: GlobalRef)
  if node.mod == ParallelAccelerator.API
    if in(node.name, ParallelAccelerator.API.reduce_operators) ||
       in(node.name, ParallelAccelerator.API.unary_map_operators)
      return true
    end
  end
  return false
end

function isSideEffectFreeAPI(node :: ANY)
  return false
end

"""
Sometimes statements we exist in the AST of the form a=Expr where a is a Symbol that isn't live past the assignment
and we'd like to eliminate the whole assignment statement but we have to know that the right-hand side has no
side effects before we can do that.  This function says whether the right-hand side passed into it has side effects
or not.  Several common function calls that otherwise we wouldn't know are safe are explicitly checked for.
"""
function hasNoSideEffects(node :: Union{Symbol, LHSVar, RHSVar, GlobalRef, DataType})
    return true
end

if VERSION >= v"0.6.0-pre"
import CompilerTools.LambdaHandling.LambdaInfo
end

function hasNoSideEffects(node :: Union{QuoteNode, LambdaInfo, Number, Function})
    return true
end

function hasNoSideEffects(node :: Any)
    return false
end

function hasNoSideEffects(node :: Expr)
    if node.head == :select || node.head == :ranges || node.head == :range || node.head == :tomask
        return all(Bool[hasNoSideEffects(a) for a in node.args])
    elseif node.head == :(=)
        return true
    elseif node.head == :alloc
        return true
    elseif node.head == :lambda
        return true
    elseif node.head == :new
        newtyp::Any = node.args[1]
        if isa(newtyp, GlobalRef) && isdefined(newtyp.mod, newtyp.name)
            newtyp = getfield(newtyp.mod, newtyp.name)
        end
        return isa(newtyp, Type) && (newtyp <: Range || newtyp <: Function)
    elseif node.head == :foreigncall
        func = CompilerTools.Helper.getCallFunction(node)
        args = CompilerTools.Helper.getCallArguments(node)
        if func == QuoteNode(:jl_alloc_array_1d) ||
           func == QuoteNode(:jl_alloc_array_2d) ||
           func == QuoteNode(:jl_alloc_array_3d)
            @dprintln(3,"hasNoSideEffects found allocation returning true")
            return true
        end
    elseif node.head == :call1 || node.head == :call || node.head == :invoke
        func = CompilerTools.Helper.getCallFunction(node)
        args = CompilerTools.Helper.getCallArguments(node)
        @dprintln(3,"hasNoSideEffects func=", func, " ", typeof(func))
        if isBaseFunc(func, :box) ||
            isBaseFunc(func, :tuple) ||
            isBaseFunc(func, :getindex_bool_1d) ||
            isBaseFunc(func, :arrayref) ||
            isBaseFunc(func, :arraylen) ||
            isBaseFunc(func, :arraysize) ||
            isBaseFunc(func, :getfield) ||
            isBaseFunc(func, :getindex) ||
            isBaseFunc(func, :not_int) ||
            isBaseFunc(func, :sub_int) ||
            isBaseFunc(func, :add_int) ||
            isBaseFunc(func, :mul_int) ||
            isBaseFunc(func, :neg_int) ||
            isBaseFunc(func, :xor_int) ||
            isBaseFunc(func, :flipsign_int) ||
            isBaseFunc(func, :checked_sadd) ||
            isBaseFunc(func, :checked_sadd_int) ||
            isBaseFunc(func, :checked_ssub) ||
            isBaseFunc(func, :checked_ssub_int) ||
            isBaseFunc(func, :checked_smul) ||
            isBaseFunc(func, :checked_smul_int) ||
            isBaseFunc(func, :checked_trunc_sint) ||
            isBaseFunc(func, :neg_float) ||
            isBaseFunc(func, :sub_float) ||
            isBaseFunc(func, :add_float) ||
            isBaseFunc(func, :mul_float) ||
            isBaseFunc(func, :div_float) ||
            isBaseFunc(func, :sitofp) ||
            isBaseFunc(func, :sle_int) ||
            isBaseFunc(func, :ule_int) ||
            isBaseFunc(func, :slt_int) ||
            isBaseFunc(func, :ult_int) ||
            isBaseFunc(func, :(===)) ||
            isBaseFunc(func, :<:) ||
            isBaseFunc(func, :apply_type) ||
            isBaseFunc(func, :nfields) ||
            isBaseFunc(func, :_apply) || # Core._apply is used for tallking to codegen, e.g. applyting promote_typeof
            isBaseFunc(func, :promote_type) ||
            isBaseFunc(func, :select_value) ||
            isBaseFunc(func, :powi_llvm) ||
            isBaseFunc(func, :svec) ||
            isSideEffectFreeAPI(func)
            @dprintln(3,"hasNoSideEffects returning true")
            return all(Bool[hasNoSideEffects(a) for a in args])
        elseif isBaseFunc(func, :ccall)
            @dprintln(3,"hasNoSideEffects found ccall")
            func = args[1]
            if func == QuoteNode(:jl_alloc_array_1d) ||
               func == QuoteNode(:jl_alloc_array_2d) ||
               func == QuoteNode(:jl_alloc_array_3d)
                @dprintln(3,"hasNoSideEffects found allocation returning true")
                return true
            end
        elseif isa(func, TopNode)
            @dprintln(3, "Found TopNode in hasNoSideEffects. type = ", typeof(func.name))
            if func.name == :getfield
                @dprintln(3,"hasNoSideEffects returning true")
                return true
            end
        end
    end

    @dprintln(3,"hasNoSideEffects returning false")
    return false
end

type SideEffectWalkState
    hasSideEffect

    function SideEffectWalkState()
        new(false)
    end
end

function hasNoSideEffectWalk(node :: ANY, data :: SideEffectWalkState, top_level_number, is_top_level, read)
    if !hasNoSideEffects(node)
        @dprintln(3,"hasNoSideEffectWalk found side effect for node ", node)
        data.hasSideEffect = true
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function from_assignment_fusion(args::Array{Any,1}, depth, state)
    lhs = args[1]
    rhs = args[2]
    @dprintln(3,"from_assignment lhs = ", lhs)
    @dprintln(3,"from_assignment rhs = ", rhs)
    if isa(rhs, Expr) && rhs.head == :lambda
        # skip handling rhs lambdas
        rhs = [rhs]
    else
        rhs = from_expr(rhs, depth, state, false)
    end
    @dprintln(3,"from_assignment rhs after = ", rhs)
    assert(isa(rhs,Array))
    assert(length(rhs) == 1)
    rhs = rhs[1]

    # Eliminate assignments to variables which are immediately dead.
    # The variable name.
    lhsName = toLHSVar(lhs)
    # Get liveness information for the current statement.
    statement_live_info = CompilerTools.LivenessAnalysis.find_top_number(state.top_level_number, state.block_lives)
    @assert statement_live_info!=nothing "$(state.top_level_number) $(state.block_lives)"

    @dprintln(3,statement_live_info)
    @dprintln(3,"def = ", statement_live_info.def)

    # Make sure this variable is listed as a "def" for this statement.
    assert(CompilerTools.LivenessAnalysis.isDef(lhsName, statement_live_info))

    # If the lhs symbol is not in the live out information for this statement then it is dead.
    if !in(lhsName, statement_live_info.live_out) && hasNoSideEffects(rhs)
        @dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
        # Eliminate the statement.
        return [], nothing
    end

    @assert typeof(rhs)==Expr && rhs.head==:parfor "Expected :parfor assignment"
    out_typ = rhs.typ
    #@dprintln(3, "from_assignment rhs is Expr, type = ", out_typ, " rhs.head = ", rhs.head, " rhs = ", rhs)
    # If we have "a = parfor(...)" then record that array "a" has the same length as the output array of the parfor.
    the_parfor = rhs.args[1]
    for i = 4:length(args)
        rhs_entry = the_parfor.postParFor[end][i-3]
        if isArrayType(getType(rhs_entry, state.LambdaVarInfo))
            add_merge_correlations(toLHSVar(rhs_entry), toLHSVar(args[i]), state)
        end
    end

    return [toRHSVar(lhs, out_typ, state.LambdaVarInfo); rhs], out_typ
end

"""
Process an assignment expression.
Starts by recurisvely processing the right-hand side of the assignment.
Eliminates the assignment of a=b if a is dead afterwards and b has no side effects.
    Does some array equivalence class work which may be redundant given that we now run a separate equivalence class pass so consider removing that part of this code.
"""
function from_assignment(lhs, rhs, depth, state)
    # :(=) assignment
    # ast = [ ... ]
    @dprintln(3,"from_assignment lhs = ", lhs)
    @dprintln(3,"from_assignment rhs = ", rhs)
    if isa(rhs, Expr) && rhs.head == :lambda
        # skip handling rhs lambdas
        rhs = [rhs]
    else
        rhs = from_expr(rhs, depth, state, false)
    end
    @dprintln(3,"from_assignment rhs after = ", rhs)
    assert(isa(rhs,Array))
    assert(length(rhs) == 1)
    rhs = rhs[1]

    # Eliminate assignments to variables which are immediately dead.
    # The variable name.
    lhsName = toLHSVar(lhs)
    # Get liveness information for the current statement.
    statement_live_info = CompilerTools.LivenessAnalysis.find_top_number(state.top_level_number, state.block_lives)
    @assert statement_live_info!=nothing "$(state.top_level_number) $(state.block_lives)"

    @dprintln(3,statement_live_info)
    @dprintln(3,"def = ", statement_live_info.def)

    # Make sure this variable is listed as a "def" for this statement.
    assert(CompilerTools.LivenessAnalysis.isDef(lhsName, statement_live_info))

    # If the lhs symbol is not in the live out information for this statement then it is dead.
    if !in(lhsName, statement_live_info.live_out) && hasNoSideEffects(rhs)
        @dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
        # Eliminate the statement.
        return [], nothing
    end

    if isa(rhs, Expr)
        out_typ = rhs.typ
        #@dprintln(3, "from_assignment rhs is Expr, type = ", out_typ, " rhs.head = ", rhs.head, " rhs = ", rhs)

        if rhs.head == :call || rhs.head == :invoke
            @dprintln(3, "Detected call rhs in from_assignment.")
            @dprintln(3, "from_assignment call, arg1 = ", rhs.args[1])
            if length(rhs.args) > 1
                @dprintln(3, " arg2 = ", rhs.args[2])
            end
            fun = CompilerTools.Helper.getCallFunction(rhs)
            args = CompilerTools.Helper.getCallArguments(rhs)
            if isBaseFunc(fun, :tuple)
                if haskey(state.tuple_assigns, lhsName)
                    @dprintln(3, "tuple assignment already remembered for ", lhsName)
                else
                    @dprintln(3, "Remembering tuple assignment for ", lhsName)
                    state.tuple_assigns[lhsName] = args
                end
            end
        end
    elseif isa(rhs, TypedVar)
        out_typ = getType(rhs, state.LambdaVarInfo)
        if false
        if isArrayType(out_typ)
            # Add a length correlation of the form "a = b".
            @dprintln(3,"Adding array length correlation ", lhs, " to ", rhs)
            add_merge_correlations(toLHSVar(rhs), lhsName, state)
        end
        end
    else
        # Get the type of the lhs from its metadata declaration.
        out_typ = CompilerTools.LambdaHandling.getType(lhs, state.LambdaVarInfo)
    end

    return [lhs; rhs], out_typ
end

"""
Process a call AST node. Note that it takes an Expr as input because it can be either :call or :invoke.
"""
function from_call(head, ast::Expr, depth, state)
    fun  = getCallFunction(ast)
    args = getCallArguments(ast)
    @dprintln(2,"from_call fun = ", fun, " typeof fun = ", typeof(fun))
    if length(args) > 0
        @dprintln(2,"first arg = ",args[1], " type = ", typeof(args[1]))
    end

    if isBaseFunc(fun, :broadcast!) && args[1] == GlobalRef(Base, :identity)
        @dprintln(3,"Detected call to broadcast! with identity argument.")
        if length(args) == 3
            arr1 = args[2]
            arr2 = args[3]
            arrtyp1 = CompilerTools.LambdaHandling.getType(arr1, state.LambdaVarInfo)
            arrtyp2 = CompilerTools.LambdaHandling.getType(arr2, state.LambdaVarInfo)
            eltyp1 = eltype(arrtyp1)
            eltyp2 = eltype(arrtyp2)
            @dprintln(3,"arr1 = ", arr1, " arr2 = ", arr2, " hasCorrelation1 = ", haskey(state.array_length_correlation, arr1), " hasCorrelation2 = ", haskey(state.array_length_correlation, arr2))
            if haskey(state.array_length_correlation, arr1) && 
               haskey(state.array_length_correlation, arr2) &&
               state.array_length_correlation[arr1] == state.array_length_correlation[arr2]
                @dprintln(3,"Arrays are equivalent in length.  Switch to copy! here.")
                new_domain_expr = ParallelAccelerator.DomainIR.mk_mmap!(args[2:3], ParallelAccelerator.DomainIR.DomainLambda(Type[eltyp1,eltyp2], Type[eltyp1], params->Any[Expr(:tuple, params[2])], state.LambdaVarInfo))
                @dprintln(3,"New mmap! = ", new_domain_expr)

                head = :parfor
                domain_oprs = [DomainOperation(:mmap!, args)]
                args = mk_parfor_args_from_mmap!(new_domain_expr.args[1], new_domain_expr.args[2], false, domain_oprs, state)
                return head, args
            end
        end
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

    return head, (ast.head == :invoke ? [ast.args[1]; fun; args] : [fun; args])
end

"""
Process a foreigncall AST node.
"""
function from_foreigncall(ast::Expr, depth, state)
    fun  = getCallFunction(ast)
    args = getCallArguments(ast)
    @dprintln(2,"from_foreigncall fun = ", fun, " typeof fun = ", typeof(fun))
    if length(args) > 0
        @dprintln(2,"first arg = ",args[1], " type = ", typeof(args[1]))
    end
    # Recursively process the arguments to the call.
    args = from_exprs(args, depth+1, state)

    return [fun; args]
end

"""
Apply a function "f" that takes the :body from the :lambda and returns a new :body that is stored back into the :lambda.
"""
function processAndUpdateBody(body, f :: Function, state)
    body.args = f(body.args, state)
end

include("parallel-ir-simplify.jl")
include("parallel-ir-fusion.jl")


mmap_to_mmap! = 1
"""
If set to non-zero, perform the phase where non-inplace maps are converted to inplace maps to reduce allocations.
"""
function PIRInplace(x)
    global mmap_to_mmap! = x
end

hoist_allocation = 1
"""
If set to non-zero, perform the rearrangement phase that tries to moves alllocations outside of loops.
"""
function PIRHoistAllocation(x)
    global hoist_allocation = x
end

bb_reorder = 1
"""
If set to non-zero, perform the bubble-sort like reordering phase to coalesce more parfor nodes together for fusion.
"""
function PIRBbReorder(x)
    global bb_reorder = x
end

shortcut_array_assignment = 0
"""
Enables an experimental mode where if there is a statement a = b and they are arrays and b is not live-out then
use a special assignment node like a move assignment in C++.
"""
function PIRShortcutArrayAssignment(x)
    global shortcut_array_assignment = x
end

"""
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

"""
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

"""
Returns true if the given "ast" node is a DomainIR operation.
"""

function isDomainNode(ast :: Expr)
    head = ast.head
    args = ast.args

    if head == :mmap || head == :mmap! || head == :reduce || head == :stencil! || head == :parallel_for
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


"""
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

"""
Debug print the parts of a DomainLambda.
"""
function pirPrintDl(dbg_level, dl)
    dprintln(dbg_level, "inputs = ", dl.inputs)
    dprintln(dbg_level, "output = ", dl.outputs)
    dprintln(dbg_level, "linfo  = ", dl.linfo)
end

"""
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

"""
A nested lambda may contain labels that conflict with labels in the top-level statements of the function being processed.
We take the maxLabel or those top-level statements and re-number labels in the nested lambda and update maxLabel.
"""
function integrateLabels(body, maxLabel)
  max_label = CompilerTools.OptFramework.updateLabels!(body.args, maxLabel)
  return (body, maxLabel)
end

"""
A routine similar to the main parallel IR entry put but designed to process DomainLambda.
"""
function nested_function_exprs(domain_lambda, out_state)
    unique_node_id = get_unique_num()

    @dprintln(2,"domain_lambda = ", domain_lambda, " " , unique_node_id)
    LambdaVarInfo = domain_lambda.linfo
    @dprintln(2,"LambdaVarInfo = ", LambdaVarInfo, " " , unique_node_id)
    body = domain_lambda.body
    @dprintln(1,"(Starting nested_function_exprs. body = ", body, " " , unique_node_id)

    start_time = time_ns()

    @dprintln(2,"nested_function_exprs out_state.max_label = ", out_state.max_label, " " , unique_node_id)
    (body, max_label) = integrateLabels(body, out_state.max_label)
    @dprintln(2,"nested_function_exprs max_label = ", max_label, " body = ", body, " " , unique_node_id)

    # Re-create the body minus any dead basic blocks.
    cfg = CompilerTools.CFGs.from_lambda(body; opt=false)
    @dprintln(3, "nested_function_exprs cfg = ", cfg)
    body = CompilerTools.LambdaHandling.getBody(CompilerTools.CFGs.createFunctionBody(cfg), CompilerTools.LambdaHandling.getReturnType(LambdaVarInfo))
    @dprintln(1,"AST after dead blocks removed, body = ", body, " " , unique_node_id)

    @dprintln(1,"Starting liveness analysis.", " " , unique_node_id)
    lives = computeLiveness(body, LambdaVarInfo)
    @dprintln(1,"Finished liveness analysis.", " " , unique_node_id)

    body = AstWalk(body, copy_propagate, CopyPropagateState(lives,LambdaVarInfo))
    @dprintln(1,"AST after copy propagation, body = ", body, " " , unique_node_id)
    lives = computeLiveness(body, LambdaVarInfo)

    if print_times
    @dprintln(1,"Liveness Analysis time = ", ns_to_sec(time_ns() - start_time), " " , unique_node_id)
    end

    mtm_start = time_ns()

    if mmap_to_mmap! != 0
        @dprintln(1, "starting mmap to mmap! transformation.", " " , unique_node_id)
        uniqSet = AliasAnalysis.from_lambda(LambdaVarInfo, body, lives, pir_alias_cb, nothing)
        @dprintln(3, "uniqSet = ", uniqSet, " " , unique_node_id)
        mmapToMmap!(LambdaVarInfo, body, lives, uniqSet)
        @dprintln(1, "Finished mmap to mmap! transformation.", " " , unique_node_id)
        @dprintln(3, "body = ", body, " " , unique_node_id)
    end

    if print_times
    @dprintln(1,"mmap_to_mmap! time = ", ns_to_sec(time_ns() - mtm_start), " " , unique_node_id)
    end

    # We pass only the non-array params to the rearrangement code because if we pass array params then
    # the code will detect statements that depend only on array params and move them to the top which
    # leaves other non-array operations after that and so prevents fusion.
    input_arrays = getArrayParams(LambdaVarInfo)
    non_array_params = Set{LHSVar}()
    non_array_escaping = Set{LHSVar}()
    for param in CompilerTools.LambdaHandling.getInputParameters(LambdaVarInfo)
        if !in(param, input_arrays) && CompilerTools.LivenessAnalysis.countSymbolDefs(param, lives) == 0
            push!(non_array_params, lookupLHSVarByName(param, LambdaVarInfo))
        end
    end
    for param in CompilerTools.LambdaHandling.getEscapingVariables(LambdaVarInfo)
        if !in(param, input_arrays) && CompilerTools.LivenessAnalysis.countSymbolDefs(param, lives) == 0
            push!(non_array_escaping, lookupLHSVarByName(param, LambdaVarInfo))
        end
    end
    @dprintln(3,"Non-array params = ", non_array_params, " " , unique_node_id)
    @dprintln(3,"Non-array escaping = ", non_array_escaping, " " , unique_node_id)

    # Find out max_label.
    assert(isa(body, Expr) && (body.head === :body))
    max_label = getMaxLabel(max_label, body.args)

    eq_start = time_ns()

    new_vars = expr_state("nested_function", lives, max_label, input_arrays)
    new_vars.in_nested = true
    # import correlations of escaping variables to enable optimizations
    # TODO: fix imported GenSym symbols
    setEscCorrelations!(new_vars, LambdaVarInfo, out_state, length(input_arrays))
    # meta may have changed, need to update ast
    @dprintln(3,"Creating nested equivalence classes. Imported correlations:", " " , unique_node_id)
    print_correlations(3, new_vars)
    genEquivalenceClasses(LambdaVarInfo, body, new_vars)
    @dprintln(3,"Done creating nested equivalence classes.", " " , unique_node_id)
    print_correlations(3, new_vars)
    if print_times
    @dprintln(1,"Creating nested equivalence classes time = ", ns_to_sec(time_ns() - eq_start), " " , unique_node_id)
    end

    rep_start = time_ns()

    changed = true
    while changed
        @dprintln(1,"Removing statement with no dependencies from the AST with parameters"), " " , unique_node_id
#        rnd_state = RemoveNoDepsState(lives, non_array_params)
        rnd_state = RemoveNoDepsState(lives, union(non_array_params, non_array_escaping))
        body = AstWalk(body, remove_no_deps, rnd_state)
        @dprintln(3,"body after no dep stmts removed = ", body, " " , unique_node_id)

        @dprintln(3,"top_level_no_deps = ", rnd_state.top_level_no_deps, " " , unique_node_id)

        @dprintln(1,"Adding statements with no dependencies to the start of the AST.", " " , unique_node_id)
        body = CompilerTools.LambdaHandling.prependStatements(body, rnd_state.top_level_no_deps)
        @dprintln(3,"body after no dep stmts re-inserted = ", body, " " , unique_node_id)

        @dprintln(1,"Re-starting liveness analysis.", " " , unique_node_id)
        lives = computeLiveness(body, LambdaVarInfo)
        @dprintln(1,"Finished liveness analysis.", " " , unique_node_id)

        changed = rnd_state.change
    end

    if print_times
    @dprintln(1,"Rearranging passes time = ", ns_to_sec(time_ns() - rep_start), " " , unique_node_id)
    end

    processAndUpdateBody(body, removeNothingStmts, nothing)
    @dprintln(1,"Re-starting liveness analysis.", " " , unique_node_id)
    lives = computeLiveness(body, LambdaVarInfo)
    @dprintln(1,"Finished liveness analysis.", " " , unique_node_id)

    @dprintln(1,"Doing conversion to parallel IR.", " " , unique_node_id)
    @dprintln(3,"body = ", body, " " , unique_node_id)

    new_vars.block_lives = lives

    # Do the main work of Parallel IR.
    body = get_one(from_expr(LambdaVarInfo, body, 1, new_vars, false))

    @dprintln(3,"Final ParallelIR = ", body, " ) " , unique_node_id)

    #throw(string("STOPPING AFTER PARALLEL IR CONVERSION"))
    out_state.max_label = new_vars.max_label
    domain_lambda.body = body
    #return new_vars.block_lives
    return
end

"""
Import correlations of escaping variables from outer lambda
to enable further optimizations.
"""
function setEscCorrelations!(new_vars, linfo, out_state, input_length)
    # intersection of escaping variables and out lambda's correlation variables
    # esc_vars is Symbol names of escaping variables
    esc_vars::Vector{Symbol} = getEscapingVariables(linfo)
    @dprintln(3, "setEscCorrelations escaping variables = ", esc_vars)
    out_length_vars = collect(keys(out_state.array_length_correlation))
    # convert LHSVar to Symbols (if possible)
    out_arr_names = map(x->lookupVariableName(x, out_state.LambdaVarInfo), out_length_vars)
    @dprintln(3, "setEscCorrelations out lambda variables = ", out_length_vars," names: ", out_arr_names)
    intersect_var_names = intersect(esc_vars, out_arr_names)
    for arr_name in intersect_var_names
        out_lhs_var = lookupLHSVarByName(arr_name, out_state.LambdaVarInfo)
        in_lhs_var = lookupLHSVarByName(arr_name, linfo)
        corr_class = out_state.array_length_correlation[out_lhs_var]
        @dprintln(3, "setEscCorrelations correlation found ", out_lhs_var, " ", in_lhs_var, " ", corr_class)
        # greater than number of arrays to avoid conflict with inside correlations
        # FIXME: -1 means shouldn't make correlation just in out lambda?
        if corr_class==-1 continue end
        new_corr_class = corr_class+input_length+1
        new_vars.array_length_correlation[in_lhs_var] = new_corr_class
        for (s,c) in out_state.symbol_array_correlation
            # add symbol correlations only if all size variables are constants or escaping
            # TODO: import GenSym, resolve name conflicts with params and local variables
            if c==corr_class && mapreduce(x-> isValidInnerVariable(x,linfo,out_state.LambdaVarInfo), &, s)
                # convert outer LHSVars (slotnumbers) to inner ones
                new_syms = map(x->isa(x,Int) ? x : lookupVariableName(x, out_state.LambdaVarInfo), s)
                new_syms = map(x->isa(x,Int) ? x : lookupLHSVarByName(x, linfo), new_syms)
                new_vars.symbol_array_correlation[new_syms] = new_corr_class
                @dprintln(3, "setEscCorrelations symbol correlation found ", s, " -> ", new_syms, " class ", new_corr_class)
            end
        end
    end
    if length(new_vars.array_length_correlation)!=0
        new_vars.next_eq_class = maximum(values(new_vars.array_length_correlation))+1
    end
    nothing
end

# integers are always valid to import
isValidInnerVariable(x::Int, linfo, out_linfo) = true
isValidInnerVariable(x::TypedVar, linfo, out_linfo) = isValidInnerVariable(toLHSVar(x), linfo)
# GenSyms are not valid to import
isValidInnerVariable(x::GenSym, linfo, out_linfo) = false

function isValidInnerVariable(x::LHSRealVar, linfo, out_linfo)
    name = lookupVariableName(x, out_linfo)
    # TODO: import GenSym, resolve name conflicts with params and local variables
    if isInputParameter(name, linfo) || isLocalVariable(name, linfo)
        return false
    end
    if !isEscapingVariable(name, linfo)
        typ = LambdaHandling.getType(x, out_linfo)
        desc = LambdaHandling.getDesc(x, out_linfo)
        @dprintln(3, "setEscCorrelations addEscapingVariable for ", x, " ",name)
        addEscapingVariable(name, typ, desc, linfo)
    end
    return true
end

doRemoveAssertEqShape = true
generalSimplification = true

"""
Returns a set of RHSVar's that are arrays for which there are multiple statements that could define that
array and by implication change its size.
FIX ME!!!!!!!!!!!!!!!!!!!
TO-DO!!!!!!!!!!!!!!!!!!!!
This code doesn't work because it conflates changing the size of the array with changing its contents.
Instead of scanning liveness information, we need to do precisely one scan of the AST looking for array
creation points.  If there are multiple of such then we'll preclude that array from having a useful
equivalence class.
"""
function findMultipleArrayAssigns(lives :: CompilerTools.LivenessAnalysis.BlockLiveness, LambdaVarInfo)
    ret   = Set{RHSVar}()
    found = Set{RHSVar}()

    # For every basic block.
    for bb in lives.basic_blocks
      BB = bb[2]    # get the BasicBlock
      # Look at one statement at a time.
      for stmt in BB.statements
        # For each RHSVar that is defined in that statement.
        for d in stmt.def
          # Get that RHSVar's type.
          dtyp = CompilerTools.LambdaHandling.getType(d, LambdaVarInfo)
          # If it is an array.
          if isArrayType(dtyp)
            # See if we've previously seen another statement in which that RHSVar was defined.
            if in(d, found)
              # If so, then there are multiple statements that define this RHSVar and so add that to the return set.
              push!(ret, d)
            else
              # We hadn't previously seen this RHSVar defined in a statement so add it to "found".
              push!(found, d)
            end
          end
        end
      end
    end

    return ret
end

wellknown_all_unmodified = Set{Any}()
wellknown_only_first_modified = Set{Any}()

function __init__()
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(./)), force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(.*)), force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(.+)), force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(.-)), force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(/)),  force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(*)),  force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(+)),  force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(-)),  force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(<=)), force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(<)),  force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(>=)), force = true))
  push!(wellknown_all_unmodified, Base.resolve(GlobalRef(ParallelAccelerator.API,:(>)),  force = true))
end

function no_mod_impl(func :: GlobalRef, arg_type_tuple :: Array{DataType,1})
    @dprintln(3, "no_mod_impl func = ", func, " arg_type_tuple = ", arg_type_tuple)
    if in(func, wellknown_all_unmodified)
        @dprintln(3, "found in wellknown_all_unmodified")
        return ones(Int64, length(arg_type_tuple))
    end

    return nothing
end

function computeLiveness(body, linfo :: CompilerTools.LambdaHandling.LambdaVarInfo)
    @dprintln(2, "computeLiveness typeof(body) = ", typeof(body))
    return CompilerTools.LivenessAnalysis.from_lambda(linfo, body, pir_live_cb, linfo, no_mod_cb=no_mod_impl)
end

#=
type markMultState
    LambdaVarInfo
    assign_dict
end

function mark_multiple_assign_equiv(node :: Expr, state :: ParallelAccelerator.ParallelIR.markMultState, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    if node.head == :lambda
        # TODD: I'm not sure about this.  Should arrays in a nested lambda be considered multiply assigned?
        save_LambdaVarInfo  = state.LambdaVarInfo
        linfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(node)
        state.LambdaVarInfo = linfo
        body = CompilerTools.LambdaHandling.getBody(node)
        AstWalk(body, mark_multiple_assign_equiv, state)
        state.LambdaVarInfo = save_LambdaVarInfo
        return node
    end

    if is_top_level
        @dprintln(3,"mark_multiple_assign_equiv is_top_level")

        if isAssignmentNode(node)
            lhs = toLHSVar(node.args[1])
            if isArrayType(CompilerTools.LambdaHandling.getType(lhs, state.LambdaVarInfo))
                if !haskey(state.assign_dict, lhs)
                    state.assign_dict[lhs] = 0
                end
                state.assign_dict[lhs] = state.assign_dict[lhs] + 1
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function mark_multiple_assign_equiv(node :: ANY, state :: ParallelAccelerator.ParallelIR.markMultState, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end
=#
function genEquivalenceClasses(linfo, body, new_vars)
    new_vars.LambdaVarInfo = linfo

    # no need to empty them since equivalence class will be overwritten if it needs to be negative
    #empty!(new_vars.array_length_correlation)
    #empty!(new_vars.symbol_array_correlation)
    #empty!(new_vars.range_correlation)
    # multiple assignment detection moved to create_equivalence_class_assignment
    #=
    mms = markMultState(new_vars.LambdaVarInfo, Dict{LHSVar,Int64}())
    AstWalk(body, mark_multiple_assign_equiv, mms)
    @dprintln(3, "Result of mark_multiple_assign_equiv = ", mms.assign_dict)
    multi_correlation = -1
    for array_assign in mms.assign_dict
        if array_assign[2] > 1
            @dprintln(3, "Giving array ", array_assign[1], " a negative equivalence class so it will never merge.")
            new_vars.array_length_correlation[array_assign[1]] = multi_correlation
            multi_correlation -= 1
        end
    end
    =#
    AstWalk(body, create_equivalence_classes, new_vars)
    # Using equivalence class info, replace Base.arraysize() calls for constant size arrays
    # A separate pass is better since this doesn't have to worry about statements being top level
    # replace range correlations if possible since they can be 1:arraysize()
    ra_data = ReplaceConstArraysizesData(
      new_vars.block_lives, new_vars.LambdaVarInfo, new_vars.array_length_correlation,
        new_vars.symbol_array_correlation)
    replaceConstArraysizesRangeCorrelations(new_vars.range_correlation, ra_data)
    AstWalk(body, replaceConstArraysizes, ra_data)
end

"""
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
function from_root(function_name, ast)
    #assert(isfunctionhead(ast))
    @dprintln(1,"Starting main ParallelIR.from_expr.  function = ", function_name, " ast = ", ast, " typeof(ast) = ", typeof(ast))

    start_time = time_ns()

    # Create CFG from AST.  This will automatically filter out dead basic blocks.
    if isa(ast, Tuple)
        (LambdaVarInfo, body) = ast
    else
        LambdaVarInfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
    end
    @dprintln(2,"LambdaVarInfo = ", LambdaVarInfo)
    cfg = CompilerTools.CFGs.from_lambda(body)
    #body = CompilerTools.LambdaHandling.getBody(ast)
    # Re-create the body minus any dead basic blocks.
    body = CompilerTools.LambdaHandling.getBody(CompilerTools.CFGs.createFunctionBody(cfg), CompilerTools.LambdaHandling.getReturnType(LambdaVarInfo))
    @dprintln(1,"body after dead blocks removed function = ", function_name, " body = ", body)

    #CompilerTools.LivenessAnalysis.set_debug_level(3)

    @dprintln(1,"Starting liveness analysis. function = ", function_name)
    lives = computeLiveness(body, LambdaVarInfo)

    # propagate transpose() calls to gemm() calls
    # copy propagation is need so that the output of transpose is directly used in gemm()
    body = AstWalk(body, copy_propagate, CopyPropagateState(lives,LambdaVarInfo))
    @dprintln(1,"body after copy_propagate = ", function_name, " body = ", body)
    lives = computeLiveness(body, LambdaVarInfo)
    body  = AstWalk(body, transpose_propagate, TransposePropagateState(lives))
    lives = computeLiveness(body, LambdaVarInfo)

    #  udinfo = CompilerTools.UDChains.getUDChains(lives)
    @dprintln(3,"lives = ", lives)
    #  @dprintln(3,"udinfo = ", udinfo)
    @dprintln(1,"Finished liveness analysis. function = ", function_name)

    if print_times
    @dprintln(1,"Liveness Analysis time = ", ns_to_sec(time_ns() - start_time))
    end

    mtm_start = time_ns()

    if mmap_to_mmap! != 0
        @dprintln(1, "starting mmap to mmap! transformation.")
        uniqSet = AliasAnalysis.from_lambda(LambdaVarInfo, body, lives, pir_alias_cb, nothing)
        @dprintln(3, "uniqSet = ", uniqSet)
        mmapToMmap!(LambdaVarInfo, body, lives, uniqSet)
        @dprintln(1, "Finished mmap to mmap! transformation. function = ", function_name)
        printLambda(3, LambdaVarInfo, body)
    end

    if print_times
    @dprintln(1,"mmap_to_mmap! time = ", ns_to_sec(time_ns() - mtm_start))
    end

    # We pass only the non-array params to the rearrangement code because if we pass array params then
    # the code will detect statements that depend only on array params and move them to the top which
    # leaves other non-array operations after that and so prevents fusion.
    input_parameters = CompilerTools.LambdaHandling.getInputParameters(LambdaVarInfo)
    @dprintln(3,"All params = ", input_parameters)
    input_arrays = getArrayParams(LambdaVarInfo)
    @dprintln(3,"input_arrays = ", input_arrays, " type = ", typeof(input_arrays))
    non_array_params = Set{LHSVar}()
    for param in input_parameters
        if !in(param, input_arrays) && CompilerTools.LivenessAnalysis.countSymbolDefs(param, lives) == 0
            push!(non_array_params, lookupLHSVarByName(param, LambdaVarInfo))
        end
    end
    @dprintln(3,"Non-array params = ", non_array_params, " function = ", function_name)

    # Find out max_label
    assert(isa(body, Expr) && (body.head === :body))
    max_label = getMaxLabel(0, body.args)
    @dprintln(3,"maxLabel = ", max_label, " body type = ", body.typ)

    rep_start = time_ns()

    changed = true
    while changed
        @dprintln(1,"Removing statement with no dependencies from the AST with parameters = ", input_parameters, " function = ", function_name)
        rnd_state = RemoveNoDepsState(lives, non_array_params)
        body = AstWalk(body, remove_no_deps, rnd_state)
        @dprintln(3,"AST after no dep stmts removed = ", " function = ", function_name)
        printLambda(3, LambdaVarInfo, body)

        @dprintln(3,"top_level_no_deps = ", rnd_state.top_level_no_deps)

        @dprintln(1,"Adding statements with no dependencies to the start of the AST.", " function = ", function_name)
        body = CompilerTools.LambdaHandling.prependStatements(body, rnd_state.top_level_no_deps)
        @dprintln(3,"body after no dep stmts re-inserted = ", body, " function = ", function_name)

        @dprintln(1,"Re-starting liveness analysis.", " function = ", function_name)
        lives = computeLiveness(body, LambdaVarInfo)
        @dprintln(1,"Finished liveness analysis.", " function = ", function_name)
        @dprintln(3,"lives = ", lives)

        changed = rnd_state.change
    end

    if print_times
    @dprintln(1,"Rearranging passes time = ", ns_to_sec(time_ns() - rep_start))
    end

    processAndUpdateBody(body, removeNothingStmts, nothing)
    @dprintln(3,"AST after removing nothing stmts = ", " function = ", function_name)
    printLambda(3, LambdaVarInfo, body)
    lives = computeLiveness(body, LambdaVarInfo)

    #multipleArrayAssigns = findMultipleArrayAssigns(lives, LambdaVarInfo)
    #@dprintln(3,"Arrays that are assigned multiple times = ", multipleArrayAssigns)

    if generalSimplification
        # motivated by logistic regression example
        # initial round of copy propagation so array size variables are propagated for arraysize() replacement
        # initial round of size analysis (create_equivalence_classes) so arraysize() calls are replaced
        # main copy propagation round after arraysize() replacement
        # main size analysis after all size variables are propagated
        body = AstWalk(body, copy_propagate, CopyPropagateState(lives,LambdaVarInfo))
        lives = computeLiveness(body, LambdaVarInfo)

        new_vars = expr_state(function_name, lives, max_label, input_arrays)
        @dprintln(3,"Creating equivalence classes.", " function = ", function_name)
        genEquivalenceClasses(LambdaVarInfo, body, new_vars)
        @dprintln(3,"Done creating equivalence classes.", " function = ", function_name)
        print_correlations(3, new_vars)

        lives = computeLiveness(body, LambdaVarInfo)
        body = AstWalk(body, copy_propagate, CopyPropagateState(lives,LambdaVarInfo))
        lives = computeLiveness(body, LambdaVarInfo)
        @dprintln(3,"AST after copy_propagate = ", " function = ", function_name)
        printLambda(3, LambdaVarInfo, body)
    end

    body = AstWalk(body, remove_dead, RemoveDeadState(lives,LambdaVarInfo))
    lives = computeLiveness(body, LambdaVarInfo)
    @dprintln(3,"AST after remove_dead = ", " function = ", function_name)
    printLambda(3, LambdaVarInfo, body)

    eq_start = time_ns()

    new_vars = expr_state(function_name, lives, max_label, input_arrays)
    @dprintln(3,"Creating equivalence classes.", " function = ", function_name)
    genEquivalenceClasses(LambdaVarInfo, body, new_vars)
    @dprintln(3,"Done creating equivalence classes.", " function = ", function_name)
    print_correlations(3, new_vars)

    if print_times
    @dprintln(1,"Creating equivalence classes time = ", ns_to_sec(time_ns() - eq_start))
    end

    if doRemoveAssertEqShape
        processAndUpdateBody(body, removeAssertEqShape, new_vars)
        @dprintln(3,"AST after removing assertEqShape = ", " function = ", function_name)
        printLambda(3, LambdaVarInfo, body)
        lives = computeLiveness(body, LambdaVarInfo)
    end

    if bb_reorder != 0
        maxFusion(lives)
        # Set the array of statements in the Lambda body to a new array constructed from the updated basic blocks.
        body = CompilerTools.LambdaHandling.getBody(CompilerTools.CFGs.createFunctionBody(lives.cfg), CompilerTools.LambdaHandling.getReturnType(LambdaVarInfo))
        @dprintln(3,"AST after maxFusion = ", " function = ", function_name)
        printLambda(3, LambdaVarInfo, body)
        lives = computeLiveness(body, LambdaVarInfo)
    end

    @dprintln(1,"Doing conversion to parallel IR.", " function = ", function_name)
    printLambda(3, LambdaVarInfo, body)

    new_vars.block_lives = lives
    @dprintln(3,"max_label before main Parallel IR = ", new_vars.max_label)
    @dprintln(3,"Lives before main Parallel IR = ")
    @dprintln(3,lives)

    # Do the main work of Parallel IR.
    body = get_one(from_expr(LambdaVarInfo, body, 1, new_vars, false))
    #assert(isa(ast,Array))
    #assert(length(ast) == 1)
    #body = body[1]

    @dprintln(1,"After from_expr function = ", function_name, " body = ")
    printLambda(1, LambdaVarInfo, body)

    body = remove_extra_allocs(LambdaVarInfo, body)
    lives = computeLiveness(body, LambdaVarInfo)
    body = AstWalk(body, remove_dead, RemoveDeadState(lives,LambdaVarInfo))

    if late_simplify
        @dprintln(3,"AST before last copy_propagate = ", " function = ", function_name)
        printLambda(3, LambdaVarInfo, body)
        lives = computeLiveness(body, LambdaVarInfo)
        new_vars = expr_state(function_name, lives, new_vars.max_label, input_arrays)
        genEquivalenceClasses(LambdaVarInfo, body, new_vars)
        print_correlations(3, new_vars)
        #AstWalk(body, replaceConstArraysizes, new_vars)
        lives = computeLiveness(body, LambdaVarInfo)
        body = AstWalk(body, copy_propagate, CopyPropagateState(lives,LambdaVarInfo))
        lives = computeLiveness(body, LambdaVarInfo)
        body = AstWalk(body, remove_dead, RemoveDeadState(lives,LambdaVarInfo))
        body = update_lambda_vars(LambdaVarInfo, body)
        @dprintln(3,"AST after late_simplify = ", " function = ", function_name)
        printLambda(3, LambdaVarInfo, body)
    end

    if unroll_small_parfors
        @dprintln(3,"AST at start of unroll_small_pafors = ", " function = ", function_name)
        printLambda(3, LambdaVarInfo, body)
        body = AstWalk(body, unroll_const_parfors, nothing)
        # flatten body since unroll can return :block nodes
        body = AstWalk(body, flatten_blocks, nothing)
        # TODO: replace small arrays with variables
        lives = computeLiveness(body, LambdaVarInfo)
        body = AstWalk(body, copy_propagate, CopyPropagateState(lives,LambdaVarInfo))
        lives = computeLiveness(body, LambdaVarInfo)
        body = AstWalk(body, remove_dead, RemoveDeadState(lives,LambdaVarInfo))
        @dprintln(3,"AST at end of unroll_small_parfors = ", " function = ", function_name)
        printLambda(3, LambdaVarInfo, body)
    end

    @dprintln(1,"Final ParallelIR function = ", function_name, " body = ")
    printLambda(1, LambdaVarInfo, body)

    set_pir_stats(body)
    #if pir_stop != 0
    #    throw(string("STOPPING AFTER PARALLEL IR CONVERSION"))
    #end
    return LambdaVarInfo, body
end

"""
Returns true if input is assignment expression with allocation
"""
function isAllocationAssignment(node::Expr)
    if node.head==:(=) && isAllocation(node.args[2])
        return true
    end
    return false
end

function isAllocationAssignment(node::ANY)
    return false
end

"""
Calculates statistics (number of allocations and parfors)
of the accelerated AST.
"""
function set_pir_stats(body)
    allocs = 0
    parfors = 0
    # count number of high-level allocations and assignment
    for expr in body.args
        if isAllocationAssignment(expr)
            allocs += 1
        elseif isBareParfor(expr)
            parfors +=1
        end
    end
    # make stats available to user
    ParallelAccelerator.set_num_acc_allocs(allocs);
    ParallelAccelerator.set_num_acc_parfors(parfors);
    return
end

type rm_allocs_state
    defs::Set{LHSVar}
    uniqsets::Set{LHSVar}
    removed_arrs::Dict{LHSVar,Array{Any,1}}
    LambdaVarInfo
end


"""
Removes extra allocations
Find arrays that are only allocated and not written to, and remove them.
"""
function remove_extra_allocs(LambdaVarInfo, body)
    @dprintln(3,"starting remove extra allocs")
    printLambda(3, LambdaVarInfo, body)
    old_lives = computeLiveness(body, LambdaVarInfo)
    # rm_allocs_live_cb callback ignores allocation calls but finds other defs of arrays
    lives = CompilerTools.LivenessAnalysis.from_lambda(LambdaVarInfo, body, rm_allocs_live_cb, LambdaVarInfo)
    @dprintln(3,"remove extra allocations lives ", lives)
    defs = Set{LHSVar}()
    for i in values(lives.basic_blocks)
        defs = union(defs, i.def)
    end
    # only consider those that are not aliased
    @dprintln(3, "starting alias analysis")
    uniqsets = CompilerTools.AliasAnalysis.from_lambda(LambdaVarInfo, body, old_lives, pir_alias_cb, nothing, noReAssign=true)
    @dprintln(3, "remove extra allocations defs ", defs, " uniqsets = ", uniqsets)
    rm_state = rm_allocs_state(defs, uniqsets, Dict{LHSVar,Array{Any,1}}(), LambdaVarInfo)
    AstWalk(body, rm_allocs_cb, rm_state)
    return body
end

"""
Remove arrays that are only allocated but not written to.
Keep shape information and replace arraysize() and arraylen() calls accordingly.
"""
function rm_allocs_cb(ast::Expr, state::rm_allocs_state, top_level_number, is_top_level, read)
    head = ast.head
    args = ast.args
    if head == :(=) && isAllocation(args[2])
        @dprintln(3,"rm_allocs_cb isAllocation ast = ", ast)
        arr = toLHSVar(args[1])
        # do not remove those that are being re-defined, or potentially aliased
        if in(arr, state.defs) || !in(arr, state.uniqsets)
            return CompilerTools.AstWalker.ASTWALK_RECURSE
        end
        call_offset = getCallOffset(args[2])
        alloc_args = args[2].args[call_offset:end]
        @dprintln(3,"alloc_args =", alloc_args)
        sh::Array{Any,1} = get_alloc_shape(alloc_args)
        shape = map(x -> if isa(x, Expr) x else toLHSVarOrInt(x) end, sh)
        @dprintln(3,"rm alloc shape ", shape)
        ast.args[2] = 0 #Expr(:call,TopNode(:tuple), shape...)
        CompilerTools.LambdaHandling.setType(arr, Int, state.LambdaVarInfo)
        state.removed_arrs[arr] = shape
        return ast
    elseif head==:call
        if length(args)>=2
            return rm_allocs_cb_call(state, args[1], args[2], args[3:end])
        end
    # remove extra arrays from parfor data structures
    elseif head==:parfor
        rm_allocs_cb_parfor(state, args[1])
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function rm_allocs_cb_call(state::rm_allocs_state, func, arr::RHSVar, rest_args::Array{Any,1})
    arr = toLHSVar(arr)
    if isBaseFunc(func, :arraysize) && haskey(state.removed_arrs, arr)
        shape = state.removed_arrs[arr]
        return shape[rest_args[1]]
    elseif isBaseFunc(func, :unsafe_arrayref) && haskey(state.removed_arrs, arr)
        return 0
    elseif isBaseFunc(func, :arraylen) && haskey(state.removed_arrs, arr)
        shape = state.removed_arrs[arr]
        dim = length(shape)
        @dprintln(3, "arraylen found")
        return mk_mult_int_expr(shape)
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function rm_allocs_cb_call(state::rm_allocs_state, func::ANY, arr::ANY, rest_args::Array{Any,1})
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end


"""
Update parfor data structures for removed arrays.
"""
function rm_allocs_cb_parfor(state::rm_allocs_state, parfor::PIRParForAst)
    if in(parfor.first_input, keys(state.removed_arrs))
        #TODO parfor.first_input = NoArrayInput
    end
end

function rm_allocs_cb(ast :: ANY, cbdata :: ANY, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function get_alloc_shape(args)
    @dprintln(3, "get_alloc_shape args = ", args)
    # tuple
    if args[1]==:(:jl_new_array) && length(args)==7
        return args[6].args[2:end]
    else
        shape_arr = Any[]
        i = 1
        while 6+(i-1)*2 <= length(args)
            push!(shape_arr, args[6+(i-1)*2])
            i+=1
        end
        return shape_arr
    end
    return Any[]
end

function rm_allocs_live_cb(ast :: Expr, cbdata :: ANY)
    head = ast.head
    args = ast.args
    @dprintln(3, "rm_allocs_live_cb called with ast ", ast)
    if head == :(=) && isAllocation(args[2])
        @dprintln(3, "rm_allocs_live_cb ignore allocation ", ast)
        return Any[args[2]]
    end
    return pir_live_cb(ast,cbdata)
end

function rm_allocs_live_cb(ast :: ANY, cbdata :: ANY)
    return pir_live_cb(ast,cbdata)
end


function from_expr(ast :: LambdaInfo, depth, state :: expr_state, top_level)
    @dprintln(3,"from_expr for LambdaInfo does nothing! Any LambdaInfo should have been turned into AST prior to ParallelIR")
    return [ast]
end

function from_expr(LambdaVarInfo, body :: Expr, depth, state :: expr_state, top_level)
    assert(body.head == :body)
    @dprintln(3,"from_expr for LambdaVarInfo body starting")
    state.LambdaVarInfo = LambdaVarInfo
    body = from_lambda_body(body, 0, state)
    return [body]
end

function from_expr(ast::Union{RHSVar,TopNode,LineNumberNode,LabelNode,Char,
    GotoNode,DataType,AbstractString,NewvarNode,Void,Module}, depth, state :: expr_state, top_level)
    #skip
    return [ast]
end

function from_expr(ast::GlobalRef, depth, state :: expr_state, top_level)
    mod = ast.mod
    name = ast.name
    typ = typeof(mod)
    @dprintln(2,"GlobalRef type ",typeof(mod))
    return [ast]
end


function from_expr(ast::QuoteNode, depth, state :: expr_state, top_level)
    value = ast.value
    #TODO: fields: value
    @dprintln(2,"QuoteNode type ",typeof(value))
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

function from_expr(ast::ANY, depth, state :: expr_state, top_level)
    typ = typeof(ast)
    if typeof(typ) == DataType
        @dprintln(4, "from_expr object encountered ", ast, " of type ", typ)
        return [ast]
    else
        throw(string("from_expr unknown type for ", ast, " of type ", typ))
    end
end

"""
The main ParallelIR function for processing some node in the AST.
"""
function from_expr(ast ::Expr, depth, state :: expr_state, top_level)
    if (ast === nothing)
        return [nothing]
    end
    @dprintln(2,"from_expr depth=",depth," ")
    @dprint(2,"Expr ")
    head = ast.head
    args = ast.args
    typ  = ast.typ
    @dprintln(2,head, " ", args)
    if head == :lambda
        ast = from_lambda(ast, depth, state)
        @dprintln(3,"After from_lambda = ", ast)
        return [ast]
    elseif head == :body
        @dprintln(3,"Processing body start")
        args = from_exprs(args,depth+1,state)
        @dprintln(3,"Processing body end")
    elseif head == :(=)
        @dprintln(3,"Before from_assignment typ is ", typ)
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
        args = from_exprs(args, depth, state)
    elseif head == :invoke || head == :call || head == :call1
        head, args = from_call(head, ast, depth, state)
        # TODO: catch domain IR result here
    elseif head == :foreigncall
        args = from_foreigncall(ast, depth, state)
    elseif head == :line
        # remove line numbers
        return []
        # skip
    elseif head == :select
        # translate dangling :select (because most other :select would have been inlined and then removed when no longer live) as an mmap into parfor
        head = :parfor
        @assert (length(args) == 2) "expect Domain IR select expr to have two arguments, but got " * args
        typ = getType(args[1], state.LambdaVarInfo)
        etyp = eltype(typ)
        args = Any[[DomainIR.mk_select(args...)], DomainIR.DomainLambda(Type[etyp], Type[etyp], as -> Any[Expr(:tuple, as...)], CompilerTools.LambdaHandling.LambdaVarInfo())]
        domain_oprs = [DomainOperation(:mmap, args)]
        args = mk_parfor_args_from_mmap(args[1], args[2], domain_oprs, state)
        @dprintln(1,"switching to parfor node for :select, got ", args)
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
        @dprintln(1,"switching to parfor node for mmap, got ", args)
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
        @dprintln(1,"switching to parfor node for mmap!")
    elseif head == :reduce
        head = :parfor
        args = mk_parfor_args_from_reduce(args, state)
        @dprintln(1,"switching to parfor node for reduce")
    elseif head == :parallel_for
        head = :parfor
        args = mk_parfor_args_from_parallel_for(args, state)
        @dprintln(1,"switching to parfor node for parallel_for")
    elseif head == :copy
        # turn array copy back to plain Julia call
        head = :call
        args = vcat(:copy, args)
    elseif head == :arraysize
        # turn array size back to plain Julia call
        head = :call
        args = vcat(GlobalRef(Base, :arraysize), args)
    elseif head == :alloc
        # turn array alloc back to plain Julia ccall
        head, args = from_alloc(args, state)
    elseif head == :stencil!
        head = :parfor
        ast = mk_parfor_args_from_stencil(typ, head, args, state)
        @dprintln(1,"switching to parfor node for stencil")
        return ast
    elseif head == :copyast
        @dprintln(2,"copyast type")
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
    elseif head == :boundscheck || head == :inbounds
        # skip
    elseif head == :meta
        # skip
    elseif head == :static_parameter
        # skip
    elseif head == :type_goto
        # skip
    elseif head == :llvmcall
        # skip
    elseif head == :simdloop
        # skip
    elseif head in DomainIR.exprHeadIgnoreList
        # other packages like HPAT can generate new nodes like :alloc, :join
    else
        throw(string("ParallelAccelerator.ParallelIR.from_expr: unknown Expr head :", head))
    end
    ast = Expr(head, args...)
    @dprintln(3,"New expr type = ", typ, " ast = ", ast)
    ast.typ = typ
    return [ast]
end

function createForeignCall(realArgs)
if VERSION >= v"0.6.0-pre"
    return :foreigncall, realArgs
else
    return :call, vcat(GlobalRef(Core,:ccall), realArgs)
end
end

function from_alloc(args::Array{Any,1}, state)
    elemTyp = args[1]
    sizes = args[2]
    n = length(sizes)
    assert(n >= 1 && n <= 3)

    @dprintln(3, "from_alloc: elemTyp = ", elemTyp, " sizes = ", sizes)

    if n == 1
        only_size = sizes[1] 
        os_type = CompilerTools.LambdaHandling.getType(only_size, state.LambdaVarInfo)
        @dprintln(3, "from_alloc: only_size ", only_size, " has typ = ", os_type)
        if os_type <: Tuple
            if haskey(state.tuple_assigns, only_size)
                @dprintln(3, "from_alloc: found declaration for tuple")
                sizes = state.tuple_assigns[only_size]
                n = length(sizes)
                assert(n >= 1 && n <= 3)
            else
                @dprintln(3, "from_alloc: did not find declaration for tuple")
            end
        end
    end

    name = Symbol(string("jl_alloc_array_", n, "d"))
    new_svec = TypedExpr(SimpleVector, :call, GlobalRef(Core, :svec), GlobalRef(Base, :Any), [ GlobalRef(Base, :Int) for i=1:n ]...)
if VERSION >= v"0.6.0-pre"
    appTypExpr = Array{elemTyp,n}
    new_svec = eval(new_svec)
else
    appTypExpr = TypedExpr(Type{Array{elemTyp,n}}, :call, GlobalRef(Core, :apply_type), GlobalRef(Core,:Array), elemTyp, n)
end
    realArgs = Any[QuoteNode(name), appTypExpr, new_svec, Array{elemTyp,n}, 0]
    for i=1:n
        push!(realArgs, sizes[i])
        push!(realArgs, 0)
    end
    return createForeignCall(realArgs)
end


"""
Take something returned from AstWalk and assert it should be an array but in this
context that the array should also be of length 1 and then return that single element.
"""
function get_one(ast::Array)
    assert(length(ast) == 1)
    ast[1]
end

"""
Wraps the callback and opaque data passed from the user of ParallelIR's AstWalk.
"""
type DirWalk
    callback
    cbdata
end

"""
Return one element array with element x.
"""
function asArray(x)
    ret = Any[]
    push!(ret, x)
    return ret
end

function AstWalkCallback(cur_parfor :: PIRParForAst, dw :: DirWalk, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    @dprintln(4,"PIR AstWalkCallback PIRParForAst starting")
    ret = dw.callback(cur_parfor, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(4,"PIR AstWalkCallback PIRParForAst ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end

    for i = 1:length(cur_parfor.preParFor)
        cur_parfor.preParFor[i] = AstWalk(cur_parfor.preParFor[i], dw.callback, dw.cbdata)
    end
    for i = 1:length(cur_parfor.hoisted)
        cur_parfor.hoisted[i] = AstWalk(cur_parfor.hoisted[i], dw.callback, dw.cbdata)
    end
    for i = 1:length(cur_parfor.loopNests)
        cur_parfor.loopNests[i].indexVariable = AstWalk(cur_parfor.loopNests[i].indexVariable, dw.callback, dw.cbdata)
        # There must be some reason that I was faking an assignment expression although this really shouldn't happen in an AstWalk. In liveness callback yes, but not here.
        #AstWalk(mk_assignment_expr(cur_parfor.loopNests[i].indexVariable, 1, Int64), dw.callback, dw.cbdata)
        cur_parfor.loopNests[i].lower = AstWalk(cur_parfor.loopNests[i].lower, dw.callback, dw.cbdata)
        cur_parfor.loopNests[i].upper = AstWalk(cur_parfor.loopNests[i].upper, dw.callback, dw.cbdata)
        cur_parfor.loopNests[i].step  = AstWalk(cur_parfor.loopNests[i].step, dw.callback, dw.cbdata)
    end
    for i = 1:length(cur_parfor.reductions)
        cur_parfor.reductions[i].reductionVar     = AstWalk(cur_parfor.reductions[i].reductionVar, dw.callback, dw.cbdata)
        cur_parfor.reductions[i].reductionVarInit = AstWalk(cur_parfor.reductions[i].reductionVarInit, dw.callback, dw.cbdata)
        cur_parfor.reductions[i].reductionFunc    = AstWalk(cur_parfor.reductions[i].reductionFunc, dw.callback, dw.cbdata)
    end
    for i = 1:length(cur_parfor.body)
        cur_parfor.body[i] = AstWalk(cur_parfor.body[i], dw.callback, dw.cbdata)
    end
    for i = 1:length(cur_parfor.postParFor)
        cur_parfor.postParFor[i] = AstWalk(cur_parfor.postParFor[i], dw.callback, dw.cbdata)
    end

    return cur_parfor
end

"""
AstWalk callback that handles ParallelIR AST node types.
"""
function AstWalkCallback(x :: Expr, dw :: DirWalk, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    @dprintln(4,"PIR AstWalkCallback Expr starting")
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(4,"PIR AstWalkCallback Expr ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end

    head = x.head
    args = x.args
    if head == :parfor
        cur_parfor = args[1]
        AstWalkCallback(cur_parfor, dw, top_level_number, is_top_level, read)
        return x
    elseif head == :parfor_start || head == :parfor_end
        @dprintln(3, "parfor_start or parfor_end walking, dw = ", dw)
        @dprintln(3, "pre x = ", x)
        cur_parfor = args[1]
        for i = 1:length(cur_parfor.loopNests)
            x.args[1].loopNests[i].indexVariable = AstWalk(cur_parfor.loopNests[i].indexVariable, dw.callback, dw.cbdata)
            #AstWalk(mk_assignment_expr(cur_parfor.loopNests[i].indexVariable, 1, Int), dw.callback, dw.cbdata)
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
        @dprintln(3, "post x = ", x)
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

# task mode commeted out
#=
function AstWalkCallback(x :: pir_range_actual, dw :: DirWalk, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    @dprintln(4,"PIR AstWalkCallback starting")
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(4,"PIR AstWalkCallback ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end

    for i = 1:length(x.dim)
        x.lower_bounds[i] = AstWalk(x.lower_bounds[i], dw.callback, dw.cbdata)
        x.upper_bounds[i] = AstWalk(x.upper_bounds[i], dw.callback, dw.cbdata)
    end
    return x
end
=#

function AstWalkCallback(x :: DelayedFunc, dw :: DirWalk, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    @dprintln(4,"PIR AstWalkCallback DelayedFunc starting")
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(4,"PIR AstWalkCallback DelayedFunc ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end
    if isa(dw.cbdata, rm_allocs_state) # skip traversal if it is for rm_allocs
        return x
    end
    for i = 1:length(x.args)
        y = x.args[i]
        if isa(y, Array)
            for j=1:length(y)
                y[j] = AstWalk(y[j], dw.callback, dw.cbdata)
            end
        else
            x.args[i] = AstWalk(x.args[i], dw.callback, dw.cbdata)
        end
    end
    return x
end

function AstWalkCallback(x :: ANY, dw :: DirWalk, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    @dprintln(4,"PIR AstWalkCallback ANY starting, x = ", x, " type = ", typeof(x))
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(4,"PIR AstWalkCallback ANY ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
ParallelIR version of AstWalk.
Invokes the DomainIR version of AstWalk and provides the parallel IR AstWalk callback AstWalkCallback.

Parallel IR AstWalk calls Domain IR AstWalk which in turn calls CompilerTools.AstWalker.AstWalk.
For each AST node, CompilerTools.AstWalker.AstWalk calls Domain IR callback to give it a chance to handle the node if it is a Domain IR node.
Likewise, Domain IR callback first calls Parallel IR callback to give it a chance to handle Parallel IR nodes.
The Parallel IR callback similarly first calls the user-level callback to give it a chance to process the node.
If a callback returns "ASTWALK_RECURSE" it means it didn't modify that node and that the previous code should process it.
The Parallel IR callback will return "ASTWALK_RECURSE" if the node isn't a Parallel IR node.
The Domain IR callback will return "ASTWALK_RECURSE" if the node isn't a Domain IR node.
"""
function AstWalk(ast::Any, callback, cbdata)
    dw = DirWalk(callback, cbdata)
    DomainIR.AstWalk(ast, AstWalkCallback, dw)
end

"""
An AliasAnalysis callback (similar to LivenessAnalysis callback) that handles ParallelIR introduced AST node types.
For each ParallelIR specific node type, form an array of expressions that AliasAnalysis
    can analyze to reflect the aliases of the given AST node.
    If we read a symbol it is sufficient to just return that symbol as one of the expressions.
    If we write a symbol, then form a fake mk_assignment_expr just to get liveness to realize the symbol is written.
"""
function pir_alias_cb(ast::Expr, state, cbdata)
    @dprintln(4,"pir_alias_cb")

    head = ast.head
    args = ast.args
    if head == :parfor
        @dprintln(3,"pir_alias_cb for :parfor")
        expr_to_process = Any[]

        assert(typeof(args[1]) == ParallelAccelerator.ParallelIR.PIRParForAst)
        this_parfor = args[1]

        AliasAnalysis.increaseNestLevel(state);
        AliasAnalysis.from_exprs(state, this_parfor.preParFor, pir_alias_cb, cbdata)
        AliasAnalysis.from_exprs(state, this_parfor.hoisted, pir_alias_cb, cbdata)
        AliasAnalysis.from_exprs(state, this_parfor.body, pir_alias_cb, cbdata)
        ret = AliasAnalysis.from_exprs(state, this_parfor.postParFor, pir_alias_cb, cbdata)
        AliasAnalysis.decreaseNestLevel(state);

        return ret[end]

    elseif head == :call
        if isBaseFunc(args[1], :unsafe_arrayref)
            return AliasAnalysis.NotArray
        elseif isBaseFunc(args[1], :unsafe_arrayset)
            return AliasAnalysis.NotArray
        end
    # flattened parfor nodes are ignored
    elseif head == :parfor_start || head == :parfor_end
        return AliasAnalysis.NotArray
    end

    return DomainIR.dir_alias_cb(ast, state, cbdata)
end

function pir_alias_cb(ast::ANY, state, cbdata)
    @dprintln(4,"pir_alias_cb")
    return DomainIR.dir_alias_cb(ast, state, cbdata)
end

function dependenceCB(ast::Expr, cbdata)
    @dprintln(3, "dependenceCB")

    head = ast.head
    args = ast.args
    if head == :parfor
        @dprintln(3,"dependenceCB for :parfor")
#        ParallelAccelerator.show_backtrace()

        assert(typeof(args[1]) == ParallelAccelerator.ParallelIR.PIRParForAst)
        this_parfor = args[1]
        postvars = []
        for pv in this_parfor.postParFor[end]
            if isa(pv, RHSVar)
                push!(postvars, toLHSVar(pv))
            end
        end

        return CompilerTools.TransitiveDependence.CallbackResult(
                 CompilerTools.TransitiveDependence.mergeDepSet(   # FIX FIX FIX...do something here with hoisted???
                   CompilerTools.TransitiveDependence.computeDependenciesAST(TypedExpr(nothing, :body, this_parfor.preParFor...), ParallelAccelerator.ParallelIR.dependenceCB, nothing),
                   CompilerTools.TransitiveDependence.computeDependenciesAST(TypedExpr(nothing, :body, this_parfor.body...), ParallelAccelerator.ParallelIR.dependenceCB, nothing)
                 ),
                 Set{LHSVar}(),
                 Set{LHSVar}(postvars),
                 [], []
               )
    elseif head == :call
        if isBaseFunc(args[1], :unsafe_arrayref)
            @dprintln(3,"dependenceCB for :unsafe_arrayref")
#            ParallelAccelerator.show_backtrace()
            return CompilerTools.TransitiveDependence.CallbackResult(
                 Dict{LHSVar,Set{LHSVar}}(),
                 Set{LHSVar}(),
                 Set{LHSVar}(),
                 args[2:end], []
               )
        elseif isBaseFunc(args[1], :unsafe_arrayset)
            @dprintln(3,"dependenceCB for :unsafe_arrayset")
#            ParallelAccelerator.show_backtrace()
            return CompilerTools.TransitiveDependence.CallbackResult(
                 Dict{LHSVar,Set{LHSVar}}(),
                 Set{LHSVar}(),
                 Set{LHSVar}(),
                 args[3:end], args[2:2]
               )
        end
    # flattened parfor nodes are ignored
    elseif head == :parfor_start || head == :parfor_end
        # FIX FIX FIX...is this right?  Maybe we need to do something for reductions here?
        @dprintln(3,"dependenceCB for :parfor_start or :parfor_end")
#        ParallelAccelerator.show_backtrace()
        return CompilerTools.TransitiveDependence.CallbackResult(
                 Dict{LHSVar,Set{LHSVar}}(),
                 Set{LHSVar}(),
                 Set{LHSVar}(),
                 [], []
               )
    end

    return nothing
end

function dependenceCB(ast::ANY, cbdata)
    return nothing
end

function getType(node :: Expr, linfo :: CompilerTools.LambdaHandling.LambdaVarInfo)
    return node.typ
end

function getType(node :: RHSVar, linfo :: CompilerTools.LambdaHandling.LambdaVarInfo)
    return CompilerTools.LambdaHandling.getType(node, linfo)
end

function getType(node :: ANY, linfo :: CompilerTools.LambdaHandling.LambdaVarInfo)
    throw(string("Don't know how to getType for node of type ", typeof(node)))
end

end
