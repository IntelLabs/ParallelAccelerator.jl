module ParallelIR
export num_threads_mode

using CompilerTools
using ..DomainIR
using ..AliasAnalysis
using ..IntelPSE
#if IntelPSE.client_intel_pse_mode == 5
#using Base.Threading
#end

import Base.show
import CompilerTools.AstWalker
import CompilerTools.ReadWriteSet
import CompilerTools.LivenessAnalysis
import CompilerTools.Loops

# This controls the debug print level.  0 prints nothing.  At the moment, 2 prints everything.
DEBUG_LVL=0

function set_debug_level(x)
    global DEBUG_LVL = x
end

# A debug print routine.
function dprint(level,msgs...)
    if(DEBUG_LVL >= level)
        print(msgs...)
    end
end

function ns_to_sec(x)
  x / 1000000000.0
end

# A debug print routine.
function dprintln(level,msgs...)
    if(DEBUG_LVL >= level)
        println(msgs...)
    end
end

ISCAPTURED = 1
ISASSIGNED = 2
ISASSIGNEDBYINNERFUNCTION = 4
ISCONST = 8
ISASSIGNEDONCE = 16
ISPRIVATEPARFORLOOP = 32

unique_num = 1

# This should pretty always be used instead of Expr(...) to form an expression as it forces the typ to be provided.
function TypedExpr(typ, rest...)
    res = Expr(rest...)
    res.typ = typ
    res
end

# Holds the information about a loop in a parfor node.
type PIRLoopNest
    indexVariable :: SymbolNode
    lower
    upper
    step
end

# Holds the information about a reduction in a parfor node.
type PIRReduction
    reductionVar  :: SymbolNode
    reductionVarInit
    reductionFunc
end

type DomainOperation
    operation
    input_args :: Array{Any,1}
end

type EquivalenceClasses
  data :: Dict{Symbol,Int64}

  function EquivalenceClasses()
    new(Dict{Symbol,Int64}())
  end
end

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

function EquivalenceClassesAdd(ec :: EquivalenceClasses, sym :: Symbol)
  if !haskey(ec.data, sym)
    a = collect(values(ec.data))
    m = length(a) == 0 ? 0 : maximum(a)
    ec.data[sym] = m + 1
  end
  ec.data[sym]
end

function EquivalenceClassesClear(ec :: EquivalenceClasses)
  empty!(ec.data)
end

# The parfor AST type.
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
    array_aliases :: Dict{Symbol, Symbol}

    # instruction count estimate of the body
    # To get the total loop instruction count, multiply this value by (upper_limit - lower_limit)/step for each loop nest
    # This will be "nothing" if we don't know how to estimate.  If not "nothing" then it is an expression which may
    # include calls.
    instruction_count_expr

    function PIRParForAst(b, pre, nests, red, post, orig, t, unique)
      r = CompilerTools.ReadWriteSet.from_exprs(b)
      new(b, pre, nests, red, post, orig, [t], r, unique, Dict{Symbol,Symbol}(), nothing)
    end

    function PIRParForAst(b, pre, nests, red, post, orig, t, r, unique)
      new(b, pre, nests, red, post, orig, [t], r, unique, Dict{Symbol,Symbol}(), nothing)
    end
end

type PIRParForStartEnd
    loopNests  :: Array{PIRLoopNest,1}      # holds information about the loop nests
    reductions :: Array{PIRReduction,1}     # holds information about the reductions
    instruction_count_expr
end

include("parallel-ir-stencil.jl")

# Pretty printing routine for parfor AST nodes.
function show(io::IO, pnode::IntelPSE.ParallelIR.PIRParForAst)
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
    if length(pnode.original_domain_nodes) > 0 && DEBUG_LVL >= 3
        println(io,"Domain nodes: ")
        for i = 1:length(pnode.original_domain_nodes)
            println(io,pnode.original_domain_nodes[i])
        end
    end
    if DEBUG_LVL >= 3
        println(io, pnode.rws)
    end
end

export PIRLoopNest, PIRReduction, from_exprs, PIRParForAst, set_debug_level, AstWalk, PIRSetFuseLimit, PIRNumSimplify, PIRInplace, PIRRunAsTasks, PIRLimitTask, PIRReduceTasks, PIRStencilTasks, createVarSet, createVarDict, PIRFlatParfor, PIRNumThreadsMode, PIRShortcutArrayAssignment, PIRPreEq, PIRTaskGraphMode, PIRPolyhedral

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

# Create an assignment expression AST node given a left and right-hand side.
function mk_assignment_expr(lhs, rhs)
    if(typeof(lhs) == SymbolNode)
        expr_typ = lhs.typ
    else
        throw(string("mk_assignment_expr lhs is not of type SymbolNode, is of this type instead: ", typeof(lhs)))
    end
    dprintln(2,"mk_assignment_expr lhs type = ", typeof(lhs))
    TypedExpr(expr_typ, symbol('='), lhs, rhs)
end

# Create an expression whose value is the length of the input array.
function mk_arraylen_expr(x, dim)
    TypedExpr(Int, :call, TopNode(:arraysize), :($x), dim)
end

function mk_parallelir_ref(sym, ref_type=Function)
    inner_call = TypedExpr(Module, :call, TopNode(:getfield), :IntelPSE, QuoteNode(:ParallelIR))
    TypedExpr(ref_type, :call, TopNode(:getfield), inner_call, QuoteNode(sym))
end

function mk_convert(new_type, ex)
    TypedExpr(new_type, :call, TopNode(:convert), new_type, ex)
end

# Create an expression which returns the index'th element of the tuple whose name is contained in tuple_var.
function mk_tupleref_expr(tuple_var, index, typ)
    TypedExpr(typ, :call, TopNode(:tupleref), tuple_var, index)
end

# Allocate and initialize a 1D Julia array.
function mk_alloc_array_1d_expr(elem_type, atype, length)
  dprintln(2,"mk_alloc_array_1d_expr atype = ", atype, " elem_type = ", elem_type, " length = ", length, " typeof(length) = ", typeof(length))
  ret_type  = TypedExpr(Type{atype}, :call1, TopNode(:apply_type), :Array, elem_type, 1)
  arg_types = TypedExpr((Type{Any},Type{Int}), :call1, TopNode(:tuple), :Any, :Int)

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
       arg_types,
       atype,
       0,
       length_expr,
       0)
end

# Allocate and initialize a 2D Julia array.
function mk_alloc_array_2d_expr(elem_type, atype, length1, length2)
  dprintln(2,"mk_alloc_array_2d_expr atype = ", atype)
  ret_type  = TypedExpr(Type{atype}, :call1, TopNode(:apply_type), :Array, elem_type, 2)
  arg_types = TypedExpr((Type{Any},Type{Int}), :call1, TopNode(:tuple), :Any, :Int, :Int)

  TypedExpr(
       atype,
       :call,
       TopNode(:ccall),
       QuoteNode(:jl_alloc_array_2d),
       ret_type,
       arg_types,
       atype,
       0,
       SymbolNode(length1,Int),
       0,
       SymbolNode(length2,Int),
       0)
end


function isArrayType(typ)
  return (typ.name == Array.name)
end

# Return the type of an Array
function getArrayElemType(array::SymbolNode)
  assert(typeof(array) == SymbolNode)
  if array.typ.name == Array.name
    array.typ.parameters[1]
  elseif array.typ.name == BitArray.name
    Bool
  else
    assert(false)
  end
end

# Return the number of dimensions of an Array
function getArrayNumDims(array::SymbolNode)
  assert(typeof(array) == SymbolNode)
  assert(array.typ.name == Array.name)
  array.typ.parameters[2]
end

# Return the type of an Array
function getArrayElemType(array::DataType)
  if array.name == Array.name
    array.parameters[1]
  elseif array.name == BitArray.name
    Bool
  else
    assert(false)
  end
end

function augment_sn(x)
  if typeof(x) == Int64
    return x
  else
    return SymbolNode(x,Int)
  end
end

# Return a new AST node that corresponds to getting the index_var index from the array array_name.
function mk_arrayref1(array_name, index_vars, inbounds)
  dprintln(3,"mk_arrayref1 typeof(index_vars) = ", typeof(index_vars))
  dprintln(3,"mk_arrayref1 array_name = ", array_name, " typeof(array_name) = ", typeof(array_name))
  elem_typ = getArrayElemType(array_name)
  dprintln(3,"mk_arrayref1 array_name.typ = ", array_name.typ, " element type = ", elem_typ)

  if inbounds
    fname = :unsafe_arrayref
  else
    fname = :arrayref
  end

  indsyms = map(x -> augment_sn(x), index_vars)

  TypedExpr(
       elem_typ,
       :call,
       TopNode(fname),
       :($array_name),
       indsyms...)
end

function createStateVar(state, name, typ, access)
  new_temp_sym = symbol(name)
  addStateVar(state, new_var(new_temp_sym, typ, access))
  SymbolNode(new_temp_sym, typ)
end

function createTempForArray(array_sn, unique_id, state, temp_map)
  if !haskey(temp_map, array_sn.name)
    temp_type = getArrayElemType(array_sn.typ)
    temp_map[array_sn.name] = createStateVar(state, string("parallel_ir_temp_", array_sn.name, "_", unique_id), temp_type, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP)
  end
  temp_map[array_sn.name]
end

function makePrivateParfor(var_name, state)
  assert(typeof(var_name) == Symbol)
  if !haskey(state.meta2_typed, var_name)
    dprintln(0, "meta2_typed = ", state.meta2_typed)
    throw(string("Could not make variable ", var_name, " private since it wasn't found in the variable metadata list."))
  end
  cur_access_type = state.meta2_typed[var_name][3]
  if cur_access_type | ISPRIVATEPARFORLOOP != 0
    dprintln(3, "adding private flag for variable ", var_name)
  end
  state.meta2_typed[var_name][3] = state.meta2_typed[var_name][3] | ISPRIVATEPARFORLOOP
end

# Return a new AST node that corresponds to setting the index_var index from the array array_name with value.
function mk_arrayset1(array_name, index_vars, value, inbounds)
  dprintln(3,"mk_arrayset1 array_name = ", array_name, " typeof(array_name) = ", typeof(array_name))
  if(typeof(array_name) == SymbolNode)
      dprintln(3,"mk_arrayset1 array_name.typ = ", array_name.typ, " param len = ", length(array_name.typ.parameters))
  end
  assert(typeof(array_name) == SymbolNode)
  elem_typ = array_name.typ.parameters[1]

  if inbounds
    fname = :unsafe_arrayset
  else
    fname = :arrayset
  end

  indsyms = map(x -> augment_sn(x), index_vars)

  TypedExpr(
       elem_typ,
       :call,
       TopNode(fname),
       array_name,
       :($value),
       indsyms...)
end

# Returns true if all array references use singular index variables and nothing more complicated involving,
# for example, addition or subtraction by a constant.
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

# Represents the typed section of the lambda header where the variables of a function are listed along with their type and access style
type new_var
  name
  typ
  access_info
end

# State passed around while converting an AST from domain to parallel IR.
type expr_state
  block_lives :: CompilerTools.LivenessAnalysis.BlockLiveness    # holds the output of liveness analysis at the block and top-level statement level
  top_level_number :: Int                          # holds the current top-level statement number...used to correlate with stmt liveness info
  # Arrays created from each other are known to have the same size. Store such correlations here.
  # If two arrays have the same dictionary value, they are equal in size.
  array_length_correlation :: Dict{Symbol,Int}
  symbol_array_correlation :: Dict{Array{Symbol,1},Int}
  param :: Array{Symbol}
  meta  :: Array{Any}
  meta2 :: Set
  meta2_typed :: Dict{Symbol,Array{Any,1}}
  num_var_assignments :: Dict{Symbol,Int}
  max_label :: Int # holds the max number of all LabelNodes

  # Initialize the state for parallel IR translation.
  function expr_state(bl, max_label, input_arrays)
    init_corr = Dict{Symbol,Int}()
    # For each input array, insert into the correlations table with a different value.
    for i = 1:length(input_arrays)
      init_corr[input_arrays[i]] = i
    end
    new(bl, 0, init_corr, Dict{Array{Symbol,1},Int}(), Symbol[], Any[], Set(), Dict{Symbol,Array{Any,1}}(), Dict{Symbol,Int}(), max_label)
  end
end

function next_label(state::IntelPSE.ParallelIR.expr_state)
  state.max_label = state.max_label + 1
  return state.max_label
end

function createVarDict(incoming)
  iv = Dict{Symbol,Array{Any,1}}()
  for i = 1:length(incoming)
    iv[incoming[i][1]] = incoming[i]
  end
  iv
end

function createVarSet(incoming)
  iv = Set()
  for i = 1:length(incoming)
    push!(iv, incoming[i])
  end
  iv
end


function addUnknownArray(x, state)
  a = collect(values(state.array_length_correlation))
  m = length(a) == 0 ? 0 : maximum(a)
  state.array_length_correlation[x] = m + 1
end

# If we somehow determine that two sets of correlations are actually the same length then merge one into the other.
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

function add_merge_correlations(old_sym :: Symbol, new_sym :: Symbol, state)
  dprintln(3, "add_merge_correlations ", old_sym, " ", new_sym, " ", state.array_length_correlation)
  old_corr = getOrAddArrayCorrelation(old_sym, state)
  new_corr = getOrAddArrayCorrelation(new_sym, state)
  merge_correlations(state, old_corr, new_corr)
  dprintln(3, "add_merge_correlations post ", state.array_length_correlation)
end

# Return a correlation set for an array.  If the array was not previously added then add it and return it.
function getOrAddArrayCorrelation(x, state)
  if !haskey(state.array_length_correlation, x)
    dprintln(3,"Correlation for array not found = ", x)
    addUnknownArray(x, state)
  end
  state.array_length_correlation[x]
end

# A new array is being created with an explicit size specification in dims.
function getOrAddSymbolCorrelation(array, state, dims)
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

function get_unique_num()
  ret = unique_num
  global unique_num = unique_num + 1
  ret
end

# ===============================================================================================================================

# The main routine that converts an reduce AST node to a parfor AST node.
function mk_parfor_args_from_reduce(input_args::Array{Any,1}, state)
  # Make sure we get what we expect from domain IR.
  # There should be three entries in the array, how to initialize the reduction variable, the arrays to work on and a DomainLambda.
  assert(length(input_args) == 3)

  zero_val = input_args[1]
  input_array = input_args[2]
  input_array_ranges = nothing
  if isa(input_array, Expr) && is(input_array.head, :select)
    dprintln(3,"mk_parfor_args_from_reduce. head is :select")
    input_array_ranges = input_array.args[2] # range object
    input_array = input_array.args[1]
    assert(isa(input_array_ranges, Expr)) # TODO: may need to handle SymbolNodes in the future
    if input_array_ranges.head == :ranges
      dprintln(3,"mk_parfor_args_from_reduce. input_array_ranges.head is :ranges")
      input_array_ranges = input_array_ranges.args
    else
      dprintln(3,"mk_parfor_args_from_reduce. input_array_ranges.head is NOT :ranges")
      input_array_ranges = Any[ input_array_ranges ]
    end
  end
  dl = input_args[3]
  assert(isa(dl, DomainLambda))

  dprintln(3,"mk_parfor_args_from_reduce. zero_val = ", zero_val, " type = ", typeof(zero_val))
  dprintln(3,"mk_parfor_args_from_reduce. input array = ", input_array)
  dprintln(3,"mk_parfor_args_from_reduce. DomainLambda = ", dl)

  # verify the number of input arrays matches the number of input types in dl
  assert(length(dl.inputs) == 2)

  # Get a unique number to embed in generated code for new variables to prevent name conflicts.
  unique_node_id = get_unique_num()

  num_dim_inputs = getArrayNumDims(input_array)
  loopNests = Array(PIRLoopNest, num_dim_inputs)
  if is(input_array_ranges, nothing)
    input_array_ranges = Any[ Expr(:range, 1, 1, mk_arraylen_expr(input_array,i)) for i = 1:num_dim_inputs ]
  end
  assert(length(input_array_ranges) == num_dim_inputs)
  dprintln(3,"input_array_ranges = ", input_array_ranges)
   
  # Create variables to use for the loop indices.
  parfor_index_syms = Symbol[]
  for i = 1:num_dim_inputs
    parfor_index_var = string("parfor_index_", i, "_", unique_node_id)
    parfor_index_sym = symbol(parfor_index_var)
    addStateVar(state,new_var(parfor_index_sym,Int,ISASSIGNED))
    push!(parfor_index_syms, parfor_index_sym)
  end

  # Make sure each input array is a SymbolNode
  # Also, create indexed versions of those symbols for the loop body
  argtyp = typeof(input_array)
  dprintln(3,"mk_parfor_args_from_reduce input_array[1] = ", input_array, " type = ", argtyp)
  assert(argtyp == SymbolNode)

  array_temp_map = Dict{Symbol,SymbolNode}()
  reduce_body = Any[]
  atm = createTempForArray(input_array, 1, state, array_temp_map)
  push!(reduce_body, mk_assignment_expr(atm, mk_arrayref1(input_array, parfor_index_syms, true)))

  # Create an expression to access one element of this input array with index symbols parfor_index_syms
  indexed_array = atm
  #indexed_array = mk_arrayref1(input_array, parfor_index_syms, true)

  # Create empty arrays to hold pre and post statements.
  pre_statements  = Any[]
  post_statements = Any[]
  save_array_lens  = String[]
  input_array_rangeconds = Array(Any, num_dim_inputs)

  # Insert a statement to assign the length of the input arrays to a var
  for i = 1:num_dim_inputs
    save_array_start = string("parallel_ir_save_array_start_", i, "_", unique_node_id)
    save_array_step  = string("parallel_ir_save_array_step_", i, "_", unique_node_id)
    save_array_len   = string("parallel_ir_save_array_len_", i, "_", unique_node_id)
    if isa(input_array_ranges[i], Expr) && is(input_array_ranges[i].head, :range)
      array1_start = mk_assignment_expr(SymbolNode(symbol(save_array_start), Int), input_array_ranges[i].args[1])
      array1_step  = mk_assignment_expr(SymbolNode(symbol(save_array_step), Int), input_array_ranges[i].args[2])
      array1_len   = mk_assignment_expr(SymbolNode(symbol(save_array_len), Int), input_array_ranges[i].args[3])
      input_array_rangeconds[i] = nothing
    elseif isa(input_array_ranges[i], Expr) && is(input_array_ranges[i].head, :tomask)
      assert(length(input_array_ranges[i].args) == 1)
      assert(isa(input_array_ranges[i].args[1], SymbolNode) && DomainIR.isbitarray(input_array_ranges[i].args[1].typ))
      mask_array = input_array_ranges[i].args[1]
      if isa(mask_array, SymbolNode) # a hack to change type to Array{Bool}
        mask_array = SymbolNode(mask_array.name, Array{Bool, mask_array.typ.parameters[1]})
      end
      # TODO: generate dimension check on mask_array
      array1_start = mk_assignment_expr(SymbolNode(symbol(save_array_start), Int), 1)
      array1_step  = mk_assignment_expr(SymbolNode(symbol(save_array_step), Int), 1)
      array1_len   = mk_assignment_expr(SymbolNode(symbol(save_array_len), Int), mk_arraylen_expr(input_array,i))
      input_array_rangeconds[i] = TypedExpr(Bool, :call, TopNode(:unsafe_arrayref), mask_array, SymbolNode(parfor_index_syms[i], Int))
    end 
    # add that assignment to the set of statements to execute before the parfor
    push!(pre_statements,array1_start)
    push!(pre_statements,array1_step)
    push!(pre_statements,array1_len)
    addStateVar(state,new_var(symbol(save_array_start),Int,ISASSIGNEDONCE | ISASSIGNED))
    addStateVar(state,new_var(symbol(save_array_step),Int,ISASSIGNEDONCE | ISASSIGNED))
    addStateVar(state,new_var(symbol(save_array_len),Int,ISASSIGNEDONCE | ISASSIGNED))
    push!(save_array_lens, save_array_len)

    loopNests[num_dim_inputs - i + 1] =
      PIRLoopNest(SymbolNode(parfor_index_syms[i],Int),
                  SymbolNode(symbol(save_array_start), Int),
                  SymbolNode(symbol(save_array_len),Int),
                  SymbolNode(symbol(save_array_step), Int))
  end

  assert(length(dl.outputs) == 1)
  out_type = dl.outputs[1]
  dprintln(3,"mk_parfor_args_from_reduce dl.outputs = ", out_type)
  reduction_output_name  = string("parallel_ir_reduction_output_",unique_node_id)
  reduction_output_snode = SymbolNode(symbol(reduction_output_name), out_type)
  dprintln(3, "Creating variable to hold reduction output = ", reduction_output_snode)
  addStateVar(state,new_var(symbol(reduction_output_name),out_type,ISASSIGNED))
  push!(post_statements, reduction_output_snode)

  # Call Domain IR to generate most of the body of the function (except for saving the output)
  dl_inputs = [reduction_output_snode, atm]
  (max_label, nested_lambda, temp_body) = nested_function_exprs(state.max_label, dl.genBody(dl_inputs), dl, dl_inputs)
  state.max_label = max_label
  assert(isa(temp_body,Array))
  assert(length(temp_body) == 1)
  temp_body = temp_body[1]
  assert(typeof(temp_body) == Expr)
  assert(temp_body.head == :tuple)
  assert(length(temp_body.args) == 1)
  temp_body = temp_body.args[1]
  mergeLambdaIntoOuterState(state, nested_lambda)

  #dprintln(3,"reduce_body = ", reduce_body, " type = ", typeof(reduce_body))
  out_body = [reduce_body, mk_assignment_expr(reduction_output_snode, temp_body)]

  fallthroughLabel = next_label(state)
  condExprs = Any[]
  for i = 1:num_dim_inputs
    if input_array_rangeconds[i] != nothing
      push!(condExprs, Expr(:gotoifnot, input_array_rangeconds[i], fallthroughLabel))
    end
  end
  if length(condExprs) > 0
    out_body = [ condExprs, out_body, LabelNode(fallthroughLabel) ]
  end
  #out_body = TypedExpr(out_type, :call, TopNode(:parallel_ir_reduce), reduction_output_snode, indexed_array)
  dprintln(2,"typeof(out_body) = ",typeof(out_body), " out_body = ", out_body)

  # Compute which scalars and arrays are ever read or written by the body of the parfor
  rws = CompilerTools.ReadWriteSet.from_exprs(out_body, pir_live_cb, nothing)
 
  # Make sure that for reduce that the array indices are all of the simple variety
  if(!simpleIndex(rws.readSet.arrays))
    throw(string("mk_parfor_args_from_reduce readSet arrays are all not simply indexed"))
  end
  if(!simpleIndex(rws.writeSet.arrays))
    throw(string("mk_parfor_args_from_reduce writeSet arrays are all not simply indexed"))
  end
  dprintln(2,rws)

  dprintln(2,"mk_parfor_args_from_reduce with out_type = ", out_type)

  reduce_func = nothing

  dprintln(3,"type of reduce_body = ", typeof(temp_body))
  if typeof(temp_body) == Expr
    dprintln(3,"head of reduce body = ", temp_body.head)

    dprintln(3,"length(reduce_body.args) = ", length(temp_body.args))
    for k = 1:length(temp_body.args)
      dprintln(3,"reduce_body.args[", k, "] = ", temp_body.args[k], " type = ", typeof(temp_body.args[k]))
    end
    if temp_body.head == :call
      dprintln(3,"Found a call")
      if length(temp_body.args) != 3
        throw(string("Non-binary reduction function used."))
      end
      op = temp_body.args[1]

      if op == TopNode(:add_float) || op == TopNode(:add_int)
        reduce_func = :+
      elseif op == TopNode(:mul_float) || op == TopNode(:mul_int)
        reduce_func = :*
      end
    end
  end

  if reduce_func == nothing
    throw(string("Parallel IR only supports + and * reductions right now."))
  end

  makeLhsPrivate(out_body, state)

  new_parfor = IntelPSE.ParallelIR.PIRParForAst(
      out_body,
      pre_statements,
      loopNests,
      [PIRReduction(reduction_output_snode, zero_val, reduce_func)],
      post_statements,
      [DomainOperation(:reduce, input_args)],
      state.top_level_number,
      rws,
      unique_node_id)

  dprintln(3,"array_temp_map = ", array_temp_map)
  dprintln(3,"Lowered parallel IR = ", new_parfor)
#  throw(string("debugging"))

  [new_parfor], out_type
end


# ===============================================================================================================================

# The main routine that converts an mmap! AST node to a parfor AST node.
function mk_parfor_args_from_mmap!(input_args::Array{Any,1}, state)
  # Make sure we get what we expect from domain IR.
  # There should be two entries in the array, another array of input array
  # symbols and a DomainLambda type
  if(length(input_args) < 2)
    throw(string("mk_parfor_args_from_mmap! input_args length should be at least 2 but is ", length(input_args)))
  end

  # First arg is an array of input arrays to the mmap!
  input_arrays = input_args[1]
  len_input_arrays = length(input_args[1])
  dprintln(1,"Number of input arrays: ", len_input_arrays)
  assert(len_input_arrays > 0)

  # Second arg is a DomainLambda
  ftype = typeof(input_args[2])
  dprintln(1,"mk_parfor_args_from_mmap! function = ",input_args[2])
  if(ftype != DomainLambda)
    throw(string("mk_parfor_args_from_mmap! second input_args should be a DomainLambda but is of type ", typeof(input_args[2])))
  end

  # third arg is withIndices
  with_indices = length(input_args) >= 3 ? input_args[3] : false

  indexed_arrays = Any[]

  # Get a unique number to embed in generated code for new variables to prevent name conflicts.
  unique_node_id = get_unique_num()

  first_input = input_arrays[1]
  num_dim_inputs = getArrayNumDims(first_input)
  loopNests = Array(PIRLoopNest, num_dim_inputs)

  # Make the DomainLambda easier to access
  dl::DomainLambda = input_args[2]
  # verify the number of input arrays matches the number of input types in dl
  assert(length(dl.inputs) == len_input_arrays || (with_indices && length(dl.inputs) == num_dim_inputs + len_input_arrays))

  # Create variables to use for the loop indices.
  parfor_index_syms = Symbol[]
  for i = 1:num_dim_inputs
    parfor_index_var = string("parfor_index_", i, "_", unique_node_id)
    parfor_index_sym = symbol(parfor_index_var)
    addStateVar(state,new_var(parfor_index_sym,Int,ISASSIGNED))
    push!(parfor_index_syms, parfor_index_sym)
  end

  # Get the correlation set of the first input array.
  main_length_correlation = getOrAddArrayCorrelation(input_arrays[1].name, state)

  array_temp_map = Dict{Symbol,SymbolNode}()
  out_body = Any[]

  # Make sure each input array is a SymbolNode
  # Also, create indexed versions of those symbols for the loop body
  for i = 1:len_input_arrays
    argtyp = typeof(input_arrays[i])
    dprintln(3,"mk_parfor_args_from_mmap! input_array[i] = ", input_arrays[i], " type = ", argtyp)
    assert(argtyp == SymbolNode)

    atm = createTempForArray(input_arrays[i], 1, state, array_temp_map)
    push!(out_body, mk_assignment_expr(atm, mk_arrayref1(input_arrays[i], parfor_index_syms, true)))

    # Create an expression to access one element of this input array with index symbols parfor_index_syms
    push!(indexed_arrays, atm)

    this_correlation = getOrAddArrayCorrelation(input_arrays[i].name, state)
    # Verify that all the inputs are the same size by verifying they are in the same correlation set.
    if this_correlation != main_length_correlation
      merge_correlations(state, main_length_correlation, this_correlation)
    end
  end

  # Create empty arrays to hold pre and post statements.
  pre_statements  = Any[]
  post_statements = Any[]

  save_array_lens = String[]

  # Insert a statement to assign the length of the input arrays to a var
  for i = 1:num_dim_inputs
    save_array_len = string("parallel_ir_save_array_len_", i, "_", unique_node_id)
    array1_len = mk_assignment_expr(SymbolNode(symbol(save_array_len), Int), mk_arraylen_expr(input_arrays[1],i))
    # add that assignment to the set of statements to execute before the parfor
    push!(pre_statements,array1_len)
    addStateVar(state,new_var(symbol(save_array_len),Int,ISASSIGNEDONCE | ISASSIGNED))
    push!(save_array_lens, save_array_len)

    loopNests[num_dim_inputs - i + 1] =
      PIRLoopNest(SymbolNode(parfor_index_syms[i],Int),
                  1,
                  SymbolNode(symbol(save_array_len),Int),
                  1)
  end

  # add local vars to state
  for (v, d) in dl.locals
    addStateVar(state, new_var(v, d.typ, d.flag))
  end

  dprintln(3,"indexed_arrays = ", indexed_arrays)
  dl_inputs = with_indices ? vcat(indexed_arrays, [SymbolNode(s, Int) for s in parfor_index_syms ]) : indexed_arrays
  dprintln(3,"dl_inputs = ", dl_inputs)
  # Call Domain IR to generate most of the body of the function (except for saving the output)
  (max_label, nested_lambda, nested_body) = nested_function_exprs(state.max_label, dl.genBody(dl_inputs), dl, dl_inputs)
  state.max_label = max_label
  out_body = [out_body, nested_body...]
  dprintln(2,"typeof(out_body) = ",typeof(out_body))
  assert(isa(out_body,Array))
  oblen = length(out_body)
  # the last output of genBody is a tuple of the outputs of the mmap!
  last_body = out_body[oblen]
  assert(typeof(last_body) == Expr)
  lbexpr::Expr = last_body
  assert(lbexpr.head == :tuple)
  assert(length(lbexpr.args) == length(dl.outputs))
  mergeLambdaIntoOuterState(state, nested_lambda)

  dprintln(2,"out_body is of length ",length(out_body))
  printBody(3,out_body)

  out_body = out_body[1:oblen-1]
  array_temp_map2 = Dict{Symbol,SymbolNode}()
  for i = 1:length(dl.outputs)
    tfa = createTempForArray(input_arrays[i], 2, state, array_temp_map2)
    push!(out_body, mk_assignment_expr(tfa, lbexpr.args[i]))
    push!(out_body, mk_arrayset1(input_arrays[i], parfor_index_syms, tfa, true))
  end

  # Compute which scalars and arrays are ever read or written by the body of the parfor
  rws = CompilerTools.ReadWriteSet.from_exprs(out_body, pir_live_cb, nothing)

  # Make sure that for mmap! that the array indices are all of the simple variety
  if(!simpleIndex(rws.readSet.arrays))
    throw(string("mk_parfor_args_from_mmap! readSet arrays are all not simply indexed"))
  end
  if(!simpleIndex(rws.writeSet.arrays))
    throw(string("mk_parfor_args_from_mmap! writeSet arrays are all not simply indexed"))
  end
  dprintln(2,rws)

  # Is there a universal output representation that is generic and doesn't depend on the kind of domain IR input?
  #if(len_input_arrays == 1)
  if length(dl.outputs) == 1
    # If there is only one output then put that output in the post_statements
    push!(post_statements,input_arrays[1])
    out_type = input_arrays[1].typ
  else
    # FIXME: multi mmap! return values are not handled properly below
    tt = Expr(:tuple)
    tt.args = map( x -> x.typ, input_arrays)
    temp_type = eval(tt)

    new_temp_name  = string("parallel_ir_tuple_ret_",get_unique_num())
    new_temp_snode = SymbolNode(symbol(new_temp_name), temp_type)
    out_type = temp_type
    dprintln(3, "Creating variable for tuple return from parfor = ", new_temp_snode)
    addStateVar(state,new_var(symbol(new_temp_name),temp_type,ISASSIGNEDONCE | ISCONST | ISASSIGNED))

    append!(post_statements, [mk_assignment_expr(new_temp_snode, mk_tuple_expr(map( x -> x.name, input_arrays), temp_type)), new_temp_snode])
  end

  dprintln(2,"mk_parfor_args_from_mmap! with out_type = ", out_type)

  makeLhsPrivate(out_body, state)

  new_parfor = IntelPSE.ParallelIR.PIRParForAst(
      out_body,
      pre_statements,
      loopNests,
      PIRReduction[],
      post_statements,
      [DomainOperation(:mmap!, input_args)],
      state.top_level_number,
      rws,
      unique_node_id)
  dprintln(3,"array_temp_map = ", merge(array_temp_map, array_temp_map2))
  dprintln(3,"Lowered parallel IR = ", new_parfor)

  [new_parfor], out_type
end

# ===============================================================================================================================

# The main routine that converts an mmap AST node to a parfor AST node.
function mk_parfor_args_from_mmap(input_args::Array{Any,1}, state)
  # Make sure we get what we expect from domain IR.
  # There should be two entries in the array, another array of input array
  # symbols and a DomainLambda type
  if(length(input_args) != 2)
    throw(string("mk_parfor_args_from_mmap input_args length should be 2 but is ", length(input_args)))
  end

  # First arg is an array of input arrays to the mmap
  input_arrays = input_args[1]
  dprintln(2,"Number of input arrays: ",length(input_arrays))
  assert(length(input_arrays) > 0)

  # handle range selector
  input_array_ranges = nothing
  if isa(input_arrays[1], Expr) && is(input_arrays[1].head, :select)
    input_array_ranges = input_arrays[1].args[2] # range object
    input_arrays[1] = input_arrays[1].args[1]
    assert(isa(input_array_ranges, Expr)) # TODO: may need to handle SymbolNodes in the future
    if input_array_ranges.head == :ranges
      input_array_ranges = input_array_ranges.args
    else
      input_array_ranges = Any[ input_array_ranges ]
    end
  end

  # Second arg is a DomainLambda
  ftype = typeof(input_args[2])
  dprintln(2,"mk_parfor_args_from_mmap function = ",input_args[2])
  if(ftype != DomainLambda)
    throw(string("mk_parfor_args_from_mmap second input_args should be a DomainLambda but is of type ", typeof(input_args[2])))
  end

  # Make the DomainLambda easier to access
  dl::DomainLambda = input_args[2]
  # verify the number of input arrays matches the number of input types in dl
  assert(length(dl.inputs) == length(input_arrays))

  indexed_arrays = Any[]
  input_element_sizes = 0

  # Get a unique number to embed in generated code for new variables to prevent name conflicts.
  unique_node_id = get_unique_num()

  first_input = input_arrays[1]
  num_dim_inputs = getArrayNumDims(first_input)
  loopNests = Array(PIRLoopNest, num_dim_inputs)

  # Create variables to use for the loop indices.
  parfor_index_syms = Symbol[]
  for i = 1:num_dim_inputs
    parfor_index_var = string("parfor_index_", i, "_", unique_node_id)
    parfor_index_sym = symbol(parfor_index_var)
    addStateVar(state,new_var(parfor_index_sym,Int,ISASSIGNED))
    push!(parfor_index_syms, parfor_index_sym)
  end

  # Get the correlation set of the first input array.
  main_length_correlation = getOrAddArrayCorrelation(input_arrays[1].name, state)

  array_temp_map = Dict{Symbol,SymbolNode}()
  out_body = Any[]

  # Make sure each input array is a SymbolNode
  # Also, create indexed versions of those symbols for the loop body
  for(i = 1:length(input_arrays))
    argtyp = typeof(input_arrays[i])
    dprintln(3,"mk_parfor_args_from_mmap input_array[i] = ", input_arrays[i], " type = ", argtyp)
    assert(argtyp == SymbolNode)

    atm = createTempForArray(input_arrays[i], 1, state, array_temp_map)
    push!(out_body, mk_assignment_expr(atm, mk_arrayref1(input_arrays[i], parfor_index_syms, true)))

    # Create an expression to access one element of this input array with index symbols parfor_index_syms
    push!(indexed_arrays,atm)
    # Keep a sum of the size of each arrays individual element sizes.
    input_element_sizes = input_element_sizes + sizeof(argtyp)

    this_correlation = getOrAddArrayCorrelation(input_arrays[i].name, state)
    # Verify that all the inputs are the same size by verifying they are in the same correlation set.
    if this_correlation != main_length_correlation
      merge_correlations(state, main_length_correlation, this_correlation)
    end
  end

  # Create empty arrays to hold pre and post statements.
  pre_statements  = Any[]
  post_statements = Any[]
  # To hold the names of newly created output arrays.
  new_array_symbols = Symbol[]

  save_array_lens = String[]

  # Insert a statement to assign the length of the input arrays to a var
  for i = 1:num_dim_inputs
    save_array_len = string("parallel_ir_save_array_len_", i, "_", unique_node_id)
    array1_len = mk_assignment_expr(SymbolNode(symbol(save_array_len), Int), mk_arraylen_expr(input_arrays[1],i))
    # add that assignment to the set of statements to execute before the parfor
    push!(pre_statements,array1_len)
    addStateVar(state,new_var(symbol(save_array_len),Int,ISASSIGNEDONCE | ISASSIGNED))
    push!(save_array_lens, save_array_len)

    loopNests[num_dim_inputs - i + 1] =
      PIRLoopNest(SymbolNode(parfor_index_syms[i],Int),
                  1,
                  SymbolNode(symbol(save_array_len),Int),
                  1)
  end

  # add local vars to state
  for (v, d) in dl.locals
    addStateVar(state, new_var(v, d.typ, d.flag))
  end

  # Call Domain IR to generate most of the body of the function (except for saving the output)
  (max_label, nested_lambda, nested_body) = nested_function_exprs(state.max_label, dl.genBody(indexed_arrays), dl, indexed_arrays)
  state.max_label = max_label
  out_body = [out_body, nested_body...]
  dprintln(2,"typeof(out_body) = ",typeof(out_body))
  assert(isa(out_body,Array))
  oblen = length(out_body)
  # the last output of genBody is a tuple of the outputs of the mmap
  last_body = out_body[oblen]
  assert(typeof(last_body) == Expr)
  lbexpr::Expr = last_body
  assert(lbexpr.head == :tuple)
  assert(length(lbexpr.args) == length(dl.outputs))
  mergeLambdaIntoOuterState(state, nested_lambda)

  dprintln(2,"out_body is of length ",length(out_body), " ", out_body)

  # To hold the sum of the sizes of the individual output array elements
  output_element_sizes = 0

  out_body = out_body[1:oblen-1]

  # Create each output array
  number_output_arrays = length(dl.outputs)
  for(i = 1:number_output_arrays)
    new_array_name = string("parallel_ir_new_array_name_", unique_node_id, "_", i)
    dprintln(2,"new_array_name = ", new_array_name, " element type = ", dl.outputs[i])
    # create the expression that create a new array and assigns it to a variable whose name is in new_array_name
    if num_dim_inputs == 1
      new_ass_expr = mk_assignment_expr(SymbolNode(symbol(new_array_name), Array{dl.outputs[i],num_dim_inputs}), mk_alloc_array_1d_expr(dl.outputs[i], Array{dl.outputs[i], num_dim_inputs}, symbol(save_array_lens[1])))
    elseif num_dim_inputs == 2
      new_ass_expr = mk_assignment_expr(SymbolNode(symbol(new_array_name), Array{dl.outputs[i],num_dim_inputs}), mk_alloc_array_2d_expr(dl.outputs[i], Array{dl.outputs[i], num_dim_inputs}, symbol(save_array_lens[1]), symbol(save_array_lens[2])))
    else
      throw(string("Only arrays up to two dimensions supported in parallel IR."))
    end
    # remember the array variable as a new variable added to the function and that it is assigned once (the 18)
    addStateVar(state,new_var(symbol(new_array_name),Array{dl.outputs[i],num_dim_inputs},ISASSIGNEDONCE | ISASSIGNED))
    # add the statement to create the new output array to the set of statements to execute before the parfor
    push!(pre_statements,new_ass_expr)
    nans = symbol(new_array_name)
    push!(new_array_symbols,nans)
    nans_sn = SymbolNode(nans, Array{dl.outputs[i], num_dim_inputs})

    tfa = createTempForArray(nans_sn, 1, state, array_temp_map)
    push!(out_body, mk_assignment_expr(tfa, lbexpr.args[i]))
    push!(out_body, mk_arrayset1(nans_sn, parfor_index_syms, tfa, true))

    # keep the sum of the sizes of the individual output array elements
    output_element_sizes = output_element_sizes + sizeof(dl.outputs)
  end
  dprintln(3,"out_body = ", out_body)

  # Compute which scalars and arrays are ever read or written by the body of the parfor
  rws = CompilerTools.ReadWriteSet.from_exprs(out_body, pir_live_cb, nothing)

  # Make sure that for mmap that the array indices are all of the simple variety
  if(!simpleIndex(rws.readSet.arrays))
    throw(string("mk_parfor_args_from_mmap readSet arrays are all not simply indexed"))
  end
  if(!simpleIndex(rws.writeSet.arrays))
    throw(string("mk_parfor_args_from_mmap writeSet arrays are all not simply indexed"))
  end
  dprintln(2,rws)

  # Is there a universal output representation that is generic and doesn't depend on the kind of domain IR input?
  if(number_output_arrays == 1)
    # If there is only one output then put that output in the post_statements
    push!(post_statements,SymbolNode(new_array_symbols[1],Array{dl.outputs[1],num_dim_inputs}))
    out_type = Array{dl.outputs[1],num_dim_inputs}
    state.array_length_correlation[new_array_symbols[1]] = main_length_correlation
  else
    tt = Expr(:tuple)
    tt.args = map( x -> Array{x,num_dim_inputs}, dl.outputs)
    temp_type = eval(tt)

    new_temp_name  = string("parallel_ir_tuple_ret_",get_unique_num())
    new_temp_snode = SymbolNode(symbol(new_temp_name), temp_type)
    dprintln(3, "Creating variable for tuple return from parfor = ", new_temp_snode)
    addStateVar(state,new_var(symbol(new_temp_name),temp_type,ISASSIGNEDONCE | ISCONST | ISASSIGNED))

    append!(post_statements, [mk_assignment_expr(new_temp_snode, mk_tuple_expr(new_array_symbols, temp_type)), new_temp_snode])

    throw(string("mk_parfor_args_from_mmap with multiple output not fully implemented."))
  end

  dprintln(2,"mk_parfor_args_from_mmap with out_type = ", out_type)

  makeLhsPrivate(out_body, state)

  new_parfor = IntelPSE.ParallelIR.PIRParForAst(
      out_body,
      pre_statements,
      loopNests,
      PIRReduction[],
      post_statements,
      [DomainOperation(:mmap, input_args)],
      state.top_level_number,
      rws,
      unique_node_id)

  dprintln(3,"array_temp_map = ", array_temp_map)
  dprintln(3,"Lowered parallel IR = ", new_parfor)

  [new_parfor], out_type
end

# ===============================================================================================================================

function makeLhsPrivateInner(x, state, top_level_number, is_top_level, read)
  if isAssignmentNode(x) || isLoopheadNode(x)
    lhs = x.args[1]
    sname = getSName(lhs)
    red_var_start = "parallel_ir_reduction_output_"
    red_var_len = length(red_var_start)
    sstr = string(sname)
    if length(sstr) >= red_var_len
      if sstr[1:red_var_len] == red_var_start
        return nothing
      end
    end
    makePrivateParfor(sname, state)
  end
  nothing
end

function makeLhsPrivate(body, state)
  for i = 1:length(body)
    AstWalk(body[i], makeLhsPrivateInner, state)
  end
end

# ===============================================================================================================================

function updateAssignState(var_decl, ns)
  var_decl[3] = (var_decl[3] & (~(ISASSIGNED | ISASSIGNEDONCE))) | ns
  var_decl
end

uncompressed_ast(l::LambdaStaticData) =
  isa(l.ast,Expr) ? l.ast : ccall(:jl_uncompress_ast, Any, (Any,Any), l, l.ast)

function count_assignments(x, symbol_assigns, top_level_number, is_top_level, read)
  if isAssignmentNode(x) || isLoopheadNode(x)
    lhs = x.args[1]
    sname = getSName(lhs)
    if !haskey(symbol_assigns, sname)
      symbol_assigns[sname] = 0
    end
    symbol_assigns[sname] = symbol_assigns[sname] + 1
  end
  nothing
end

# :lambda expression
# ast = [ parameters, meta (local, types, etc), body ]
function from_lambda(ast::Array{Any,1}, depth, state)
  dprintln(4,"from_lambda starting")
  assert(length(ast) == 3)
  param = ast[1]
  meta  = ast[2]
  body  = ast[3]

  save_param = state.param
  save_meta  = state.meta
  save_meta2 = state.meta2
  save_meta2_typed = state.meta2_typed

  state.param = param
  state.meta  = meta
  state.meta2 = createVarSet(meta[1])
  state.meta2_typed = createVarDict(meta[2])
  dprintln(3,"state.meta2 = ",state.meta2)
  dprintln(3,"state.meta2_typed = ",state.meta2_typed)
  body = get_one(from_expr(body, depth, state, false))
  dprintln(4,"from_lambda after from_expr")
  ast = Array(Any, 3)
  assert(length(meta) >= 2)
  dprintln(3,"meta[1] = ", meta[1], " type = ", typeof(meta[1]))
  dprintln(3,"meta[2] = ", meta[2], " type = ", typeof(meta[2]))
  meta[1] = Any[]
  meta[2] = Any[]

  symbol_assigns = Dict{Symbol,Int}()
  AstWalk(body, count_assignments, symbol_assigns)

  for i in state.meta2
    dprintln(3, "from_lambda inspecting variable usage for ", i)
    if haskey(symbol_assigns, i)
      num_assigns = symbol_assigns[i]
      dprintln(3, "symbols_assigns = ", num_assigns)
      push!(meta[1], i)
      if num_assigns > 1
        new_assign_state = ISASSIGNED
      else
        new_assign_state = ISASSIGNED | ISASSIGNEDONCE
      end
      push!(meta[2], updateAssignState(state.meta2_typed[i], new_assign_state))
    end
  end
  for i in param
    push!(meta[2], state.meta2_typed[i])
  end

  dprintln(3,"meta[1] = ", meta[1], " type = ", typeof(meta[1]))
  dprintln(3,"meta[2] = ", meta[2], " type = ", typeof(meta[2]))
  ast[1] = param
  ast[2] = meta
  ast[3] = body

  state.param = save_param
  state.meta  = save_meta
  state.meta2 = save_meta2
  state.meta2_typed = save_meta2_typed

#throw(string("testing"))

  dprintln(4,"from_lambda ending")
  return ast
end

# Is a node an assignment expression node.
function isAssignmentNode(node)
  return typeof(node) == Expr && node.head == :(=)
end

# Is a node a loophead expression node (a form of assignment)
function isLoopheadNode(node)
  return typeof(node) == Expr && node.head == :loophead
end

function isBareParfor(node)
  if(typeof(node) == Expr && node.head == :parfor)
      dprintln(4,"Found a bare parfor.")
      return true
  end
  return false
end

# Is a node an assignment expression with a parfor node as the right-hand side.
function isParforAssignmentNode(node)
  dprintln(4,"isParforAssignmentNode")
  dprintln(4,node)

  if isAssignmentNode(node)
      assert(length(node.args) >= 2)
      lhs = node.args[1]
      dprintln(4,lhs)
      rhs = node.args[2]
      dprintln(4,rhs)

      if(typeof(lhs) == Symbol || typeof(lhs) == SymbolNode)
          if(typeof(rhs) == Expr && rhs.head == :parfor)
              dprintln(4,"Found a parfor assignment node.")
              return true
          else
              dprintln(4,"rhs is not a parfor")
          end
      else
          dprintln(4,"lhs is not a SymbolNode")
      end
  else
      dprintln(4,"node is not an Expr")
  end

  return false
end

function getParforNode(node)
  if isBareParfor(node)
    return node.args[1]
  else
    return node.args[2].args[1]
  end
end

# Get the parfor information out of an assignment expression where the parfor is the right-hand side.
function getParforNodeFromAssignment(assignment)
  assignment.args[2].args[1]
end

# Get the right-hand side of an assignment expression.
function getRhsFromAssignment(assignment)
  assignment.args[2]
end

# Get the left-hand side of an assignment expression.
function getLhsFromAssignment(assignment)
  assignment.args[1]
end

function iterations_equals_inputs(node::IntelPSE.ParallelIR.PIRParForAst)
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

function getOutputSet(node::IntelPSE.ParallelIR.PIRParForAst)
  ret = Set(collect(keys(node.rws.writeSet.arrays)))
  dprintln(3,"Output set = ", ret)
  ret
end

function getInputSet(node::IntelPSE.ParallelIR.PIRParForAst)
  ret = Set(collect(keys(node.rws.readSet.arrays)))
  dprintln(3,"Input set = ", ret)
  ret
end

# Get the real outputs of an assignment statement.
# If the assignment expression is normal then the output is just the left-hand side.
# If the assignment expression is augmented with a FusionSentinel then the real outputs
# are the 4+ arguments to the expression.
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

# Return an expression which creates a tuple.
function mk_tuple_expr(tuple_fields, typ)
    # Tuples are formed with a call to :tuple.
    TypedExpr(typ, :call, TopNode(:tuple), tuple_fields...)
end

function nameToSymbolNode(name, sym_to_type)
  SymbolNode(name, sym_to_type[name])
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
      return (new_lhs, [new_lhs], true, new_rhs, [new_rhs])
    end
  end

  lhs_order = SymbolNode[]
  rhs_order = SymbolNode[]
  for i in output_map
    push!(lhs_order, nameToSymbolNode(i[1], sym_to_type))
    push!(rhs_order, nameToSymbolNode(getAliasMap(loweredAliasMap, i[2]), sym_to_type))
  end
  num_map = length(lhs_order)

  # Multiple outputs.

  # First, form the type of the tuple for those multiple outputs.
  tt = Expr(:tuple)
  for i = 1:num_map
    push!(tt.args, rhs_order[i].typ)
  end
  temp_type = eval(tt)

  # Create a new variable to hold the return tuple for this new parfor node.
  new_temp_name  = string("parallel_ir_tuple_ret_",unique_id)
  new_temp_snode = SymbolNode(symbol(new_temp_name), temp_type)
  dprintln(3, "Creating variable for tuple return from parfor = ", new_temp_snode)
  # Add the new var to the list of variables for the function.
  addStateVar(state,new_var(symbol(new_temp_name),temp_type,ISASSIGNEDONCE | ISCONST | ISASSIGNED))

  # Two postParFor statements are needed.  The first to assign a newly created tuple with the outputs
  # into a tuple variable and then in the second to return that tuple variable.
  ( createRetTupleType(lhs_order, unique_id, state), lhs_order, false, [mk_assignment_expr(new_temp_snode, mk_tuple_expr(rhs_order, temp_type)), new_temp_snode], rhs_order)
end

function eliminateStateVar(state, x)
  delete!(state.meta2, x)
  delete!(state.meta2_typed, x)
end

# Add a new variable to the set of variables that get merged into the function's variable list.
function addStateVar(state, x)
  if in(x.name, state.meta2)
    state.meta2_typed[x.name][2] = x.typ
    state.meta2_typed[x.name][3] = x.access_info
    dprintln(3, "addStateVar variable ", x.name, " already exists with type ", state.meta2_typed[x.name])
    return true
  end

  push!(state.meta2, x.name)
  state.meta2_typed[x.name] = [x.name, x.typ, x.access_info]
  dprintln(3,"addStateVar = ", x, " length = ", length(state.meta2_typed))

  if (x.access_info & ISASSIGNEDONCE) != 0
    state.num_var_assignments[x.name] = 2
  else
    state.num_var_assignments[x.name] = 1
  end

  return false
end

function mergeLambdaIntoOuterState(state, inner_lambda :: Expr)
  assert(inner_lambda.head == :lambda)
  inner_meta2 = inner_lambda.args[2][1]
  inner_meta2_typed = inner_lambda.args[2][2]
  inner_typed_dict = createVarDict(inner_meta2_typed)

  for i = 1:length(inner_meta2)
    var = inner_meta2[i]
    vartype = inner_typed_dict[var]

    if !in(var, state.meta2)
      push!(state.meta2, var)
      state.meta2_typed[var] = vartype      
    end
  end
end

# Create a variable for a left-hand side of an assignment to hold the multi-output tuple of a parfor.
function createRetTupleType(rets :: Array{SymbolNode,1}, unique_id, state)
  # Form the type of the tuple var.
  tt = Expr(:tuple)
  tt.args = map( x -> x.typ, rets)
  temp_type = eval(tt)

  new_temp_name  = string("parallel_ir_ret_holder_",unique_id)
  new_temp_snode = SymbolNode(symbol(new_temp_name), temp_type)
  dprintln(3, "Creating variable for multiple return from parfor = ", new_temp_snode)
  addStateVar(state,new_var(symbol(new_temp_name),temp_type,ISASSIGNEDONCE | ISCONST | ISASSIGNED))

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

function getAllAliases(input :: Set, aliases :: Dict{Symbol,Symbol})
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

function isAllocation(expr)
  if typeof(expr) == Expr && expr.head == :call && expr.args[1] == TopNode(:ccall) && (expr.args[2] == QuoteNode(:jl_alloc_array_1d) || expr.args[2] == QuoteNode(:jl_alloc_array_2d))
    return true
  end
  return false
end

# Takes one statement in the preParFor of a parfor and a set of variables that we've determined we can eliminate.
# Returns true if this statement is an allocation of one such variable.
function is_eliminated_allocation_map(x, all_aliased_outputs :: Set)
  eliminated_allocations = Set()
  dprintln(4,"is_eliminated_allocation_map: x = ", x, " typeof(x) = ", typeof(x), " all_aliased_outputs = ", all_aliased_outputs)
  if typeof(x) == Expr
    dprintln(4,"is_eliminated_allocation_map: head = ", x.head)
    if x.head == symbol('=')
      assert(typeof(x.args[1]) == SymbolNode)
      lhs = x.args[1]
      rhs = x.args[2]
      if isAllocation(rhs)
        dprintln(4,"is_eliminated_allocation_map: lhs = ", lhs)
        if !in(lhs.name, all_aliased_outputs)
          push!(eliminated_allocations, lhs.name)
          dprintln(4,"is_eliminated_allocation_map: this will be removed => ", x)
          return true
        end
      elseif typeof(rhs) == SymbolNode
        if in(rhs.name, eliminated_allocations)
          push!(eliminated_allocations, lhs.name)
          dprintln(4,"is_eliminated_allocation_map: this will be removed => ", x)
          return true
        end
      end
    end
  end

  return false
end

# Holds data for modifying arrayset calls.
type sub_arrayset_data
  arrays_set_in_cur_body #remove_arrayset
  output_items_with_aliases
end

function isArrayset(x)
  if x == TopNode(:arrayset) || x == TopNode(:unsafe_arrayset)
    return true
  end
  return false
end

function isArrayref(x)
  if x == TopNode(:arrayref) || x == TopNode(:unsafe_arrayref)
    return true
  end
  return false
end

function isArraysetCall(x)
  if typeof(x) == Expr && x.head == :call && isArrayset(x.args[1])
    return true
  end
  return false
end

function isArrayrefCall(x)
  if typeof(x) == Expr && x.head == :call && isArrayref(x.args[1])
    return true
  end
  return false
end

# Does the work of substitute_arrayset on a node-by-node basis.
function sub_arrayset_walk(x, cbd, top_level_number, is_top_level, read)
  use_dbg_level = 3
  dprintln(use_dbg_level,"sub_arrayset_walk ", x, " ", cbd.arrays_set_in_cur_body, " ", cbd.output_items_with_aliases)

  if typeof(x) == Expr
    dprintln(use_dbg_level,"sub_arrayset_walk is Expr")
    if x.head == :call
      dprintln(use_dbg_level,"sub_arrayset_walk is :call")
      if x.args[1] == TopNode(:arrayset) || x.args[1] == TopNode(:unsafe_arrayset)
        # Here we have a call to arrayset.
        dprintln(use_dbg_level,"sub_arrayset_walk is :arrayset")
        array_name = x.args[2]
        value      = x.args[3]
        index      = x.args[4]
        assert(typeof(array_name) == SymbolNode)
        # If the array being assigned to is in temp_map.
        if in(array_name.name, cbd.arrays_set_in_cur_body)
          return [nothing]
        elseif !in(array_name.name, cbd.output_items_with_aliases)
          return [nothing]
        else
          dprintln(use_dbg_level,"sub_arrayset_walk array_name will not substitute ", array_name)
        end
      end
    end
  end

  nothing
end

# Modify the body of a parfor.
# temp_map holds a map of array names whose arraysets should be turned into a mapped variable instead of the arrayset. a[i] = b. a=>c. becomes c = b
# map_for_non_eliminated holds arrays for which we need to add a variable to save the value but we can't eiminate the arrayset. a[i] = b. a=>c. becomes c = a[i] = b
# map_drop_arrayset drops the arrayset without replacing with a variable.  This is because a variable was previously added here with a map_for_non_eliminated case.
#     a[i] = b. becomes b
function substitute_arrayset(x, arrays_set_in_cur_body, output_items_with_aliases)
  dprintln(3,"substitute_arrayset ", x, " ", arrays_set_in_cur_body, " ", output_items_with_aliases)
  # Walk the AST and call sub_arrayset_walk for each node.
  res = AstWalk(x, sub_arrayset_walk, sub_arrayset_data(arrays_set_in_cur_body, output_items_with_aliases))
  assert(isa(res,Array))
  assert(length(res) == 1)
  res[1]
end

# Get the variable which holds the length of the first input array to a parfor.
function getFirstArrayLens(prestatements, num_dims)
  ret = Any[]

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

# Holds the data for substitute_cur_body AST walk.
type cur_body_data
  temp_map
  index_map
  map_replace_array_temps
  map_replace_array_name
  arrays_set_in_cur_body
  map_for_non_eliminated
  replace_array_name_in_arrayset
end

# Do the work of substitute_cur_body on a node-by-node basis.
function sub_cur_body_walk(x, cbd, top_level_number, is_top_level, read)
  dbglvl = 3
  temp_map  = cbd.temp_map
  index_map = cbd.index_map
  map_replace_array_temps = cbd.map_replace_array_temps
  map_replace_array_name = cbd.map_replace_array_name
  map_for_non_eliminated = cbd.map_for_non_eliminated
  replace_array_name_in_arrayset = cbd.replace_array_name_in_arrayset

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
        assert(typeof(array_name) == SymbolNode)
        lowered_array_name = lowerAlias(map_replace_array_name, array_name.name)
        assert(typeof(lowered_array_name) == Symbol)
        dprintln(dbglvl, "array_name = ", array_name, " index = ", index, " lowered_array_name = ", lowered_array_name)
        # If the array name is in temp_map or map_for_non_eliminated then replace the arrayref call with the mapped variable.
        if haskey(temp_map, lowered_array_name)
          dprintln(dbglvl,"sub_cur_body_walk IS substituting ", temp_map[lowered_array_name])
          return temp_map[lowered_array_name]
        elseif haskey(map_for_non_eliminated, array_name.name)
          dprintln(dbglvl,"sub_cur_body_walk IS substituting non-eliminated ", map_for_non_eliminated[array_name.name])
          return map_for_non_eliminated[array_name.name]
        end
      elseif x.args[1] == TopNode(:arrayset) || x.args[1] == TopNode(:unsafe_arrayset)
        array_name = x.args[2]
        assert(typeof(array_name) == SymbolNode)
        push!(cbd.arrays_set_in_cur_body, lowerAlias(map_replace_array_name, array_name.name))
        if haskey(replace_array_name_in_arrayset, array_name.name)
          x.args[2].name = replace_array_name_in_arrayset[array_name.name]
        end
      end
    end
  elseif xtype == Symbol
    dprintln(dbglvl,"sub_cur_body_walk xtype is Symbol")
    if haskey(index_map, x)
      # Detected the use of an index variable.  Change it to the first parfor's index variable.
      dprintln(dbglvl,"sub_cur_body_walk IS substituting ", index_map[x])
      return index_map[x]
    elseif haskey(map_replace_array_temps, x)
      return map_replace_array_temps[x].name
    end
  elseif xtype == SymbolNode
    dprintln(dbglvl,"sub_cur_body_walk xtype is SymbolNode")
    if haskey(index_map, x.name)
      # Detected the use of an index variable.  Change it to the first parfor's index variable.
      dprintln(dbglvl,"sub_cur_body_walk IS substituting ", index_map[x.name])
      x.name = index_map[x.name]
      return x
    elseif haskey(map_replace_array_name, x.name)
      dprintln(dbglvl,"sub_cur_body_walk IS substituting from map_replace_array_name ", map_replace_array_name[x.name], " for ", x.name)
      x.name = map_replace_array_name[x.name]
      return x
    elseif haskey(map_replace_array_temps, x.name)
      dprintln(dbglvl,"sub_cur_body_walk IS substituting from map_replace_array_temps ", map_replace_array_temps[x.name], " for ", x.name)
      x.name = map_replace_array_temps[x.name].name
      return x
    end
  end
  dprintln(dbglvl,"sub_cur_body_walk not substituting")
  nothing
end

# Make changes to the second parfor body in the process of parfor fusion.
# temp_map holds array names for which arrayrefs should be converted to a variable.  a[i].  a=>b. becomes b
# map_for_non_eliminated has the same behavior as temp_map but comes from modifications to the first parfor in the fusion where the arrayset isn't eliminated.
# index_map holds maps between index variables.  The second parfor is modified to use the index variable of the first parfor.
function substitute_cur_body(x, temp_map, index_map, map_replace_array_temps, map_replace_array_name, arrays_set_in_cur_body, map_for_non_eliminated, replace_array_name_in_arrayset)
  dprintln(3,"substitute_cur_body ", x)
  dprintln(3,"temp_map = ", temp_map)
  dprintln(3,"index_map = ", index_map)
  dprintln(3,"map_replace_array_temps = ", map_replace_array_temps)
  dprintln(3,"map_replace_array_name = ", map_replace_array_name)
  dprintln(3,"arrays_set_in_cur_body = ", arrays_set_in_cur_body)
  dprintln(3,"map_for_non_eliminated = ", map_for_non_eliminated)
  dprintln(3,"replace_array_name_in_array_set = ", replace_array_name_in_arrayset)
  # Walk the AST and call sub_cur_body_walk for each node.
  res = DomainIR.AstWalk(x, sub_cur_body_walk, cur_body_data(temp_map, index_map, map_replace_array_temps, map_replace_array_name, arrays_set_in_cur_body, map_for_non_eliminated, replace_array_name_in_arrayset))
  assert(isa(res,Array))
  assert(length(res) == 1)
  res[1]
end

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

# Does the work of substitute_arraylen on a node-by-node basis.
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
  nothing
end

# Modify the preParFor statements to remove calls to arraylen on eliminated arrays.
# Replace with the variable holding the length of the first input array.
function substitute_arraylen(x, replacement)
  dprintln(3,"substitute_arraylen ", x, " ", replacement)
  # Walk the AST and call sub_arraylen_walk for each node.
  res = DomainIR.AstWalk(x, sub_arraylen_walk, replacement)
  assert(isa(res,Array))
  assert(length(res) == 1)
  res[1]
end

function update_alias_walk(x, inverse_aliases, top_level_number, is_top_level, read)
  dprintln(3, "update_alias_walk ", x)
  if isAssignmentNode(x)
    dprintln(3, "update_alias_walk found assignment node")
    lhs = x.args[1]
    assert(typeof(lhs) == SymbolNode)
    rhs = x.args[2]
    if isAllocation(rhs)
      dprintln(3, "update_alias_walk found allocation")
      while haskey(inverse_aliases, lhs.name)
        lhs.name = inverse_aliases[lhs.name]
      end
      x.args[1] = lhs
      return x
    end
  end
  dprintln(3, "update_alias_walk not substituting")
  nothing
end

function update_alias(x, inverse_aliases)
  dprintln(3,"update_alias ", x, " ", inverse_aliases)
  # Walk the AST and call sub_arraylen_walk for each node.
  res = DomainIR.AstWalk(x, update_alias_walk, inverse_aliases)
  assert(isa(res,Array))
  assert(length(res) == 1)
  res[1]
end

fuse_limit = -1
#fuse_limit = 0
function PIRSetFuseLimit(x)
  global fuse_limit = x
end

rearrange_passes = 2
function PIRNumSimplify(x)
  global rearrange_passes = x
end

# Add to the map of symbol names to types.
function rememberTypeForSym(sym_to_type, sym, typ)
  assert(typ != Any)
  sym_to_type[sym] = typ
end

# Just used to hold a spot in an array to indicate the this is a special assignment expression with embedded real array output names from a fusion.
type FusionSentinel
end

# Check if an assignement is a fusion assignment.
function isFusionAssignment(x)
  assert(typeof(x) == Expr)
  if x.head != symbol('=')
    return false
  elseif length(x.args) <= 2
    return false
  else
    assert(typeof(x.args[3]) == FusionSentinel)
    return true
  end
end

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

function getParforCorrelation(parfor, state)
  if length(parfor.preParFor) == 0
    return nothing
  end
  # FIXME: is this reliable?? -- PaulLiu
  for first_stmt in parfor.preParFor
    # first_stmt = parfor.preParFor[1]
    if (typeof(first_stmt) == Expr) && (first_stmt.head == symbol('='))
      rhs = first_stmt.args[2]
      if (typeof(rhs) == Expr) && (rhs.head == :call) && (rhs.args[1] == TopNode(:arraysize)) && (typeof(rhs.args[2]) == SymbolNode)
        # state.array_length_correlation[rhs.args[2].name]
        dprintln(3,"Getting parfor array correlation for array = ", rhs.args[2].name)
        return getOrAddArrayCorrelation(rhs.args[2].name, state) 
      end
    end
  end
  assert(false)
end

function createMapLhsToParfor(parfor_assignment, the_parfor, is_multi, sym_to_type)
  map_lhs_post_array     = Dict{Symbol,Symbol}()
  map_lhs_post_reduction = Dict{Symbol,Symbol}()

  if is_multi
    for i = 4:length(parfor_assignment.args)
      assert(typeof(parfor_assignment.args[i]) == SymbolNode)
      dprintln(3,"Remembering type for parfor_assignment sym ", parfor_assignment.args[i].name, "=>", parfor_assignment.args[i].typ)
      rememberTypeForSym(sym_to_type, parfor_assignment.args[i].name, parfor_assignment.args[i].typ)
      rememberTypeForSym(sym_to_type, the_parfor.postParFor[end-1].args[2].args[i-2].name, the_parfor.postParFor[end-1].args[2].args[i-2].typ)
      if parfor_assignment.args[i].typ.name == Array.name
        # For fused parfors, the last post statement is a tuple variable.
        # That tuple variable is declared in the previous statement (end-1).
        # The statement is an Expr with head == :call and top(:tuple) as the first arg.
        # So, the first member of the tuple is at offset 2 which corresponds to index 4 of this loop, ergo the "i-2".
        map_lhs_post_array[parfor_assignment.args[i].name] = the_parfor.postParFor[end-1].args[2].args[i-2].name
      else
        map_lhs_post_reduction[parfor_assignment.args[i].name] = the_parfor.postParFor[end-1].args[2].args[i-2].name
      end
    end
  else
    if !isBareParfor(parfor_assignment)
      lhs_pa = getLhsFromAssignment(parfor_assignment)
      if typeof(lhs_pa) == SymbolNode
        assert(typeof(the_parfor.postParFor[end]) == SymbolNode)
        rememberTypeForSym(sym_to_type, lhs_pa.name, lhs_pa.typ)
        rhs = the_parfor.postParFor[end]
        rememberTypeForSym(sym_to_type, rhs.name, rhs.typ)

        if lhs_pa.typ.name == Array.name
          map_lhs_post_array[lhs_pa.name] = the_parfor.postParFor[end].name
        else
          map_lhs_post_reduction[lhs_pa.name] = the_parfor.postParFor[end].name
        end
      elseif typeof(lhs_pa) == Symbol
        throw(string("lhs_pa as a symbol not longer supported"))
      else
        dprintln(3,"typeof(lhs_pa) = ", typeof(lhs_pa))
        assert(false)
      end
    end
  end

  map_lhs_post_array, map_lhs_post_reduction
end

function lowerAlias(dict, input :: Symbol)
  if haskey(dict, input)
    return dict[input]
  else
    return input
  end
end

function fullyLowerAlias(dict, input :: Symbol)
  while haskey(dict, input)
    input = dict[input]
  end
  input
end

function createLoweredAliasMap(dict1)
  ret = Dict{Symbol,Symbol}()

  for i in dict1
    assert(typeof(i[1]) == Symbol)
    ret[i[1]] = fullyLowerAlias(dict1, i[2])
  end

  ret
end

run_as_tasks = 0
function PIRRunAsTasks(x)
  global run_as_tasks = x
end

# Test whether we can fuse the two most recent parfor statements and if so to perform that fusion.
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

  sym_to_type   = Dict{Symbol,DataType}()

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
  in_correlation  = state.array_length_correlation[first_in]
  dprintln(3,"first_in = ", first_in)
  dprintln(3,"Fusion correlations ", out_correlation, " ", in_correlation)

  is_prev_multi = isFusionAssignment(prev)
  is_cur_multi  = isFusionAssignment(cur)

  prev_num_dims = length(prev_parfor.loopNests)
  cur_num_dims  = length(cur_parfor.loopNests)

  map_prev_lhs_post, map_prev_lhs_reduction = createMapLhsToParfor(prev, prev_parfor, is_prev_multi, sym_to_type)
  map_prev_lhs_all = merge(map_prev_lhs_post, map_prev_lhs_reduction)
  map_cur_lhs_post,  map_cur_lhs_reduction  = createMapLhsToParfor(cur,  cur_parfor,  is_cur_multi,  sym_to_type)
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
    !reduction_var_used
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
    dprintln(2,"live_in_prev = ", live_in_prev)

    # Get the variables live after the previous parfor.
    live_out_prev = prev_stmt_live_last.live_out
    dprintln(2,"live_out_prev = ", live_out_prev)

    # Get the live variables into the current parfor.
    live_in_cur   = cur_stmt_live.live_in
    dprintln(2,"live_in_cur = ", live_in_cur)

    # Get the variables live after the current parfor.
    live_out_cur  = cur_stmt_live.live_out
    dprintln(2,"live_out_cur = ", live_out_cur)

    new_in_prev = setdiff(live_out_prev, live_in_prev)
    new_in_cur  = setdiff(live_out_cur,  live_in_cur)
    dprintln(3,"new_in_prev = ", new_in_prev)
    dprintln(3,"new_in_cur = ", new_in_cur)

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
    output_map = Dict{Symbol,Symbol}()
    for i in map_prev_lhs_all
      if !in(i[1], not_used_after_cur)
        output_map[i[1]] = i[2]
      end
    end
    for i in map_cur_lhs_all
      output_map[i[1]] = i[2]
    end

    new_aliases = Dict{Symbol,Symbol}()
    for i in map_prev_lhs_post
      if !in(i[1], not_used_after_cur)
        new_aliases[i[1]] = i[2]
      end
    end

    outputs = collect(values(output_map))
    dprintln(3,"output_map = ", output_map)
    dprintln(3,"new_aliases = ", new_aliases)

    first_arraylen = getFirstArrayLens(prev_parfor.preParFor, prev_num_dims)

    # Merge each part of the two parfor nodes.

    # loopNests - nothing needs to be done to the loopNests
    # But we use them to establish a mapping between corresponding indices in the two parfors.
    # Then, we use that map to convert indices in the second parfor to the corresponding ones in the first parfor.
    index_map = Dict{Symbol,Symbol}()
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
    (new_lhs, all_rets, single, merged_output, output_items) = create_merged_output_from_map(output_map, unique_id, state, sym_to_type, loweredAliasMap)
# This is an old version.  Can be removed once we're confident the newer version is better.
#    output_items_set = Set()
#    for i in output_items
#      push!(output_items_set, i.name)
#    end
    output_items_set = live_out_cur
    output_items_with_aliases = getAllAliases(output_items_set, prev_parfor.array_aliases)

    dprintln(3,"output_items_set = ", output_items_set)
    dprintln(3,"output_items_with_aliases = ", output_items_with_aliases)

    # Create a dictionary of arrays to the last variable containing the array's value at the current index space.
    save_body = prev_parfor.body
    arrayset_dict = Dict{Symbol,SymbolNode}()
    for i = 1:length(save_body)
      x = save_body[i]
      if isArraysetCall(x)
        # Here we have a call to arrayset.
        array_name = x.args[2]
        value      = x.args[3]
        assert(typeof(array_name) == SymbolNode)
        assert(typeof(value) == SymbolNode)
        arrayset_dict[array_name.name] = value
      elseif typeof(x) == Expr && x.head == :(=)
        lhs = x.args[1]
        rhs = x.args[2]
        assert(typeof(lhs) == SymbolNode)
        if isArrayrefCall(rhs)
          array_name = rhs.args[2]
          assert(typeof(array_name) == SymbolNode)
          arrayset_dict[array_name.name] = lhs
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
    empty_dict = Dict{Symbol,Symbol}()
    arrays_set_in_cur_body = Set()
    # Convert the cur_parfor body.
    new_cur_body = map(x -> substitute_cur_body(x, arrayset_dict, index_map, empty_dict, empty_dict, arrays_set_in_cur_body, empty_dict, loweredAliasMap), cur_parfor.body)
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
    prev_parfor.preParFor = [ filter(x -> !is_eliminated_allocation_map(x, output_items_with_aliases), prev_parfor.preParFor),
                              map(x -> substitute_arraylen(x,first_arraylen) , filter(x -> !is_eliminated_arraylen(x), cur_parfor.preParFor)) ]
    dprintln(2,"New preParFor = ", prev_parfor.preParFor)

    # reductions - a simple append with the caveat that you can't fuse parfor where the first has a reduction that the second one uses
    # need to check this caveat above.
    append!(prev_parfor.reductions, cur_parfor.reductions)
    dprintln(2,"New reductions = ", prev_parfor.reductions)

    dprintln(3,"merged_output = ", merged_output)
    if is_prev_multi
      num_prev_sub = 2
    else
      num_prev_sub = 1
    end
    if is_cur_multi
      num_cur_sub = 2
    else
      num_cur_sub = 1
    end
    prev_parfor.postParFor = [ prev_parfor.postParFor[1:end-num_prev_sub], cur_parfor.postParFor[1:end-num_cur_sub], merged_output ]
    dprintln(2,"New postParFor = ", prev_parfor.postParFor, " typeof(postParFor) = ", typeof(prev_parfor.postParFor), " ", typeof(prev_parfor.postParFor[end]))

    # original_domain_nodes - simple append
    append!(prev_parfor.original_domain_nodes, cur_parfor.original_domain_nodes)
    dprintln(2,"New domain nodes = ", prev_parfor.original_domain_nodes)

    # top_level_number - what to do here? is this right?
    push!(prev_parfor.top_level_number, cur_parfor.top_level_number[1])

    # rws
    prev_parfor.rws = CompilerTools.ReadWriteSet.from_exprs(prev_parfor.body, pir_live_cb, nothing)

    dprintln(3,"New lhs = ", new_lhs)
    if prev_assignment
      # The prev parfor was of the form "var = parfor(...)".
      if new_lhs != nothing
        dprintln(2,"prev was assignment and is staying an assignment")
        # The new lhs is not empty and so this is the normal case where "prev" stays an assignment expression and we update types here and if necessary FusionSentinel.
        prev.args[1] = new_lhs
        prev.typ = new_lhs.typ
        prev.args[2].typ = new_lhs.typ
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
        body[body_index] = mk_assignment_expr(new_lhs, prev)
        prev = body[body_index]
        prev.args[2].typ = new_lhs.typ
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

    return true
  else
    dprintln(3, "Fusion could not happen here.")
  end

  return false

  false
end

# Get the name of a symbol whether the input is a Symbol or SymbolNode.
function getSName(ssn)
  stype = typeof(ssn)
  if stype == Symbol
    return ssn
  elseif stype == SymbolNode
    return ssn.name
  elseif stype == Expr && ssn.head == :(::)
    return ssn.args[1]
  end

  dprintln(0, "getSName ssn = ", ssn, " stype = ", stype)
  if stype == Expr
    dprintln(0, "ssn.head = ", ssn.head)
  end
  throw(string("getSName called with something of type ", stype))
end

type TaskGraphSection
  start_body_index :: Int
  end_body_index :: Int
  exprs :: Array{Any,1}
end

# sequence of expressions
# ast = [ expr, ... ]
function from_exprs(ast::Array{Any,1}, depth, state)
  # Is this the first node in the AST with an array of expressions, i.e., is it the top-level?
  top_level = (state.top_level_number == 0)
  if top_level
    return top_level_from_exprs(ast, depth, state)
  else
    return intermediate_from_exprs(ast, depth, state)
  end
end

# sequence of expressions
# ast = [ expr, ... ]
function intermediate_from_exprs(ast::Array{Any,1}, depth, state)
  len  = length(ast)
  body = Any[]

  for i = 1:len
    dprintln(2,"Processing ast #",i," depth=",depth)

    # Convert the current expression.
    new_exprs = from_expr(ast[i], depth, state, false)
    assert(isa(new_exprs,Array))

    append!(body, new_exprs)
  end

  return body
end

function getNonBlock(head_preds, back_edge)
  assert(length(head_preds) == 2)

  for i in head_preds
    if i.label != back_edge
      return i
    end
  end

  throw(string("All entries in head preds list were back edges."))
end

type ReplacedRegion
  start_index
  end_index
  bb
  tasks
end

type TaskInfo
  task_func :: Function
  function_sym
  join_func :: String
  input_symbols   :: Array{SymbolNode,1}
  modified_inputs :: Array{SymbolNode,1}
  io_symbols      :: Array{SymbolNode,1}
  reduction_vars  :: Array{SymbolNode,1}
  code
  loopNests       :: Array{PIRLoopNest,1}      # holds information about the loop nests
end

# translated to pert_range_Nd_t
type pir_range
  dim :: Int
  lower_bounds :: Array{Any,1}
  upper_bounds :: Array{Any,1}
  function pir_range()
    new(0, Any[], Any[])
  end
end

#actual_type = Int64
#actual_type = Uint64

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

type pir_array_access_desc
  dim_info :: Array{pir_aad_dim, 1}
  row_major :: Bool

  function pir_array_access_desc()
    new(pir_aad_dim[],false)
  end
end

function create1D_array_access_desc(array :: SymbolNode)
  ret = pir_array_access_desc()
  push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 1), 1, 0, 0, 0 ))
  ret
end

function create2D_array_access_desc(array :: SymbolNode)
  ret = pir_array_access_desc()
  push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 1), 1, 0, 0, 0 ))
  push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 2), 1, 0, 0, 0 ))
  ret
end

function create_array_access_desc(array :: SymbolNode)
  if array.typ.parameters[2] == 1
    return create1D_array_access_desc(array)
  elseif array.typ.parameters[2] == 2
    return create2D_array_access_desc(array)
  else
    throw(string("Greater than 2D arrays not supported in create_array_access_desc."))
  end
end

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

type InsertTaskNode
  ranges :: pir_range
  args   :: Array{pir_arg_metadata,1}
  task_func :: Any
  join_func :: String  # empty string for no join function
  task_options :: Int
  host_grain_size :: pir_grain_size
  phi_grain_size :: pir_grain_size

  function InsertTaskNode()
    new(pir_range(), pir_arg_metadata[], nothing, string(""), 0, pir_grain_size(), pir_grain_size())
  end
end

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

flat_parfor = 1
function PIRFlatParfor(x)
  global flat_parfor = x
end

pre_eq = 1
function PIRPreEq(x)
  global pre_eq = x
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
  if num_threads_mode != x
    ccall(:set_j2c_num_threads_mode, Void, (Cint,), x)
  end
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

function taskableParfor(node)
  dprintln(3,"taskableParfor for: ", node)
  if limit_task == 0
    dprintln(3,"task limit exceeded so won't convert parfor to task")
    return false
  end
  if isParforAssignmentNode(node) || isBareParfor(node)
    dprintln(3,"Found parfor node, stencil: ", stencil_tasks, " reductions: ", reduce_tasks)
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
  non_calls :: Float64    # estimated instruction count for non-calls
  fully_analyzed :: Bool
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

function call_instruction_count(args, state, debug_level)
  func  = args[1]
  fargs = args[2:end]

  sig_expr = Expr(:tuple)
  sig_expr.args = map(x -> DomainIR.typeOfOpr(x), fargs)
  signature = eval(sig_expr)
  fs = (func, signature)

  if haskey(call_costs, fs)
    res = call_costs[fs]
    if res == nothing
      dprintln(debug_level, "Didn't process call to function ", func, " ", signature, " ", args)
      state.fully_analyzed = false
      return nothing
    end
  else
    res = generate_instr_count(func, signature)
    call_costs[fs] = res
    if res == nothing
      dprintln(debug_level, "Didn't process call to function ", func, " ", signature, " ", args)
      state.fully_analyzed = false
      return nothing
    end
  end

  assert(typeof(res) == Float64)
  state.non_calls = state.non_calls + res
  return nothing
end

function generate_instr_count(function_name, signature)
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
    dprintln(3,"Calling getfield")
    function_name = getfield(function_name.mod, function_name.name)
  elseif ftyp == IntrinsicFunction
    dprintln(3, "generate_instr_count: found IntrinsicFunction = ", function_name)
    call_costs[(function_name, signature)] = nothing
    return call_costs[(function_name, signature)]
  end

  if typeof(function_name) != Function
    dprintln(3, "generate_instr_count: function_name is not a Function = ", function_name)
    call_costs[(function_name, signature)] = nothing
    return call_costs[(function_name, signature)]
  end

  m = methods(function_name, signature)
  if length(m) < 1
    return nothing
#    error("Method for ", function_name, " with signature ", signature, " is not found")
  end

  ct = code_typed(function_name, signature)      # get information about code for the given function and signature

  dprintln(2,"generate_instr_count ", function_name, " ", signature)
  state = eic_state(0, true)
  AstWalk(ct[1], estimateInstrCount, state)
  dprintln(2,"instruction count estimate for parfor = ", state)
  if state.fully_analyzed
    call_costs[(function_name, signature)] = state.non_calls
  else
    call_costs[(function_name, signature)] = nothing
  end
  return call_costs[(function_name, signature)]
end

function estimateInstrCount(ast, state, top_level_number, is_top_level, read)
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
      # skip
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
  elseif asttyp == Nothing
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

  nothing
end

function createInstructionCountEstimate(the_parfor :: IntelPSE.ParallelIR.PIRParForAst)
  if num_threads_mode == 1 || num_threads_mode == 2 || num_threads_mode == 3
    dprintln(2,"instruction count estimate for parfor = ", the_parfor)
    state = eic_state(0, true)
    for i = 1:length(the_parfor.body)
      AstWalk(the_parfor.body[i], estimateInstrCount, state)
    end
    if state.fully_analyzed
      the_parfor.instruction_count_expr = state.non_calls
    else
      the_parfor.instruction_count_expr = nothing
    end
    dprintln(2,"instruction count estimate for parfor = ", the_parfor.instruction_count_expr)
  end
end

type RhsDead
end

# Task Graph Modes
SEQUENTIAL_TASKS = 1
ONE_AT_A_TIME = 2
MULTI_PARFOR_SEQ_NO = 3

task_graph_mode = ONE_AT_A_TIME
function PIRTaskGraphMode(x)
  global task_graph_mode = x
end

# TOP_LEVEL
# sequence of expressions
# ast = [ expr, ... ]
function top_level_from_exprs(ast::Array{Any,1}, depth, state)
  len  = length(ast)
  body = Any[]
  pre_next_parfor = Any[]
  fuse_number = 1

  main_proc_start = time_ns()

  # Process the top-level expressions of a function and do fusion and useless assignment elimination.
  for i = 1:len
    # Record the top-level statement number in the processing state.
    state.top_level_number = i
    dprintln(2,"Processing top-level ast #",i," depth=",depth)

    # Convert the current expression.
    new_exprs = from_expr(ast[i], depth, state, true)
    assert(isa(new_exprs,Array))
    # If conversion of current statement resulted in anything.
    if length(new_exprs) != 0
      # If this isn't the first statement processed that created something.
      if length(body) != 0
        last_node = body[end]
        dprintln(3, "Should fuse?")
        dprintln(3, "new = ", new_exprs[1])
        dprintln(3, "last = ", last_node)

        # See if the previous expression is a parfor.
        is_last_parfor = isParforAssignmentNode(last_node)    || isBareParfor(last_node)
        # See if the new expression is a parfor.
        is_new_parfor  = isParforAssignmentNode(new_exprs[1]) || isBareParfor(new_exprs[1])
        dprintln(3,"is_new_parfor = ", is_new_parfor, " is_last_parfor = ", is_last_parfor)

        if is_last_parfor && !is_new_parfor
          simple = false
          for j = 1:length(new_exprs)
            e = new_exprs[j]
            if isa(e, Expr) && is(e.head, :(=)) && isa(e.args[2], Expr) && (e.args[2].args[1] == TopNode(:box))
              dprintln(3, "box operation detected")
              simple = true
            else
              simple = false
              break
            end
          end
          if simple
            dprintln(3, "insert into pre_next_parfor")
            append!(pre_next_parfor, new_exprs)
            continue
          end
        end

        # If both are parfors then try to fuse them.
        if is_new_parfor && is_last_parfor
          dprintln(3,"Starting fusion ", fuse_number)
          new_exprs[1]
          fuse_number = fuse_number + 1
          if length(pre_next_parfor) > 0
            dprintln(3, "prepend statements to new parfor: ", pre_next_parfor)
            new_parfor = getParforNode(new_exprs[1])
            new_parfor.preParFor = [ pre_next_parfor, new_parfor.preParFor ]
          end
          if fuse(body, length(body), new_exprs[1], state)
            pre_next_parfor = Any[]
            # If fused then the new node is consumed and no new node is added to the body.
            continue
          end
        end

        new_exprs = [ pre_next_parfor, new_exprs ]
        pre_next_parfor = Any[]
        # Do this transformation:  a = ...; b = a; becomes b = ...
        # Eliminate the variable a if it is never used again.
        for expr in new_exprs
          if isAssignmentNode(last_node) &&
             isAssignmentNode(expr)
            # Detected two assignments in a row.
            new_lhs  = expr.args[1]    # The left-hand side of the second assignment.
            new_rhs  = expr.args[2]    # The right-hand side of the second assignment.
            nrhstype = typeof(new_rhs)         # The type of the right-hand side of the second assignment.
            # Find the liveness information for the second assignment.
            new_stmt_lives = CompilerTools.LivenessAnalysis.find_top_number(state.top_level_number, state.block_lives)
            # Get the live out information for the second assignment.
            new_stmt_live_out = new_stmt_lives.live_out
            dprintln(3,"Found two assignments in a row.")
            dprintln(3,"new_lhs  = ", new_lhs)
            dprintln(3,"new_rhs  = ", new_rhs)
            dprintln(3,"nrhstype = ", nrhstype)
            dprintln(3,"lives = ", new_stmt_lives)
            dprintln(3,"lives.out = ", new_stmt_lives.live_out)

            # If the right-hand side is a simple symbol and that symbol isn't used after this statement.
            if nrhstype == SymbolNode && !in(getSName(new_rhs), new_stmt_live_out)
              # How we remove a depends on where a is...whether embedded in a fusion assignment or by itself.
              if isFusionAssignment(last_node)
                dprintln(3,"Last node is a fusion assignment.")
                removed_assignment = false
                # The left-hand side of the previous assignment is a Fusion assignment node.
                # So, look at the real output names stored in that node.
                for i = 4:length(last_node.args)
                  assert(typeof(last_node.args[i]) == SymbolNode)
                  dprintln(3,"Testing against ", last_node.args[i])
                  if getSName(last_node.args[i]) == getSName(new_rhs)
                    dprintln(3,"Removing an unnecessary assignment statement in a fusion extended assignment.")
                    # Found the variable to replace in the real output list so replace it.
                    eliminateStateVar(state, last_node.args[i].name)
                    last_node.args[i].name = getSName(new_lhs)
                    removed_assignment = true
                    break
                  end
                end
                if removed_assignment
                  # If we found an item to remove then update the parfor nodes top-level list to encompass this assignment.
                  # This is really a fusion of an assignment statement into a parfor.
                  last_parfor = getParforNode(last_node)
                  push!(last_parfor.top_level_number, state.top_level_number)
                  continue
                end
              else
                # Left-hand side of previous assignment was simple.
                prev_lhs = last_node.args[1]
                dprintln(3,"prev_lhs = ", prev_lhs)

                # If the names match...
                if getSName(prev_lhs) == getSName(new_rhs)
                  dprintln(3,"Removing an unnecessary assignment statement.")
                  # ...replace with the left-hand side of the second assignment.
                  last_node.args[1] = new_lhs
                  eliminateStateVar(state, getSName(prev_lhs))
                  continue
                end
              end
            end
          end
          push!(body, expr)
          last_node = expr
        end
      else
        append!(body, new_exprs)
      end
    end
  end

  dprintln(1,"Main parallel conversion loop time = ", ns_to_sec(time_ns() - main_proc_start))

  dprintln(3,"Body after first pass before task graph creation.")
  for j = 1:length(body)
    dprintln(3,body[j])
  end

  expanded_body = Any[]

  # TASK GRAPH

  if polyhedral != 0
    # Anand: you can insert code here.
  end

  rr = ReplacedRegion[]

  expand_start = time_ns()

  # Remove the pre-statements from parfor nodes and expand them into the top-level expression array.
  for i = 1:length(body)
    if isParforAssignmentNode(body[i])
      parfor_assignment = body[i]
      dprintln(3,"Expanding a parfor assignment node")

      the_parfor = getParforNodeFromAssignment(parfor_assignment)
      lhs = getLhsFromAssignment(parfor_assignment)
      rhs = getRhsFromAssignment(parfor_assignment)

      # Add all the pre-parfor statements to the expanded body.
      append!(expanded_body, the_parfor.preParFor)
      the_parfor.preParFor = Any[]

      # Add just the parfor to the expanded body.  The post-parfor part removed below.
      assert(typeof(rhs) == Expr)
      rhs.typ = typeof(0) # a few lines down you can see that the last post-statement of 0 is added.
      push!(expanded_body, rhs)

      # All the post parfor part to the expanded body.
      # The regular part of the post parfor is just added.
      # The part that indicates the return values creates individual assignment statements for each thing returned.
      if isFusionAssignment(parfor_assignment)
        append!(expanded_body, the_parfor.postParFor[1:end-2])
        for j = 4:length(parfor_assignment.args)
          push!(expanded_body, mk_assignment_expr(parfor_assignment.args[j], the_parfor.postParFor[end-1].args[2].args[j-2]))
        end
      else
        append!(expanded_body, the_parfor.postParFor[1:end-1])
        push!(expanded_body, mk_assignment_expr(lhs, the_parfor.postParFor[end]))
      end
      the_parfor.postParFor = Any[]
      push!(the_parfor.postParFor, 0)
      createInstructionCountEstimate(the_parfor)
    elseif isBareParfor(body[i])
      rhs = body[i]
      the_parfor = rhs.args[1]

      # Add all the pre-parfor statements to the expanded body.
      append!(expanded_body, the_parfor.preParFor)
      the_parfor.preParFor = Any[]

      # Add just the parfor to the expanded body.  The post-parfor part removed below.
      assert(typeof(rhs) == Expr)
      rhs.typ = typeof(0) # a few lines down you can see that the last post-statement of 0 is added.
      push!(expanded_body, rhs)

      the_parfor.postParFor = Any[]
      push!(the_parfor.postParFor, 0)
      createInstructionCountEstimate(the_parfor)
    else
      push!(expanded_body, body[i])
    end
  end

  dprintln(1,"Expanding parfors time = ", ns_to_sec(time_ns() - expand_start))

  body = expanded_body

  dprintln(3,"expanded_body = ")
  for j = 1:length(body)
    dprintln(3,body[j])
  end

  fake_body = lambdaFromStmtsMeta(body, state.param, state.meta)
  dprintln(3,"fake_body = ", fake_body)
  new_lives = CompilerTools.LivenessAnalysis.from_expr(fake_body, pir_live_cb, nothing)
  dprintln(1,"Starting loop analysis.")
  loop_info = CompilerTools.Loops.compute_dom_loops(new_lives.cfg)
  dprintln(1,"Finished loop analysis.")

  if hoist_allocation == 1
    body = hoistAllocation(body, new_lives, loop_info, state)
    fake_body = lambdaFromStmtsMeta(body, state.param, state.meta)
    new_lives = CompilerTools.LivenessAnalysis.from_expr(fake_body, pir_live_cb, nothing)
    dprintln(1,"Starting loop analysis again.")
    loop_info = CompilerTools.Loops.compute_dom_loops(new_lives.cfg)
    dprintln(1,"Finished loop analysis.")
  end

  dprintln(3,"new_lives = ", new_lives)
  dprintln(3,"loop_info = ", loop_info)

  if IntelPSE.client_intel_pse_mode == 5 || IntelPSE.client_intel_task_graph || run_as_task()
    task_start = time_ns()

    # TODO: another pass of alias analysis to re-use dead but uniquely allocated arrays
    # AliasAnalysis.analyze_lambda_body(fake_body, state.param, state.meta2_typed, new_lives)

    # Create a mapping between the minimized basic block numbers and indices in body that are in those correponding basic blocks.
    map_reduced_bb_num_to_body = Dict{Int,Array{Int,1}}()
    for i = 1:length(body)
      # Get the basic block number for the first top level number associated with this entry in body.
      bb_num = CompilerTools.LivenessAnalysis.find_bb_for_statement(i, new_lives)
      if bb_num == nothing
          if typeof(body[i]) != LabelNode
              dprintln(0,"statement that couldn't be found in liveness analysis ", body[i])
              throw(string("find_bb_for_statement should not fail for non-LabelNodes"))
          end
          continue
      end
      # If not previously in the map then initialize it with the current body index.  Otherwise, add the current body index
      # as also mapping to its containing basic block.
      if !haskey(map_reduced_bb_num_to_body, bb_num)
          map_reduced_bb_num_to_body[bb_num] = [i]
      else
          map_reduced_bb_num_to_body[bb_num] = [map_reduced_bb_num_to_body[bb_num], i]
      end
    end

    dprintln(3,"map_reduced_bb_num_to_body = ", map_reduced_bb_num_to_body)

    bbs_in_task_graph_loops = Set()
    bbs = new_lives.basic_blocks
    tgsections = TaskGraphSection[]

    if put_loops_in_task_graph
      # For each loop that loop analysis identified.
      for one_loop in loop_info.loops
        # If the loop is "simple", i.e., has no just a body and a back-edge and no conditionals or nested loops.
        if length(one_loop.members) == 2
          # This is sort of sanity checking because the way Julia creates loops, these conditions should probably always hold.
          head_bb = bbs[one_loop.head]
          back_bb = bbs[one_loop.back_edge]

          if length(head_bb.preds) == 2 &&
             length(head_bb.succs) == 1 &&
             length(back_bb.preds) == 1 &&
             length(back_bb.succs) == 2
            before_head = getNonBlock(head_bb.preds, one_loop.back_edge)
            assert(typeof(before_head) == CompilerTools.LivenessAnalysis.BasicBlock)
            dprintln(3,"head_bb.preds = ", head_bb.preds, " one_loop.back_edge = ", one_loop.back_edge, " before_head = ", before_head)
            # assert(length(before_head) == 1)
            after_back  = getNonBlock(back_bb.succs, one_loop.head)
            assert(typeof(after_back) == CompilerTools.LivenessAnalysis.BasicBlock)
            #assert(length(after_back) == 1)

            head_indices = map_reduced_bb_num_to_body[one_loop.head]
            head_first_parfor = nothing
            for j = 1:length(head_indices)
              if isParforAssignmentNode(body[head_indices[j]]) || isBareParfor(body[head_indices[j]])
                head_first_parfor = j
                break
              end
            end

            back_indices = map_reduced_bb_num_to_body[one_loop.back_edge]
            back_first_parfor = nothing
            for j = 1:length(back_indices)
              if isParforAssignmentNode(body[back_indices[j]]) || isBareParfor(body[back_indices[j]])
                back_first_parfor = j
                break
              end
            end

            if head_first_parfor != nothing || back_first_parfor != nothing
              new_bbs_for_set = Set(one_loop.head, one_loop.back_edge, before_head.label, after_back.label)
              assert(length(intersect(bbs_in_task_graph_loops, new_bbs_for_set)) == 0)
              bbs_in_task_graph_loops = union(bbs_in_task_graph_loops, new_bbs_for_set)

              before_indices = map_reduced_bb_num_to_body[before_head.label]
              before_first_parfor = nothing
              for j = 1:length(before_indices)
                if isParforAssignmentNode(body[before_indices[j]]) || isBareParfor(body[body_indices[j]])
                  before_first_parfor = j
                  break
                end
              end

              after_indices = map_reduced_bb_num_to_body[after_back.label]
              after_first_parfor = nothing
              for j = 1:length(after_indices)
                if isParforAssignmentNode(body[after_indices[j]]) || isBareParfor(body[after_indices[j]])
                  after_first_parfor = j
                  break
                end
              end

    #          bb_live_info = new_lives.basic_blocks[bb_num]
    #          push!(replaced_regions, (first_parfor, last_parfor, makeTasks(first_parfor, last_parfor, body, bb_live_info)))

            end
          else
            dprintln(1,"Found a loop with 2 members but unexpected head or back_edge structure.")
            dprintln(1,"head = ", head_bb)
            dprintln(1,"back_edge = ", back_bb)
          end
        end
      end
    end

    for i in map_reduced_bb_num_to_body
      bb_num = i[1]
      body_indices = i[2]
      bb_live_info = new_lives.basic_blocks[new_lives.cfg.basic_blocks[bb_num]]

      if !in(bb_num, bbs_in_task_graph_loops)
        if task_graph_mode == SEQUENTIAL_TASKS
          # Find the first parfor in the block.
          first_parfor = nothing
          for j = 1:length(body_indices)
            if isParforAssignmentNode(body[body_indices[j]]) || isBareParfor(body[body_indices[j]])
              first_parfor = body_indices[j]
              break
            end
          end

          # If we found a parfor in the block.
          if first_parfor != nothing
            # Find the last parfor in the block...it might be the same as the first.
            last_parfor = nothing
            for j = length(body_indices):-1:1
              if isParforAssignmentNode(body[body_indices[j]]) || isBareParfor(body[body_indices[j]])
                last_parfor = body_indices[j]
                break
              end
            end
            assert(last_parfor != nothing)

            # Remember this section of code as something to transform into task graph format.
            push!(tgsections, TaskGraphSection(first_parfor, last_parfor, body[first_parfor:last_parfor]))
            dprintln(3,"Adding TaskGraphSection ", tgsections[end])

            push!(rr, ReplacedRegion(first_parfor, last_parfor, bb_num, makeTasks(first_parfor, last_parfor, body, bb_live_info, state, task_graph_mode)))
          end
        elseif task_graph_mode == ONE_AT_A_TIME
          for j = 1:length(body_indices)
            if taskableParfor(body[body_indices[j]])
              # Remember this section of code as something to transform into task graph format.
              cur_start = cur_end = body_indices[j]
              push!(tgsections, TaskGraphSection(cur_start, cur_end, body[cur_start:cur_end]))
              dprintln(3,"Adding TaskGraphSection ", tgsections[end])

              push!(rr, ReplacedRegion(body_indices[j], body_indices[j], bb_num, makeTasks(cur_start, cur_end, body, bb_live_info, state, task_graph_mode)))
            end
          end
        elseif task_graph_mode == MULTI_PARFOR_SEQ_NO
          cur_start = nothing
          cur_end   = nothing
          stmts_in_batch = Int64[]

          for j = 1:length(body_indices)
            if taskableParfor(body[body_indices[j]])
              if cur_start == nothing
                cur_start = cur_end = body_indices[j]
              else
                cur_end = body_indices[j]
              end 
              push!(stmts_in_batch, body_indices[j])
            else
              if cur_start != nothing
                dprintln(3,"Non-taskable parfor ", stmts_in_batch, " ", body[body_indices[j]])
                in_vars, out, locals = io_of_stmts_in_batch = getIO(stmts_in_batch, bb_live_info.statements)
                dprintln(3,"in_vars = ", in_vars)
                dprintln(3,"out_vars = ", out)
                dprintln(3,"local_vars = ", locals)

                cur_in_vars, cur_out, cur_locals = io_of_stmts_in_batch = getIO([body_indices[j]], bb_live_info.statements)
                dprintln(3,"cur_in_vars = ", cur_in_vars)
                dprintln(3,"cur_out_vars = ", cur_out)
                dprintln(3,"cur_local_vars = ", cur_locals)

                if isempty(intersect(out, cur_in_vars))
                  dprintln(3,"Sequential statement doesn't conflict with batch.")
                  push!(stmts_in_batch, body_indices[j])
                  cur_end = body_indices[j]
                else
                  # Remember this section of code (excluding current statement) as something to transform into task graph format.
                  push!(tgsections, TaskGraphSection(cur_start, cur_end, body[cur_start:cur_end]))
                  dprintln(3,"Adding TaskGraphSection ", tgsections[end])
  
                  push!(rr, ReplacedRegion(cur_start, cur_end, bb_num, makeTasks(cur_start, cur_end, body, bb_live_info, state, task_graph_mode)))
                
                  cur_start = cur_end = nothing
                  stmts_in_batch = Int64[]
                end
              end
            end
          end

          if cur_start != nothing
            # Remember this section of code as something to transform into task graph format.
            push!(tgsections, TaskGraphSection(cur_start, cur_end, body[cur_start:cur_end]))
            dprintln(3,"Adding TaskGraphSection ", tgsections[end])

            push!(rr, ReplacedRegion(cur_start, cur_end, bb_num, makeTasks(cur_start, cur_end, body, bb_live_info, state, task_graph_mode)))
          end
        else
          throw(string("Unknown Parallel IR task graph formation mode."))
        end
      end
    end

    dprintln(3,"Regions prior to sorting.")
    dprintln(3,rr)
    # We replace regions in reverse order of index so that we don't mess up indices that we need to replace later.
    sort!(rr, by=x -> x.end_index, rev=true)
    dprintln(3,"Regions after sorting.")
    dprintln(3,rr)

    printBody(3,body)

    dprintln(2, "replaced_regions")
    for i = 1:length(rr)
      dprintln(2, rr[i])

      if IntelPSE.client_intel_pse_mode == 5
        # new body starts with the pre-task graph portion
        new_body = body[1:rr[i].start_index-1]
        copy_back = Any[]

        # then adds calls for each task
        for j = 1:length(rr[i].tasks)
          cur_task = rr[i].tasks[j]
          dprintln(3,"cur_task = ", cur_task, " type = ", typeof(cur_task))
          if typeof(cur_task) == TaskInfo
            range_var = string(cur_task.task_func,"_range_var")
            range_sym = symbol(range_var)

            dprintln(3,"Inserting call to jl_threading_run ", range_sym)
            dprintln(3,cur_task.function_sym, " type = ", typeof(cur_task.function_sym))

            in_len  = length(cur_task.input_symbols)
            mod_len = length(cur_task.modified_inputs)
            io_len  = length(cur_task.io_symbols)
            red_len = length(cur_task.reduction_vars)
            dprintln(3, "inputs, modifieds, io_sym, reductions = ", cur_task.input_symbols, " ", cur_task.modified_inputs, " ", cur_task.io_symbols, " ", cur_task.reduction_vars)

            dims = length(cur_task.loopNests)
            if dims > 0
              dprintln(3,"dims > 0")
              assert(dims <= 3)
              #whole_iteration_range = pir_range_actual()
              #whole_iteration_range.dim = dims
              cstr_params = Any[]
              for l = 1:dims
                # Note that loopNest is outer-dimension first
                # Should this still be outer-dimension first?  FIX FIX FIX
                push!(cstr_params, cur_task.loopNests[dims - l + 1].lower)
                push!(cstr_params, cur_task.loopNests[dims - l + 1].upper)
                #push!(whole_iteration_range.lower_bounds, cur_task.loopNests[dims - l + 1].lower)
                #push!(whole_iteration_range.upper_bounds, cur_task.loopNests[dims - l + 1].upper)
              end
              dprintln(3, "cstr_params = ", cstr_params)
              cstr_expr = mk_parallelir_ref(:pir_range_actual, Any)
              whole_range_expr = mk_assignment_expr(SymbolNode(range_sym, pir_range_actual), TypedExpr(pir_range_actual, :call, cstr_expr, cstr_params...))
              dprintln(3,"whole_range_expr = ", whole_range_expr)
              push!(new_body, whole_range_expr) 

#    push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "whole_range = ", SymbolNode(range_sym, pir_range_actual)))

              real_args_build = Any[]
              args_type = Expr(:tuple)

              # Fill in the arg metadata.
              for l = 1:in_len
                push!(real_args_build, cur_task.input_symbols[l].name)
                push!(args_type.args,  cur_task.input_symbols[l].typ)
              end
              for l = 1:mod_len
                push!(real_args_build, cur_task.modified_inputs[l].name)
                push!(args_type.args,  cur_task.modified_inputs[l].typ)
              end
              for l = 1:io_len
                push!(real_args_build, cur_task.io_symbols[l].name)
                push!(args_type.args,  cur_task.io_symbols[l].typ)
              end
              for l = 1:red_len
                push!(real_args_build, cur_task.reduction_vars[l].name)
                push!(args_type.args,  cur_task.reduction_vars[l].typ)
              end

              dprintln(3,"task_func = ", cur_task.task_func)
              #dprintln(3,"whole_iteration_range = ", whole_iteration_range)
              dprintln(3,"real_args_build = ", real_args_build)
              
              tup_var = string(cur_task.task_func,"_tup_var")
              tup_sym = symbol(tup_var)

              if false
                real_args_tuple_expr = TypedExpr(eval(args_type), :call, TopNode(:tuple), real_args_build...)
                call_tup = (Function,pir_range_actual,Any)
                push!(new_body, mk_assignment_expr(SymbolNode(tup_sym, call_tup), TypedExpr(call_tup, :call, TopNode(:tuple), cur_task.task_func, SymbolNode(range_sym, pir_range_actual), real_args_tuple_expr)))
              else
                call_tup_expr = Expr(:tuple, Function, pir_range_actual, args_type.args...)
                call_tup = eval(call_tup_expr)
                dprintln(3, "call_tup = ", call_tup)
                push!(new_body, mk_assignment_expr(SymbolNode(tup_sym, call_tup), TypedExpr(call_tup, :call, TopNode(:tuple), cur_task.task_func, SymbolNode(range_sym, pir_range_actual), real_args_build...)))
              end

              if false
              insert_task_expr = TypedExpr(Any,
                                           :call,
                                           cur_task.task_func,
                                           SymbolNode(range_sym, pir_range_actual),
                                           real_args_build...)
              else
              insert_task_expr = TypedExpr(Any, 
                                           :call, 
                                           TopNode(:ccall), 
                                           QuoteNode(:jl_threading_run), 
                                           :Void, 
                                           TypedExpr((Any,Any), :call1, TopNode(:tuple), Any, Any), 
                                           mk_parallelir_ref(:isf), 0, 
                                           tup_sym, 0)
              end
              push!(new_body, insert_task_expr)
            else
              throw(string("insert sequential task not implemented yet"))
            end
          else
            push!(new_body, cur_task)
          end
        end

        # Insert call to wait on the scheduler to complete all tasks.
        #push!(new_body, TypedExpr(Cint, :call, TopNode(:ccall), QuoteNode(:pert_wait_all_task), Type{Cint}, ()))

        # Add the statements that copy results out of temp arrays into real variables.
        append!(new_body, copy_back)

        # Then appends the post-task graph portion
        append!(new_body, body[rr[i].end_index+1:end])

        body = new_body

        dprintln(3,"new_body after region ", i, " replaced")
        printBody(3,body)
      elseif IntelPSE.client_intel_task_graph
        # new body starts with the pre-task graph portion
        new_body = body[1:rr[i].start_index-1]
        copy_back = Any[]

        # then adds calls for each task
        for j = 1:length(rr[i].tasks)
          cur_task = rr[i].tasks[j]
          dprintln(3,"cur_task = ", cur_task, " type = ", typeof(cur_task))
          if typeof(cur_task) == TaskInfo
            dprintln(3,"Inserting call to insert_divisible_task")
            dprintln(3,cur_task.function_sym, " type = ", typeof(cur_task.function_sym))

            in_len  = length(cur_task.input_symbols)
            mod_len = length(cur_task.modified_inputs)
            io_len  = length(cur_task.io_symbols)
            red_len = length(cur_task.reduction_vars)
            dprintln(3, "inputs, modifieds, io_sym, reductions = ", cur_task.input_symbols, " ", cur_task.modified_inputs, " ", cur_task.io_symbols, " ", cur_task.reduction_vars)

            dims = length(cur_task.loopNests)
            if dims > 0
              itn = InsertTaskNode()

              if IntelPSE.client_intel_task_graph_mode == 0
                itn.task_options = TASK_STATIC_SCHEDULER | TASK_AFFINITY_XEON
              elseif IntelPSE.client_intel_task_graph_mode == 1
                itn.task_options = TASK_STATIC_SCHEDULER | TASK_AFFINITY_PHI
              elseif IntelPSE.client_intel_task_graph_mode == 2
                itn.task_options = 0
              else
                throw(string("Unknown task graph mode option ", IntelPSE.client_intel_task_graph_mode))
              end

              if task_graph_mode == SEQUENTIAL_TASKS
                # intentionally do nothing
              elseif task_graph_mode == ONE_AT_A_TIME
                itn.task_options |= TASK_FINISH
              elseif task_graph_mode == MULTI_PARFOR_SEQ_NO
                # intentionally do nothing
              else
                throw(string("Unknown task_graph_mode."))
              end

              # Fill in pir_range
              # Fill in the min and max grain sizes.
              itn.ranges.dim = dims
              itn.host_grain_size.dim = dims
              itn.phi_grain_size.dim = dims
              for l = 1:dims
                # Note that loopNest is outer-dimension first
                push!(itn.ranges.lower_bounds, TypedExpr(Int64, :call, TopNode(:sub_int), cur_task.loopNests[dims - l + 1].lower, 1))
                push!(itn.ranges.upper_bounds, TypedExpr(Int64, :call, TopNode(:sub_int), cur_task.loopNests[dims - l + 1].upper, 1))
                push!(itn.host_grain_size.sizes, 2)
                push!(itn.phi_grain_size.sizes, 2)
              end

              # Fill in the arg metadata.
              for l = 1:in_len
                if cur_task.input_symbols[l].typ.name == Array.name
                  dprintln(3, "is array")
                  push!(itn.args, pir_arg_metadata(cur_task.input_symbols[l], ARG_OPT_IN, create_array_access_desc(cur_task.input_symbols[l])))
                else
                  dprintln(3, "is not array")
                  push!(itn.args, pir_arg_metadata(cur_task.input_symbols[l], ARG_OPT_IN))
                end
              end
              for l = 1:mod_len
                dprintln(3, "mod_len loop: ", l, " ", cur_task.modified_inputs[l])
                #if isa(cur_task.modified_inputs[l], Array)
                if cur_task.modified_inputs[l].typ.name == Array.name
                  dprintln(3, "is array")
                  push!(itn.args, pir_arg_metadata(cur_task.modified_inputs[l], ARG_OPT_OUT, create_array_access_desc(cur_task.modified_inputs[l])))
                else
                  dprintln(3, "is not array")
                  push!(itn.args, pir_arg_metadata(cur_task.modified_inputs[l], ARG_OPT_OUT))
                end
              end
              for l = 1:io_len
                dprintln(3, "io_len loop: ", l, " ", cur_task.io_symbols[l])
                if cur_task.io_symbols[l].typ.name == Array.name
                  dprintln(3, "is array")
                  push!(itn.args, pir_arg_metadata(cur_task.io_symbols[l], ARG_OPT_INOUT, create_array_access_desc(cur_task.io_symbols[l])))
                else
                  dprintln(3, "is not array")
                  push!(itn.args, pir_arg_metadata(cur_task.io_symbols[l], ARG_OPT_INOUT))
                end
              end
              for l = 1:red_len
                dprintln(3, "red_len loop: ", l, " ", cur_task.reduction_vars[l])
                push!(itn.args, pir_arg_metadata(cur_task.reduction_vars[l], ARG_OPT_ACCUMULATOR))
              end

              # Fill in the task function.
              itn.task_func = cur_task.task_func
              itn.join_func = cur_task.join_func

              dprintln(3,"InsertTaskNode = ", itn)

              insert_task_expr = TypedExpr(Int, :insert_divisible_task, itn) 
              push!(new_body, insert_task_expr)
            else
              throw(string("insert sequential task not implemented yet"))
            end
          else
            push!(new_body, cur_task)
          end
        end

        # Insert call to wait on the scheduler to complete all tasks.
        #push!(new_body, TypedExpr(Cint, :call, TopNode(:ccall), QuoteNode(:pert_wait_all_task), Type{Cint}, ()))

        # Add the statements that copy results out of temp arrays into real variables.
        append!(new_body, copy_back)

        # Then appends the post-task graph portion
        append!(new_body, body[rr[i].end_index+1:end])

        body = new_body

        dprintln(3,"new_body after region ", i, " replaced")
        printBody(3,body)
      elseif run_as_task_decrement()
        # new body starts with the pre-task graph portion
        new_body = body[1:rr[i].start_index-1]

        # then adds calls for each task
        for j = 1:length(rr[i].tasks)
          cur_task = rr[i].tasks[j]
          assert(typeof(cur_task) == TaskInfo)

          dprintln(3,"Inserting call to function")
          dprintln(3,cur_task, " type = ", typeof(cur_task))
          dprintln(3,cur_task.function_sym, " type = ", typeof(cur_task.function_sym))

          in_len = length(cur_task.input_symbols)
          out_len = length(cur_task.output_symbols)

          real_out_params = Any[]

          for k = 1:out_len
            this_param = cur_task.output_symbols[k]
            assert(typeof(this_param) == SymbolNode)
            atype = Array{this_param.typ, 1}
            temp_param_array = createStateVar(state, string(this_param.name, "_out_array"), atype, ISASSIGNED)
            push!(real_out_params, temp_param_array)
            new_temp_array = mk_alloc_array_1d_expr(this_param.typ, atype, 1)
            push!(new_body, mk_assignment_expr(temp_param_array, new_temp_array))
          end

          #push!(new_body, TypedExpr(Nothing, :call, cur_task.function_sym, cur_task.input_symbols..., real_out_params...))
          #push!(new_body, TypedExpr(Nothing, :call, TopNode(cur_task.function_sym), cur_task.input_symbols..., real_out_params...))
          push!(new_body, TypedExpr(Nothing, :call, mk_parallelir_ref(cur_task.function_sym), TypedExpr(pir_range_actual, :call, :pir_range_actual), cur_task.input_symbols..., real_out_params...))

          for k = 1:out_len
            push!(new_body, mk_assignment_expr(cur_task.output_symbols[k], mk_arrayref1(real_out_params[k], 1, false)))
          end
        end

        # then appends the post-task graph portion
        append!(new_body, body[rr[i].end_index+1:end])

        body = new_body

        dprintln(3,"new_body after region ", i, " replaced")
        printBody(3,body)
      end
    end

  #    throw(string("debugging task graph"))
    dprintln(1,"Task formation time = ", ns_to_sec(time_ns() - task_start))
  end  # end of task graph formation section

  if flat_parfor != 0
    flatten_start = time_ns()

    expanded_body = Any[]

    for i = 1:length(body)
      dprintln(3,"Flatten index ", i, " ", body[i], " type = ", typeof(body[i]))
      if isBareParfor(body[i])
        flattenParfor(expanded_body, body[i].args[1])
      else
        push!(expanded_body, body[i])
      end
    end

    body = expanded_body
    dprintln(1,"Flattening parfor bodies time = ", ns_to_sec(time_ns() - flatten_start))
  end

  if shortcut_array_assignment != 0
    fake_body = lambdaFromStmtsMeta(body, state.param, state.meta)
    new_lives = CompilerTools.LivenessAnalysis.from_expr(fake_body, pir_live_cb, nothing)

    for i = 1:length(body)
      node = body[i]
      if isAssignmentNode(node)
        lhs = node.args[1]
        rhs = node.args[2]
        dprintln(3,"shortcut_array_assignment = ", node)
        if typeof(lhs) == SymbolNode && isArrayType(lhs) && typeof(rhs) == SymbolNode
          dprintln(3,"shortcut_array_assignment to array detected")
          live_info = CompilerTools.LivenessAnalysis.find_top_number(i, new_lives)
          if !in(rhs.name, live_info.live_out)
            dprintln(3,"rhs is dead")
            # The RHS of the assignment is not live out so we can do a special assignment where the j2c_array for the LHS takes over the RHS and the RHS is nulled.
            push!(node.args, RhsDead())
          end
        end
      end
    end
  end

  return body
end


function isf(t::Function, 
             full_iteration_space::IntelPSE.ParallelIR.pir_range_actual,
             rest...)
    tid = Base.Threading.threadid()

#    println("isf ", t, " ", full_iteration_space, " ", args)
    if full_iteration_space.dim == 1
        num_iters = full_iteration_space.upper_bounds[1] - full_iteration_space.lower_bounds[1] + 1

        println("num_iters = ", num_iters)
        if num_iters <= nthreads()
if true
    if tid == 1
      #println("ISF tid = ", tid, " t = ", t, " fis = ", full_iteration_space, " args = ", rest...)
      return t(IntelPSE.ParallelIR.pir_range_actual(1,2), rest...)
      #return t(full_iteration_space, rest...)
    else
      return nothing
    end
else
          if tid <= num_iters
              return t(IntelPSE.ParallelIR.pir_range_actual(1,2), rest...)
              #return t(IntelPSE.ParallelIR.pir_range_actual(tid,tid), rest...)
          else
              #return t(pir_range_actual([0],[-1]), rest...)
              return nothing
          end
end
        else
          # one dimensional scheduling
          len, rem = divrem(num_iters, nthreads())
          ls = len * tid
          if tid == nthreads()
             le = full_iteration_space.upper_bounds[1]
          else
             le = (len * (tid+1)) - 1
          end
          return t(pir_range_actual([ls],[le]), rest...)
        end
    elseif full_iteration_space.dim == 2
        assert(0)
    else
        assert(0)
    end
end

function isArrayType(x :: SymbolNode)
  the_type = x.typ
  if typeof(the_type) == DataType
    return x.typ.name == Array.name
  end
  return false
end

function makeTasks(start_index, stop_index, body, bb_live_info, state, task_graph_mode)
  task_list = Any[]
  seq_accum = Any[]
  dprintln(3,"makeTasks starting")

  # ONE_AT_A_TIME mode should only have parfors in the list of things to replace so we assert if we see a non-parfor in this mode.
  # SEQUENTIAL_TASKS mode bunches up all non-parfors into one sequential task.  Maybe it would be better to do one sequential task per non-parfor stmt?
  # MULTI_PARFOR_SEQ_NO mode converts parfors to tasks but leaves non-parfors as non-tasks.  This implies that the code calling this function has ensured
  #     that none of the non-parfor stmts depend on the completion of a parfor in the batch.

  if IntelPSE.client_intel_pse_mode != 5
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
  
  if IntelPSE.client_intel_pse_mode != 5
    # If each task doesn't wait to finish then add a call to pert_wait_all_task to wait for the batch to finish.
    if !task_finish
      #julia_root      = IntelPSE.getJuliaRoot()
      #runtime_libpath = string(julia_root, "/intel-runtime/lib/libintel-runtime.so")
      #runtime_libpath = IntelPSE.runtime_libpath

      #call_wait = Expr(:ccall, Expr(:tuple, QuoteNode(:pert_wait_all_task), runtime_libpath), :Void, Expr(:tuple))
      #push!(task_list, call_wait) 

      call_wait = TypedExpr(Void, 
                            :call, 
                            TopNode(:ccall), 
                            Expr(:call1, TopNode(:tuple), QuoteNode(:pert_wait_all_task), runtime_libpath), 
                            :Void, 
                            Expr(:call1, TopNode(:tuple)))
      push!(task_list, call_wait)

#    call_wait = quote ccall((:pert_wait_all_task, $runtime_libpath), Void, ()) end
#    assert(typeof(call_wait) == Expr && call_wait.head == :block)
#    append!(task_list, call_wait.args) 
    end
  end

  task_list
end

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
    cur_inputs = union(cur_inputs, setdiff(stmts_for_ids[i].use, cur_defs))
    cur_defs   = union(cur_defs, stmts_for_ids[i].def)
  end
  IntrinsicSet = Set()
  push!(IntrinsicSet, :Intrinsics)
  outputs = setdiff(intersect(cur_defs, stmts_for_ids[end].live_out), IntrinsicSet)
  cur_defs = setdiff(cur_defs, IntrinsicSet)
  cur_inputs = setdiff(filter(x -> !(is(x, :Int64) || is(x, :Float32)), cur_inputs), IntrinsicSet)
  cur_inputs, outputs, setdiff(cur_defs, union(cur_inputs, outputs))
end

# Declare a function as a task function.
function make_task_function(function_name, signature)
  m = methods(function_name, signature)
  if length(m) < 1
    error("Method for ", function_name, " with signature ", signature, " is not found")
  end
  def = m[1].func.code
  if def.j2cflag != 4 && def.j2cflag != 0
    error("method for ", function_name, " with signature ", signature, " cannot be combined with other j2c flags")
  end
  def.j2cflag = convert(Int32, 4)
end

function removePound(sym :: Symbol)
  s = string(sym)
  ret = ""
  for i = 1:length(s)
    if s[i] == '\#'
      ret = string(ret, 'p')
    else
      ret = string(ret, s[i])
    end
  end
  symbol(ret)
end

function mk_colon_expr(start_expr, skip_expr, end_expr)
    TypedExpr(Any, :call, :colon, start_expr, skip_expr, end_expr)
end

function mk_start_expr(colon_sym)
    TypedExpr(Any, :call, TopNode(:start), colon_sym)
end

function mk_next_expr(colon_sym, start_sym)
    TypedExpr(Any, :call, TopNode(:next), colon_sym, start_sym)
end

function mk_gotoifnot_expr(cond, goto_label)
    TypedExpr(Any, :gotoifnot, cond, goto_label)
end

type cuw_state
  found
  function cuw_state()
    new(false)
  end
end

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

  nothing
end

function convertUnsafe(stmt)
  dprintln(3,"convertUnsafe: ", stmt)
  state = cuw_state() 
  res = AstWalk(stmt, convertUnsafeWalk, state)
  if state.found
    assert(isa(res,Array))
    assert(length(res) == 1)
    dprintln(3,"Replaced unsafe: ", res[1])
    return res[1]
  else
    return nothing
  end
end

function convertUnsafeOrElse(stmt)
  res = convertUnsafe(stmt)
  if res == nothing
    res = stmt
  end
  return res
end

function recreateLoopsInternal(new_body, the_parfor :: IntelPSE.ParallelIR.PIRParForAst, loop_nest_level, next_available_label)
  if loop_nest_level > length(the_parfor.loopNests)
    for i = 1:length(the_parfor.body)
      cu_res = convertUnsafe(the_parfor.body[i])
      if cu_res != nothing
        push!(new_body, Expr(:boundscheck, false)) 
        push!(new_body, cu_res)
        push!(new_body, Expr(:boundscheck, Expr(:call, TopNode(:getfield), Base, QuoteNode(:pop))))
      else
        push!(new_body, the_parfor.body[i])
      end
    end
  else
    this_nest = the_parfor.loopNests[loop_nest_level]

    label_after_first_unless   = next_available_label
    label_before_second_unless = next_available_label + 1
    label_after_second_unless  = next_available_label + 2
    label_last                 = next_available_label + 3

    colon_var = string("#recreate_colon_", (loop_nest_level-1) * 3 + 0)
    colon_sym = symbol(colon_var)
    start_var = string("#recreate_start_", (loop_nest_level-1) * 3 + 1)
    start_sym = symbol(start_var)
    next_var  = string("#recreate_next_",  (loop_nest_level-1) * 3 + 1)
    next_sym  = symbol(next_var)

    #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "ranges = ", SymbolNode(:ranges, pir_range_actual)))
    #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.lower = ", this_nest.lower))
    #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.step  = ", this_nest.step))
    #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.upper = ", this_nest.upper))

    push!(new_body, mk_assignment_expr(SymbolNode(colon_sym,Any), mk_colon_expr(convertUnsafeOrElse(this_nest.lower), convertUnsafeOrElse(this_nest.step), convertUnsafeOrElse(this_nest.upper))))
    push!(new_body, mk_assignment_expr(SymbolNode(start_sym,Any), mk_start_expr(colon_sym)))
    push!(new_body, mk_gotoifnot_expr(TypedExpr(Any, :call, TopNode(:(!)), TypedExpr(Any, :call, TopNode(:done), colon_sym, start_sym) ), label_after_second_unless))
    push!(new_body, LabelNode(label_after_first_unless))

    push!(new_body, mk_assignment_expr(SymbolNode(next_sym,Any),  mk_next_expr(colon_sym, start_sym)))
    push!(new_body, mk_assignment_expr(this_nest.indexVariable,   mk_tupleref_expr(next_sym, 1, Any)))
    push!(new_body, mk_assignment_expr(SymbolNode(start_sym,Any), mk_tupleref_expr(next_sym, 2, Any)))

    #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "loopIndex = ", this_nest.indexVariable))
    #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), colon_sym, " ", start_sym))
    recreateLoopsInternal(new_body, the_parfor, loop_nest_level + 1, next_available_label + 4)

    push!(new_body, LabelNode(label_before_second_unless))
    push!(new_body, mk_gotoifnot_expr(TypedExpr(Any, :call, TopNode(:(!)), TypedExpr(Any, :call, TopNode(:(!)), TypedExpr(Any, :call, TopNode(:done), colon_sym, start_sym))), label_after_first_unless))
    push!(new_body, LabelNode(label_after_second_unless))
    push!(new_body, LabelNode(label_last))
  end
end

function recreateLoops(new_body, the_parfor :: IntelPSE.ParallelIR.PIRParForAst)
  max_label = getMaxLabel(0, the_parfor.body)
  dprintln(2,"recreateLoops ", the_parfor, " max_label = ", max_label)
  recreateLoopsInternal(new_body, the_parfor, 1, max_label + 1)
  nothing
end
 
function flattenParfor(new_body, the_parfor :: IntelPSE.ParallelIR.PIRParForAst)
  dprintln(2,"Flattening ", the_parfor)
  push!(new_body, TypedExpr(Int64, :parfor_start, PIRParForStartEnd(the_parfor.loopNests, the_parfor.reductions, the_parfor.instruction_count_expr)))
  append!(new_body, the_parfor.body)
  push!(new_body, TypedExpr(Int64, :parfor_end, PIRParForStartEnd(the_parfor.loopNests, the_parfor.reductions, the_parfor.instruction_count_expr)))
  nothing
end

function parforToTask(parfor_index, bb_statements, body, state)
  assert(typeof(body[parfor_index]) == Expr)
  assert(body[parfor_index].head == :parfor)
  the_parfor = body[parfor_index].args[1]
  dprintln(3,"parforToTask = ", the_parfor)

  reduction_vars = Symbol[]
  for i in the_parfor.reductions
    push!(reduction_vars, i.reductionVar.name)
  end

  in_vars , out, locals = getIO([parfor_index], bb_statements)
  dprintln(3,"in_vars = ", in_vars)
  dprintln(3,"out_vars = ", out)
  dprintln(3,"local_vars = ", locals)

  # Convert Set to Array
  in_array_names   = Any[]
  modified_symbols = Any[]
  io_symbols       = Any[]
  for i in in_vars
    #stype    = state.meta2_typed[i][2]
    swritten = CompilerTools.ReadWriteSet.isWritten(i, the_parfor.rws)
    sread    = CompilerTools.ReadWriteSet.isRead(i, the_parfor.rws)
    sio      = swritten & sread
    if in(i, reduction_vars)
      assert(sio)   # reduction_vars must be initialized before the parfor and updated during the parfor so must be io
    elseif sio
      push!(io_symbols, i)
    elseif swritten
      push!(modified_symbols, i)
    else
      push!(in_array_names, i)
    end
  end
  # Convert Set to Array
  if length(out) != 0
    throw(string("out variable of parfor task not supported right now."))
  end
  # Convert Set to Array
  locals_array = Any[]
  for i in locals
    push!(locals_array, i)
  end

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
  dprintln(3,"arg_types = ", arg_types)

  # Form an array including symbols for all the in and output parameters plus the additional iteration control parameter "ranges".
  all_arg_names = [:ranges,
                   map(x -> symbol(x), in_array_names),
                   map(x -> symbol(x), modified_symbols),
                   map(x -> symbol(x), io_symbols),
                   map(x -> symbol(x), reduction_vars)]
  # Form a tuple that contains the type of each parameter.
  all_arg_types_tuple = Expr(:tuple)
  all_arg_types_tuple.args = [pir_range_actual,
                              map(x -> state.meta2_typed[x][2], in_array_names),
                              map(x -> state.meta2_typed[x][2], modified_symbols),
                              map(x -> state.meta2_typed[x][2], io_symbols),
                              map(x -> state.meta2_typed[x][2], reduction_vars)]
  all_arg_type = eval(all_arg_types_tuple)
  # Forms part of the meta[2] array for the task function's parameters.
  args_var = Array{Any,1}[]
  push!(args_var, [:ranges, pir_range_actual, 0])
  append!(args_var, [ map(x -> [x,state.meta2_typed[x][2],0], in_array_names),
                      map(x -> [x,state.meta2_typed[x][2],0], modified_symbols),
                      map(x -> [x,state.meta2_typed[x][2],0], io_symbols),
                      map(x -> [x,state.meta2_typed[x][2],0], reduction_vars)])
  dprintln(3,"all_arg_names = ", all_arg_names)
  dprintln(3,"all_arg_type = ", all_arg_type)
  dprintln(3,"args_var = ", args_var)

  unique_node_id = get_unique_num()

  # The name of the new task function.
  task_func_name = string("task_func_",unique_node_id)
  task_func_sym  = symbol(task_func_name)

  # Just stub out the new task function...the body will be replaced below.
  task_func = @eval function ($task_func_sym)($(all_arg_names...))
                      throw(string("Some task function's body was not replaced."))
                    end
  dprintln(3,"task_func = ", task_func)

  # DON'T DELETE.  Forces function into existence.
  unused_ct = code_typed(task_func, all_arg_type)
  dprintln(3, "unused_ct = ", unused_ct)

  meta = Any[]
  # The regular out parameters names act as locals and are then copied into their *_out_arg array entries.
  all_array_names = [locals_array...]
  # This creates meta[1] which is the local variables used in the function.
  push!(meta, all_array_names)
  # This creates meta[2] which is an array of info about variables.  Here we add the info for the non-parameters.
  push!(meta, map(x -> state.meta2_typed[x], all_array_names))
  # This creates meta[3].
  push!(meta, Any[])
  # This adds the type information about the parameters to meta[2].
  append!(meta[2], args_var)

  # Creating the new body for the task function.
  task_body = TypedExpr(Int, :body)
  saved_loopNests = deepcopy(the_parfor.loopNests)

#  for i in all_arg_names
#    push!(task_body.args, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), string(i), " = ", Symbol(i)))
#  end
 
  dprintln(3, "meta = ", meta)

  if IntelPSE.client_intel_task_graph
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
  elseif IntelPSE.client_intel_pse_mode == 5
    for i = 1:length(the_parfor.loopNests)
      # Put outerloop first in the loopNest
      j = length(the_parfor.loopNests) - i + 1
      the_parfor.loopNests[j].lower = TypedExpr(Int64, :call, TopNode(:unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, TopNode(:getfield), :ranges, QuoteNode(:lower_bounds)), i)
      the_parfor.loopNests[j].upper = TypedExpr(Int64, :call, TopNode(:unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, TopNode(:getfield), :ranges, QuoteNode(:upper_bounds)), i)
    end
  end

  dprintln(3, "Before recreation or flattening")

  # Add the parfor stmt to the task function body.
  if IntelPSE.client_intel_pse_mode == 5
    recreateLoops(task_body.args, the_parfor)
  else
    if flat_parfor != 0
      flattenParfor(task_body.args, the_parfor)
    else
      push!(task_body.args, body[parfor_index])
    end
  end

  push!(task_body.args, TypedExpr(Int, :return, 0))

  code = Expr(:lambda,
           all_arg_names,   # the parameters to the function
           meta,            # the type and access information for variables used in the function
           task_body)       # the code for the function

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
  if IntelPSE.client_intel_pse_mode != 5
    def.j2cflag = convert(Int32,6)
    ccall(:set_j2c_task_arg_types, Void, (Ptr{Uint8}, Cint, Ptr{Cint}), task_func_name, length(arg_types), arg_types)
  end
  if IntelPSE.client_intel_pse_mode == 5
    precompile(task_func, all_arg_type)
  else
    cfunction(task_func, Int, all_arg_type)
  end
  dprintln(3, "def post = ", def, " type = ", typeof(def))

  if DEBUG_LVL >= 3
    task_func_ct = code_typed(task_func, all_arg_type)
    if length(task_func_ct) == 0
      println("Error getting task func code.\n")
    else
      task_func_ct = task_func_ct[1]
      println("Task func code for ", task_func)
      println(task_func_ct)    
    end
  end
 
  reduction_func_name = string("")
  if length(the_parfor.reductions) > 0
    # The name of the new reduction function.
    reduction_func_name = string("reduction_func_",unique_node_id)

    the_types = String[]
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

    c_reduction_func = string("void _")
    c_reduction_func = string(c_reduction_func, reduction_func_name)
    c_reduction_func = string(c_reduction_func, "(void **accumulator, void **new_reduction_vars) {\n")
    for i = 1:length(the_parfor.reductions)
      if the_parfor.reductions[i].reductionFunc == :+
        this_op = "+"
      elseif the_parfor.reductions[i].reductionFunc == :*
        this_op = "*"
      else
        throw(string("Unsupported reduction function ", the_parfor.reductions[i].reductionFunc, " during join function generation."))
      end
      c_reduction_func = string(c_reduction_func, "*((", the_types[i], "*)accumulator[", i-1, "]) = *((", the_types[i], "*)accumulator[", i-1, "]) ", this_op, " *((", the_types[i], "*)new_reduction_vars[", i-1, "]);\n")
    end
    c_reduction_func = string(c_reduction_func, "}\n")
    dprintln(3,"Created reduction function is:")
    dprintln(3,c_reduction_func)

    ccall(:set_j2c_add_c_code, Void, (Ptr{Uint8},), c_reduction_func)
  end

  return TaskInfo(task_func,
                  task_func_sym,
                  reduction_func_name,
                  map(x -> SymbolNode(x, state.meta2_typed[x][2]), in_array_names),
                  map(x -> SymbolNode(x, state.meta2_typed[x][2]), modified_symbols),
                  map(x -> SymbolNode(x, state.meta2_typed[x][2]), io_symbols),
                  map(x -> SymbolNode(x, state.meta2_typed[x][2]), reduction_vars),
                  code,
                  saved_loopNests)
end

function seqTask(body_indices, bb_statements, body, state)
  getIO(body_indices, bb_statements)  
  throw(string("seqTask construction not implemented yet."))
  TaskInfo(:FIXFIXFIX, :FIXFIXFIX, Any[], Any[], nothing, PIRLoopNest[])
end

function printBody(dlvl, body)
  for i = 1:length(body)
    dprintln(dlvl, "    ", body[i])
  end
end

function printLambda(dlvl, node)
  assert(typeof(node) == Expr)
  assert(node.head == :lambda)
  dprintln(dlvl, "Lambda:")
  dprintln(dlvl, "Input parameters: ", node.args[1])
  dprintln(dlvl, "Metadata: ", node.args[2])
  body = node.args[3]
  assert(typeof(body) == Expr)
  assert(body.head == :body)
  dprintln(dlvl, "typeof(body): ", body.typ)
  printBody(dlvl, body.args)
#  if body.typ == Any
#    throw(string("Body type should never be Any."))
#  end
end

function pir_live_cb(ast, cbdata)
  dprintln(4,"pir_live_cb")
  asttyp = typeof(ast)
  if asttyp == Expr
    head = ast.head
    args = ast.args
    if head == :parfor
      dprintln(3,"pir_live_cb for :parfor")
      expr_to_process = Any[]

      assert(typeof(args[1]) == IntelPSE.ParallelIR.PIRParForAst)
      this_parfor = args[1]

      append!(expr_to_process, this_parfor.preParFor)
      for i = 1:length(this_parfor.loopNests)
        # force the indexVariable to be treated as an rvalue
        push!(expr_to_process, mk_assignment_expr(this_parfor.loopNests[i].indexVariable, 1))
        push!(expr_to_process, this_parfor.loopNests[i].lower)
        push!(expr_to_process, this_parfor.loopNests[i].upper)
        push!(expr_to_process, this_parfor.loopNests[i].step)
      end
      #fake_body = Expr(:body)
      #fake_body.args = this_parfor.body
      fake_body = lambdaFromStmtsMeta(this_parfor.body)

      body_lives = CompilerTools.LivenessAnalysis.from_expr(fake_body, pir_live_cb, nothing)
      live_in_to_start_block = body_lives.basic_blocks[body_lives.cfg.basic_blocks[-1]].live_in
      all_defs = Set()
      for bb in body_lives.basic_blocks
        all_defs = union(all_defs, bb[2].def)
      end 
      as = CompilerTools.LivenessAnalysis.AccessSummary(setdiff(all_defs, live_in_to_start_block), live_in_to_start_block)

      push!(expr_to_process, as)
      #append!(expr_to_process, this_parfor.body)

      append!(expr_to_process, this_parfor.postParFor)

      return expr_to_process
    elseif head == :parfor_start
      dprintln(3,"pir_live_cb for :parfor_start")
      expr_to_process = Any[]

      assert(typeof(args[1]) == PIRParForStartEnd)
      this_parfor = args[1]

      for i = 1:length(this_parfor.loopNests)
        # force the indexVariable to be treated as an rvalue
        push!(expr_to_process, mk_assignment_expr(this_parfor.loopNests[i].indexVariable, 1))
        push!(expr_to_process, this_parfor.loopNests[i].lower)
        push!(expr_to_process, this_parfor.loopNests[i].upper)
        push!(expr_to_process, this_parfor.loopNests[i].step)
      end

      return expr_to_process
    elseif head == :parfor_end
      # intentionally do nothing
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
          push!(expr_to_process, mk_assignment_expr(cur_task.args[i].value, 1))
        end
      end

      return expr_to_process
    elseif head == :loophead
      dprintln(3,"pir_live_cb for :loophead")
      assert(length(args) == 3)

      expr_to_process = Any[]
      push!(expr_to_process, mk_assignment_expr(SymbolNode(args[1], Int64), 1))  # force args[1] to be seen as an rvalue
      push!(expr_to_process, args[2])
      push!(expr_to_process, args[3])

      return expr_to_process
    elseif head == :loopend
      # There is nothing really interesting in the loopend node to signify something being read or written.
      assert(length(args) == 1)
      return Any[]
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

function hasNoSideEffects(node)
  ntype = typeof(node)

  if ntype == Expr
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
    elseif node.head == :call
      func = node.args[1]
      if func == TopNode(:box) ||
         func == TopNode(:tuple) ||
         func == TopNode(:getindex_bool_1d) ||
         func == :getindex
        return true
      end
    end
  elseif ntype == Symbol || ntype == SymbolNode || ntype == LambdaStaticData
    return true
  elseif ntype == Int64 || ntype == Int32 || ntype == Float64 || ntype == Float32
    return true
  end

  false
end

function removeAssertEqShape(args :: Array{Any,1}, state)
  newBody = Any[]
  for i = 1:length(args)
    if !(typeof(args[i]) == Expr && args[i].head == :assertEqShape && from_assertEqShape(args[i], state))
      push!(newBody, args[i])
    end
  end
  return newBody
end

function from_assertEqShape(node, state)
  dprintln(3,"from_assertEqShape ", node)
  a1 = node.args[1]    # first array to compare
  a2 = node.args[2]    # second array to compare
  a1_corr = getOrAddArrayCorrelation(a1.name, state)  # get the length set of the first array
  a2_corr = getOrAddArrayCorrelation(a2.name, state)  # get the length set of the second array
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

# :(=) assignment
# ast = [ ... ]
function from_assignment(ast::Array{Any,1}, depth, state)
  assert(length(ast) == 2)
  lhs = ast[1]
  rhs = ast[2]
  dprintln(3,"from_assignment lhs = ", lhs)
  dprintln(3,"from_assignment rhs = ", rhs)
  assert(typeof(lhs) == Symbol)
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
  lhsName = getSName(lhs)
  # Get liveness information for the current statement.
  statement_live_info = CompilerTools.LivenessAnalysis.find_top_number(state.top_level_number, state.block_lives)
  assert(statement_live_info != nothing)

  dprintln(3,statement_live_info)
  dprintln(3,"def = ", statement_live_info.def)

  # Make sure this variable is listed as a "def" for this statement.
  assert(CompilerTools.LivenessAnalysis.isDef(lhsName, statement_live_info))

  # If the lhs symbol is not in the live out information for this statement then it is dead.
  if !in(lhsName, statement_live_info.live_out) && hasNoSideEffects(rhs)
    dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
    # FIX FIX FIX...is this safe?
    eliminateStateVar(state, lhsName)
    # Eliminate the statement.
    return [], nothing
  end

  if typeof(rhs) == Expr
    out_typ = rhs.typ
    #dprintln(3, "from_assignment rhs is Expr, type = ", out_typ, " rhs.head = ", rhs.head, " rhs = ", rhs)

    # If we have "a = parfor(...)" then record that array "a" has the same length as the output array of the parfor.
    if rhs.head == :parfor
      the_parfor = rhs.args[1]
      if length(ast) > 2
        assert(typeof(ast[3]) == FusionSentinel)
        for i = 4:length(ast)
          rhs_entry = the_parfor.postParFor[end-1].args[2].args[i-2]
          assert(typeof(ast[i]) == SymbolNode)
          assert(typeof(rhs_entry) == SymbolNode)
          if rhs_entry.typ.name == Array.name
            add_merge_correlations(rhs_entry.name, ast[i].name, state)
          end
        end
      else
        if !(isa(out_typ, Tuple)) && out_typ.name == Array.name # both lhs and out_typ could be a tuple
          dprintln(3,"Adding parfor array length correlation ", lhs, " to ", rhs.args[1].postParFor[end].name)
          add_merge_correlations(the_parfor.postParFor[end].name, lhs, state)
        end
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
            si1 = state.meta2_typed[dim1.name]
            if si1[3] & ISASSIGNEDONCE == ISASSIGNEDONCE
              dprintln(3, "Will establish array length correlation for const size ", dim1)
              getOrAddSymbolCorrelation(lhs, state, [dim1.name])
            end
          end
        elseif rhs.args[2] == QuoteNode(:jl_alloc_array_2d)
          dim1 = rhs.args[7]
          dim2 = rhs.args[9]
          dprintln(3, "Detected 2D array allocation. dim1 = ", dim1, " dim2 = ", dim2)
          if typeof(dim1) == SymbolNode && typeof(dim2) == SymbolNode
            si1 = state.meta2_typed[dim1.name]
            si2 = state.meta2_typed[dim2.name]
            if (si1[3] & ISASSIGNEDONCE == ISASSIGNEDONCE) && (si2[3] & ISASSIGNEDONCE == ISASSIGNEDONCE)
              dprintln(3, "Will establish array length correlation for const size ", dim1, " ", dim2)
              getOrAddSymbolCorrelation(lhs, state, [dim1.name, dim2.name])
              dprintln(3, "correlations = ", state.array_length_correlation)
            end
          end
        end
      end
    end
  elseif typeof(rhs) == SymbolNode
    out_typ = rhs.typ
    if out_typ.name == Array.name
      # Add a length correlation of the form "a = b".
      dprintln(3,"Adding array length correlation ", lhs, " to ", rhs.name)
      add_merge_correlations(rhs.name, lhs, state)
    end
  else
    # Get the type of the lhs from its metadata declaration.
    out_typ = state.meta2_typed[lhs][2]
#    out_typ = nothing
#    throw(string("unknown RHS type in from_assignment."))
  end

  if haskey(state.num_var_assignments, lhs)
    state.num_var_assignments[lhs] = state.num_var_assignments[lhs] + 1
  else
    state.num_var_assignments[lhs] = 1
  end

  return [SymbolNode(lhs, out_typ), rhs], out_typ
end

function from_call(ast::Array{Any,1}, depth, state)
  assert(length(ast) >= 1)
  fun  = ast[1]
  args = ast[2:end]
  dprintln(2,"from_call fun = ", fun, " typeof fun = ", typeof(fun))
  if length(args) > 0
    dprintln(2,"first arg = ",args[1], " type = ", typeof(args[1]))
  end
  # symbols don't need to be translated
  if typeof(fun) != Symbol
      fun = from_expr(fun, depth, state, false)
      assert(isa(fun,Array))
      assert(length(fun) == 1)
      fun = fun[1]
  end
  args = from_exprs(args,depth+1,state)

  return [fun, args]
end

# Expands FusionSentinel assignments on a node-by-node basis.
function expand_assignment_tupleref(node, data, top_level_number, is_top_level)
  dprintln(3,"expand_assignment_tupleref: node = ", node)
  if typeof(node) == Expr
    dprintln(3,"expand_assignment_tupleref: head = ", node.head)
    if node.head == symbol('=')
      dprintln(3,"expand_assignment_tupleref: found assignment length(node.args) = ", length(node.args))
      # For the moment, parallel IR fusion is the only thing that generates assignment expressions with more than 2 args.
      if length(node.args) > 2
        # Sanity check that third arg is a sentinel for fusion.
        assert(typeof(node.args[3]) == FusionSentinel)
        # Then there must be at least two things in the tuple for it to have been generated.
        assert(length(node.args) >= 5)
        # Save part of the tuple.
        items = node.args[4:end]
        # Strip off the extra stuff that assignment doesn't expect and save it back.
        node.args = node.args[1:2]

        lhs = node.args[1]

        assert(typeof(lhs) == SymbolNode)

        # An array of expressions will be returned...the original plus one for each extraction from the tuple.
        ret = Any[]
        push!(ret, node)

        num_items = length(items)
        for i = 1:num_items
          if typeof(items[i]) != SymbolNode
             dprintln(0,"expand_assignment_tupleref error, node = ", node, " items = ", items)
          end
          assert(typeof(items[i]) == SymbolNode)
          new_tupleref = mk_assignment_expr(items[i], mk_tupleref_expr(lhs, i, items[i].typ))
          push!(ret, new_tupleref)
        end

        dprintln(3,"Expanding assignment into = ", ret)
        return ret
      end
    end
  end
  nothing
end

type CopyPropagateState
  lives  :: CompilerTools.LivenessAnalysis.BlockLiveness
  copies :: Dict{Symbol,Symbol}

  function CopyPropagateState(l, c)
    new(l,c)
  end
end

# In each basic block, if there is a "copy" (i.e., something of the form "a = b") then put
# that in copies as copies[a] = b.  Then, later in the basic block if you see the symbol
# "a" then replace it with "b".  Note that this is not SSA so "a" may be written again
# and if it is then it must be removed from copies.
function copy_propagate(node, data::CopyPropagateState, top_level_number, is_top_level, read)
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
          if in(def, copy)
            dprintln(3,"LHS or RHS of data.copies is modified so removing ", copy," from data.copies.")
            # Then remove the lhs = rhs entry from copies.
            delete!(data.copies, copy[1])
          end
        end
      end
    end

    if isa(node, LabelNode) || isa(node, GotoNode) || (isa(node, Expr) && is(node.head, :gotoifnot))
      # Only copy propagate within a basic block.  this is now a new basic block.
      data.copies = Dict{Symbol,Symbol}() 
    elseif isAssignmentNode(node)
      dprintln(3,"Is an assignment node.")
      lhs = node.args[1] = get_one(AstWalk(node.args[1], copy_propagate, data))
      dprintln(4,lhs)
      rhs = node.args[2] = get_one(AstWalk(node.args[2], copy_propagate, data))
      dprintln(4,rhs)

      if typeof(rhs) == Symbol || typeof(rhs) == SymbolNode
        dprintln(3,"Creating copy, lhs = ", lhs, " rhs = ", rhs)
        data.copies[getSName(lhs)] = getSName(rhs)
      end
      return [node]
    end
  end

  if isa(node, Symbol)
    if haskey(data.copies, node)
      dprintln(3,"Replacing ", node, " with ", data.copies[node])
      return [data.copies[node]]
    end
  elseif isa(node, SymbolNode)
    if haskey(data.copies, node.name)
      dprintln(3,"Replacing ", node.name, " with ", data.copies[node.name])
      return [SymbolNode(data.copies[node.name], node.typ)]
    end
  elseif isa(node, DomainLambda)
    dprintln(3,"Found DomainLambda in copy_propagate, dl = ", node)
    intersection_dict = Dict{Symbol,Any}()
    for copy in data.copies
      if haskey(node.escapes, copy[1])
        ed = node.escapes[copy[1]]
        intersection_dict[copy[1]] = SymbolNode(copy[2], ed.typ)
        delete!(node.escapes, copy[1])
        node.escapes[copy[2]] = ed
      end
    end 
    dprintln(3,"Intersection dict = ", intersection_dict)
    if !isempty(intersection_dict)
      origBody      = node.genBody
      newBody(args) = DomainIR.replaceWithDict(origBody(args), intersection_dict)
      node.genBody  = newBody
      return [node]
    end 
  end
  nothing
end

type RemoveDeadState
  lives :: CompilerTools.LivenessAnalysis.BlockLiveness
end

function remove_dead(node, data::RemoveDeadState, top_level_number, is_top_level, read)
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
          # remove a dead store
          if !in(lhs_sym, live_info.live_out)
            dprintln(3,"remove_dead lhs is NOT live out")
            if hasNoSideEffects(rhs)
              # FIX FIX FIX...eliminateStateVar here for lhs?
              dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
              return Any[]
            else
              # just eliminate the assignment but keep the rhs
              dprintln(3,"Eliminating dead variable but keeping rhs, dead = ", lhs_sym, " rhs = ", rhs)
              return [rhs]
            end
          end
        end
      end
    end
  end
  nothing
end


type RemoveNoDepsState
  lives :: CompilerTools.LivenessAnalysis.BlockLiveness
  top_level_no_deps
  hoistable_scalars
  dict_sym
end

# Works with remove_no_deps below to move statements with no dependencies to the beginning of the AST.
function insert_no_deps_beginning(node, data::RemoveNoDepsState, top_level_number, is_top_level, read)
  if is_top_level && top_level_number == 1
    return [data.top_level_no_deps, node]
  end
  nothing
end

# Not sure if this will be useful or not.  The idea is that it gathers up nodes that do not use
# any variable and removes them from the AST into top_level_no_deps.  This works in conjunction with
# insert_no_deps_beginning above to move these statements with no dependencies to the beginning of the AST
# where they can't prevent fusion.
function remove_no_deps(node, data::RemoveNoDepsState, top_level_number, is_top_level, read)
  dprintln(3,"remove_no_deps starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
  dprintln(3,"remove_no_deps node = ", node, " type = ", typeof(node))
  if typeof(node) == Expr
    dprintln(3,"node.head = ", node.head)
  end
  ntype = typeof(node)

  if is_top_level
    dprintln(3,"remove_no_deps is_top_level")
    live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
    if live_info == nothing
      dprintln(3,"remove_no_deps no live_info")
      # Remove line number statements.
      if ntype == LineNumberNode || (ntype == Expr && node.head == :line)
        return Any[]
      end
    else
      dprintln(3,"remove_no_deps live_info = ", live_info)
      dprintln(3,"remove_no_deps live_info.use = ", live_info.use)

      if isa(node, LabelNode) || isa(node, GotoNode) || (isa(node, Expr) && is(node.head, :gotoifnot))
        # empty the state at the end or begining of a basic block
        data.dict_sym = Dict{Symbol,Any}()
      elseif isAssignmentNode(node)
        dprintln(3,"Is an assignment node.")
        lhs = node.args[1]
        dprintln(4,lhs)
        rhs = node.args[2]
        dprintln(4,rhs)

        if isa(rhs, Expr) && (is(rhs.head, :parfor) || is(rhs.head, :mmap!))
          # always keep parfor assignment in order to work with fusion
          dprintln(3, "keep assignment due to parfor or mmap! node")
          return [node]
        end
        if typeof(lhs) == SymbolNode || typeof(lhs) == Symbol
          lhs_sym = getSName(lhs)
          dprintln(3,"remove_no_deps found assignment with lhs symbol ", lhs, " ", rhs, " typeof(rhs) = ", typeof(rhs))
          # remove a dead store
          if !in(lhs_sym, live_info.live_out)
            dprintln(3,"remove_no_deps lhs is NOT live out")
            if hasNoSideEffects(rhs)
              # FIX FIX FIX...eliminateStateVar here for lhs?
              dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
              return Any[]
            else
              # just eliminate the assignment but keep the rhs
              dprintln(3,"Eliminating dead variable but keeping rhs, dead = ", lhs_sym)
              return [rhs]
            end
          else
            dprintln(3,"remove_no_deps lhs is live out")
            if typeof(rhs) == SymbolNode || typeof(rhs) == Symbol
              rhs_sym = getSName(rhs)
              dprintln(3,"remove_no_deps rhs is symbol ", rhs_sym)
              if !in(rhs_sym, live_info.live_out)
                dprintln(3,"remove_no_deps rhs is NOT live out")
                if haskey(data.dict_sym, rhs_sym)
                  prev_expr = data.dict_sym[rhs_sym]
                  prev_expr.args[1] = lhs_sym
                  delete!(data.dict_sym, rhs_sym)
                  data.dict_sym[lhs_sym] = prev_expr
                  dprintln(3,"Lhs is live but rhs is not so substituting rhs for lhs ", lhs_sym, " => ", rhs_sym)
                  dprintln(3,"New expr = ", prev_expr)
                  return Any[]
                end
              else
                dprintln(3,"Lhs and rhs are live so forgetting assignment ", lhs_sym, " ", rhs_sym)
                delete!(data.dict_sym, rhs_sym)
              end
            else
              data.dict_sym[lhs_sym] = node
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
            return Any[]
          end
        end
      end
    end
  end
  nothing
end

function extractArrayEquivalencies(node, state)
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
  assert(len_input_arrays > 0)

  # Second arg is a DomainLambda
  ftype = typeof(input_args[2])
  dprintln(2,"extractArrayEquivalencies function = ",input_args[2])
  if(ftype != DomainLambda)
    throw(string("extractArrayEquivalencies second input_args should be a DomainLambda but is of type ", typeof(input_args[2])))
  end

  # Get the correlation set of the first input array.
  main_length_correlation = getOrAddArrayCorrelation(input_arrays[1].name, state)

  # Make sure each input array is a SymbolNode
  # Also, create indexed versions of those symbols for the loop body
  for i = 2:len_input_arrays
    argtyp = typeof(input_arrays[i])
    dprintln(3,"extractArrayEquivalencies input_array[i] = ", input_arrays[i], " type = ", argtyp)
    assert(argtyp == SymbolNode)
    this_correlation = getOrAddArrayCorrelation(input_arrays[i].name, state)
    # Verify that all the inputs are the same size by verifying they are in the same correlation set.
    if this_correlation != main_length_correlation
      merge_correlations(state, main_length_correlation, this_correlation)
    end
  end

  return main_length_correlation
end

# Make sure all the dimensions are SymbolNodes.
# Make sure each dimension variable is assigned to only once in the function.
# Extract just the dimension variables names into dim_names and then register the correlation from lhs to those dimension names.
function checkAndAddSymbolCorrelation(lhs, state, dim_array)
  dim_names = Symbol[]
  for i = 1:length(dim_array)
    if typeof(dim_array[i]) != SymbolNode
      return false
    end
    if !haskey(state.meta2_typed, dim_array[i].name)
      #dprintln(0, "Didn't find ", dim_array[i].name, " in state.meta2_typed.")
      dprintln(3, state.meta2_typed)
      throw(string("Didn't find ", dim_array[i].name, " in state.meta2_typed."))
    end
    si = state.meta2_typed[dim_array[i].name]
    if si[3] & ISASSIGNEDONCE != ISASSIGNEDONCE
      return false
    end
    push!(dim_names, dim_array[i].name)
  end

  dprintln(3, "Will establish array length correlation for const size lhs = ", lhs, " dims = ", dim_names)
  getOrAddSymbolCorrelation(lhs, state, dim_names)
  return true
end

function processAndUpdateBody(lambda :: Expr, f :: Function, state)
  assert(lambda.head == :lambda) 
  lambda.args[3].args = f(lambda.args[3].args, state)
  return lambda
end

function removeNothingStmts(args :: Array{Any,1}, state)
  newBody = Any[]
  for i = 1:length(args)
    if args[i] != nothing
      push!(newBody, args[i])
    end
  end
  return newBody
end

function create_equivalence_classes(node, state::IntelPSE.ParallelIR.expr_state, top_level_number, is_top_level, read)
  dprintln(3,"create_equivalence_classes starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
  dprintln(3,"create_equivalence_classes node = ", node, " type = ", typeof(node))
  if typeof(node) == Expr
    dprintln(3,"node.head = ", node.head)
  end
  ntype = typeof(node)

  if ntype == Expr && node.head == :lambda
    assert(length(node.args) == 3)
    param = node.args[1]
    meta  = node.args[2]
    body  = node.args[3]
    dprintln(3,"Is a lambda node. typeof(param) = ", typeof(param), " typeof(meta) = ", typeof(meta))
    dprintln(3,param)
    dprintln(3,meta)
    assert(typeof(body) == Expr)
    assert(body.head == :body)

    save_param = state.param
    save_meta2 = state.meta2
    save_meta2_typed = state.meta2_typed

    state.param = map(x -> getSName(x), param)
    state.meta2 = createVarSet(meta[1])
    state.meta2_typed = createVarDict(meta[2])
    
    AstWalk(body, create_equivalence_classes, state)

    state.param = save_param
    state.meta2 = save_meta2
    state.meta2_typed = save_meta2_typed

    return node
  end

  if is_top_level
    dprintln(3,"create_equivalence_classes is_top_level")

    if isAssignmentNode(node)
      dprintln(3,"Is an assignment node.")
      lhs = node.args[1]
      dprintln(4,lhs)
      rhs = node.args[2]
      dprintln(4,rhs)

      if isa(rhs, Expr)
        if rhs.head == :assertEqShape
          dprintln(3,"Creating array length assignment from assertEqShape in remove_no_deps")
          from_assertEqShape(rhs, state)
        elseif rhs.head == :alloc
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
            array_param = rhs.args[2]                  # length takes one param, which is the array
            assert(typeof(array_param) == SymbolNode)  # should be a SymbolNode
            array_param_type = array_param.typ         # get its type
            if ndims(array_param_type) == 1            # can only associate when number of dimensions is 1
              dim_symbols = [getSName(lhs)]
              dprintln(3,"Adding symbol correlation from arraylen, name = ", rhs.args[2].name, " dims = ", dim_symbols)
              checkAndAddSymbolCorrelation(rhs.args[2].name, state, dim_symbols)
            end
          elseif rhs.args[1] == TopNode(:arraysize)
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
          rhs_corr = extractArrayEquivalencies(rhs, state)
          dprintln(3,"lhs = ", lhs, " type = ", typeof(lhs))
          if typeof(lhs) == SymbolNode
            assert(isArrayType(lhs))
            lhs_corr = getOrAddArrayCorrelation(lhs.name, state) 
            merge_correlations(state, lhs_corr, rhs_corr)
            dprintln(3,"Correlations after map merge into lhs = ", state.array_length_correlation)
          elseif typeof(lhs) == Symbol
            lhs_corr = getOrAddArrayCorrelation(lhs, state) 
            merge_correlations(state, lhs_corr, rhs_corr)
            dprintln(3,"Correlations after map merge into lhs = ", state.array_length_correlation)
          end
        end
      end
    elseif isa(node, Expr)
      if node.head == :mmap! || node.head == :mmap || node.head == :map! || node.head == :map
        extractArrayEquivalencies(node, state)
      end
    else
      dprintln(3,"Not an assignment or expr node.")
    end
  end
  nothing
end


# If a definition of a mmap is only used once and not aliased, it can be inlined into its
# use side as long as its dependencies have not been changed.
# FIXME: is the implementation still correct when branches are present?
function mmapInline(ast, lives, uniqSet)
  body = ast.args[3]
  defs = Dict{Symbol, Int}()
  usedAt = Dict{Symbol, Int}()
  modifiedAt = Dict{Symbol, Array{Int}}()
  shapeAssertAt = Dict{Symbol, Array{Int}}()
  function modify!(dict, lhs, i)
    if haskey(dict, lhs)
      push!(dict[lhs], i)
    else
      push!(dict, lhs, Int[i])
    end
  end
  assert(isa(body, Expr) && is(body.head, :body))
  # first do a loop to see which def is only referenced once
  for i =1:length(body.args)
    expr = body.args[i]
    head = isa(expr, Expr) ? expr.head : nothing
    function check_used(expr)
      if isa(expr, Expr)
        if is(expr.head, :assertEqShape)
        # handle assertEqShape separately, do not consider them
        # as valid references
          for arg in expr.args
            s = isa(arg, SymbolNode) ? arg.name : arg 
            if isa(s, Symbol)
              modify!(shapeAssertAt, s, i)
            end
          end
        else
          for arg in expr.args
            check_used(arg)
          end
        end
      elseif isa(expr, Symbol)
        if haskey(usedAt, expr) # already used? remove from defs
          delete!(defs, expr)
        else
          push!(usedAt, expr, i)
        end 
      elseif isa(expr, SymbolNode)
        if haskey(usedAt, expr.name) # already used? remove from defs
          dprintln(3, "MI: def ", expr.name, " removed due to multi-use")
          delete!(defs, expr.name)
        else
          push!(usedAt, expr.name, i)
        end 
      elseif isa(expr, Array) || isa(expr, Tuple)
        for e in expr
          check_used(e)
        end
      end
    end
    # record usedAt, and reject those used more than once
    # record definition
    if is(head, :(=))
      lhs = expr.args[1]
      rhs = expr.args[2]
      check_used(rhs)
      assert(isa(lhs, Symbol))
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
        if (ok) push!(defs, lhs, i) end
        dprintln(3, "MI: def for ", lhs, " ok=", ok, " defs=", defs)
      end
    else
      check_used(expr)
    end
    # check if args may be modified in place
    if is(head, :mmap!)
      for j in 1:length(expr.args[2].outputs)
        v = expr.args[1][j]
        if isa(v, SymbolNode)
          v = v.name
        end
        if isa(v, Symbol)
          modify!(modifiedAt, v, i)
        end
      end
    elseif is(head, :stencil!)
      krnStat = expr.args[1]
      iterations = expr.args[2]
      bufs = expr.args[3]
      for j in krnStat.modified
        k = krnStat.modified[j]
        s = bufs[k]
        if isa(s, SymbolNode) s = s.name end
        modify!(modifiedAt, s, i)
      end
      if !((isa(iterations, Number) && iterations == 1) || krnStat.rotateNum == 0)
        for j in 1:min(krnStat.rotatNum, length(bufs))
          s = bufs[j]
          if isa(s, SymbolNode) s = s.name end
          modify!(modifiedAt, s, i)
        end
      end
    end
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
      for v in src.args
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
      # function that inlines from src into dst
      function inline!(src, dst)
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
        assert(pos > 0)
        args = [args[1:pos-1], args[pos+1:end], src.args[1]]
        f = src.args[2]
        assert(length(f.outputs)==1)
        tmp_v = gensym()
        tmp_t = f.outputs[1]
        dst.args[1] = args
        g = dst.args[2]
        inputs = g.inputs
        g_inps = length(inputs) 
        inputs = [inputs[1:pos-1], inputs[pos+1:end], f.inputs]
        locals = merge(f.locals, g.locals)
        escapes = merge(f.escapes, g.escapes)
        push!(locals, tmp_v, DomainIR.VarDef(tmp_t, ISASSIGNED | ISASSIGNEDONCE, nothing))
        dst.args[2] = DomainLambda(inputs, g.outputs, 
          args -> begin
            fb = f.genBody(args[g_inps:end])
            expr = TypedExpr(tmp_t, :(=), tmp_v, fb[end].args[1])
            gb = g.genBody([args[1:pos-1], [SymbolNode(tmp_v, tmp_t)], args[pos:g_inps-1]])
            return [fb[1:end-1], [expr], gb]
          end, locals, escapes)
        DomainIR.mmapRemoveDupArg!(dst)
      end
      function eliminateShapeAssert(dict, lhs)
        if haskey(dict, lhs)
          for k in dict[lhs]
            dprintln(3, "MI: eliminate shape assert at line ", k)
            body.args[k] = nothing
          end
        end
      end
      if isa(dst, Expr) && is(dst.head, :mmap)
         # inline mmap into mmap
        inline!(src, dst)
        body.args[i] = nothing
        eliminateShapeAssert(shapeAssertAt, lhs)
        dprintln(3, "MI: result: ", body.args[j])
      elseif isa(dst, Expr) && is(dst.head, :mmap!)
        s = dst.args[1][1]
        if isa(s, SymbolNode) s = s.name end
        if s == lhs 
          # when lhs is the inplace array that dst operates on
          # change dst to mmap
          inline!(src, dst)
          dst.head = :mmap
        else
          # otherwise just normal inline
          inline!(src, dst)
        end
        body.args[i] = nothing
        eliminateShapeAssert(shapeAssertAt, lhs)
        # inline mmap into mmap!
        dprintln(3, "MI: result: ", body.args[j])
      else
        # otherwise ignore, e.g., when dst is some simple assignments.
      end
    end
  end
end

# Try to hoist allocation outside the loop if possible.
function hoistAllocation(ast, lives, domLoop, state::IntelPSE.ParallelIR.expr_state)
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

# If an arguments of a mmap dies aftewards, and is not aliased, then
# we can safely change the mmap to mmap!.
function mmapToMmap!(ast, lives, uniqSet)
  body = ast.args[3]
  assert(isa(body, Expr) && is(body.head, :body))
  for i =1:length(body.args)
    expr = body.args[i]
    if isa(expr, Expr) && is(expr.head, :(=))
      lhs = expr.args[1]
      rhs = expr.args[2]
      # right now assume all
      assert(isa(lhs, Symbol))
      if isa(rhs, Expr) && is(rhs.head, :mmap)
        args = rhs.args[1]
        tls = CompilerTools.LivenessAnalysis.find_top_number(i, lives)
        assert(tls != nothing)
        assert(CompilerTools.LivenessAnalysis.isDef(lhs, tls))
        dprintln(4, "mmap lhs=", lhs, " args=", args, " live_out = ", tls.live_out)
        reuse = nothing
        j = 0
        while j < length(args)
          j = j + 1
          v = args[j]
          if isa(v, SymbolNode) && !in(v.name, tls.live_out) && in(v.name, uniqSet)
            reuse = v
            break
          end
        end
        if !is(reuse, nothing)
          rhs.head = :mmap!
          dprintln(2, "mmapToMMap!: successfully reuse ", reuse, " for ", lhs)
          if j != 1
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

mmap_to_mmap! = 0
function PIRInplace(x)
  global mmap_to_mmap! = x
end

hoist_allocation = 0
function PIRHoistAllocation(x)
  global hoist_allocation = x
end

bb_reorder = 0
function PIRBbReorder(x)
  global bb_reorder = x
end 

shortcut_array_assignment = 0
function PIRShortcutArrayAssignment(x)
  global shortcut_array_assignment = x
end

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

function isDomainNode(ast)
  asttyp = typeof(ast)
  if asttyp == Expr
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
  end 
  return false
end

function mustRemainLastStatementInBlock(node)
  if typeof(node) == GotoNode
    return true
  elseif typeof(node) == Expr && node.head == :gotoifnot
    return true
  end
  return false
end

function maxFusion(bl :: CompilerTools.LivenessAnalysis.BlockLiveness)
  # We will try to optimize the order in each basic block.
  for bb in collect(values(bl.basic_blocks))
    if false
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
      earliest_parfor = 1
      found_change = true

      while found_change
        found_change = false

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

function pirPrintDl(dbg_level, dl)
  dprintln(dbg_level, "inputs = ", dl.inputs)
  dprintln(dbg_level, "output = ", dl.outputs)
  dprintln(dbg_level, "locals = ", dl.locals)
  dprintln(dbg_level, "escapes = ", dl.escapes)
end

function getMaxLabel(max_label, stmts :: Array{Any, 1})
  for i =1:length(stmts)
    if isa(stmts[i], LabelNode)
      max_label = max(max_label, stmts[i].label)
    end
  end
  return max_label
end

function lambdaFromStmtsMeta(stmts, params=Any[], meta=Any[Any[],Any[],Any[]])
  Expr(:lambda, deepcopy(params), deepcopy(meta), Expr(:body, stmts...))
  end

function lambdaFromDomainLambda(stmts, domain_lambda, dl_inputs)
  inputs_as_symbols = map(x -> x.name, dl_inputs)
  type_data = Any[]
  input_arrays = Symbol[]
  for di in dl_inputs
    push!(type_data, [di.name, di.typ, 0])
    if isArrayType(di.typ)
      push!(input_arrays, di.name)
    end
  end
  dprintln(3,"inputs = ", inputs_as_symbols)
  dprintln(3,"types = ", type_data)
  dprintln(3,"DomainLambda is:")
  pirPrintDl(3, domain_lambda)
  ast = Expr(:lambda, inputs_as_symbols, Any[Symbol[],type_data,Any[]], Expr(:body, stmts...))
  assert(typeof(ast) == Expr)
  assert(ast.head == :lambda)
  return (ast, input_arrays) 
end

function nested_function_exprs(max_label, stmts, domain_lambda, dl_inputs)
  dprintln(2,"nested_function_exprs called with ", stmts, " of type = ", typeof(stmts))
  if !isa(stmts, Array)
    return stmts
  end
  (ast, input_arrays) = lambdaFromDomainLambda(stmts, domain_lambda, dl_inputs)
  dprintln(1,"Starting nested_function_exprs. ast = ", ast, " input_arrays = ", input_arrays)

  start_time = time_ns()

  dprintln(1,"Starting liveness analysis.")
  lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
  dprintln(1,"Finished liveness analysis.")

  dprintln(1,"Liveness Analysis time = ", ns_to_sec(time_ns() - start_time))

  mtm_start = time_ns()

  if mmap_to_mmap! != 0
    dprintln(1, "starting mmap to mmap! transformation.")
    uniqSet = AliasAnalysis.analyze_lambda(ast, lives)
    dprintln(3, "uniqSet = ", uniqSet)
    mmapInline(ast, lives, uniqSet)
    lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
    uniqSet = AliasAnalysis.analyze_lambda(ast, lives)
    mmapToMmap!(ast, lives, uniqSet)
    dprintln(1, "Finished mmap to mmap! transformation.")
    dprintln(3, "AST = ", ast)
  end

  dprintln(1,"mmap_to_mmap! time = ", ns_to_sec(time_ns() - mtm_start))

  # We pass only the non-array params to the rearrangement code because if we pass array params then
  # the code will detect statements that depend only on array params and move them to the top which
  # leaves other non-array operations after that and so prevents fusion.
  dprintln(3,"All params = ", ast.args[1])
  non_array_params = Set{Symbol}()
  for param in ast.args[1]
    if !in(param, input_arrays) && CompilerTools.LivenessAnalysis.countSymbolDefs(param, lives) == 0
      push!(non_array_params, param)
    end
  end
  dprintln(3,"Non-array params = ", non_array_params)

  # find out max_label
  body = ast.args[3]
  assert(isa(body, Expr) && is(body.head, :body))
  max_label = getMaxLabel(max_label, body.args)

  eq_start = time_ns()

  new_vars = expr_state(lives, max_label, input_arrays)
  if pre_eq != 0
    dprintln(3,"Creating equivalence classes.")
    AstWalk(ast, create_equivalence_classes, new_vars)
    dprintln(3,"Done creating equivalence classes.")
  end

  dprintln(1,"Creating equivalence classes time = ", ns_to_sec(time_ns() - eq_start))

  rep_start = time_ns()

  for i = 1:rearrange_passes
    dprintln(1,"Removing statement with no dependencies from the AST with parameters = ", ast.args[1])
    rnd_state = RemoveNoDepsState(lives, Any[], non_array_params, Dict{Symbol,Any}())
    ast = get_one(AstWalk(ast, remove_no_deps, rnd_state))
    dprintln(3,"ast after no dep stmts removed = ", ast)
      
    dprintln(3,"top_level_no_deps = ", rnd_state.top_level_no_deps)

    dprintln(1,"Adding statements with no dependencies to the start of the AST.")
    ast = get_one(AstWalk(ast, insert_no_deps_beginning, rnd_state))
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

# ENTRY
function from_expr(function_name, ast::Any, input_arrays)
  assert(typeof(ast) == Expr)
  assert(ast.head == :lambda)
  dprintln(1,"Starting main ParallelIR.from_expr.  function = ", function_name, " ast = ", ast, " input_arrays = ", input_arrays)

  start_time = time_ns()

  #CompilerTools.LivenessAnalysis.set_debug_level(3)

  dprintln(1,"Starting liveness analysis.")
  lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
#  udinfo = CompilerTools.UDChains.getUDChains(lives)
  dprintln(3,"lives = ", lives)
#  dprintln(3,"udinfo = ", udinfo)
  dprintln(1,"Finished liveness analysis.")

  dprintln(1,"Liveness Analysis time = ", ns_to_sec(time_ns() - start_time))

  mtm_start = time_ns()

  if mmap_to_mmap! != 0
    dprintln(1, "starting mmap to mmap! transformation.")
    uniqSet = AliasAnalysis.analyze_lambda(ast, lives)
    dprintln(3, "uniqSet = ", uniqSet)
    mmapInline(ast, lives, uniqSet)
    lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
    uniqSet = AliasAnalysis.analyze_lambda(ast, lives)
    mmapToMmap!(ast, lives, uniqSet)
    dprintln(1, "Finished mmap to mmap! transformation.")
#    dprintln(3, "AST = ", ast)
    printLambda(3, ast)
  end

  dprintln(1,"mmap_to_mmap! time = ", ns_to_sec(time_ns() - mtm_start))

  # We pass only the non-array params to the rearrangement code because if we pass array params then
  # the code will detect statements that depend only on array params and move them to the top which
  # leaves other non-array operations after that and so prevents fusion.
  dprintln(3,"All params = ", ast.args[1])
  non_array_params = Set{Symbol}()
  for param in ast.args[1]
    if !in(param, input_arrays) && CompilerTools.LivenessAnalysis.countSymbolDefs(param, lives) == 0
      push!(non_array_params, param)
    end
  end
  dprintln(3,"Non-array params = ", non_array_params)

  # find out max_label
  body = ast.args[3]
  assert(isa(body, Expr) && is(body.head, :body))
  max_label = getMaxLabel(0, body.args)

  rep_start = time_ns()

  for i = 1:rearrange_passes
    dprintln(1,"Removing statement with no dependencies from the AST with parameters = ", ast.args[1])
    rnd_state = RemoveNoDepsState(lives, Any[], non_array_params, Dict{Symbol,Any}())
    ast = get_one(AstWalk(ast, remove_no_deps, rnd_state))
    dprintln(3,"ast after no dep stmts removed = ")
    printLambda(3, ast)
      
    dprintln(3,"top_level_no_deps = ", rnd_state.top_level_no_deps)

    dprintln(1,"Adding statements with no dependencies to the start of the AST.")
    ast = get_one(AstWalk(ast, insert_no_deps_beginning, rnd_state))
    dprintln(3,"ast after no dep stmts re-inserted = ")
    printLambda(3, ast)

    dprintln(1,"Re-starting liveness analysis.")
    lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
    dprintln(1,"Finished liveness analysis.")
  end

  dprintln(1,"Rearranging passes time = ", ns_to_sec(time_ns() - rep_start))

  processAndUpdateBody(ast, removeNothingStmts, nothing)
  dprintln(3,"ast after removing nothing stmts = ")
  printLambda(3, ast)
  lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)

  ast   = get_one(AstWalk(ast, copy_propagate, CopyPropagateState(lives, Dict{Symbol,Symbol}())))
  lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
  dprintln(3,"ast after copy_propagate = ")
  printLambda(3, ast)

  ast   = get_one(AstWalk(ast, remove_dead, RemoveDeadState(lives)))
  lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
  dprintln(3,"ast after remove_dead = ")
  printLambda(3, ast)

  eq_start = time_ns()

  new_vars = expr_state(lives, max_label, input_arrays)
  if pre_eq != 0
    dprintln(3,"Creating equivalence classes.")
    AstWalk(ast, create_equivalence_classes, new_vars)
    dprintln(3,"Done creating equivalence classes.")
    dprintln(3,"symbol_correlations = ", new_vars.symbol_array_correlation)
    dprintln(3,"array_correlations  = ", new_vars.array_length_correlation)
  end

  dprintln(1,"Creating equivalence classes time = ", ns_to_sec(time_ns() - eq_start))
 
  processAndUpdateBody(ast, removeAssertEqShape, new_vars)
  dprintln(3,"ast after removing assertEqShape = ")
  printLambda(3, ast)
  lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)

  if bb_reorder != 0
    maxFusion(lives)
    # Set the array of statements in the Lambda body to a new array constructed from the updated basic blocks.
    ast.args[3].args = CompilerTools.CFGs.createFunctionBody(lives.cfg)
    lives = CompilerTools.LivenessAnalysis.from_expr(ast, DomainIR.dir_live_cb, nothing)
    dprintln(3,"ast after maxFusion = ")
    printLambda(3, ast)
  end

  dprintln(1,"Doing conversion to parallel IR.")

  new_vars.block_lives = lives

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

function isbitstuple(a)
  if isa(a, Tuple)
    for i in a
      if !isbits(i)
        return false
      end
    end
    return true
  end
  return false
end

function from_expr(ast::Any, depth, state, top_level)
  if is(ast, nothing)
    return [nothing]
  end
  if typeof(ast) == LambdaStaticData
      ast = uncompressed_ast(ast)
  end
  dprintln(2,"from_expr depth=",depth," ")
  asttyp = typeof(ast)
  if asttyp == Expr
    dprint(2,"Expr ")
    head = ast.head
    args = ast.args
    typ  = ast.typ
    dprintln(2,head, " ", args)
    if head == :lambda
        args = from_lambda(args,depth,state)
    elseif head == :body
        dprintln(3,"Processing body start")
        args = from_exprs(args,depth+1,state)
        dprintln(3,"Processing body end")
    elseif head == :(=)
        dprintln(3,"Before from_assignment typ is ", typ)
        args, new_typ = from_assignment(args,depth,state)
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
        args, typ = mk_parfor_args_from_mmap(args, state)
        dprintln(1,"switching to parfor node for mmap")
    elseif head == :mmap!
        head = :parfor
        args, typ = mk_parfor_args_from_mmap!(args, state)
        dprintln(1,"switching to parfor node for mmap!")
    elseif head == :reduce
        head = :parfor
        args, typ = mk_parfor_args_from_reduce(args, state)
        dprintln(1,"switching to parfor node for reduce")
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
        elemTyp = args[1]
        sizes = args[2]
        n = length(sizes)
        assert(n >= 1 && n <= 3)
        name = symbol(string("jl_alloc_array_", n, "d"))
        appTypExpr = Expr(:call1, TopNode(:apply_type), :Array, elemTyp, n)
        appTypExpr.typ = Type{Array{elemTyp,n}}
        tupExpr = Expr(:call1, TopNode(:tuple), :Any, [ :Int for i=1:n ]...)
        tupExpr.typ = ntuple(n+1, i -> (i==1) ? Type{Any} : Type{Int})
        realArgs = Any[QuoteNode(name), appTypExpr, tupExpr, Array{elemTyp,n}, 0]
        for i=1:n
          push!(realArgs, sizes[i])
          push!(realArgs, 0)
        end
        args = vcat(TopNode(:ccall), realArgs)
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
    else
        #println("from_expr: unknown Expr head :", head)
        throw(string("from_expr: unknown Expr head :", head))
    end
    ast = Expr(head, args...)
    dprintln(3,"New expr type = ", typ)
    ast.typ = typ
  elseif asttyp == Symbol
    dprintln(2,"Symbol type")
    #skip
  elseif asttyp == SymbolNode # name, typ
    dprintln(2,"SymbolNode type")
    #skip
  elseif asttyp == TopNode    # name
    dprintln(2,"TopNode type")
    #skip
  elseif asttyp == GlobalRef
    mod = ast.mod
    name = ast.name
   # typ = ast.typ  # FIXME: is this type needed?
   typ = typeof(mod)
    dprintln(2,"GlobalRef type ",typeof(mod))
  elseif asttyp == QuoteNode
    value = ast.value
    #TODO: fields: value
    dprintln(2,"QuoteNode type ",typeof(value))
  elseif asttyp == LineNumberNode
    # remove line numbers
    return []
  elseif asttyp == LabelNode
    #skip
  elseif asttyp == GotoNode
    #skip
  elseif asttyp == DataType
    #skip
  elseif asttyp == ()
    #skip
  elseif asttyp == ASCIIString
    #skip
  elseif asttyp == NewvarNode
    #skip
  elseif asttyp == Nothing
    #skip
  elseif asttyp == Module
    #skip
  #elseif asttyp == Int64 || asttyp == Int32 || asttyp == Float64 || asttyp == Float32
  elseif isbits(asttyp)
    #skip
  elseif isbitstuple(ast)
    #skip
  else
#    dprintln(2,"from_expr: unknown AST (", typeof(ast), ",", ast, ")")
    throw(string("from_expr: unknown AST (", typeof(ast), ",", ast, ")"))
  end
  return [ast]
end

function get_one(ast)
  assert(isa(ast,Array))
  assert(length(ast) == 1)
  ast[1]
end

type DirWalk
  callback
  cbdata
end

function AstWalkCallback(x, dw::DirWalk, top_level_number, is_top_level, read)
  dprintln(4,"PIR AstWalkCallback starting")
  ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
  dprintln(4,"PIR AstWalkCallback ret = ", ret)
  if ret != nothing
    return [ret]
  end

  asttyp = typeof(x)
  if asttyp == Expr
    head = x.head
    args = x.args
#    typ  = x.typ
    if head == :parfor
      cur_parfor = args[1]
      for i = 1:length(cur_parfor.preParFor)
        #x.args[1].preParFor[i] = AstWalker.get_one(AstWalk(cur_parfor.preParFor[i], dw.callback, dw.cbdata))
        AstWalk(cur_parfor.preParFor[i], dw.callback, dw.cbdata)
      end
      for i = 1:length(cur_parfor.loopNests)
        AstWalk(mk_assignment_expr(cur_parfor.loopNests[i].indexVariable,1), dw.callback, dw.cbdata)
        AstWalk(cur_parfor.loopNests[i].lower, dw.callback, dw.cbdata)
        AstWalk(cur_parfor.loopNests[i].upper, dw.callback, dw.cbdata)
        AstWalk(cur_parfor.loopNests[i].step, dw.callback, dw.cbdata)
      end
      for i = 1:length(cur_parfor.reductions)
        AstWalk(cur_parfor.reductions[i].reductionVar, dw.callback, dw.cbdata)
        AstWalk(cur_parfor.reductions[i].reductionVarInit, dw.callback, dw.cbdata)
        AstWalk(cur_parfor.reductions[i].reductionFunc, dw.callback, dw.cbdata)
      end
      for i = 1:length(cur_parfor.body)
        AstWalk(cur_parfor.body[i], dw.callback, dw.cbdata)
      end
      for i = 1:length(cur_parfor.postParFor)-1
        AstWalk(cur_parfor.postParFor[i], dw.callback, dw.cbdata)
      end
      return x
    elseif head == :parfor_start # || head == :parfor_end
      cur_parfor = args[1]
      for i = 1:length(cur_parfor.loopNests)
        AstWalk(mk_assignment_expr(cur_parfor.loopNests[i].indexVariable,1), dw.callback, dw.cbdata)
        AstWalk(cur_parfor.loopNests[i].lower, dw.callback, dw.cbdata)
        AstWalk(cur_parfor.loopNests[i].upper, dw.callback, dw.cbdata)
        AstWalk(cur_parfor.loopNests[i].step, dw.callback, dw.cbdata)
      end
      for i = 1:length(cur_parfor.reductions)
        AstWalk(cur_parfor.reductions[i].reductionVar, dw.callback, dw.cbdata)
        AstWalk(cur_parfor.reductions[i].reductionVarInit, dw.callback, dw.cbdata)
        AstWalk(cur_parfor.reductions[i].reductionFunc, dw.callback, dw.cbdata)
      end
      return x
    elseif head == :parfor_end
      # intentionally do nothing
      return x
    elseif head == :insert_divisible_task
      cur_task = args[1]
      for i = 1:length(cur_task.args)
        AstWalk(cur_task.args[i].value, dw.callback, dw.cbdata)
      end
      return x
    elseif head == :loophead
      # intentionally do nothing
      return x
    elseif head == :loopend
      # intentionally do nothing
      return x
    end
  elseif asttyp == pir_range_actual
    for i = 1:length(x.dim)
      AstWalk(x.lower_bounds[i], dw.callback, dw.cbdata)
      AstWalk(x.upper_bounds[i], dw.callback, dw.cbdata)
    end
    return x
  end
  return nothing
end

function AstWalk(ast::Any, callback, cbdata)
  dw = DirWalk(callback, cbdata)
  DomainIR.AstWalk(ast, AstWalkCallback, dw)
end

end

