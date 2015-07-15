module IntelPSE

export DomainIR, ParallelIR, AliasAnalysis, LD
export decompose, offload, set_debug_level, getenv

# MODE for offload
const TOPLEVEL=1
const PROXYONLY=3

# 0 = none
# 1 = host
# 2 = offload1
# 3 = offload2
# 4 = task
# 5 = threads
client_intel_pse_mode = 5
# true if use task graph, false if use old-style whole-function conversion.
client_intel_task_graph = false
import Base.show

if haskey(ENV,"INTEL_PSE_MODE")
  mode = ENV["INTEL_PSE_MODE"]
  if mode == 0 || mode == "none"
    client_intel_pse_mode = 0
  elseif mode == 1 || mode == "host"
    client_intel_pse_mode = 1
  elseif mode == 2 || mode == "offload1"
    client_intel_pse_mode = 2
  elseif mode == 3 || mode == "offload2"
    client_intel_pse_mode = 3
  elseif mode == 4 || mode == "task"
    client_intel_pse_mode = 4
  elseif mode == 5 || mode == "threads"
    client_intel_pse_mode = 5
  else
    println("Unknown INTEL_PSE_MODE = ", mode)
  end
end

# a hack to make offload function available to domain IR. The assumption is that this
# is neither a TOPLEVEL function for J2C, nor a PROXYONLY compilation.
_offload(func, sig) = offload(func, sig, 0)

type CallInfo
  func_sig                             # a tuple of (function, signature)
  could_be_global :: Set               # the set of indices of arguments to the function that are or could be global arrays
  array_parameters :: Dict{Int,Any}    # the set of indices of arguments to the function that are or could be array parameters to the caller
end

type FunctionInfo
  func_sig                             # a tuple of (function, signature)
  calls :: Array{CallInfo,1}
  array_params_set_or_aliased :: Set   # the indices of the parameters that are set or aliased
  can_parallelize :: Bool              # false if a global is written in this function
  recursive_parallelize :: Bool
  params
  locals
  types
end

function getenv(var::String)
  ENV[var]
end

function getJuliaRoot()
  julia_root   = getenv("JULIA_ROOT")
  # Strip trailing /
  len_root     = endof(julia_root)
  if(julia_root[len_root] == '/')
    julia_root = julia_root[1:len_root-1]
  end

  return julia_root
end

function __init__()
  julia_root = getJuliaRoot()
  ENV["LD_LIBRARY_PATH"] = string(ENV["LD_LIBRARY_PATH"], ":" , julia_root, "/intel-runtime/lib", ":", julia_root, "/j2c")
end

julia_root      = IntelPSE.getJuliaRoot()
runtime_libpath = string(julia_root, "/intel-runtime/lib/libintel-runtime.so")

using CompilerTools
include("domain-ir.jl")
include("alias-analysis.jl")
include("parallel-ir.jl")
include("cgen.jl")

import .DomainIR.isarray
import .cgen.from_root, .cgen.writec, .cgen.compile, .cgen.link

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

# A debug print routine.
function dprintln(level,msgs...)
    if(DEBUG_LVL >= level)
        println(msgs...)
    end
end

function pert_shutdown(julia_root)
  #runtime_libpath = string(julia_root, "/intel-runtime/lib/libintel-runtime")
  eval(quote ccall((:pert_shutdown, $runtime_libpath), Cint, ()) end)
  eval(quote ccall((:FinalizeTiming, $runtime_libpath), Void, ()) end)
end

pert_inited = false

function pert_init(julia_root, double_buffer::Bool)
  global pert_inited
  if !pert_inited
    pert_inited = true
    #runtime_libpath = string(julia_root, "/intel-runtime/lib/libintel-runtime")
    eval(quote (ccall((:InitializeTiming, $runtime_libpath), Void, ())) end)
    eval(quote (ccall((:pert_init,$runtime_libpath), Cint, (Cint,), convert(Cint, $double_buffer))) end)
    shutdown() = pert_shutdown(julia_root) 
    atexit(shutdown)
  end
end

include("pse-ld.jl")

# Convert regular Julia types to make them appropriate for calling C code.
# Note that it only handles conversion of () and Array, not tuples.
function convert_to_ccall_typ(typ)
  dprintln(3,"convert_to_ccall_typ typ = ", typ, " typeof(typ) = ", typeof(typ))
  # if there a better way to check for typ being an array DataType?
  if isarray(typ)
    # If it is an Array type then convert to Ptr type.
    return (Ptr{Void},ndims(typ))
  elseif is(typ, ())
    return (Void, 0)
  else
    # Else no conversion needed.
    return (typ,0)
  end
end

# Convert a whole function signature in a form of a tuple to something appropriate for calling C code.
function convert_sig(sig)
  assert(isa(sig,Tuple))   # make sure we got a tuple
  new_tuple = Expr(:tuple)         # declare the new tuple
  # fill in the new_tuple args/elements by converting the individual elements of the incoming signature
  new_tuple.args = [ convert_to_ccall_typ(sig[i])[1] for i = 1:length(sig) ]
  sig_ndims      = [ convert_to_ccall_typ(sig[i])[2] for i = 1:length(sig) ]
  dprintln(3,"new_tuple.args = ", new_tuple.args)
  dprintln(3,"sig_ndims = ", sig_ndims)
  return (eval(new_tuple), sig_ndims)
end

function get_input_arrays(input_vars, var_types)
  ret = Symbol[]

  dprintln(3,"input_vars = ", input_vars)
  dprintln(3,"var_types = ", var_types)

  for i = 1:length(input_vars)
    iv = input_vars[i]
    dprintln(3,"iv = ", iv, " type = ", typeof(iv))
    found = false
    for j = 1:length(var_types)
      dprintln(3,"vt = ", var_types[j][1], " type = ", typeof(var_types[j][1]))
      if iv == var_types[j][1]
        dprintln(3,"Found matching name.")
        if var_types[j][2].name == Array.name
          dprintln(3,"Parameter is an Array.")
          push!(ret, iv)
        end
        found = true
        break
      end
    end
    if !found
      throw(string("Didn't find parameter variable in type list."))
    end
  end

  ret
end

function ns_to_sec(x)
  x / 1000000000.0
end

# Converts a given function and signature to use library decomposition.
# It does NOT generates a stub/proxy like the offload function, or go through J2C.
function decompose(function_name, signature)
  if client_intel_pse_mode == 0
    return function_name
  end

  start_time = time_ns()

  dprintln(2, "Starting decomposition for ", function_name)
  ct           = code_typed(function_name, signature)      # get information about code for the given function and signature
  dprintln(3, "Initial typed code = ", ct)
  domain_start = time_ns()
  code         = LD.decompose(ct[1])                 # decompose that code
  dir_time     = time_ns() - domain_start
  dprintln(1, "decomposition: time = ", ns_to_sec(dir_time))
  m            = methods(function_name, signature)
  if length(m) < 1
    error("Method for ", function_name, " with signature ", signature, " is not found")
  end
  def          = m[1].func.code
  def.tfunc[2] = ccall(:jl_compress_ast, Any, (Any,Any), def, code)
end

type extractStaticCallGraphState
  cur_func_sig
  mapNameFuncInfo :: Dict{Any,FunctionInfo}
  functionsToProcess :: Set

  calls :: Array{CallInfo,1}
  globalWrites :: Set
  params
  locals
  types
  array_params_set_or_aliased :: Set   # the indices of the parameters that are set or aliased
  cant_analyze :: Bool

  local_lambdas :: Dict{Symbol,LambdaStaticData}

  function extractStaticCallGraphState(fs, mnfi, ftp)
    new(fs, mnfi, ftp, CallInfo[], Set(), nothing, nothing, nothing, Set(), false, Dict{Symbol,LambdaStaticData}())
  end

  function extractStaticCallGraphState(fs, mnfi, ftp, calls, gw, ll)
    new(fs, mnfi, ftp, calls, gw, nothing, nothing, nothing, Set(), false, ll)
  end
end

function indexin(a, b::Array{Any,1})
  for i = 1:length(b)
    if b[i] == a
      return i
    end
  end
  throw(string("element not found in input array in indexin"))
end

function resolveFuncByName(cur_func_sig , func :: Symbol, call_sig_arg_tuple, local_lambdas)
  dprintln(4,"resolveFuncByName ", cur_func_sig, " ", func, " ", call_sig_arg_tuple, " ", local_lambdas)
  ftyp = typeof(cur_func_sig[1])
  if ftyp == Function
    cur_module = Base.function_module(cur_func_sig[1])
  elseif ftyp == LambdaStaticData
    cur_module = cur_func_sig[1].module
  else
    throw(string("Unsupported current function type in resolveFuncByName."))
  end
  dprintln(4,"Starting module = ", Base.module_name(cur_module))

  if haskey(local_lambdas, func)
    return local_lambdas[func]
  end

  while true
    if isdefined(cur_module, func)
      return getfield(cur_module, func)
    else
      if cur_module == Main
        break
      else
        cur_module = Base.module_parent(cur_module)
      end
    end
  end

  if isdefined(Base, func)
    return getfield(Base, func)
  end
  if isdefined(Main, func)
    return getfield(Main, func)
  end
  throw(string("Failed to resolve symbol ", func, " in ", Base.function_name(cur_func_sig[1])))
end

function isGlobal(sym :: Symbol, varDict :: Dict{Symbol,Array{Any,1}})
  return !haskey(varDict, sym)
end

function getGlobalOrArrayParam(varDict :: Dict{Symbol, Array{Any,1}}, params, real_args)
  possibleGlobalArgs = Set()
  possibleArrayArgs  = Dict{Int,Any}()

  dprintln(3,"getGlobalOrArrayParam ", real_args, " varDict ", varDict, " params ", params)
  for i = 1:length(real_args)
    atyp = typeof(real_args[i])
    dprintln(3,"getGlobalOrArrayParam arg ", i, " ", real_args[i], " type = ", atyp)
    if atyp == Symbol || atyp == SymbolNode
      aname = ParallelIR.getSName(real_args[i])
      if isGlobal(aname, varDict)
        # definitely a global so add
        push!(possibleGlobalArgs, i)
      elseif in(aname, params)
        possibleArrayArgs[i] = aname
      end
    else
      # unknown so add
      push!(possibleGlobalArgs, i)
      possibleArrayArgs[i] = nothing
    end
  end

  [possibleGlobalArgs, possibleArrayArgs]
end

function processFuncCall(state, func_expr, call_sig_arg_tuple, possibleGlobals, possibleArrayParams)
  fetyp = typeof(func_expr)

  dprintln(3,"processFuncCall ", func_expr, " ", call_sig_arg_tuple, " ", fetyp, " ", possibleGlobals)
  if fetyp == Symbol || fetyp == SymbolNode
    func = resolveFuncByName(state.cur_func_sig, ParallelIR.getSName(func_expr), call_sig_arg_tuple, state.local_lambdas)
  elseif fetyp == TopNode
    return nothing
  elseif fetyp == DataType
    return nothing
  elseif fetyp == GlobalRef
    func = getfield(func_expr.mod, func_expr.name)
  elseif fetyp == Expr
    dprintln(3,"head = ", func_expr.head)
    if func_expr.head == :call && func_expr.args[1] == TopNode(:getfield)
      dprintln(3,"args2 = ", func_expr.args[2], " type = ", typeof(func_expr.args[2]))
      dprintln(3,"args3 = ", func_expr.args[3], " type = ", typeof(func_expr.args[3]))
      fsym = func_expr.args[3]
      if typeof(fsym) == QuoteNode
        fsym = fsym.value
      end
      func = getfield(Main.(func_expr.args[2]), fsym)
    else
      func = eval(func_expr)
    end
    dprintln(3,"eval of func_expr is ", func, " type = ", typeof(func))
  else
    throw(string("Unhandled func expression type ", fetyp," for ", func_expr, " in ", Base.function_name(state.cur_func_sig.func)))
  end

  ftyp = typeof(func)
  dprintln(4,"After name resolution: func = ", func, " type = ", ftyp)
  if ftyp == DataType
    return nothing
  end
  assert(ftyp == Function || ftyp == IntrinsicFunction || ftyp == LambdaStaticData)

  if ftyp == Function
    fs = (func, call_sig_arg_tuple)

    if !haskey(state.mapNameFuncInfo, fs)
      dprintln(3,"Adding ", func, " to functionsToProcess")
      push!(state.functionsToProcess, fs)
      dprintln(3,state.functionsToProcess)
    end
    push!(state.calls, CallInfo(fs, possibleGlobals, possibleArrayParams))
  elseif ftyp == LambdaStaticData
    fs = (func, call_sig_arg_tuple)

    if !haskey(state.mapNameFuncInfo, fs)
      dprintln(3,"Adding ", func, " to functionsToProcess")
      push!(state.functionsToProcess, fs)
      dprintln(3,state.functionsToProcess)
    end
    push!(state.calls, CallInfo(fs, possibleGlobals, possibleArrayParams))
  end
end

function extractStaticCallGraphWalk(node, state, top_level_number, is_top_level, read)
  asttyp = typeof(node)
  dprintln(4,"escgw: ", node, " type = ", asttyp)
  if asttyp == Expr
    head = node.head
    args = node.args
    if head == :lambda
      if state.params == nothing
        param = args[1]
        meta  = args[2]
        state.params = args[1]
        state.locals = ParallelIR.createVarSet(meta[1])
        state.types  = ParallelIR.createVarDict(meta[2])
        dprintln(3,"params = ", state.params)
        dprintln(3,"locals = ", state.locals)
        dprintln(3,"types = ", state.types)
      else
        new_lambda_state = extractStaticCallGraphState(state.cur_func_sig, state.mapNameFuncInfo, state.functionsToProcess, state.calls, state.globalWrites, state.local_lambdas)
        AstWalker.AstWalk(node, extractStaticCallGraphWalk, new_lambda_state)
        return node
      end
    elseif head == :call || head == :call1
      func_expr = args[1]
      call_args = args[2:end]
      call_sig = Expr(:tuple)
      call_sig.args = map(DomainIR.typeOfOpr, call_args)
      call_sig_arg_tuple = eval(call_sig)
      dprintln(4,"func_expr = ", func_expr)
      dprintln(4,"Arg tuple = ", call_sig_arg_tuple)

      pmap_name = symbol("__pmap#39__")
      if func_expr == TopNode(pmap_name)
        func_expr = call_args[3]
        assert(typeof(func_expr) == SymbolNode)
        func_expr = func_expr.name
        call_sig = Expr(:tuple)
        push!(call_sig.args, ParallelIR.getArrayElemType(call_args[4]))
        call_sig_arg_tuple = eval(call_sig)
        dprintln(3,"Found pmap with func = ", func_expr, " and arg = ", call_sig_arg_tuple)
        processFuncCall(state, func_expr, call_sig_arg_tuple, getGlobalOrArrayParam(state.types, state.params, [call_args[4]])...)
        return node
      elseif func_expr == :map
        func_expr = call_args[1]
        call_sig = Expr(:tuple)
        push!(call_sig.args, ParallelIR.getArrayElemType(call_args[2]))
        call_sig_arg_tuple = eval(call_sig)
        dprintln(3,"Found map with func = ", func_expr, " and arg = ", call_sig_arg_tuple)
        processFuncCall(state, func_expr, call_sig_arg_tuple, getGlobalOrArrayParam(state.types, state.params, [call_args[2]])...)
        return node
      elseif func_expr == :broadcast!
        assert(length(call_args) == 4)
        func_expr = call_args[1]
        dprintln(3,"ca types = ", typeof(call_args[2]), " ", typeof(call_args[3]), " ", typeof(call_args[4]))
        catype = call_args[2].typ
        assert(call_args[3].typ == catype)
        assert(call_args[4].typ == catype)
        call_sig = Expr(:tuple)
        push!(call_sig.args, ParallelIR.getArrayElemType(catype))
        push!(call_sig.args, ParallelIR.getArrayElemType(catype))
        call_sig_arg_tuple = eval(call_sig)
        dprintln(3,"Found broadcast! with func = ", func_expr, " and arg = ", call_sig_arg_tuple)
        processFuncCall(state, func_expr, call_sig_arg_tuple, getGlobalOrArrayParam(state.types, state.params, call_args[2:4])...)
        return node
      elseif func_expr == :cartesianarray
        new_lambda_state = extractStaticCallGraphState(state.cur_func_sig, state.mapNameFuncInfo, state.functionsToProcess, state.calls, state.globalWrites, state.local_lambdas)
        AstWalker.AstWalk(call_args[1], extractStaticCallGraphWalk, new_lambda_state)
        for i = 2:length(call_args)
          AstWalker.AstWalk(call_args[i], extractStaticCallGraphWalk, state)
        end
        return node
      elseif ParallelIR.isArrayset(func_expr) || func_expr == :setindex!
        dprintln(3,"arrayset or setindex! found")
        array_name = call_args[1]
        if in(ParallelIR.getSName(array_name), state.params)
          dprintln(3,"can't analyze due to write to array")
          # Writing to an array parameter which could be a global passed to this function makes parallelization analysis unsafe.
          #state.cant_analyze = true
          push!(state.array_params_set_or_aliased, indexin(ParallelIR.getSName(array_name), state.params))
        end
        return node
      end

      processFuncCall(state, func_expr, call_sig_arg_tuple, getGlobalOrArrayParam(state.types, state.params, call_args)...)
    elseif head == :(=)
      lhs = args[1]
      rhs = args[2]
      rhstyp = typeof(rhs)
      dprintln(3,"Found assignment: lhs = ", lhs, " rhs = ", rhs, " type = ", typeof(rhs))
      if rhstyp == LambdaStaticData
        state.local_lambdas[lhs] = rhs
        return node
      elseif rhstyp == SymbolNode || rhstyp == Symbol
        rhsname = ParallelIR.getSName(rhs)
        if in(rhsname, state.params)
          rhsname_typ = state.types[rhsname][2]
          if rhsname_typ.name == Array.name
            # if an array parameter is the right-hand side of an assignment then we give up on tracking whether some parallelization requirement is violated for this parameter
            # state.cant_analyze = true
            push!(state.array_params_set_or_aliased, indexin(rhsname, state.params))
          end
        end
      end
    end
  elseif asttyp == Symbol
    dprintln(4,"Symbol ", node, " found in extractStaticCallGraphWalk. ", read, " ", isGlobal(node, state.types))
    if !read && isGlobal(node, state.types)
      dprintln(3,"Detected a global write to ", node, " ", state.types)
      push!(state.globalWrites, node)
    end
  elseif asttyp == SymbolNode
    dprintln(4,"SymbolNode ", node, " found in extractStaticCallGraphWalk.", read, " ", isGlobal(node.name, state.types))
    if !read && isGlobal(node.name, state.types)
      dprintln(3,"Detected a global write to ", node.name, " ", state.types)
      push!(state.globalWrites, node.name)
    end
    if node.typ == Function
      throw(string("Unhandled function in extractStaticCallGraphWalk."))
    end
  end

  # Don't make any changes to the AST, just record information in the state variable.
  return nothing
end

function show(io::IO, cg :: Dict{Any,FunctionInfo})
  for i in cg
    println(io,"Function: ", i[2].func_sig[1], " ", i[2].func_sig[2], " local_par: ", i[2].can_parallelize, " recursive_par: ", i[2].recursive_parallelize, " array_params: ", i[2].array_params_set_or_aliased)
    for j in i[2].calls
      print(io,"    ", j)
      if haskey(cg, j.func_sig)
        jfi = cg[j.func_sig]
        print(io, " (", jfi.can_parallelize, ",", jfi.recursive_parallelize, ",", jfi.array_params_set_or_aliased, ")")
      end
      println(io,"")
    end
  end
end

function propagate(mapNameFuncInfo)
  # The can_parallelize field is purely local but the recursive_parallelize field includes all the children of this function.
  # Initialize the recursive state to the local state.
  for i in mapNameFuncInfo
    i[2].recursive_parallelize = i[2].can_parallelize
  end

  changes = true
  loop_iter = 1

  # Keep iterating so long as we find changes to a some recursive_parallelize field.
  while changes
    dprintln(3,"propagate ", loop_iter)
    loop_iter = loop_iter + 1

    changes = false
    # For each function in the call graph
    for i in mapNameFuncInfo  # i of type FunctionInfo
      this_fi = i[2]

      dprintln(3,"propagate for ", i)
      # If the function so far thinks it can parallelize.
      if this_fi.recursive_parallelize
        # Check everything it calls.
        for j in this_fi.calls # j of type CallInfo
          dprintln(3,"    call ", j)
          if haskey(mapNameFuncInfo, j.func_sig)
            dprintln(3,"    call ", j, " found")
            other_func = mapNameFuncInfo[j.func_sig]
            # If one of its callees can't be parallelized then it can't either.
            if !other_func.recursive_parallelize
              # Record we found a change so the outer iteration must continue.
              changes = true
              # Record that the current function can't parallelize.
              dprintln(3,"Function ", i[1], " not parallelizable since callee ", j.func_sig," isn't.")
              this_fi.recursive_parallelize = false
              break
            end
            if !isempty(intersect(j.could_be_global, other_func.array_params_set_or_aliased))
              # Record we found a change so the outer iteration must continue.
              changes = true
              # Record that the current function can't parallelize.
              dprintln(3,"Function ", i[1], " not parallelizable since callee ", j.func_sig," intersects at ", intersect(j.could_be_global, other_func.array_params_set_or_aliased))
              this_fi.recursive_parallelize = false
              break
            end
            for k in j.array_parameters
              dprintln(3,"    k = ", k)
              arg_index = k[1]
              arg_array_name = k[2]
              if in(arg_index, other_func.array_params_set_or_aliased)
                if typeof(arg_array_name) == Symbol
                  push!(this_fi.array_params_set_or_aliased, indexin(arg_array_name, this_fi.params))
                else
                  for l = 1:length(this_fi.params)
                    ptype = this_fi.types[this_fi.params[l]][2]
                    if ptype.name == Array.name
                      push!(this_fi.array_params_set_or_aliased, l)
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  end
end


function extractStaticCallGraph(func, sig)
  assert(typeof(func) == Function)

  functionsToProcess = Set()
  push!(functionsToProcess, (func,sig))

  mapNameFuncInfo = Dict{Any,FunctionInfo}()

  while length(functionsToProcess) > 0
    cur_func_sig = first(functionsToProcess)
    delete!(functionsToProcess, cur_func_sig)
    the_func = cur_func_sig[1]
    the_sig  = cur_func_sig[2]
    ftyp     = typeof(the_func)
    dprintln(3,"functionsToProcess left = ", functionsToProcess)

    if ftyp == Function
      dprintln(3,"Processing function ", the_func, " ", Base.function_name(the_func), " ", the_sig)

      method = methods(the_func, the_sig)
      if length(method) < 1
        dprintln(3,"Skipping function not found. ", the_func, " ", Base.function_name(the_func), " ", the_sig)
        continue
      end
      ct = code_typed(the_func, the_sig)
      dprintln(4,ct[1])
      state = extractStaticCallGraphState(cur_func_sig, mapNameFuncInfo, functionsToProcess)
      AstWalker.AstWalk(ct[1], extractStaticCallGraphWalk, state)
      dprintln(4,state)
      mapNameFuncInfo[cur_func_sig] = FunctionInfo(cur_func_sig, state.calls, state.array_params_set_or_aliased, !state.cant_analyze && length(state.globalWrites) == 0, true, state.params, state.locals, state.types)
    elseif ftyp == LambdaStaticData
      dprintln(3,"Processing lambda static data ", the_func, " ", the_sig)
      ast = ParallelIR.uncompressed_ast(the_func)
      dprintln(4,ast)
      state = extractStaticCallGraphState(cur_func_sig, mapNameFuncInfo, functionsToProcess)
      AstWalker.AstWalk(ast, extractStaticCallGraphWalk, state)
      dprintln(4,state)
      mapNameFuncInfo[cur_func_sig] = FunctionInfo(cur_func_sig, state.calls, state.array_params_set_or_aliased, !state.cant_analyze && length(state.globalWrites) == 0, true, state.params, state.locals, state.types)
    end
  end

  propagate(mapNameFuncInfo)
  mapNameFuncInfo
end

# Create a new j2c array object with element size in bytes and given dimension.
# It will share the data pointer of the given inp array, and if inp is nothing,
# the j2c array will allocate fresh memory to hold data.
# NOTE: when elem_bytes is 0, it means the elements must be j2c array type
function j2c_array_new(elem_bytes::Int, inp::Union(Array, Nothing), ndim::Int, dims::Tuple)
  # note that C interface mandates Int64 for dimension data
  _dims = Int64[ convert(Int64, x) for x in dims ]
  _inp = is(inp, nothing) ? C_NULL : convert(Ptr{Void}, pointer(inp))
  ccall(:j2c_array_new, Ptr{Void}, (Cint, Ptr{Void}, Cuint, Ptr{Uint64}),
        convert(Cint, elem_bytes), _inp, convert(Cuint, ndim), pointer(_dims))
end

# Array size in the given dimension.
function j2c_array_size(arr::Ptr{Void}, dim::Int)
  l = ccall(:j2c_array_size, Cuint, (Ptr{Void}, Cuint),
            arr, convert(Cuint, dim))
  return convert(Int, l)
end

# Retrieve j2c array data pointer, and parameter "own" means caller will
# handle the memory deallocation of this pointer.
function j2c_array_to_pointer(arr::Ptr{Void}, own::Bool)
  ccall(:j2c_array_to_pointer, Ptr{Void}, (Ptr{Void}, Bool), arr, own)
end

# Read the j2c array element of given type at the given (linear) index.
# If T is Ptr{Void}, treat the element type as j2c array, and the
# returned array is merely a pointer, not a new object.
function j2c_array_get(arr::Ptr{Void}, idx::Int, T::Type)
  nbytes = is(T, Ptr{Void}) ? 0 : sizeof(T)
  _value = Array(T, 1)
  ccall(:j2c_array_get, Void, (Cint, Ptr{Void}, Cuint, Ptr{Void}),
        convert(Cint, nbytes), arr, convert(Cuint, idx), convert(Ptr{Void}, pointer(_value)))
  return _value[1]
end

# Set the j2c array element at the given (linear) index to the given value.
# If T is Ptr{Void}, treat value as a pointer to j2c array.
function j2c_array_set{T}(arr::Ptr{Void}, idx::Int, value::T)
  nbytes = is(T, Ptr{Void}) ? 0 : sizeof(T)
  _value = nbytes == 0 ? value : convert(Ptr{Void}, pointer(T[ value ]))
  ccall(:j2c_array_set, Void, (Cint, Ptr{Void}, Cuint, Ptr{Void}),
        convert(Cint, nbytes), arr, convert(Cuint, idx), _value)
end

# Delete an j2c array object.
# Note that this only works for scalar array or arrays of
# objects whose derefrence will definite not trigger nested
# deletion (either data pointer is NULL, or refcount > 1).
# Currently there is no way to cleanly delete an nested j2c
# array without first converting back to a julia array.
function j2c_array_delete(arr::Ptr{Void})
  ccall(:j2c_array_delete, Void, (Ptr{Void},), arr)
end

# Dereference a j2c array data pointer.
# Require that the j2c array data pointer points to either
# a scalar array, or an array of already dereferenced array
# (whose data pointers are NULL).
function j2c_array_deref(arr::Ptr{Void})
  ccall(:j2c_array_deref, Void, (Ptr{Void},), arr)
end

# Convert Julia array to J2C array object.
# Note that Julia array data are not copied but shared by the J2C array
# The caller needs to make sure these arrays stay alive so long as the
# returned j2c array is alive.
function to_j2c_array{T, N}(inp :: Array{T, N}, ptr_array_dict :: Dict{Ptr, Array})
  dims = size(inp)
  _isbits = isbits(T)
  nbytes = _isbits ? sizeof(T) : 0
  _inp = _isbits ? inp : nothing
  arr = j2c_array_new(nbytes, _inp, N, dims)
  ptr_array_dict[convert(Ptr{Void}, pointer(inp))] = inp  # establish a mapping between pointer and the original array
  if !(_isbits)
    for i = 1:length(inp)
      obj = to_j2c_array(inp[i], ptr_array_dict) # obj is a new j2c array
      j2c_array_set(arr, i, obj) # obj is duplicated during this set
      j2c_array_delete(obj)      # safe to delete obj without triggering free
    end
  end
  return arr
end

# Convert J2C array object to Julia array.
# Note that:
# 1. We assume the input j2c array object contains no data pointer aliases.
# 2. The returned Julia array will share the pointer to J2C array data at leaf level.
# 3. The input j2c array object will be de-referenced before return, and shall
#    be later manually freed. 
function _from_j2c_array(inp::Ptr{Void}, elem_typ::DataType, N::Int, ptr_array_dict :: Dict{Ptr, Array})
  dims = Array(Int, N)
  len  = 1
  for i = 1:N
    dims[i] = j2c_array_size(inp, i)
    len = len * dims[i]
  end
  if isbits(elem_typ)
    array_ptr = convert(Ptr{Void}, j2c_array_to_pointer(inp, true))
    if haskey(ptr_array_dict, array_ptr)
      arr = ptr_array_dict[array_ptr]
    else
      arr = pointer_to_array(convert(Ptr{elem_typ}, array_ptr), tuple(dims...))
    end
  elseif isarray(elem_typ)
    arr = Array(elem_typ, dims...)
    sub_type = elem_typ.parameters[1]
    sub_dim  = elem_typ.parameters[2]
    for i = 1:len
      ptr = j2c_array_get(inp, i, Ptr{Void})
      arr[i] = _from_j2c_array(ptr, sub_type, sub_dim, ptr_array_dict)
      j2c_array_deref(ptr)
    end
  end
  return arr
end

function from_j2c_array(inp::Ptr{Void}, elem_typ::DataType, N::Int, ptr_array_dict :: Dict{Ptr, Array})
   arr = _from_j2c_array(inp, elem_typ, N, ptr_array_dict)
   j2c_array_delete(inp)
   return arr
end 
  
latest_func_id = 0
function getNextFuncId()
  cur = latest_func_id
  global latest_func_id = latest_func_id + 1
  return cur
end

function Optimize(ast, call_sig_arg_tuple, call_sig_args)
  assert(typeof(ast) == Expr)
  assert(ast.head == :lambda)

  dprintln(2, "Starting Optimize with args = ", call_sig_arg_tuple, " names = ", call_sig_args, "\n", ast, "\n")

  # Overall approach:
  # 1: create a copy of the incoming AST as a new function.
  # 2: pass that new function to the existing offload function.
  # 3: force the j2c function to compile
  # 4: get the AST of the proxy returned by offload and use that as the AST to return.
  func_copy_to_j2c = string("_IntelPSE_optimize_", getNextFuncId())
  func_copy_sym    = symbol(func_copy_to_j2c)
  #arg_array        = [ symbol(string("arg", i)) for i=1:length(call_sig_arg_tuple) ]
  #dprintln(2,"func_copy = ", func_copy_to_j2c, " arg_array = ", arg_array)

  # Create an empty function with the right number of arguments (which probably isn't necessary to get arg num right).
  new_func = @eval function ($func_copy_sym)($(call_sig_args...)) end
  new_ct = code_typed(new_func, call_sig_arg_tuple)
  m = methods(new_func, call_sig_arg_tuple)
  if length(m) < 1
    error("Method for ", function_name, " with signature ", signature, " is not found")
  end
  m = m[1]
  new_def = m.func.code
  # Write the AST that we got into the newly created function.
  new_def.tfunc[2] = ccall(:jl_compress_ast, Any, (Any,Any), new_def, ast)

  # Call the normal offload method and capture the proxy name.
  proxy = offload(new_func, call_sig_arg_tuple)

  if proxy != nothing
    # Force j2c to run on the target method.
    precompile(new_func, call_sig_arg_tuple)

    dprintln(3,"offload in Optimize returned something.")
    # Get the AST of the proxy function.
    cps   = string("_IntelPSE_optimize_call_proxy_", getNextFuncId())
    cpsym = symbol(cps)
    # Create a temporary function which just calls the proxy.
                              #println("======> ", $cps, " ", $call_sig_args)
    call_proxy_func = @eval function ($cpsym)($(call_sig_args...))
                              $proxy($(call_sig_args...))
                            end
    # Get the code for this temporary function.
    cpfct = code_typed(call_proxy_func, call_sig_arg_tuple)
    cpfct = cpfct[1]
    # Use that code for current function (ast).
    ast.args = cpfct.args
    return ast
  else
    dprintln(3,"offload in Optimize returned nothing.")
  end
  return ast
end

use_extract_static_call_graph = 0
function use_static_call_graph(x)
    global use_extract_static_call_graph = x
end

stop_after_offload = 0
function stopAfterOffload(x)
  global stop_after_offload = x
end

no_precompile = 0
function noPrecompile(x)
  global no_precompile = x;
end

previouslyOptimized = Set()

function replace_gensym_nodes(node, state, top_level_number, is_top_level, read)
	dprintln(9,"replace gensym in node:",node)
	#xdump(node,1000)
	if !isa(node,GenSym)
		return nothing
	end
	return SymbolNode(Symbol("loc_sym_"*string(node.id)), state[node.id+1])
end

function remove_gensym(ast)

	# gensym types are in the 3rd array in lambda's metadata
	gensym_types = ast.args[2][3]
	# go through function body and rename gensyms with symbols
	AstWalker.AstWalk(ast.args[3], replace_gensym_nodes, gensym_types)
	for i in 1:length(gensym_types)
		gensym_types[i] = [Symbol("loc_sym_"*string(i-1)), gensym_types[i], 18]
	end

	new_meta = Any[]
	push!(new_meta, Any[])
	# add local GenSym types to meta data
	push!(new_meta,append!(ast.args[2][1],gensym_types))
	# assume no free variables
	push!(new_meta, Any[])
	# construct local variables
	for i in 1:length(new_meta[2])
		# extract symbol from metadata
		sym = new_meta[2][i][1]
		if findfirst(ast.args[1],sym) == 0
			push!(new_meta[1],sym)
		end
	end
	ast.args[2] = new_meta
end

# Converts a given function and signature to use domain IR and parallel IR.
# It also generates a stub/proxy with the same signature as the original that you can call to get you
# to the j2c version of the code.
function offload(function_name, signature, offload_mode=TOPLEVEL)
  if client_intel_pse_mode == 0
    return function_name
  end
  start_time = time_ns()

  dprintln(2, "Starting offload for ", function_name)
  if !isgeneric(function_name)
    dprintln(1, "method ", function_name, " is not generic.")
    return nothing
  end
  m = methods(function_name, signature)
  if length(m) < 1
    error("Method for ", function_name, " with signature ", signature, " is not found")
  end

  def = m[1].func.code
  if in((function_name, signature), previouslyOptimized)
    dprintln(2, "method ", function_name, " already offloaded")
    return
  end

  if use_extract_static_call_graph != 0
    callgraph = extractStaticCallGraph(function_name, signature)
    if DEBUG_LVL >= 3
      println("Callgraph:")
      println(callgraph)
#    #throw(string("stop after cb"))
    end
  end

  cur_module   = def.module
  ct           = code_typed(function_name, signature)      # get information about code for the given function and signature
  
  dprintln(3,"before remove_gensym()",ct[1])
  
  remove_gensym(ct[1])

  dprintln(3,"after remove_gensym()",ct[1])

  if offload_mode & PROXYONLY != PROXYONLY
    push!(previouslyOptimized, (function_name, signature))
    dprintln(3, "Initial typed code = ", ct)
    domain_start = time_ns()
    domain_ir    = DomainIR.from_expr(string(function_name), cur_module, ct[1])     # convert that code to domain IR
    dir_time = time_ns() - domain_start
    dprintln(3, "domain code = ", domain_ir)
    dprintln(1, "offload: DomainIR conversion time = ", ns_to_sec(dir_time))
    input_arrays = get_input_arrays(ct[1].args[1],ct[1].args[2][2])
    pir_start = time_ns()
    if DEBUG_LVL >= 5
      code = @profile ParallelIR.from_expr(string(function_name), domain_ir, input_arrays) # convert that code to parallel IR
      Profile.print()
    else
      code = ParallelIR.from_expr(string(function_name), domain_ir, input_arrays) # convert that code to parallel IR
    end
    pir_time = time_ns() - pir_start
    dprintln(3, "parallel code = ", code)
    dprintln(1, "offload: ParallelIR conversion time = ", ns_to_sec(pir_time))
    def.tfunc[2] = ccall(:jl_compress_ast, Any, (Any,Any), def, code)
  else
    dprintln(1, "IntelPSE code optimization skipped")
    code = ct[1]
  end

  if client_intel_pse_mode == 5
    if no_precompile == 0
      precompile(function_name, signature)
    end
    return nothing
  end

  if offload_mode & TOPLEVEL == TOPLEVEL
     
    off_time_start = time_ns()
  
    julia_root   = getJuliaRoot()

	# cgen path
  #  if client_intel_pse_cgen == 1
        cgen.writec(from_root(code, string(function_name)))
        cgen.compile()
        cgen.link()
 #   end 
 
    # The proxy function name is the original function name with "_j2c_proxy" appended.
    proxy_name   = string("_",function_name,"_j2c_proxy")
    proxy_sym    = symbol(proxy_name)
    dprintln(2, "IntelPSE.offload for ", proxy_name)
  
    # This is the name of the function that j2c generates.
    j2c_name     = string("_",function_name,"_")
    # This is where the j2c dynamic library should be.
    dyn_lib      = string(julia_root, "/j2c/libout.so.1.0")
    dprintln(2, "dyn_lib = ", dyn_lib)
  
    # Same the number of statements so we can get the last one.
    num_stmts    = length(ct[1].args[3].args)
    # Get the return type of the function by looking at the last statement of the lambda and getting its type.
    n            = num_stmts
    ret_typs     = nothing
    while n > 0
       last_stmt = ct[1].args[3].args[n]
       if isa(last_stmt, LabelNode)
         n = n - 1
       elseif isa(last_stmt, Expr) && is(last_stmt.head, :return)
         typ = DomainIR.typeOfOpr(last_stmt.args[1])
         dprintln(3,"offload return typ = ", typ, " typeof(typ) = ", typeof(typ), " last_stmt = ", last_stmt)
         ret_typs = isa(typ, Tuple) ? [ (x, isarray(x)) for x in typ ] : [ (typ, isarray(typ)) ]
         break
       else
         error("Last statement is not a return: ", last_stmt)
       end
    end
    if ret_typs == nothing
      error("Cannot figure out function return type")
    end
    # Convert Arrays in signature to Ptr and add extra arguments for array dimensions
    (modified_sig, sig_dims) = convert_sig(signature)
    dprintln(2, "modified_sig = ", modified_sig)
    dprintln(2, "sig_dims = ", sig_dims)
    dprintln(3, "len? ", length(code.args[1]), length(sig_dims))
  
    original_args = Symbol[ gensym(string(s)) for s in code.args[1] ]
    assert(length(original_args) == length(sig_dims))
    modified_args = Array(Any, length(sig_dims))
    extra_inits = Array(Any, 0)
    j2c_array = gensym("j2c_arr")
    for i = 1:length(sig_dims)
      arg = original_args[i]
      if sig_dims[i] > 0 
        j = length(extra_inits) + 1
        push!(extra_inits, :(to_j2c_array($arg, ptr_array_dict)))
        modified_args[i] = :($(j2c_array)[$j])
      else
        modified_args[i] = arg
      end
    end
  
    # Create a set of expressions to pass as arguments to specify the array dimension sizes.
    if client_intel_pse_mode == 1
      if client_intel_task_graph
          # task mode, init runtime without double buffer
          pert_init(julia_root, false)
      end
      run_where = -1
    elseif client_intel_pse_mode == 2
      run_where = 0
    else
      assert(0)
    end
  
    num_rets = length(ret_typs)
    assert(n > 0)
    ret_arg_exps = Array(Any, 0)
    extra_sig = Array(Type, 0)
    for i = 1:num_rets
      (typ, is_array) = ret_typs[i]
      push!(extra_sig, is_array ? Ptr{Ptr{Void}} : Ptr{typ})
      push!(ret_arg_exps, Expr(:call, TopNode(:pointer), Expr(:call, TopNode(:arrayref), SymbolNode(:ret_args, Array{Any,1}), i)))
    end
  
    dprintln(2,"signature = ", signature, " -> ", ret_typs)
    dprintln(2,"modified_args = ", typeof(modified_args), " ", modified_args)
    dprintln(2,"extra_sig = ", extra_sig)
    dprintln(2,"ret_arg_exps = ", ret_arg_exps)
    tuple_sig_expr = Expr(:tuple,Cint,modified_sig...,extra_sig...)
    dprintln(2,"tuple_sig_expr = ", tuple_sig_expr)
    proxy_func = @eval function ($proxy_sym)($(original_args...))
      # As we convert arrays into pointers that are stored in j2c_array objects, we remember in this
      # dictionary a mapping between an array's data pointer and the array object itself.  Later,
      # when processing arrays returned by the function, we see if some data pointer returned is
      # equal to one of the pointers in the ptr_array_dict.  If so, then the C code has returned an
      # array we passed to it as input and so from_j2c_array will get the original array from
      # ptr_array_dict and will alias to the returned array.
      ptr_array_dict = Dict{Ptr,Array}()
      #dprintln(2,"Running proxy function.")
      ret_args = Array(Any, $num_rets)
      for i = 1:$num_rets
        (t, is_array) = $(ret_typs)[i]
        if is_array
          t = Ptr{Void}
        end
        ret_args[i] = Array(t, 1) # hold return result
      end
      $(j2c_array) = [ $(extra_inits...) ]
      #j2c_array_typs = Any[ typeof(x) for x in $(extra_inits...) ]
      #dprintln(3, "before ccall: ret_args = ", Any[$(ret_arg_exps...)])
      #dprintln(3, "before ccall: modified_args = ", Any[$(modified_args...)])
      ccall(($j2c_name, $dyn_lib), Void, $tuple_sig_expr, $run_where, $(modified_args...), $(ret_arg_exps...))
      result = Array(Any, $num_rets)
      for i = 1:$num_rets
        (t, is_array) = $(ret_typs)[i]
        if is_array
          # dprintln(3, "ret=", ret_args[i][1], "t=", t)
          result[i] = from_j2c_array(ret_args[i][1], t.parameters[1], t.parameters[2], ptr_array_dict)
        else
          result[i] = convert(t, (ret_args[i][1]))
        end
      end
      # free j2c arrays FIXME: needs a better delete for nested arrays
      for i = 1:length($(j2c_array))
        j2c_array_delete($(j2c_array)[i])
      end
      return ($num_rets == 1 ? result[1] : tuple(result...))
    end
  
    off_time = time_ns() - off_time_start
    dprintln(1, "offload: offload conversion time = ", ns_to_sec(off_time))
    
    if no_precompile == 0
      precompile(function_name, signature)
      precompile(proxy_func, signature)
    end
  
    end_time = time_ns() - start_time
    dprintln(1, "offload: total conversion time = ", ns_to_sec(end_time))
  
    if DEBUG_LVL >= 3
      proxy_ct = code_lowered(proxy_func, signature)
      if length(proxy_ct) == 0
        println("Error getting proxy code.\n")
      else
        proxy_ct = proxy_ct[1]
        println("Proxy code for ", function_name)
        println(proxy_ct)    
		println("Done printing proxy code")
      end
    end
  
    if stop_after_offload != 0
      throw(string("debuggging IntelPSE"))
    end
    return proxy_func #@eval IntelPSE.$proxy_sym
  else
    return nothing
  end
end

include("pp.jl")
include("ParallelComprehension.jl")

@eval function StartTiming(state::String)
    ccall((:StartTiming, $runtime_libpath), Void, (Ptr{Uint8},), state)
end

@eval function StopTiming(state::String)
    ccall((:StopTiming, $runtime_libpath), Void, (Ptr{Uint8},), state)
end

@eval function Register{T, N}(a :: Array{T, N})
    data = convert(Ptr{Void}, pointer(a))
    is_scalar = convert(Cint, 0)
    dim = convert(Cuint, N)  #ndims(a)
    sizes = [size(a)...]
    max_size = pointer(sizes)
    type_size = convert(Cuint, sizeof(T))

#    println("data ", data, "is_scalar ", is_scalar, "dim ", dim, "sizes ", sizes, "max_size ", max_size, "type_size ", type_size);

    ccall((:pert_register_data, $runtime_libpath), Cint, (Ptr{Void}, Cint, Cuint, Ptr{Int64}, Cuint),
    	    data, is_scalar, dim, max_size, type_size)
end

end
