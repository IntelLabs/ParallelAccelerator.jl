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

module Driver

export accelerate, toDomainIR, toParallelIR, toFlatParfors, toCGen, toCartesianArray, runStencilMacro, captureOperators

using CompilerTools
using CompilerTools.AstWalker
using CompilerTools.LambdaHandling

import ..ParallelAccelerator, ..Comprehension, ..DomainIR, ..ParallelIR, ..CGen, ..DomainIR.isarray, ..API
import ..dprint, ..dprintln, ..DEBUG_LVL
import ..CallGraph.extractStaticCallGraph, ..CallGraph.use_extract_static_call_graph
using ..J2CArray

# MODE for accelerate
const TOPLEVEL=1
const PROXYONLY=3

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

function ns_to_sec(x)
  x / 1000000000.0
end

latest_func_id = 0
function getNextFuncId()
  cur = latest_func_id
  global latest_func_id = latest_func_id + 1
  return cur
end

stop_after_accelerate = 0
function stopAfterAccelerate(x)
  global stop_after_accelerate = x
end

no_precompile = 0
function noPrecompile(x)
  global no_precompile = x;
end

alreadyOptimized = Dict{Tuple{Function,Tuple},Expr}()

@doc """
A pass that translates supported operators and function calls to
those defined in ParallelAccelerator.API.
"""
function captureOperators(func, ast, sig)
  AstWalk(ast, API.Capture.process_node, nothing)
  return ast
end

@doc """
Pass that translates runStencil call in the same way as a macro would do.
This is only used when PROSPECT_MODE is off.
"""
function runStencilMacro(func, ast, sig)
  AstWalk(ast, API.Stencil.process_node, nothing)
  return ast
end

@doc """
Pass for comprehension to cartesianarray translation.
"""
function toCartesianArray(func, ast, sig)
  AstWalk(ast, Comprehension.process_node, nothing)
  return ast
end

function toDomainIR(func :: GlobalRef, ast :: Expr, signature :: Tuple)
  dir_start = time_ns()
  code = DomainIR.from_expr(string(func.name), func.mod, ast)
  dir_time = time_ns() - dir_start
  dprintln(3, "domain code = ", code)
  dprintln(1, "accelerate: ParallelIR conversion time = ", ns_to_sec(dir_time))
  return code
end

function toParallelIR(func :: GlobalRef, ast :: Expr, signature :: Tuple)
  pir_start = time_ns()
# uncomment these 2 lines for ParallelIR profiling
#  code = @profile ParallelIR.from_root(string(func.name), ast)
#  Profile.print()
  code = ParallelIR.from_root(string(func.name), ast)
  pir_time = time_ns() - pir_start
  dprintln(3, "parallel code = ", code)
  dprintln(1, "accelerate: ParallelIR conversion time = ", ns_to_sec(pir_time))
  return code
end

function toFlatParfors(func :: GlobalRef, ast :: Expr, signature :: Tuple)
  code = ParallelIR.flattenParfors(string(func.name), ast)
  dprintln(3, "flattened code = ", code)
  return code
end

function toDistributedIR(func :: GlobalRef, ast :: Expr, signature :: Tuple)
  dir_start = time_ns()
  code = ast #DistributedIR.from_top(string(func.name), ast)
  dir_time = time_ns() - dir_start
  dprintln(3, "Distributed code = ", code)
  dprintln(1, "accelerate: DistributedIR conversion time = ", ns_to_sec(dir_time))
  return code
end

function toCGen(func :: GlobalRef, code :: Expr, signature :: Tuple)
  # In threads mode, we have already converted back to standard Julia AST so we skip this phase.
  if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
    return code
  end
  
  off_time_start = time_ns()

  package_root = ParallelAccelerator.getPackageRoot()
  function_name_string = ParallelAccelerator.CGen.canonicalize(string(func.name))

  array_types_in_sig = Dict{DataType,Int64}()
  atiskey = 1;
  for t in signature
      while isarray(t)
          array_types_in_sig[t] = atiskey;
          atiskey += 1
          t = eltype(t)
      end
  end
  dprintln(3, "array_types_in_sig = ", array_types_in_sig)
 
  outfile_name = CGen.writec(CGen.from_root(code, function_name_string, array_types_in_sig))
  CGen.compile(outfile_name)
  dyn_lib = CGen.link(outfile_name)
 
  # The proxy function name is the original function name with "_j2c_proxy" appended.
  proxy_name = string("_",function_name_string,"_j2c_proxy")
  proxy_sym = symbol(proxy_name)
  dprintln(2, "ParallelAccelerator.accelerate for ", proxy_name)
  dprintln(2, "C File  = $package_root/deps/generated/$outfile_name.cpp")
  dprintln(2, "dyn_lib = ", dyn_lib)
  
  # This is the name of the function that j2c generates.
  j2c_name = string("_",function_name_string,"_")
  
  lambdaInfo = lambdaExprToLambdaInfo(code)
  ret_type = getReturnType(lambdaInfo)
  # TO-DO: Check ret_type if it is Any or a Union in which case we probably need to abort optimization in CGen mode.
  ret_typs = DomainIR.istupletyp(ret_type) ? [ (x, isarray(x)) for x in ret_type.parameters ] : [ (ret_type, isarray(ret_type)) ]

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

  @eval begin
    # Create a new j2c array object with element size in bytes and given dimension.
    # It will share the data pointer of the given inp array, and if inp is nothing,
    # the j2c array will allocate fresh memory to hold data.
    # NOTE: when elem_bytes is 0, it means the elements must be j2c array type
    function j2c_array_new(elem_bytes::Int, inp::Union{Array, Void}, ndim::Int, dims::Tuple)
      # note that C interface mandates Int64 for dimension data
      _dims = Int64[ convert(Int64, x) for x in dims ]
      _inp = is(inp, nothing) ? C_NULL : convert(Ptr{Void}, pointer(inp))

      #ccall((:j2c_array_new, $dyn_lib), Ptr{Void}, (Cint, Ptr{Void}, Cuint, Ptr{UInt64}),
      #      convert(Cint, elem_bytes), _inp, convert(Cuint, ndim), pointer(_dims))
      ccall((:j2c_array_new, $dyn_lib), Ptr{Void}, (Cint, Ptr{Void}, Cuint, Ptr{UInt64}),
            convert(Cint, elem_bytes), _inp, convert(Cuint, ndim), pointer(_dims))
    end
  end

  for i = 1:length(sig_dims)
    arg = original_args[i]
    if sig_dims[i] > 0 
      j = length(extra_inits) + 1
      push!(extra_inits, :(to_j2c_array($arg, ptr_array_dict, $array_types_in_sig, $j2c_array_new)))
      modified_args[i] = :($(j2c_array)[$j])
    else
      modified_args[i] = arg
    end
  end
  
  # Create a set of expressions to pass as arguments to specify the array dimension sizes.
  if ParallelAccelerator.getPseMode() == ParallelAccelerator.HOST_MODE
      run_where = -1
  elseif ParallelAccelerator.getPseMode() == ParallelAccelerator.OFFLOAD1_MODE
      run_where = 0
  elseif ParallelAccelerator.getPseMode() == ParallelAccelerator.OFFLOAD2_MODE
      run_where = 1
#  elseif ParallelAccelerator.getPseMode() == ParallelAccelerator.TASK_MODE
#      pert_init(package_root, false)
  else
      throw("PSE mode error")
  end
  
  num_rets = length(ret_typs)
  ret_arg_exps = Array(Any, 0)
  extra_sig = Array(Type, 0)
  # We special-case functions that return Void/nothing since it is common.
  Void_return = (num_rets == 1 && ret_typs[1][1] == Void)
  if !Void_return
      for i = 1:num_rets
          (typ, is_array) = ret_typs[i]
          push!(extra_sig, is_array ? Ptr{Ptr{Void}} : Ptr{typ})
          push!(ret_arg_exps, Expr(:call, TopNode(:pointer), Expr(:call, TopNode(:arrayref), SymbolNode(:ret_args, Array{Any,1}), i)))
      end
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
      ptr_array_dict = Dict{Ptr{Void},Array}()
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
      # If the function returns nothing then just force it here since cgen code can't return it.
      return ($num_rets == 1 ? ($Void_return ? nothing : result[1]) : tuple(result...))
  end
  
  off_time = time_ns() - off_time_start
  dprintln(1, "accelerate: accelerate conversion time = ", ns_to_sec(off_time))
  return proxy_func
end

function code_typed(func, signature)
  global alreadyOptimized
  if haskey(alreadyOptimized, (func, signature))
    Any[ alreadyOptimized[(func, signature)] ]
  else
    mod = Base.function_module(func, signature)
    name = Base.function_name(func)
    func = eval(CompilerTools.OptFramework.findTargetFunc(mod, name))
    Base.code_typed(func, signature)
  end
end

# Converts a given function and signature to use domain IR and parallel IR, and
# remember it so that it won't be translated again.
function accelerate(func::Function, signature::Tuple, level = TOPLEVEL)
  pse_mode = ParallelAccelerator.getPseMode() 
  if pse_mode == ParallelAccelerator.OFF_MODE
    return func
  end

  dprintln(2, "Starting accelerate for ", func, " with signature ", signature)
  if !isgeneric(func)
    dprintln(1, "method ", func, " is not generic.")
    return nothing
  end
  m = methods(func, signature)
  if length(m) < 1
    error("Method for ", func, " with signature ", signature, " is not found")
  end

  def::LambdaStaticData = m[1].func.code
  cur_module = def.module
  func_ref = GlobalRef(cur_module, symbol(string(func)))

  local out::Expr
  ast = code_typed(func, signature)[1]
  global alreadyOptimized 

  try
    if !haskey(alreadyOptimized, (func, signature))
      # place holder to prevent recursive accelerate
      alreadyOptimized[(func, signature)] = ast 
      dir_ast::Expr = toDomainIR(func_ref, ast, signature)
      pir_ast::Expr = toParallelIR(func_ref, dir_ast, signature)
      pir_ast = toFlatParfors(func_ref, pir_ast, signature)
      alreadyOptimized[(func, signature)] = pir_ast
      out = pir_ast
    else
      out = ast
    end

    #if use_extract_static_call_graph != 0
    #  callgraph = extractStaticCallGraph(func, signature)
    #  if DEBUG_LVL >= 3
    #    println("Callgraph:")
    #    println(callgraph)
    #    #throw(string("stop after cb"))
    #  end
    #end

    if level & TOPLEVEL > 0
      return toCGen(func_ref, out, signature)
    else
      return nothing
    end
  catch texp
    if CompilerTools.DebugMsg.PROSPECT_DEV_MODE
      rethrow(texp)
    end

    if isa(texp, ParallelAccelerator.UnsupportedFeature)
      println("ParallelAccelerator.accelerate cannot accelerate function ", func, " because the following unsupported feature was used.")
      println(texp.text)
    else
      println("ParallelAccelerator.accelerate cannot accelerate function ", func, " due to an unhandled exception of type ", typeof(texp), " whose value is ", texp)
    end

    return func
  end
end

end
 
