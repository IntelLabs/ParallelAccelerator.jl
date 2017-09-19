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

export accelerate, toDomainIR, toParallelIR, toFlatParfors, toJulia, toCGen, toCartesianArray, runStencilMacro, captureOperators, expandParMacro, extractCallGraph

using CompilerTools
using CompilerTools.AstWalker
using CompilerTools.LambdaHandling
using CompilerTools.Helper

import ..ParallelAccelerator, ..Comprehension, ..DomainIR, ..ParallelIR, ..CGen, ..API
import ..dprint, ..dprintln, ..@dprint, ..@dprintln, ..DEBUG_LVL
#import ..CallGraph.extractStaticCallGraph, ..CallGraph.use_extract_static_call_graph
using ..J2CArray

# MODE for accelerate
const TOPLEVEL=1
const PROXYONLY=3

isArrayOrStringType(x) = isArrayType(x) || isStringType(x)

# Convert regular Julia types to make them appropriate for calling C code.
# Note that it only handles conversion of () and Array, not tuples.
function convert_to_ccall_typ(typ)
  @dprintln(3,"convert_to_ccall_typ typ = ", typ, " typeof(typ) = ", typeof(typ))
  # if there a better way to check for typ being an array DataType?
  if isArrayType(typ)
    # If it is an Array type then convert to Ptr type.
    return (Ptr{Void},ndims(typ))
  elseif (typ === ())
    return (Void, 0)
  elseif isStringType(typ)
      return (Ptr{UInt8},1)
  else
    # Else no conversion needed.
    return (typ,0)
  end
end

"""
Convert return types to C interface types of Julia such as Cchar.
"""
function convert_to_Julia_typ(typ::DataType)
    if typ==Char
        return Cchar
    else
        return typ
    end
end

# Convert a whole function signature in a form of a tuple to something appropriate for calling C code.
function convert_sig(sig)
  assert(isa(sig,Tuple))   # make sure we got a tuple
  new_tuple = Expr(:tuple)         # declare the new tuple
  # fill in the new_tuple args/elements by converting the individual elements of the incoming signature
  new_tuple.args = [ convert_to_ccall_typ(sig[i])[1] for i = 1:length(sig) ]
  sig_ndims      = [ convert_to_ccall_typ(sig[i])[2] for i = 1:length(sig) ]
  @dprintln(3,"new_tuple.args = ", new_tuple.args)
  @dprintln(3,"sig_ndims = ", sig_ndims)
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

alreadyOptimized = Dict{Tuple{Function,Tuple},Any}()

# ParallelAccelerator doesn't really function well on callsites anymore so we would like to give an
# error if somebody tries to use it in that way.  This is a little difficult with @acc because where at callsite
# or on a function declaration, what the optimization pass will see is still a Function Expr.
# There is one difference though.  If there is @acc at a callsite then the ParallelAccelerator macro passes
# will not be called on the function.  In this global, we record which functions/signatures have passed through
# the ParallelAccelerator macro passes.  Then, in the domain IR pass, we check if the macro passes saw the
# function/signature and if not then throw an error.
seenByMacroPass = Set()

"""
A pass that translates supported operators and function calls to
those defined in ParallelAccelerator.API.
"""
function captureOperators(func, ast, sig)
  # See the comment on the global varibale seenByMacroPass for a description of this code.
  push!(seenByMacroPass, func)
  @dprintln(3, "captureOperators func = ", func, "func type = ", typeof(func), " ast = ", ast, " sig = ", sig, " typeof(ast) = ", typeof(ast))
  if typeof(ast) == Expr
    @dprintln(3, "ast.head = ", ast.head)
  end
  AstWalk(ast, API.Capture.process_node, nothing)
  return ast
end

"""
Pass that translates runStencil call in the same way as a macro would do.
This is only used when PROSPECT_MODE is off.
"""
function runStencilMacro(func, ast, sig)
  AstWalk(ast, API.Stencil.process_node, nothing)
  return ast
end

"""
Pass for comprehension to cartesianarray translation.
"""
function toCartesianArray(func, ast, sig)
  AstWalk(ast, Comprehension.process_node, nothing)
  return ast
end

"""
Pass for to convert @par to cartesianmapreduce. Without this pass, @par is a no-op.
"""
function expandParMacro(func, ast, sig)
  AstWalk(ast, API.Capture.process_par_macro, nothing)
  return ast
end

function extractCallGraph(func :: GlobalRef, ast, signature :: Tuple)
#    if use_extract_static_call_graph != 0
#      callgraph = extractStaticCallGraph(func, ast, signature)
#      @dprintln(3,"Callgraph:")
#      @dprintln(3,callgraph)
#      #throw(string("stop after cb"))
#    end
    return ast
end

function toDomainIR(func :: GlobalRef, ast, signature :: Tuple)
  # See the comment on the global varibale seenByMacroPass for a description of this code.
  if !in(func, seenByMacroPass)
    #throw(string("ParallelAccelerator no longer supports @acc at callsites.  Please add @acc to the declarations of functions that you want to optimize.  Function tried to optimize = ", func, " with signature = ", signature))
    println("ParallelAccelerator no longer supports @acc at callsites.  Please add @acc to the declarations of functions that you want to optimize.  Function tried to optimize = ", func, " with signature = ", signature)
    exit(-1)
  end
  dir_start = time_ns()
  code = DomainIR.from_expr(func.mod, ast)
  #code = LambdaVarInfoToLambda(linfo, body, DomainIR.AstWalk)
  dir_time = time_ns() - dir_start
  @dprintln(3, "domain code = ", code)
  @dprintln(1, "accelerate: DomainIR conversion time = ", ns_to_sec(dir_time))
  return code
end

function toParallelIR(func :: GlobalRef, ast, signature :: Tuple)
  pir_start = time_ns()
# uncomment these 2 lines for ParallelIR profiling
#  code = @profile ParallelIR.from_root(string(func.name), ast)
#  Profile.print()
  code = ParallelIR.from_root(string(func.name), ast)
  #code = LambdaVarInfoToLambda(linfo, body, ParallelIR.AstWalk)
  pir_time = time_ns() - pir_start
  @dprintln(3, "parallel code = ", code)
  @dprintln(1, "accelerate: ParallelIR conversion time = ", ns_to_sec(pir_time))
#throw(string("Done with ParallelIR"))
  return code
end

function toFlatParfors(func :: GlobalRef, ast, signature :: Tuple)
  code = ParallelIR.flattenParfors(string(func.name), ast)
  # code = CompilerTools.LambdaHandling.LambdaVarInfoToLambda(linfo, body, ParallelIR.AstWalk)
  @dprintln(3, "flattened code = ", code)
  return code
end

function toJulia(func :: GlobalRef, ast, signature :: Tuple)
  if isa(ast, Tuple)
    (LambdaVarInfo, body) = ast
  else
    LambdaVarInfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
  end
  return (LambdaVarInfo, body)
end

function toCGen(func :: GlobalRef, code, signature :: Tuple)
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
      if isStringType(t)
          array_types_in_sig[Array{UInt8, 1}] = atiskey
          atiskey += 1
      else
          while isArrayType(t)
              array_types_in_sig[t] = atiskey;
              atiskey += 1
              t = eltype(t)
          end
      end
  end
  @dprintln(3, "array_types_in_sig from signature = ", array_types_in_sig)

  if isa(code, Tuple)
    LambdaVarInfo, body = code
  else
    LambdaVarInfo, body = lambdaToLambdaVarInfo(code)
  end
  ret_type = getReturnType(LambdaVarInfo)
  # TO-DO: Check ret_type if it is Any or a Union in which case we probably need to abort optimization in CGen mode.
  ret_typs = isTupleType(ret_type) ? [ (convert_to_Julia_typ(x), isArrayOrStringType(x)) for x in ret_type.parameters ] : [ (convert_to_Julia_typ(ret_type), isArrayOrStringType(ret_type)) ]

  # Add arrays from the return type to array_types_in_sig.
  for rt in ret_typs
      t = rt[1]
      if isStringType(t)
          array_types_in_sig[Array{UInt8,1}] = atiskey
          atiskey += 1
      else
          while isArrayType(t)
              array_types_in_sig[t] = atiskey;
              atiskey += 1
              t = eltype(t)
          end
      end
  end
  @dprintln(3, "array_types_in_sig including returns = ", array_types_in_sig)
 
  outfile_name = CGen.writec(CGen.from_root_entry(code, function_name_string, signature, array_types_in_sig))
  CGen.compile(outfile_name)
  dyn_lib = CGen.link(outfile_name)
  full_outfile_name = "$package_root/deps/generated/$outfile_name.cpp"
  full_outfile_base = "$package_root/deps/generated/$outfile_name"
 
  # The proxy function name is the original function name with "_j2c_proxy" appended.
  proxy_name = string("_",function_name_string,"_j2c_proxy")
  proxy_sym = gensym(proxy_name)
  @dprintln(2, "toCGen for ", proxy_name)
  @dprintln(2, "C File  = ", full_outfile_name)
  @dprintln(2, "dyn_lib = ", dyn_lib)
  
  # This is the name of the function that j2c generates.
  j2c_name = string("_",function_name_string,"_")
  

  # Convert Arrays in signature to Ptr and add extra arguments for array dimensions
  (modified_sig, sig_dims) = convert_sig(signature)
  @dprintln(2, "modified_sig = ", modified_sig)
  @dprintln(2, "sig_dims = ", sig_dims)
  original_args = CompilerTools.LambdaHandling.getInputParameters(LambdaVarInfo)
  @dprintln(3, "len? ", length(original_args), length(sig_dims))
 
  map!(s -> gensym(string(s)), original_args, original_args)
  assert(length(original_args) == length(sig_dims))
  modified_args = Array{Any}(length(sig_dims))
  extra_inits = Array{Any}(0)
  j2c_array = gensym("j2c_arr")

  j2c_array_new = 
    # Create a new j2c array object with element size in bytes and given dimension.
    # It will share the data pointer of the given inp array, and if inp is nothing,
    # the j2c array will allocate fresh memory to hold data.
    # NOTE: when elem_bytes is 0, it means the elements must be j2c array type
    @eval (elem_bytes::Int, inp::Union{Array, Void}, ndim::Int, dims::Tuple) -> begin
      # note that C interface mandates Int64 for dimension data
      _dims = Int64[ convert(Int64, x) for x in dims ]
      _inp = (inp === nothing) ? C_NULL : convert(Ptr{Void}, pointer(inp))

      #ccall((:j2c_array_new, $dyn_lib), Ptr{Void}, (Cint, Ptr{Void}, Cuint, Ptr{UInt64}),
      #      convert(Cint, elem_bytes), _inp, convert(Cuint, ndim), pointer(_dims))
      ccall((:j2c_array_new, $dyn_lib), Ptr{Void}, (Cint, Ptr{Void}, Cuint, Ptr{UInt64}),
            convert(Cint, elem_bytes), _inp, convert(Cuint, ndim), pointer(_dims))
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
  ret_arg_exps = Array{Any}(0)
  extra_sig = Array{Type}(0)
  # We special-case functions that return Void/nothing since it is common.
  Void_return = (num_rets == 1 && ret_typs[1][1] == Void)
  if !Void_return
      for i = 1:num_rets
          (typ, is_array) = ret_typs[i]
          push!(extra_sig, is_array ? Ptr{Ptr{Void}} : Ptr{typ})
          push!(ret_arg_exps, Expr(:call, GlobalRef(Base, :pointer), Expr(:call, GlobalRef(Base, :arrayref), :ret_args, i)))
          #push!(ret_arg_exps, Expr(:call, TopNode(:pointer), Expr(:call, TopNode(:arrayref), toRHSVar(:ret_args, Array{Any,1}, LambdaVarInfo), i)))
      end
  end
  
  @dprintln(2,"signature = ", signature, " -> ", ret_typs)
  @dprintln(2,"modified_args = ", typeof(modified_args), " ", modified_args)
  @dprintln(2,"extra_sig = ", extra_sig)
  @dprintln(2,"ret_arg_exps = ", ret_arg_exps)
  tuple_sig_expr = Expr(:tuple,Cint,modified_sig...,extra_sig...)
  @dprintln(2,"tuple_sig_expr = ", tuple_sig_expr)
  proxy_func = @eval function ($proxy_sym)($(original_args...))
      # As we convert arrays into pointers that are stored in j2c_array objects, we remember in this
      # dictionary a mapping between an array's data pointer and the array object itself.  Later,
      # when processing arrays returned by the function, we see if some data pointer returned is
      # equal to one of the pointers in the ptr_array_dict.  If so, then the C code has returned an
      # array we passed to it as input and so from_j2c_array will get the original array from
      # ptr_array_dict and will alias to the returned array.
      ptr_array_dict = Dict{Ptr{Void},Array}()
      #@dprintln(2,"Running proxy function.")
      ret_args = Array{Any}($num_rets)
      for i = 1:$num_rets
        (t, is_array) = $(ret_typs)[i]
        if is_array
          t = Ptr{Void}
        end
        ret_args[i] = Array{t}(1) # hold return result
      end
      $(j2c_array) = [ $(extra_inits...) ]
      #j2c_array_typs = Any[ typeof(x) for x in $(extra_inits...) ]
      #@dprintln(3, "before ccall: ret_args = ", Any[$(ret_arg_exps...)])
      #@dprintln(3, "before ccall: modified_args = ", Any[$(modified_args...)])
      ccall(($j2c_name, $dyn_lib), Void, $tuple_sig_expr, $run_where, $(modified_args...), $(ret_arg_exps...))
      result = Array{Any}($num_rets)
      for i = 1:$num_rets
        (t, is_array) = $(ret_typs)[i]
        dprintln(3, "ret=", ret_args[i][1], "t=", t, " is_array=", is_array)
        if is_array
          if isStringType(t)
             result[i] = convert(t, from_ascii_string(ret_args[i][1], ptr_array_dict))
          else
             result[i] = from_j2c_array(ret_args[i][1], eltype(t), ndims(t), ptr_array_dict)
             if isBitArrayType(t)
                 result[i] = convert(BitArray, result[i])
             end
          end
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
  @dprintln(1, "accelerate: accelerate conversion time = ", ns_to_sec(off_time))
  return proxy_func
end

function code_typed(func, signature)
  global alreadyOptimized
  if haskey(alreadyOptimized, (func, signature))
    @dprintln(3, "func ", func, " is already optimized")
    return alreadyOptimized[(func, signature)]
  else
    @dprintln(3, "func ", func, " is not already optimized")
    mod = Base.function_module(func, signature)
    name = Base.function_name(func)
    func = eval(CompilerTools.OptFramework.findTargetFunc(mod, name))
    @dprintln(3, "target func is ", func)
    ast = Base.code_typed(func, signature)[1]
    if VERSION >= v"0.6.0-pre"
        t1 = ast
        @dprintln(3, "t1 = ", t1, " ", typeof(t1))
        ast = LambdaInfo(func, signature, t1)
    end
    return ast
  end
end

function code_llvm(func, signature)
    mod = Base.function_module(func, signature)
    name = Base.function_name(func)
    func = eval(CompilerTools.OptFramework.findTargetFunc(mod, name))
    @dprintln(3, "target func is ", func)
    Base.code_llvm(func, signature)
end

# Converts a given function and signature to use domain IR and parallel IR, and
# remember it so that it won't be translated again.
function accelerate(func::Function, signature::Tuple, level = TOPLEVEL)
  pse_mode = ParallelAccelerator.getPseMode() 
  if pse_mode == ParallelAccelerator.OFF_MODE
    return func
  end

  @dprintln(2, "Starting accelerate for ", func, " with signature ", signature)
  #if !isgeneric(func)
  #  @dprintln(1, "method ", func, " is not generic.")
  #  return nothing
  #end
  m = methods(func, signature)
  if length(m) < 1
    error("Method for ", func, " with signature ", signature, " is not found")
  end

  cur_module = Base.function_module(func, signature)
  func_ref = GlobalRef(cur_module, Base.function_name(func))

  local out
  ast = ParallelAccelerator.Driver.code_typed(func, signature)
  global alreadyOptimized 
  global seenByMacroPass

  # In threads mode we do not accelerator functions outside @acc
  if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE && !in(func_ref, seenByMacroPass)
      return func
  end

  try
    if !haskey(alreadyOptimized, (func, signature))
      push!(seenByMacroPass, func_ref)
      # place holder to prevent recursive accelerate
      alreadyOptimized[(func, signature)] = ast 
      dir_ast = toDomainIR(func_ref, ast, signature)
      pir_ast = toParallelIR(func_ref, dir_ast, signature)
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
 
