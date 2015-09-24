module Driver

export offload, Optimize, decompose

using CompilerTools
using CompilerTools.LambdaHandling

import ..ParallelAccelerator, ..DomainIR, ..ParallelIR, ..LD, ..Pert, ..cgen, ..DomainIR.isarray
import ..dprint, ..dprintln, ..DEBUG_LVL
import ..CallGraph.extractStaticCallGraph, ..CallGraph.use_extract_static_call_graph
using ..J2CArray

# MODE for offload
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

function ns_to_sec(x)
  x / 1000000000.0
end

# Converts a given function and signature to use library decomposition.
# It does NOT generates a stub/proxy like the offload function, or go through J2C.
function decompose(function_name, signature)
  if ParallelAccelerator.getPseMode() == ParallelAccelerator.OFF_MODE
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

function Optimize(ast :: Expr, call_sig_arg_tuple :: Tuple, call_sig_args)
  assert(ast.head == :lambda)

  dprintln(2, "Starting Optimize with args = ", call_sig_arg_tuple, " names = ", call_sig_args, "\n", ast, "\n")

  # Overall approach:
  # 1: create a copy of the incoming AST as a new function.
  # 2: pass that new function to the existing offload function.
  # 3: force the j2c function to compile
  # 4: get the AST of the proxy returned by offload and use that as the AST to return.
  func_copy_to_j2c = string("_ParallelAccelerator_optimize_", getNextFuncId())
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
    dprintln(3,"offload in Optimize returned something.")
    # Get the AST of the proxy function.
    cps   = string("_ParallelAccelerator_optimize_call_proxy_", getNextFuncId())
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


latest_func_id = 0
function getNextFuncId()
  cur = latest_func_id
  global latest_func_id = latest_func_id + 1
  return cur
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

# Converts a given function and signature to use domain IR and parallel IR.
# It also generates a stub/proxy with the same signature as the original that you can call to get you
# to the j2c version of the code.
function offload(function_name, signature, offload_mode=TOPLEVEL)
  pse_mode = ParallelAccelerator.getPseMode() 
  if pse_mode == ParallelAccelerator.OFF_MODE
    return function_name
  end
  start_time = time_ns()

  bt = backtrace() ;
  s = sprint(io->Base.show_backtrace(io, bt))
  dprintln(3, "offload backtrace ")
  dprintln(3, s)

  dprintln(2, "Starting offload for ", function_name, " with signature ", signature)
  if !isgeneric(function_name)
    dprintln(1, "method ", function_name, " is not generic.")
    return nothing
  end
  m = methods(function_name, signature)
  if length(m) < 1
    error("Method for ", function_name, " with signature ", signature, " is not found")
  end

  def = m[1].func.code
  global previouslyOptimized 
  dprintln(3, "previouslyOptimized = ", previouslyOptimized)
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
  assert(length(ct) == 1)
  ct = ct[1]
  lambdaInfo   = lambdaExprToLambdaInfo(ct)
  
#  dprintln(3,"before remove_gensym()",ct)
#  remove_gensym(ct)
#  dprintln(3,"after remove_gensym()",ct)

  if offload_mode & PROXYONLY != PROXYONLY
    push!(previouslyOptimized, (function_name, signature))
    dprintln(3, "Initial typed code = ", ct)
    domain_start = time_ns()
    domain_ir    = DomainIR.from_expr(string(function_name), cur_module, ct)     # convert that code to domain IR
    dir_time = time_ns() - domain_start
    dprintln(3, "domain code = ", domain_ir)
    dprintln(1, "offload: DomainIR conversion time = ", ns_to_sec(dir_time))
    input_arrays = get_input_arrays(lambdaInfo)
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
    dprintln(1, "ParallelAccelerator code optimization skipped")
    code = ct
  end

  if pse_mode == ParallelAccelerator.THREADS_MODE
    if no_precompile == 0
      precompile(function_name, signature)
    end
    return nothing
  end

  if offload_mode & TOPLEVEL == TOPLEVEL
     
    off_time_start = time_ns()
  
    package_root   = ParallelAccelerator.getPackageRoot()

    function_name_string = ParallelAccelerator.cgen.canonicalize(string(function_name))
	# cgen path
    outfile_name = cgen.writec(cgen.from_root(code, function_name_string))
    cgen.compile(outfile_name)
    dyn_lib = cgen.link(outfile_name)
 
    # The proxy function name is the original function name with "_j2c_proxy" appended.
    proxy_name   = string("_",function_name_string,"_j2c_proxy")
    proxy_sym    = symbol(proxy_name)
    dprintln(2, "ParallelAccelerator.offload for ", proxy_name)
    dprintln(2, "C File  = $package_root/deps/generated/$outfile_name.cpp")
    dprintln(2, "dyn_lib = ", dyn_lib)
  
    # This is the name of the function that j2c generates.
    j2c_name     = string("_",function_name_string,"_")
  
    ret_type     = CompilerTools.LambdaHandling.getReturnType(lambdaInfo)
    # TO-DO: Check ret_type if it is Any or a Union in which case we probably need to abort optimization in cgen mode.
    ret_typs     = DomainIR.istupletyp(ret_type) ? [ (x, isarray(x)) for x in ret_type.parameters ] : [ (ret_type, isarray(ret_type)) ]

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
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.HOST_MODE
      run_where = -1
    elseif ParallelAccelerator.getPseMode() == ParallelAccelerator.OFFLOAD1_MODE
      run_where = 0
    elseif ParallelAccelerator.getPseMode() == ParallelAccelerator.OFFLOAD2_MODE
      run_where = 1
    elseif ParallelAccelerator.getPseMode() == ParallelAccelerator.TASK_MODE
      pert_init(package_root, false)
    else
      assert(0)
    end
  
    num_rets = length(ret_typs)
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
      return ($num_rets == 1 ? result[1] : tuple(result...))
    end
  
    off_time = time_ns() - off_time_start
    dprintln(1, "offload: offload conversion time = ", ns_to_sec(off_time))
    
    if no_precompile == 0
    #  precompile(function_name, signature)
    #  precompile(proxy_func, signature)
    end
  
    end_time = time_ns() - start_time
    dprintln(1, "offload: total conversion time = ", ns_to_sec(end_time))
  
    if DEBUG_LVL >= 3
      proxy_ct = code_lowered(proxy_func, signature)
      if length(proxy_ct) == 0
        println("Error getting proxy code.\n")
      else
        proxy_ct = proxy_ct[1]
        println("Proxy code for ", function_name_string)
        println(proxy_ct)    
		println("Done printing proxy code")
      end
    end
  
    if stop_after_offload != 0
      throw(string("debuggging ParallelAccelerator"))
    end
    return proxy_func #@eval ParallelAccelerator.$proxy_sym
  else
    return nothing
  end
end

end
 
