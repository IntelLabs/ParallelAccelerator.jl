module DomainIR

import CompilerTools.AstWalker

# List of Domain IR Operators
#
# eltype  :: Array{T, n} -> T
# ndim    :: Array{T, n} -> Int
# length  :: Array{T, n} -> Int
# arraysize :: Array{T, n} -> Int -> Int
# sizes   :: Array{T, n} -> (Int, ...)n
# strides :: Array{T, n} -> (Int, ...)n
#
# assert :: Bool -> ... -> ()  # C style assertion check on all Bool arguments
# assertEqShape :: Array{T, n} -> Array{T, n} -> (Int, ...)n
#
# alloc    :: type -> (Int, ...)n -> Array{T, n}
# copy     :: Array{T, n} -> Array{T, n}
# generate :: (Int, ...)n -> ((Int, ...)n -> T) -> Array{T, n}
#
# reshape     :: Array{T, n} -> (Int, ...)m -> Array{T, m}
# backpermute :: Array{T, n} -> (Int, ...)m -> ((Int, ...)m -> (Int, ...)n) -> Array{T, m}
# arrayref  :: Array{T, n} -> Int -> T
# arrayset! :: Array{T, n} -> Int -> T -> T
#
# range    :: (Int, Int, Int) -> Mask{1}    # argument is (start, step, final)
# ranges   :: (Mask{1}, ...)n -> Mask{n}
# tomask   :: Array{Bool,n} -> Mask{n}
# tomaskFromIndex :: Array{Int,1} -> Mask{n}
# frommask :: (Int,...)n -> Mask{n} -> Array{Bool,n}
# negMask  :: Mask{n} -> Mask{n}
# andMask  :: Mask{n} -> Mask{n} -> Mask{n}
# orMask   :: Mask{n} -> Mask{n} -> Mask{n}
# select   :: Array{T, n} -> Mask{n} -> Array{T, n}
# select!  :: Array{T, n} -> Mask{n} -> Array{T, n}
#
# map   :: Array{T, n} -> (T -> S) -> Array{S, n}
# map!  :: Array{T, n} -> (T -> T) -> Array{T, n}
# mmap  :: (Array{T1, _}, ...)m -> ((T1, ...)m -> (S1, ...)n) -> (Array{S1, _}, ...)n
# mmap! :: (Array{T1, _}, ...)m -> ((T1, ...)m -> (T1, ...)n) -> (Array{T1, _}, ...)n, where m > =  n
#
# reduce  :: T -> Array{T, n} -> ((T, T) -> T) -> T
# mreduce :: (T, ...)m -> (Array{T1, _}, ...)m -> (((T1, ...)m, (T1, ...)m) -> (T1, ...)m) -> (T1, ...)m
#
# concat  :: Int -> Array{T, n} -> Array{T, n} -> Array{T, n}
# concats :: Int -> Array{Array{T,n},m} -> Array{T, m*n}
#
# gather   :: Array{T, n} -> Array{Int, m} -> Array{T, m}
# scatter! :: Array{T, n} -> Array{T, m} -> Array{Int, m} -> Array{T, n}
#
# The IR for stencil provides a single function to run (Iterative) stencil
# over a set of buffers, that may be updated in-place.
#
# stencil! :: KernelStat -> Int -> (Array{T1, _}, ...)m ->
#             ((T1, ...)m -> (T1, ...)m) -> (Array{T1, _}, ...)m
#
# 1. The KernelStat is defined in domain-ir-stencil.jl, and should be known
#    statically.
# 2. The first argument is the number of iterations to run.
# 3. All image buffers must be created prior to applyStencil! call.
# 4. The lambda takes a set of buffers, and returns the same number of
#    buffers, but with possible position rotation, which is useful for
#    iterative stencil computation (when iterations > 1).
# 5. The lambda body contains computations that potentially update one or more
#    of the buffer.
# 6. The final return result is what is returned after the last stecil
#    iteration. By default, it is what is returned from the lambda body.
# 7. Although the signature shows the kernel lambda takes only a set of
#    of buffers, in real implementation, it takes the set of index symbol
#    nodes, as well as stride symbol nodes, in addition to the buffers.

mk_eltype(arr) = Expr(:eltype, arr)
mk_ndim(arr) = Expr(:ndim, arr)
mk_length(arr) = Expr(:length, arr)
mk_arraysize(arr, d) = Expr(:arraysize, arr, d)
mk_sizes(arr) = Expr(:sizes, arr)
mk_strides(arr) = Expr(:strides, arr)
mk_alloc(typ, s) = Expr(:alloc, typ, s)
mk_copy(arr) = Expr(:copy, arr)
mk_generate(range, f) = Expr(:generate, range, f)
mk_reshape(arr, shape) = Expr(:reshape, arr, shape)
mk_backpermute(arr, f) = Expr(:backpermute, arr, f)
mk_arrayref(arr, idx) = Expr(:arrayref, arr, idx)
mk_arrayset!(arr, idx, v) = Expr(:arrayset!, arr, idx, v)
mk_range(start, step, final) = Expr(:range, start, step, final)
mk_ranges(ranges...) = length(ranges) == 1 ? ranges[1] : Expr(:ranges, ranges...)
mk_tomask(arr) = Expr(:tomask, arr)
mk_frommask(mask) = Expr(:frommask, mask)
mk_negmask(mask) = Expr(:negmask, mask)
mk_andmask(mask1, mask2) = Expr(:andmask, mask1, mask2)
mk_ormask(mask1, mask2) = Expr(:ormask, mask1, mask2)
mk_select(arr, mask) = Expr(:select, arr, mask)
mk_select!(arr, mask) = Expr(:select!, arr, mask)
mk_map(arr, f) = Expr(:map, arr, f)
mk_map!(arr, f) = Expr(:map!, arr, f)
mk_mmap(arrs, f) = Expr(:mmap, arrs, f)
mk_mmap!(arrs, f) = Expr(:mmap!, arrs, f)
# mmap! optionally takes a third parameter to indicate whether the
# iteration indices shall be fed to the function f as extra parameters.
mk_mmap!(arrs, f, withIndices) = Expr(:mmap!, arrs, f, withIndices)
mk_reduce(zero, arr, f) = Expr(:reduce, zero, arr, f)
mk_mreduce(zero, arrs, f) = Expr(:mreduce, zero, arrs, f)
mk_concat(dim, arr1, arr2) = Expr(:concat, dim, arr1, arr2)
mk_concats(dim, arrs) = Expr(:concats, dim, arrs)
mk_gather(arr, idx) = Expr(:gather, arr, idx)
mk_scatter!(base, arr, idx) = Expr(:scatter!, base, arr, idx)
mk_stencil!(stat, iterations, bufs, f) = Expr(:stencil!, stat, iterations, bufs, f)

function mk_expr(typ, args...)
  e = Expr(args...)
  e.typ = typ
  return e
end

function type_expr(typ, expr)
  expr.typ = typ
  return expr
end

export DomainLambda, KernelStat, set_debug_level, AstWalk, arraySwap, lambdaSwapArg, typeOfOpr, isarray

# A representation for anonymous lambda used in domainIR.
#   inputs:  types of input tuple
#   outputs: types of output tuple
#   genBody: Array{Any, 1} -> Array{Expr, 1}
#   escapes: escaping variables in the body
#
# So the downstream can just call genBody, and pass it
# the list of parameters, and it will return an expression
# (with :body head) that represents the loop body
#
# genBody always returns an array of expression even when there
# is only one. The last expression is always of the form:
#   (:tuple, values...)
# where there could be either 0 or multiple values being
# returned.
type VarDef
  typ  :: Type
  flag :: Int
  rhs  :: Any
end

type DomainLambda
  inputs  :: Array{Type, 1}
  outputs :: Array{Type, 1}
  genBody :: Function
  locals  :: Dict{Symbol, VarDef}
  escapes :: Dict{Symbol, VarDef}
end

function arraySwap(arr, i, j)
  x = arr[i]
  y = arr[j]
  arr1 = copy(arr)
  arr1[i] = y
  arr1[j] = x
  return arr1
end

# swap the i-th and j-th argument to domain lambda
function lambdaSwapArg(f::DomainLambda, i, j)
  DomainLambda(f.inputs, f.outputs,
    args -> f.genBody(arraySwap(args, i, j)),
    f.locals, f.escapes)
end

type IRState
  defs   :: Dict{Symbol, VarDef}
  stmts  :: Array{Any, 1}
  parent :: Union(Nothing, IRState)
end

emptyState()=IRState(Dict{Symbol,VarDef}(),Array(Any,0), nothing)
newState(defs, state::IRState)=IRState(defs, Array(Any,0), state)

lookupDef(state::IRState, s::Symbol)=get(state.defs, s, nothing)

function lookupConstDef(state::IRState, s::Symbol)
  def = get(state.defs, s, nothing)
  # flag bits are [is assigned once][is const][is assigned by inner function][is assigned][is captured]
  if !is(def, nothing) && (def.flag & (16 + 8)) != 0
    return def
  end
  return nothing
end

function lookupConstDefForArg(state::IRState, s::Any)
  while isa(s, SymbolNode)
    d = lookupConstDef(state, s.name)
    if isa(d, VarDef)
         s = d.rhs
    end
  end
  return s
end

function lookupDefInAllScopes(state::IRState, s::Symbol)
  def = lookupDef(state, s)
  if is(def, nothing) && !is(state.parent, nothing)
    return lookupDefInAllScopes(state.parent, s)
  else
    return def
  end
end


function emitStmt(state::IRState, stmt)
  dprintln(2,"emit stmt: ", stmt)
  push!(state.stmts, stmt)
end

type IREnv
  cur_module  :: Union(Module, Nothing)
  debugLevel  :: Int
  debugIndent :: Int
end

newEnv(m)=IREnv(m,2,0)
nextEnv(env::IREnv)=IREnv(env.cur_module, env.debugLevel, env.debugIndent + 1)

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

function dprint(env::IREnv,msgs...)
    if(DEBUG_LVL >= env.debugLevel)
        print(repeat(" ", env.debugIndent*2), msgs...)
    end
end

function dprintln(env::IREnv,msgs...)
    if(DEBUG_LVL >= env.debugLevel)
        println(repeat(" ", env.debugIndent*2), msgs...)
    end
end

mapSym = Symbol[:negate, :.<=, :.>=, :.<, :.==, :.>, :.+, :.-, :.*, :./, :+, :-, :*, :/, :sin, :erf, :log10, :exp, :sqrt, :min, :max]
mapVal = Symbol[:negate, :<=,  :>=,  :<, :(==), :>,  :+,  :-,  :*,  :/,  :+, :-, :*, :/, :sin, :erf, :log10, :exp, :sqrt, :min, :max]
# * / are not point wise. it becomes point wise only when one argument is scalar.
pointWiseOps = Set{Symbol}([:negate, :.<=, :.>=, :.<, :.==, :.>, :.+, :.-, :.*, :./, :+, :-, :sin, :erf, :log10, :exp, :sqrt, :min, :max])
mapOps = Dict{Symbol,Symbol}(zip(mapSym, mapVal))
# symbols that when lifted up to array level should be changed.
liftOps = Dict{Symbol,Symbol}(zip(Symbol[:<=, :>=, :<, :(==), :>, :+,:-,:*,:/], Symbol[:.<=, :.>=, :.<, :.==, :.>, :.+, :.-, :.*, :./]))
topOpsTypeFix = Set{Symbol}([:neg_int, :add_int, :mul_int, :sub_int, :neg_float, :mul_float, :add_float, :sub_float, :div_float, :box, :fptrunc, :fpsiround, :checked_sadd, :checked_ssub, :rint_llvm, :floor_llvm, :ceil_llvm, :abs_float])

opsSym = Symbol[:negate, :+, :-, :*, :/, :(==), :!=, :<, :<=]
opsSymSet = Set{Symbol}(opsSym)
floatOps = Dict{Symbol,Symbol}(zip(opsSym, [:neg_float, :add_float, :sub_float, :mul_float, :div_float,
                                            :eq_float, :ne_float, :lt_float, :le_float]))
sintOps  = Dict{Symbol,Symbol}(zip(opsSym, [:neg_int, :add_int, :sub_int, :mul_int, :sdiv_int,
                                            :seq_int, :sne_int, :slt_int, :sle_int]))

ignoreSym = Symbol[:box]
ignoreSet = Set{Symbol}(ignoreSym)

unique_id = 0
function freshsym(s::String)
  global unique_id
  unique_id = unique_id + 1
  return symbol(string(s, "##", unique_id))
end

include("domain-ir-stencil.jl")

function isinttyp(typ)
    is(typ, Int64)  || is(typ, Int32)  || is(typ, Int16)  || is(typ, Int8)  || 
    is(typ, Uint64) || is(typ, Uint32) || is(typ, Uint16) || is(typ, Uint8)
end

function isarray(typ)
  isa(typ, DataType) && is(typ.name, Array.name)
end

function isbitarray(typ)
  isa(typ, DataType) && is(typ.name, BitArray.name)
end

function isunitrange(typ)
  isa(typ, DataType) && is(typ.name, UnitRange.name)
end

function issteprange(typ)
  isa(typ, DataType) && is(typ.name, StepRange.name)
end

function from_range(rhs)
  if isa(rhs, Expr) && is(rhs.head, :new) && isunitrange(rhs.args[1]) &&
     isa(rhs.args[3], Expr) && is(rhs.args[3].head, :call) &&
     isa(rhs.args[3].args[1], Expr) && is(rhs.args[3].args[1].head, :call) &&
     is(rhs.args[3].args[1].args[1], TopNode(:getfield)) &&
     is(rhs.args[3].args[1].args[2], :Intrinsics) &&
     is(rhs.args[3].args[1].args[3], QuoteNode(:select_value))
    # only look at final value in select_value of UnitRange
    start = rhs.args[2]
    step  = 1 # FIXME: could be wrong here!
    final = rhs.args[3].args[3]
  elseif isa(rhs, Expr) && is(rhs.head, :call) && issteprange(rhs.args[1])
    assert(length(rhs.args) == 4)
    start = rhs.args[2]
    step  = rhs.args[3]
    final = rhs.args[4]
  else
    error("expect Expr(:new, UnitRange, ...) or Expr(StepRange, ...) but got ", rhs)
  end
  return (start, step, final)
end

function rangeToMask(state, r)
  if isa(r, SymbolNode)
    if isbitarray(r.typ)
      mk_tomask(r)
    elseif isunitrange(r.typ)
      r = lookupConstDefForArg(state, r)
      (start, step, final) = from_range(r)
      mk_range(start, step, final)
    elseif isinttyp(r.typ) 
      mk_range(r, convert(r.typ, 0), r)
    else
        error("Unhandled range object: ", r)
    end
  elseif isa(r, Int)
    mk_range(r, r, 1)
  else
    error("unhandled range object: ", r)
  end
end

# check if a given function can be considered as a map operation.
# Some operations depends on types.
function verifyMapOps(fun :: Symbol, args :: Array{Any, 1})
  if !haskey(mapOps, fun)
    return false
  elseif in(fun, pointWiseOps)
    return true
  else
    # for non-pointwise operators, only one argument can be array, the rest must be scalar
    n = 0
    for i = 1:length(args)
      typ = typeOfOpr(args[i])
      if isarray(typ) || isbitarray(typ)
        n = n + 1
      end
    end
    return (n == 1)
  end     
end

function specializeOp(opr, argstyp)
  reorder = x -> x # default no reorder
  if opr == :>= 
    opr = :<=
    reorder = reverse
  elseif opr == :>
    opr = :<
    reorder = reverse
  end
  # try to guess argument type
  typ = nothing
  for i = 1:length(argstyp)
    atyp = argstyp[i]
    if typ == nothing && atyp != nothing
      typ = atyp
    elseif is(atyp, Float32) || is(atyp, Float64)
      typ = atyp
    end
  end
  dprintln(2, "specializeOp opsSymSet[", opr, "] = ", in(opr, opsSymSet), " typ=", typ)
  if in(opr, opsSymSet)
    try
    # TODO: use subtype checking here?
    if is(typ, Int) || is(typ, Int32) || is(typ, Int64)
      opr = TopNode(sintOps[opr])
    elseif is(typ, Float32) || is(typ, Float64)
      opr = TopNode(floatOps[opr])
    end
    catch err
      error(string("Cannot specialize operator ", opr, " to type ", typ))
    end
  end
  return opr, reorder
end

import Base.show
import .._offload

function show(io::IO, f::DomainLambda)
  local len = length(f.inputs)
  local syms = [ symbol(string("x", i)) for i = 1:len ]
  local symNodes = [ SymbolNode(syms[i], f.inputs[i]) for i = 1:len ]
  local body = f.genBody(syms)
  print(io, "(")
  show(io, symNodes)
  print(io, ";")
  show(io, f.locals)
  print(io, ";")
  show(io, f.escapes)
  print(io, ") -> (")
  show(io, body)
  print(io, ")::", f.outputs)
end

uncompressed_ast(l::LambdaStaticData) =
  isa(l.ast,Expr) ? l.ast : ccall(:jl_uncompress_ast, Any, (Any,Any), l, l.ast)

# Specialize non-array arguments into the body function as either constants
# or escaping variables. Arguments of array type are treated as parameters to
# the body function instead.
# It returns the list of arguments that are specialized into the body function,
# the remaining arguments (that are of array type), their types, and
# the specialized body function.
function specialize(args::Array{Any,1}, typs::Array{Type,1}, bodyf::Function)
  local j = 0
  local len = length(typs)
  local idx = Array(Int, len)
  local args_ = Array(Any, len)
  local nonarrays = Array(Any, 0)
  for i = 1:len
    local typ = typs[i]
    if isarray(typ) || isbitarray(typ)
      j = j + 1
      typs[j] = typ
      args_[j] = args[i]
      idx[j] = i
    else
      push!(nonarrays, args[i])
    end
  end
  function mkFun(params)
    local myArgs = copy(args)
    assert(length(params)==j)
    local i
    for i=1:j
      myArgs[idx[i]] = params[i]
    end
    bodyf(myArgs)
  end
  return (nonarrays, args_[1:j], typs[1:j], mkFun)
end

function typeOfOpr(x)
  if isa(x, Expr) x.typ
  elseif isa(x, SymbolNode) x.typ
  else typeof(x)
  end
end

# get elem type T from an Array{T} type
function elmTypOf(x)
  if isarray(x)
    return x.parameters[1] 
  elseif isa(x, Expr) && x.head == :call && x.args[1] == TopNode(:apply_type) && x.args[2] == :Array
    return x.args[3] # element type
  elseif isbitarray(x)
    return Bool
  else
    error("expect Array type, but got ", x)
  end
end

function metaToVarDef(meta)
  defs = Dict{Symbol,VarDef}()
  for m in meta
    assert(length(m)==3)
    defs[m[1]] = VarDef(m[2], m[3], nothing)
    #push!(defs, m[1], VarDef(m[2], m[3], nothing))
  end
  return defs
end

# :lambda expression
# (:lambda, {param, meta@{localvars, types, freevars}, body})
function from_lambda(state, env, expr)
  local env_ = nextEnv(env)
  local head = expr.head
  local ast  = expr.args
  local typ  = expr.typ
  assert(length(ast) == 3)
  local param = ast[1]
  local meta  = ast[2] # { {Symbol}, {{Symbol,Type,Int}}, {Symbol,Type,Int} }
  local body  = ast[3]
  assert(isa(body, Expr) && is(body.head, :body))
  local state_ = newState(metaToVarDef(meta[2]), state)
  body = from_expr(state_, env_, body)
  # fix return type
  typ = body.typ
  dprintln(env,"from_lambda: body=", body)
  meta[1] = Array(Any, 0)
  meta[2] = Array(Any, 0)
  for (v, def) in state_.defs
    push!(meta[1], v)
    push!(meta[2], Any[v, def.typ, def.flag])
  end
  dprintln(env,"from_lambda: meta=", meta)
  return mk_expr(typ, head, param, meta, body)
end

# sequence of expressions {expr, ...}
# unlike from_body, from_exprs do not emit the input expressions
# as statements to the state, while still allowing side effects
# of emitting statements durinfg the translation of these expressions.
function from_exprs(state, env, ast::Array{Any,1})
  local env_ = nextEnv(env)
  local len  = length(ast)
  local body = Array(Any, len)
  for i = 1:len
    body[i] = from_expr(state, env_, ast[i])
  end
  return body
end

# :body expression (:body, {expr, ... })
# Unlike from_exprs, from_body treats every expr in the body
# as separate statements, and emit them (after translation)
# to the state one by one.
function from_body(state, env, expr::Any)
  local env_ = nextEnv(env)
  # So long as :body immediate nests with :lambda, we do not
  # need to create new state.
  local head = expr.head
  local body = expr.args
  local typ  = expr.typ
  for i = 1:length(body)
    expr = from_expr(state, env_, body[i])
    emitStmt(state, expr)
  end
  # fix return type
  n = length(state.stmts)
  while n > 0
    last_exp = state.stmts[n]
    if isa(last_exp, LabelNode) 
      n = n - 1
    elseif isa(last_exp, Expr) && last_exp.head == :return
      typ = state.stmts[n].typ
      break
    else
      error("Cannot figure out return type from function body")
    end
  end
  return mk_expr(typ, head, state.stmts...)
end

function mmapRemoveDupArg!(expr)
  head = expr.head # must be :mmap or :mmap!
  arr = expr.args[1]
  f = expr.args[2]
  posMap = Dict{Symbol, Int}()
  indices = Array(Int, length(arr))
  hasDup = false
  n = 1
  newarr = Any[]
  newinp = Array(Type, 0)
  oldn = length(arr)
  for i = 1:oldn
    indices[i] = n
    s = arr[i]
    if isa(s, SymbolNode) s = s.name end
    if isa(s, Symbol)
      if haskey(posMap, s)
	hasDup = true
	indices[i] = posMap[s]
      else
	push!(posMap, s, n)
	push!(newarr, arr[i])
	push!(newinp, f.inputs[i])
        n += 1
      end
    end
  end
  if (!hasDup) return expr end
  dprintln(3, "MMRD: expr was ", expr)
  dprintln(3, "MMRD:  ", newarr, newinp, indices)
  expr.args[1] = newarr
  expr.args[2] = DomainLambda(newinp, f.outputs,
    args -> begin
	dupargs = Array(Any, oldn)
        for i=1:oldn
	  dupargs[i] = args[indices[i]]
	end
	f.genBody(dupargs)
    end, f.locals, f.escapes)
  dprintln(3, "MMRD: expr becomes ", expr)
  return expr
end

# :(=) assignment (:(=), lhs, rhs)
function from_assignment(state, env, expr::Any)
  local env_ = nextEnv(env)
  local head = expr.head
  local ast  = expr.args
  local typ  = expr.typ
  assert(length(ast) == 2)
  local lhs = ast[1]
  local rhs = ast[2]
  if isa(lhs, SymbolNode)
    lhs = lhs.name
  end
  if typeof(lhs) != Symbol
    println(expr, " lhs type = ", typeof(lhs))
  end
  assert(isa(lhs, Symbol))
  rhs = from_expr(state, env_, rhs)
  def = lookupDef(state, lhs)
  assert(isa(def, VarDef))
  dprintln(env, "from_assignment lhs=", lhs, " typ=", typ)
  # turn x = mmap((x,...), f) into x = mmap!((x,...), f)
  if isa(rhs, Expr) && is(rhs.head, :mmap) && length(rhs.args[1]) > 0 &&
     (is(lhs, rhs.args[1][1]) || (isa(rhs.args[1][1], SymbolNode) && is(lhs, rhs.args[1][1].name)))
     rhs.head = :mmap!
     lhs = freshsym(string(lhs))
  end
  # TODO: handle indirections like x = y so that x gets y's definition instead of just y.
  push!(state.defs, lhs, VarDef(def.typ, def.flag, rhs))
  return mk_expr(typ, head, lhs, rhs)
end

function from_call(state, env, expr::Any)
  local env_ = nextEnv(env)
  local head = expr.head
  local ast = expr.args
  local typ = expr.typ
  assert(length(ast) >= 1)
  local fun  = ast[1]
  local args = ast[2:end]
  dprintln(env,"from_call: fun=", fun, " typeof(fun)=", typeof(fun), " args=",args, " typ=", typ)
  fun = from_expr(state, env_, fun)
  dprintln(env,"from_call: new fun=", fun)
  (fun_, args_) = normalize_callname(state, env, fun, args)
  return translate_call(state, env, typ, :call, fun, args, fun_, args_)
end

# turn Exprs in args into variable names, and put their definition into state
# anything of void type in the argument is omitted in return value.
function normalize_args(state::IRState, env, args)
  args = from_exprs(state, env, args)
  j = 0
  for i = 1:length(args)
    local arg = args[i]
    if isa(arg, Expr) && arg.typ == Void
      # do not produce new assignment for Void values
      dprintln(3, "normalize_args got Void args[", i, "] = ", arg)
      emitStmt(state, arg)
    elseif isa(arg, Expr) || isa(arg, LambdaStaticData)
      newVar = freshsym("arg")
      typ = isa(arg, Expr) ? arg.typ : Any
      # set flag [is assigned once][is const][is assigned by inner function][is assigned][is captured]
      push!(state.defs, newVar, VarDef(typ, 16+8, arg))
      emitStmt(state, mk_expr(typ, :(=), newVar, arg))
      j = j + 1
      args[j] = SymbolNode(newVar, typ)
    else
      j = j + 1
      args[j] = args[i]
    end
  end
  return args[1:j]
end

# Fix Julia inconsistencies in call before we pattern match
function normalize_callname(state::IRState, env, fun, args)
  if isa(fun, Symbol)
    if is(fun, :broadcast!)
      dst = lookupConstDefForArg(state, args[2])
      if isa(dst, Expr) && is(dst.head, :call) && isa(dst.args[1], TopNode) &&
         is(dst.args[1].name, :ccall) && isa(dst.args[2], QuoteNode) &&
         is(dst.args[2].value, :jl_new_array)
        # now we are sure destination array is new
        fun   = args[1]
        args  = args[3:end]
        if isa(fun, Symbol)
        elseif isa(fun, SymbolNode)
          fun1 = lookupConstDef(state, fun.name)
          if isa(fun1, VarDef)
            fun = fun1.rhs
          end
        else
          error("DomainIR: cannot handle broadcast! with function ", fun)
        end
      elseif isa(dst, Expr) && is(dst.head, :call) && isa(dst.args[1], DataType) &&
         is(dst.args[1].name, BitArray.name) 
        # destination array is a new bitarray
        fun   = args[1]
        args  = args[3:end]
        if isa(fun, SymbolNode) 
          fun = fun.name
        end
        if isa(fun, Symbol)
          # fun could be a variable 
          fun_def = get(state.defs, fun, nothing)
          if isa(fun_def, VarDef)
            fun = fun_def.rhs
          end
        end
        if !isa(fun, Symbol)
          error("DomainIR: cannot handle broadcast! with function ", fun)
        end
      else
        dprintln(env, "cannot decide :broadcast! destination is temporary ")
      end
      if haskey(liftOps, fun) # lift operation to array level
        fun = liftOps[fun]
      end
    end
  elseif isa(fun, TopNode)
    fun = fun.name
    if is(fun, :ccall)
      callee = lookupConstDefForArg(state, args[1])
      if isa(callee, QuoteNode) && (is(callee.value, :jl_alloc_array_1d) || is(callee.value, :jl_alloc_array_2d) || is(callee.value, :jl_alloc_array_3d))
        local realArgs = Any[]
        for i = 4:2:length(args)
          push!(realArgs, args[i])
        end
        fun  = :alloc
        args = realArgs
      else
      end
    end
  elseif isa(fun, GetfieldNode)
    if is(fun.value, Base.Broadcast)
      if is(fun.name, :broadcast_shape)
        fun = :broadcast_shape
      end
    end
  end
  return (fun, args)
end

# if a definition of arr is getindex(a, ...), return select(a, ranges(...))
# otherwise return arr unchanged.
function inline_select(env, state, arr)
  range_extra = Any[]
  if isa(arr, SymbolNode) 
    # TODO: this requires safety check. Local lookups are only correct if free variables in the definition have not changed.
    def = lookupConstDef(state, arr.name)
    if !isa(def, Nothing)  
      if isa(def.rhs, Expr) && is(def.rhs.head, :call) 
        assert(length(def.rhs.args) > 2)
        if is(def.rhs.args[1], :getindex)
          arr = def.rhs.args[2]
          range_extra = def.rhs.args[3:end]
        elseif def.rhs.args[1] == TopNode(:_getindex!) # getindex gets desugared!
          error("we cannot handle TopNode(_getindex!) because it is effectful and hence will persist until J2C time")
        end
        dprintln(env, "inline-select: arr = ", arr, " range = ", range_extra)
        if length(range_extra) > 0
          ranges = mk_ranges([rangeToMask(state, r) for r in range_extra]...)
          arr = mk_select(arr, ranges)
        end
      end
    end
  end
  return arr
end

# translate a function call to domain IR if it matches
function translate_call(state, env, typ, head, oldfun, oldargs, fun, args)
  local env_ = nextEnv(env)
  expr = nothing
  dprintln(env, "translate_call fun=", fun, "::", typeof(fun), " args=", args, " typ=", typ)
  if isa(fun, Symbol)
    dprintln(env, "verifyMapOps -> ", verifyMapOps(fun, args))
    if verifyMapOps(fun, args) && (isarray(typ) || isbitarray(typ)) 
      # TODO: check for unboxed array type
      args = normalize_args(state, env_, args)
      etyp = elmTypOf(typ) 
      if is(fun, :-) && length(args) == 1
        fun = :negate
      end
      typs = Type[ typeOfOpr(arg) for arg in args ]
      elmtyps = Type[ (isarray(t) || isbitarray(t)) ? elmTypOf(t) : t for t in typs ]
      opr, reorder = specializeOp(mapOps[fun], elmtyps)
      typs = reorder(typs)
      args = reorder(args)
      dprintln(env,"from_lambda: before specialize, opr=", opr, " args=", args, " typs=", typs)
      (nonarrays, args, typs, f) = specialize(args, typs, 
                          as -> [Expr(:tuple, mk_expr(etyp, :call, opr, as...))])
      dprintln(env,"from_lambda: after specialize, typs=", typs)
      elmtyps = Type[ (isarray(t) || isbitarray(t)) ? elmTypOf(t) : t for t in typs ]
      # calculate escaping variables
      escapes = Dict{Symbol,VarDef}()
      for i=1:length(nonarrays)
        # At this point, they are either symbol nodes, or constants
        if isa(nonarrays[i], SymbolNode)
            push!(escapes, nonarrays[i].name, VarDef(nonarrays[i].typ, 0, nothing))
        end
      end
      domF = DomainLambda(elmtyps, [etyp], f, Dict{Symbol,VarDef}(), escapes)
      for i = 1:length(args)
        arg_ = inline_select(env, state, args[i])
        if arg_ != args[i] && i != 1 && length(args) > 1
          error("Selector array must be the only array argument to mmap: ", args)
        end
        args[i] = arg_
      end
      expr = mmapRemoveDupArg!(mk_mmap(args, domF))
      expr.typ = typ
    elseif is(fun, :cartesianarray)
      # equivalent to creating an array first, then map! with indices.
      dprintln(env, "got cartesianarray args=", args)
      # need to retrieve map lambda from inits, since it is already moved out.
      nargs = length(args)
      args = normalize_args(state, env_, args)
      assert(nargs >= 3) # needs at least a function, one or more types, and a dimension tuple
      local dimExp = args[end]     # last argument is the dimension tuple
      assert(isa(dimExp, SymbolNode))
      dimExp = lookupConstDefForArg(state, dimExp)
      dprintln(env, "dimExp = ", dimExp, " head = ", dimExp.head, " args = ", dimExp.args)
      assert(isa(dimExp, Expr) && is(dimExp.head, :call) && is(dimExp.args[1], TopNode(:tuple)))
      dimExp = dimExp.args[2:end]
      ndim = length(dimExp)   # num of dimensions
      argstyp = Any[ Int for i in 1:ndim ] 
      local mapExp = args[1]     # first argument is the lambda
      if isa(mapExp, Symbol) && !is(env.cur_module, nothing) && isdefined(env.cur_module, mapExp) && !isdefined(Base, mapExp) # only handle functions in current or Main module
        dprintln(env,"function for cartesianarray: ", mapExp, " methods=", methods(getfield(env.cur_module, mapExp)), " argstyp=", argstyp)
        m = methods(getfield(env.cur_module, mapExp), tuple(argstyp...))
        assert(length(m) > 0)
        mapExp = m[1].func.code
      elseif isa(mapExp, SymbolNode)
        mapExp = lookupConstDefForArg(state, mapExp)
      end
      assert(isa(mapExp, LambdaStaticData))
      # call typeinf since Julia doesn't do it for us
      # and we can figure out the element type from mapExp's return type
      (tree, ety)=Base.typeinf(mapExp, tuple(argstyp...), ())
      etys = isa(ety, Tuple) ? Type[ t for t in ety ] : Type[ ety ]
      mapExp.ast = tree
      # make sure we go through domain translation on the lambda too
      ast = from_expr("anonymous", env.cur_module, uncompressed_ast(mapExp))
      # dprintln(env, "ast = ", ast)
      # create tmp arrays to store results
      arrtyps = Type[ Array{t, ndim} for t in etys ]
      tmpNodes = Array(SymbolNode, length(arrtyps))
      # allocate the tmp array
      for i = 1:length(arrtyps)
        tmparr = freshsym("arr")
        arrdef = type_expr(arrtyps[i], mk_alloc(etys[i], dimExp))
        push!(state.defs, tmparr, VarDef(arrtyps[i], 2, arrdef))
        emitStmt(state, mk_expr(arrtyps[i], :(=), tmparr, arrdef))
        tmpNodes[i] = SymbolNode(tmparr, arrtyps[i])
      end
      # produce a DomainLambda
      body = ast.args[3]
      params = [ if isa(x, Expr) x.args[1] else x end for x in ast.args[1] ]
      # dprintln(env, "params = ", params)
      locals = metaToVarDef(ast.args[2][2])
      escapes = metaToVarDef(ast.args[2][3])
      assert(isa(body, Expr) && is(body.head, :body))
      # fix the return in body
      lastExp = body.args[end]
      assert(isa(lastExp, Expr) && is(lastExp.head, :return))
      # FIXME: lastExp may be returning a tuple
      # dprintln(env, "fixReturn: lastExp = ", lastExp)
      if (isa(lastExp.args[1], SymbolNode) &&
          isa(lastExp.args[1].typ, Tuple))
        # create tmp variables to store results
        local tvar = lastExp.args[1]
        local typs = tvar.typ
        local nvar = length(typs)
        local retNodes = SymbolNode[ SymbolNode(freshsym("ret"), t) for t in typs ]
        local retExprs = Array(Expr, length(retNodes))
        for i in 1:length(retNodes)
          n = retNodes[i]
          push!(locals, n.name, VarDef(n.typ, 16+2, nothing)) # tmp vars assigned only once
          retExprs[i] = mk_expr(n.typ, :(=), n.name,
                                  mk_expr(n.typ, :call, TopNode(:tupleref), tvar, i))
        end
        lastExp.head = retExprs[1].head
        lastExp.args = retExprs[1].args
        lastExp.typ  = retExprs[1].typ
        for i = 2:length(retExprs)
          push!(body.args, retExprs[i])
        end
        push!(body.args, mk_expr(typs, :tuple, retNodes...))
      else
        lastExp.head = :tuple
      end
      bodyF(args)=replaceWithDict(body, Dict{Symbol,Any}(zip(params, args[1+length(etys):end]))).args
      domF = DomainLambda([etys, argstyp], etys, bodyF, locals, escapes)
      expr = mk_mmap!(tmpNodes, domF, true)
      expr.typ = length(arrtyps) == 1 ? arrtyps[1] : tuple(arrtyps...)
    elseif is(fun, :runStencil)
      # we handle the following runStencil form:
      #  runStencil(kernelFunc, buf1, buf2, ..., [iterations], [border], buffers)
      # where kernelFunc takes the same number of bufs as inputs
      # and returns the same number of them, but in different order,
      # as a rotation specification for iterative stencils.
      dprintln(env,"got runStencil, args=", args)
      # need to retrieve stencil kernel lambda from inits, since it is already moved out.
      local nargs = length(args)
      args = normalize_args(state, env_, args)
      assert(nargs >= 3) # needs at least a function, and two buffers
      local iterations = 1               # default
      local borderExp = nothing          # default
      local kernelExp = args[1]
      local bufs = Any[]
      local bufstyp = Any[]
      local i
      for i = 2:nargs
        oprTyp = typeOfOpr(args[i])
        if isarray(oprTyp)
          push!(bufs, args[i])
          push!(bufstyp, oprTyp)
        else
          break
        end
      end
      if i == nargs
        if is(typeOfOpr(args[i]), Int)
          iterations = args[i]
        else
          borderExp = args[i]
        end
      elseif i + 1 <= nargs
        iterations = args[i]
        borderExp = args[i+1]
      end
      assert(isa(kernelExp, SymbolNode))
      kernelExp = lookupConstDefForArg(state, kernelExp)
      assert(isa(kernelExp, LambdaStaticData))
      # TODO: better infer type here
      (tree, ety)=Base.typeinf(kernelExp, tuple(bufstyp...), ())
      #etys = isa(ety, Tuple) ? Type [ t for t in ety ] : Type[ ety ]
      kernelExp.ast = tree
      kernelExp = from_expr(state, env_, uncompressed_ast(kernelExp))
      if !is(borderExp, nothing)
        borderExp = lookupConstDefForArg(state, borderExp)
      end
      dprintln(env, "bufs = ", bufs, " kernelExp = ", kernelExp, " borderExp=", borderExp, " :: ", typeof(borderExp))
      local stat, kernelF
      stat, kernelF = mkStencilLambda(bufs, kernelExp, borderExp)
      expr = mk_stencil!(stat, iterations, bufs, kernelF)
      #typ = length(bufs) > 2 ? tuple(kernelF.outputs...) : kernelF.outputs[1] 
      # force typ to be Void, which means stencil doesn't return anything
      typ = Void
      expr.typ = typ
    elseif is(fun, :copy)
      args = normalize_args(state, env_, args[1:1])
      dprintln(env,"got copy, args=", args)
      expr = mk_copy(args[1])
      expr.typ = typ
    elseif in(fun, topOpsTypeFix) && is(typ, Any) && length(args) > 0
      typ1 = typeOfOpr(args[1])
      if is(fun, :fptrunc)
        if is(args[1], :Float32) typ1 = Float32
        elseif is(args[1], :Float64) typ1 = Float64
        else throw(string("unknown target type for fptrunc: ", typ1))
        end
      elseif is(fun, :fpsiround)
        if is(typ1, Float32) typ1 = Int32
        elseif is(typ1, Float64) typ1 = Int64
        else throw(string("unknown target type for fpsiround: ", typ1))
        end
      end
      dprintln(env,"fix type ", typ, " => ", typ1)
      typ = typ1
    elseif is(fun, :arraysize)
      args = normalize_args(state, env_, args)
      dprintln(env,"got arraysize, args=", args)
      expr = mk_arraysize(args...)
      expr.typ = typ
    elseif is(fun, :alloc) || is(fun, :Array)
      typExp = lookupConstDefForArg(state, args[1])
      args = normalize_args(state, env_, args[2:end])
      if isa(typExp, QuoteNode) 
        elemTyp = typExp.value 
      elseif isa(typExp, DataType)
        elemTyp = elmTypOf(typExp)
      else
        error("Expect QuoteNode or DataType, but got typExp = ", typExp)
      end
      expr = mk_alloc(elemTyp, args)
      expr.typ = typ
    elseif is(fun, :broadcast_shape)
      dprintln(env, "got ", fun)
      args = normalize_args(state, env_, args)
      expr = mk_expr(typ, :assertEqShape, args...)
    elseif is(fun, :checkbounds)
      dprintln(env, "got ", fun)
      assert(length(args) == 2)
      args = normalize_args(state, env_, args)
      typ = typeOfOpr(args[1])
      if !isinttyp(typ)
        error("Unhandled bound in checkbounds: ", args[1])
      end
      if isinttyp(typeOfOpr(args[2]))
        expr = mk_expr(Bool, :assert, mk_expr(Bool, :call, TopNode(:sle_int), convert(typ, 1), args[2]),
                                      mk_expr(Bool, :call, TopNode(:sle_int), args[2], args[1]))
      elseif isa(args[2], SymbolNode) && (isunitrange(args[2].typ) || issteprange(args[2].typ))
        def = lookupConstDefForArg(state, args[2])
        (start, step, final) = from_range(def.rhs)
        expr = mk_expr(Bool, :assert, mk_expr(Bool, :call, TopNode(:sle_int), convert(typ, 1), start),
                                      mk_expr(Bool, :call, TopNode(:sle_int), final, args[1]))
      else
        error("Unhandled bound in checkbounds: ", args[2])
      end
    elseif is(fun, :sitofp)
      typ = args[1]
    elseif is(fun, :assign_bool_scalar_1d!) || # args = (array, scalar_value, bitarray)
           is(fun, :assign_bool_vector_1d!)    # args = (array, getindex_bool_1d(array, bitarray), bitarray) 
      etyp = elmTypOf(typ)
      args = normalize_args(state, env_, args)
      arr2 = nothing
      if is(fun, :assign_bool_vector_1d!) # must has a mattching getindex_bool_1d
        arr2 = args[2]
        if isa(arr2, SymbolNode)
          def = lookupConstDefForArg(state, arr2)
          if isa(def, Expr) && is(def.head, :call) && def.args[1] == TopNode(:getindex_bool_1d) 
            #dprintln(env, "matching :getindex_bool_1d")
            b1 = lookupConstDefForArg(state, def.args[3])
            b2 = lookupConstDefForArg(state, args[3])
            if b1 == b2 # got a match?
              arr2 = def.args[2]
            end
          end
        end
      end
      if is(arr2, args[2]) 
        error("expect :assign_bool_vector_1d! to be used with a matching :getindex_bool_1d")
      elseif !is(arr2, nothing)
        args[2] = arr2
      end
      assert(length(args) == 3)
      typs = Type[typeOfOpr(a) for a in args]
      (nonarrays, args, typs, f) = specialize(args, typs, 
            as -> [ Expr(:tuple, mk_expr(etyp, :call, TopNode(:select_value), as[3], as[2], as[1])) ])
      elmtyps = Type[ (isarray(t) || isbitarray(t)) ? elmTypOf(t) : t for t in typs ]
      escapes = Dict{Symbol,VarDef}()
      for i=1:length(nonarrays)
        # At this point, they are either symbol nodes, or constants
        if isa(nonarrays[i], SymbolNode)
            push!(escapes, nonarrays[i].name, VarDef(nonarrays[i].typ, 0, nothing))
        end
      end
      domF = DomainLambda(elmtyps, Type[etyp], f, Dict{Symbol,VarDef}(), escapes)
      expr = mmapRemoveDupArg!(mk_mmap!(args, domF))
      expr.typ = typ
    elseif is(fun, :fill!)
      args = normalize_args(state, env_, args)
      assert(length(args) == 2)
      arr = args[1]
      ival = args[2]
      typs = Type[typeOfOpr(arr)]
      f = as -> [ Expr(:tuple, ival) ]
      escapes = Dict{Symbol,VarDef}()
      if isa(ival, SymbolNode)
        def = lookupConstDef(state, ival.name)
        def = def == nothing ? VarDef(ival.typ, 0, nothing) : 
                               VarDef(ival.typ, def.flag, nothing)
        push!(escapes, ival.name, def)
      end
      domF = DomainLambda(typs, typs, f, Dict{Symbol,VarDef}(), escapes)
      expr = mmapRemoveDupArg!(mk_mmap!([arr], domF))
      expr.typ = typ
    elseif is(fun, :_getindex!) # see if we can turn getindex! back into getindex
      if isa(args[1], Expr) && args[1].head == :call && args[1].args[1] == TopNode(:ccall) && 
         (args[1].args[2] == :jl_new_array ||
          (isa(args[1].args[2], QuoteNode) && args[1].args[2].value == :jl_new_array))
        expr = mk_expr(typ, :call, :getindex, args[2:end]...)
      end
    elseif is(fun, :sum) || is(fun, :prod)
      args = normalize_args(state, env_, args)
      assert(length(args) == 1)
      arr = args[1]
      # element type is the same as typ
      etyp = is(typ, Any) ? elmTypOf(typeOfOpr(arr)) : typ;
      neutral = is(fun, :sum) ? 0 : 1
      fun = is(fun, :sum) ? :+ : :*
      typs = Type[ etyp for arg in args] # just use etyp for input element types
      opr, reorder = specializeOp(mapOps[fun], typs)
      # ignore reorder since it is always id function
      f = as -> [Expr(:tuple, mk_expr(etyp, :call, opr, as...))]
      domF = DomainLambda([etyp, etyp], [etyp], f, Dict{Symbol,VarDef}(), Dict{Symbol,VarDef}())
      # turn reduce(z, getindex(a, ...), f) into reduce(z, select(a, ranges(...)), f)
      arr = inline_select(env, state, arr)
      expr = mk_reduce(convert(etyp, neutral), arr, domF)
      expr.typ = typ
    elseif in(fun, ignoreSet)
    else
      args_typ = map(typeOfOpr, args)
      if !is(env.cur_module, nothing) && isdefined(env.cur_module, fun) && !isdefined(Base, fun) # only handle functions in Main module
        dprintln(env,"function to offload: ", fun, " methods=", methods(getfield(env.cur_module, fun)))
        _offload(getfield(env.cur_module, fun), tuple(args_typ...))
        oldfun = GetfieldNode(env.cur_module, fun, Any)
      elseif is(fun, :getfield)
        dprintln(env,"eval getfield with args ", args)
        if isa(args[1], Symbol) && isa(args[2], QuoteNode)
        # translate getfield call to getfieldNode
        local m
        try
          m = eval(Main, args[1]) # FIXME: the use of Main is not exactly correct here!
        catch err
          dprintln(env,"module name ", args[1], " fails to resolve")
          throw(err)
        end
        return GetfieldNode(m, args[2].value, typ) # TODO: fill in the type properly?
        end
      else
        dprintln(env,"function call not translated: ", fun, ", typeof(fun)=", typeof(fun), "head = ", head, "oldfun = ", oldfun, ", args typ=", args_typ)
      end
    end
  elseif isa(fun, GetfieldNode)
    if is(fun.value, Base.Math)
      # NOTE: we simply bypass all math functions for now
      dprintln(env,"by pass math function ", fun, ", typ=", typ)
      # Fix return type of math functions
      if is(typ, Any) && length(args) > 0
        dprintln(env,"fix type for ", expr, " from ", typ, " => ", args[1].typ)
        typ = args[1].typ
      end
    elseif isdefined(fun.value, fun.name)
        args_typ = map(typeOfOpr, args)
        dprintln(env,"function to offload: ", fun, " methods=", methods(getfield(fun.value, fun.name)))
        _offload(getfield(fun.value, fun.name), tuple(args_typ...))
    else
        dprintln(env,"function call not translated: ", fun, ", and is not found!")
    end
  else
    dprintln(env,"function call is not GetfieldNode and not translated: ", fun, ", return typ=", typ)
  end
  if isa(expr, Nothing)
    if !is(fun, :ccall)
      oldargs = normalize_args(state, env_, oldargs)
    end
    expr = mk_expr(typ, head, oldfun, oldargs...)
  end
  return expr
end

function from_return(state, env, expr)
  local env_ = nextEnv(env)
  local head = expr.head
  local typ  = expr.typ
  local args = normalize_args(state, env, expr.args)
  # fix return type, the assumption is there is either 0 or 1 args.
  typ = length(args) > 0 ? typeOfOpr(args[1]) : Void
  return mk_expr(typ, head, args...)
end

function from_expr(function_name, cur_module::Module, ast)
  dprintln(2,"DomainIR translation function = ", function_name, " on:")
  dprintln(2,ast)
  ast = from_expr(emptyState(), newEnv(cur_module), ast)
  dprintln(2,"DomainIR translation returns:")
  dprintln(2,ast)
  return ast
end

function from_expr(state, env, ast)
  if isa(ast, LambdaStaticData)
    dprintln(env, "from_expr: LambdaStaticData inferred = ", ast.inferred)
    if !ast.inferred
      # we return this unmodified since we want the caller to
      # type check it before conversion.
      return ast
    else
      ast = uncompressed_ast(ast)
      # (tree, ty)=Base.typeinf(ast, argstyp, ())
    end
  end
  local asttyp = typeof(ast)
  dprint(env,"from_expr: ", asttyp)
  if is(asttyp, Expr)
    local head = ast.head
    local args = ast.args
    local typ  = ast.typ
    dprintln(2, " :", head)
    if is(head, :lambda)
        return from_lambda(state, env, ast)
    elseif is(head, :body)
        return from_body(state, env, ast)
    elseif is(head, :(=))
        return from_assignment(state, env, ast)
    elseif is(head, :return)
        return from_return(state, env, ast)
    elseif is(head, :call)
        return from_call(state, env, ast)
        # TODO: catch domain IR result here
    elseif is(head, :call1)
        return from_call(state, env, ast)
        # TODO?: tuple
    elseif is(head, :method)
        # change it to assignment
        ast.head = :(=)
        n = length(args)
        ast.args = Array(Any, n-1)
        ast.args[1] = args[1]
        for i = 3:n
          ast.args[i-1] = args[i]
        end
        return from_assignment(state, env, ast)
    elseif is(head, :line)
        # skip
    elseif is(head, :new)
        # skip?
    elseif is(head, :boundscheck)
        # skip?
    elseif is(head, :gotoifnot)
        # ?
    else
        throw(string("from_expr: unknown Expr head :", head))
    end
  elseif isa(ast, SymbolNode) || isa(ast, Symbol)
    name = isa(ast, SymbolNode) ? ast.name : ast
    # if it is global const, we replace it with its const value
    def = lookupDefInAllScopes(state, name)
    if is(def, nothing) && isdefined(env.cur_module, name) && ccall(:jl_is_const, Int32, (Any, Any), env.cur_module, name) == 1
      def = getfield(env.cur_module, name)
      if isbits(def)
        return def
      end
    end
    dprintln(2, " not handled ", ast)
  else 
    dprintln(2, " not handled ", ast)
  end
  return ast
end

type DirWalk
  callback
  cbdata
end

function get_one(ast)
  assert(isa(ast,Array))
  assert(length(ast) == 1)
  ast[1]
end

function AstWalkCallback(x, dw::DirWalk, top_level_number, is_top_level, read)
  dprintln(3,"DomainIR.AstWalkCallback ", x)
  ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
  dprintln(3,"DomainIR.AstWalkCallback ret = ", ret)
  if ret != nothing
    return [ret]
  end

  local asttyp = typeof(x)
  if asttyp == Expr
    local head = x.head
    local args = x.args
    local typ  = x.typ
    if head == :mmap || head == :mmap!
      assert(length(args) >= 2)
      input_arrays = args[1]
      for i = 1:length(input_arrays)
        args[1][i] = get_one(AstWalker.AstWalk(input_arrays[i], AstWalkCallback, dw))
      end
      args[2] = get_one(AstWalker.AstWalk(args[2], AstWalkCallback, dw))
      return x
    elseif head == :reduce
      assert(length(args) == 3)
      for i = 1:3
        args[i] = get_one(AstWalker.AstWalk(args[i], AstWalkCallback, dw))
      end
      return x
    elseif head == :select 
      # it is always in the form of select(arr, mask), where range can itself be ranges(range(...), ...))
      assert(length(args) == 2)
      args[1] = get_one(AstWalker.AstWalk(args[1], AstWalkCallback, dw))
      assert(isa(args[2], Expr))
      if args[2].head == :ranges
        ranges = args[2].args
      elseif args[2].head == :range || args[2].head == :tomask
        ranges = Any[ args[2] ]
      else
        error("Unsupprted range object in select: ", args[2])
      end
      for i = 1:length(ranges)
        # dprintln(3, "ranges[i] = ", ranges[i], " ", typeof(ranges[i]))
        assert(isa(ranges[i], Expr) && (ranges[i].head == :range || ranges[i].head == :tomask))
        for j = 1:length(ranges[i].args)
          ranges[i].args[j] = get_one(AstWalker.AstWalk(ranges[i].args[j], AstWalkCallback, dw))
        end
      end
      return x
    elseif head == :stencil!
      assert(length(args) == 4)
      args[2] = get_one(AstWalker.AstWalk(args[2], AstWalkCallback, dw))
      for i in 1:length(args[3]) # buffer array
        args[3][i] = get_one(AstWalker.AstWalk(args[3][i], AstWalkCallback, dw))
      end
      return x
    elseif head == :assertEqShape
      assert(length(args) == 2)
      for i = 1:length(args)
        args[i] = get_one(AstWalker.AstWalk(args[i], AstWalkCallback, dw))
      end
      return x
    elseif head == :assert
      for i = 1:length(args)
        AstWalker.AstWalk(args[i], AstWalkCallback, dw)
      end
      return x
    end
    x = Expr(head, args...)
    x.typ = typ
  elseif asttyp == DomainLambda
    dprintln(3,"DomainIR.AstWalkCallback for DomainLambda", x)
    return x
  end
  return nothing
end

function AstWalk(ast::Any, callback, cbdata)
  dprintln(3,"DomainIR.AstWalk ", ast)
  dw = DirWalk(callback, cbdata)
  AstWalker.AstWalk(ast, AstWalkCallback, dw)
end

function dir_live_cb(ast, cbdata)
  dprintln(4,"dir_live_cb ")
  asttyp = typeof(ast)
  if asttyp == Expr
    head = ast.head
    args = ast.args
    if head == :mmap
      expr_to_process = Any[]

      assert(length(args) == 2)
      input_arrays = args[1]
      for i = 1:length(input_arrays)
        push!(expr_to_process, input_arrays[i])
      end
      assert(isa(args[2], DomainLambda))
      escapes = args[2].escapes
      for (v, d) in escapes
        push!(expr_to_process, v)
      end 

      dprintln(3, ":mmap ", expr_to_process)
      return expr_to_process
    elseif head == :mmap!
      expr_to_process = Any[]

      assert(length(args) >= 2)
      input_arrays = args[1]
      for i = 1:length(input_arrays)
        # Need to make input_arrays[1] written?
        push!(expr_to_process, input_arrays[i])
      end
      assert(isa(args[2], DomainLambda))
      escapes = args[2].escapes
      for (v, d) in escapes
        push!(expr_to_process, v)
      end 

      dprintln(3, ":mmap! ", expr_to_process)
      return expr_to_process
    elseif head == :reduce
      expr_to_process = Any[]

      assert(length(args) == 3)
      zero_val = args[1]
      input_array = args[2]
      dl = args[3]
      push!(expr_to_process, zero_val)
      push!(expr_to_process, input_array)
      assert(isa(dl, DomainLambda))
      escapes = dl.escapes
      for (v, d) in escapes
        push!(expr_to_process, v)
      end

      dprintln(3, ":reduce ", expr_to_process)
      return expr_to_process
    elseif head == :stencil!
      expr_to_process = Any[]

      sbufs = args[3]
      for i = 1:length(sbufs)
        push!(expr_to_process, sbufs[i])
      end

      dl = args[4]
      assert(isa(dl, DomainLambda))
      escapes = dl.escapes
      for (v, d) in escapes
        push!(expr_to_process, v)
      end

      dprintln(3, ":stencil! ", expr_to_process)
      return expr_to_process
    elseif head == :assertEqShape
      assert(length(args) == 2)
      #dprintln(3,"liveness: assertEqShape ", args[1], " ", args[2], " ", typeof(args[1]), " ", typeof(args[2]))
      expr_to_process = Any[]
      push!(expr_to_process, args[1].name)
      push!(expr_to_process, args[2].name)
      return expr_to_process
    end
  elseif asttyp == KernelStat
    return Any[]
  end
  return nothing
end

end
