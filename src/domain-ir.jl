#=
Copyright (c) 2015, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.^, .

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

module DomainIR

import CompilerTools.DebugMsg
DebugMsg.init()

import CompilerTools.AstWalker
using CompilerTools
using CompilerTools.LivenessAnalysis
using CompilerTools.LambdaHandling
using Core.Inference: to_tuple_type
using Base.uncompressed_ast
using CompilerTools.AliasAnalysis

# uncomment this line when using Debug.jl
#using Debug

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


function TypedExpr(typ, rest...)
    res = Expr(rest...)
    res.typ = typ
    res
end

mk_eltype(arr) = Expr(:eltype, arr)
mk_ndim(arr) = Expr(:ndim, arr)
mk_length(arr) = Expr(:length, arr)
#mk_arraysize(arr, d) = Expr(:arraysize, arr, d)
mk_arraysize(arr, dim) = TypedExpr(Int64, :call, TopNode(:arraysize), arr, dim)
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
mk_parallel_for(loopvars, ranges, f) = Expr(:parallel_for, loopvars, ranges, f)

function mk_expr(typ, args...)
    e = Expr(args...)
    e.typ = typ
    return e
end

function type_expr(typ, expr)
    expr.typ = typ
    return expr
end

export DomainLambda, KernelStat, AstWalk, arraySwap, lambdaSwapArg, isarray

# A representation for anonymous lambda used in domainIR.
#   inputs:  types of input tuple
#   outputs: types of output tuple
#   genBody: (LambdaInfo, Array{Any,1}) -> Array{Expr, 1}
#   escapes: escaping variables in the body
#
# So the downstream can just call genBody, and pass it
# the downstream's LambdaInfo and an array of parameters,
# and it will return an expression (with :body head) that 
# represents the loop body. The input LambdaInfo, if
# given, is updated inplace. 
#
# genBody always returns an array of expression even when 
# there is only one. The last expression is always of the 
# form:
#   (:tuple, values...)
# where there could be either 0 or multiple values being
# returned.
#
# Note that DomainLambda only supports Julia expressions
# and domain IR expressions, but not custom IR nodes.

type DomainLambda
    inputs  :: Array{Type, 1}
    outputs :: Array{Type, 1}
    genBody :: Function
    linfo   :: LambdaInfo

    function DomainLambda(i, o, gb, li)
        # TODO: is the following necessary?
        licopy = deepcopy(li) 
        new(i, o, gb, licopy)
    end
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
    (linfo, args) -> f.genBody(linfo, arraySwap(args, i, j)),
    f.linfo)
end

type IRState
    linfo  :: LambdaInfo
    defs   :: Dict{Union{Symbol,Int}, Any}  # stores local definition of LHS = RHS
    stmts  :: Array{Any, 1}
    parent :: Union{Void, IRState}
end

emptyState() = IRState(LambdaInfo(), Dict{Union{Symbol,Int},Any}(), Any[], nothing)
newState(linfo, defs, state::IRState)=IRState(linfo, defs, Any[], state)

@doc """
Update the definition of a variable.
"""
function updateDef(state::IRState, s::SymAllGen, rhs)
    s = isa(s, SymbolNode) ? s.name : s
    #dprintln(3, "updateDef: s = ", s, " rhs = ", rhs, " typeof s = ", typeof(s))
    @assert ((isa(s, GenSym) && isLocalGenSym(s, state.linfo)) ||
    (isa(s, Symbol) && isLocalVariable(s, state.linfo)) ||
    (isa(s, Symbol) && isInputParameter(s, state.linfo))) state.linfo
    s = isa(s, GenSym) ? s.id : s
    state.defs[s] = rhs
end

@doc """
Look up a definition of a variable.
Return nothing If none is found.
"""
function lookupDef(state::IRState, s::SymAllGen)
    s = isa(s, SymbolNode) ? s.name : (isa(s, GenSym) ? s.id : s)
    get(state.defs, s, nothing)
end

@doc """
Look up a definition of a variable only when it is const or assigned once.
Return nothing If none is found.
"""
function lookupConstDef(state::IRState, s::SymAllGen)
    def = lookupDef(state, s)
    # we assume all GenSym is assigned once 
    desc = isa(s, SymbolNode) ? getDesc(s.name, state.linfo) : (ISASSIGNEDONCE | ISASSIGNED)
    if !is(def, nothing) && (desc & (ISASSIGNEDONCE | ISCONST)) != 0
        return def
    end
    return nothing
end

@doc """
Look up a definition of a variable recursively until the RHS is no-longer just a variable.
Return the last rhs If found, or the input variable itself otherwise.
"""
function lookupConstDefForArg(state::IRState, s::Any)
    s1 = s
    while isa(s, Symbol) || isa(s, SymbolNode) || isa(s, GenSym)
        s1 = s
        s = lookupConstDef(state, s1)
    end
    is(s, nothing) ? s1 : s
end

@doc """
Look up a definition of a variable throughout nested states until a definition is found.
Return nothing If none is found.
"""
function lookupDefInAllScopes(state::IRState, s::SymAllGen)
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
    cur_module  :: Union{Module, Void}
    debugLevel  :: Int
    debugIndent :: Int
end

newEnv(m)=IREnv(m,2,0)
nextEnv(env::IREnv)=IREnv(env.cur_module, env.debugLevel, env.debugIndent + 1)

@inline function dprint(env::IREnv,msgs...)
  dprint(env.debugLevel, repeat(" ", env.debugIndent*2), msgs...)
end

@inline function dprintln(env::IREnv,msgs...)
  dprintln(env.debugLevel, repeat(" ", env.debugIndent*2), msgs...)
end

import ..API

const mapSym = vcat(Symbol[:negate], API.unary_map_operators, API.binary_map_operators)

const mapVal = Symbol[ begin s = string(x); startswith(s, '.') ? symbol(s[2:end]) : x end for x in mapSym]

# * / are not point wise. it becomes point wise only when one argument is scalar.
const pointWiseOps = setdiff(Set{Symbol}(mapSym), Set{Symbol}([:*, :/]))

const mapOps = Dict{Symbol,Symbol}(zip(mapSym, mapVal))
# symbols that when lifted up to array level should be changed.
const liftOps = Dict{Symbol,Symbol}(zip(Symbol[:<=, :>=, :<, :(==), :>, :+,:-,:*,:/], Symbol[:.<=, :.>=, :.<, :.==, :.>, :.+, :.-, :.*, :./]))

const topOpsTypeFix = Set{Symbol}([:not_int, :and_int, :or_int, :neg_int, :add_int, :mul_int, :sub_int, :neg_float, :mul_float, :add_float, :sub_float, :div_float, :box, :fptrunc, :fpsiround, :checked_sadd, :checked_ssub, :rint_llvm, :floor_llvm, :ceil_llvm, :abs_float, :cat_t, :srem_int])

const opsSym = Symbol[:negate, :+, :-, :*, :/, :(==), :!=, :<, :<=]
const opsSymSet = Set{Symbol}(opsSym)
const floatOps = Dict{Symbol,Symbol}(zip(opsSym, [:neg_float, :add_float, :sub_float, :mul_float, :div_float,
:eq_float, :ne_float, :lt_float, :le_float]))
const sintOps  = Dict{Symbol,Symbol}(zip(opsSym, [:neg_int, :add_int, :sub_int, :mul_int, :sdiv_int,
:eq_int, :ne_int, :slt_int, :sle_int]))

const reduceSym = Symbol[:sum, :prod, :maximum, :minimum, :any, :all]
const reduceVal = Symbol[:+, :*, :max, :min, :|, :&]
const reduceFun = Function[zero, one, typemin, typemax, x->false, x->true]
const reduceOps = Dict{Symbol,Symbol}(zip(reduceSym,reduceVal))
const reduceNeutrals = Dict{Symbol,Function}(zip(reduceSym,reduceFun))

const ignoreSym = Symbol[:box]
const ignoreSet = Set{Symbol}(ignoreSym)

const allocCalls = Set{Symbol}([:jl_alloc_array_1d, :jl_alloc_array_2d, :jl_alloc_array_3d, :jl_new_array])

# some part of the code still requires this
unique_id = 0
function addFreshLocalVariable(s::AbstractString, t::Any, desc, linfo::LambdaInfo)
    global unique_id
    name = :tmpvar
    unique = false
    while (!unique)
        unique_id = unique_id + 1
        name = symbol(string(s, "##", unique_id))
        unique = !isLocalVariable(name, linfo)
    end
    addLocalVariable(name, t, desc, linfo)
    return SymbolNode(name, t)
end

include("domain-ir-stencil.jl")

function isinttyp(typ)
    is(typ, Int64)  || is(typ, Int32)  || is(typ, Int16)  || is(typ, Int8)  || 
    is(typ, UInt64) || is(typ, UInt32) || is(typ, UInt16) || is(typ, UInt8)
end

function istupletyp(typ)
    isa(typ, DataType) && is(typ.name, Tuple.name)
end

function isarray(typ)
    isa(typ, DataType) && is(typ.name, Array.name)
end

function isbitarray(typ)
    (isa(typ, DataType) && is(typ.name, BitArray.name)) ||
    (isarray(typ) && is(eltype(typ), Bool))
end

function isunitrange(typ)
    isa(typ, DataType) && is(typ.name, UnitRange.name)
end

function issteprange(typ)
    isa(typ, DataType) && is(typ.name, StepRange.name)
end

function isrange(typ)
    isunitrange(typ) || issteprange(typ)
end

function ismask(state, r::SymbolNode)
    return isrange(r.typ) || isbitarray(r.typ)
end

function ismask(state, r::SymGen)
    typ = getType(r, state.linfo)
    return isrange(typ) || isbitarray(typ)
end

function ismask(state, r::GlobalRef)
    return r.mod==Main && r.name==:(:)
end

function ismask(state, r::Any)
    typ = typeOfOpr(state, r)
    return isrange(typ) || isbitarray(typ)
end

function remove_typenode(expr)
    if isa(expr, Expr)
        if is(expr.head, :(::))
            return remove_typenode(expr.args[1])
        else
            args = Any[]
            for i = 1:length(expr.args)
                push!(args, remove_typenode(expr.args[i]))
            end
            return mk_expr(expr.typ, expr.head, args...)
        end
    end
    expr
end

function from_range(rhs)
    if isa(rhs, Expr) && is(rhs.head, :new) && isunitrange(rhs.args[1]) &&
        isa(rhs.args[3], Expr) && is(rhs.args[3].head, :call) &&
        isa(rhs.args[3].args[1], Expr) && is(rhs.args[3].args[1].head, :call) &&
        is(rhs.args[3].args[1].args[1], TopNode(:getfield)) &&
        is(rhs.args[3].args[1].args[2], GlobalRef(Base, :Intrinsics)) &&
        is(rhs.args[3].args[1].args[3], QuoteNode(:select_value))
        # only look at final value in select_value of UnitRange
        start = rhs.args[2]
        step  = 1 # FIXME: could be wrong here!
        final = rhs.args[3].args[3]
    elseif isa(rhs, Expr) && is(rhs.head, :new) && issteprange(rhs.args[1])
        assert(length(rhs.args) == 4)
        start = rhs.args[2]
        step  = rhs.args[3]
        final = rhs.args[4]
    else
        error("expect Expr(:new, UnitRange, ...) or Expr(:new, StepRange, ...) but got ", rhs)
    end
    return (start, step, final)
end

function rangeToMask(state, r::Int, arraysize)
    return mk_range(r, r, 1)
end

function rangeToMask(state, r::SymAllGen, arraysize)
    typ = getType(r, state.linfo)
    if isbitarray(typ)
        mk_tomask(r)
    elseif isunitrange(typ)
        r = lookupConstDefForArg(state, r)
        (start, step, final) = from_range(r)
        mk_range(start, step, final)
    elseif isinttyp(typ) 
        mk_range(r, convert(typ, 1), r)
    else
        error("Unhandled range object: ", r)
    end
end

function rangeToMask(state, r::GlobalRef, arraysize)
    # FIXME: why doesn't this assert work?
    #@assert (r.mod!=Main || r.name!=symbol(":")) "unhandled GlobalRef range"
    if r.mod==Main && r.name==symbol(":")
        return mk_range(1, 1, arraysize)
    else
        error("unhandled GlobalRef range object: ", r)
    end
end

function rangeToMask(state, r::ANY, arraysize)
    error("unhandled range object: ", r)
end

# check if a given function can be considered as a map operation.
# Some operations depends on types.
function verifyMapOps(state, fun :: Symbol, args :: Array{Any, 1})
    if !haskey(mapOps, fun)
        return false
    elseif in(fun, pointWiseOps)
        return true
    else
        # for non-pointwise operators, only one argument can be array, the rest must be scalar
        n = 0
        for i = 1:length(args)
            typ = typeOfOpr(state, args[i])
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
    if isa(opr, Symbol)
        opr = TopNode(opr)
    end
    return opr, reorder
end

import Base.show
import .._accelerate

function show(io::IO, f::DomainLambda)
    local len = length(f.inputs)
    local syms = [ symbol(string("x", i)) for i = 1:len ]
    local symNodes = [ SymbolNode(syms[i], f.inputs[i]) for i = 1:len ]
    local body = f.genBody(LambdaInfo(), syms) # use a dummy LambdaInfo for show purpose
    print(io, "(")
    show(io, symNodes)
    print(io, ";")
    show(io, f.linfo)
    print(io, ") -> (")
    show(io, body)
    print(io, ")::", f.outputs)
end

# Specialize non-array arguments into the body function as either constants
# or escaping variables. Arguments of array type are treated as parameters to
# the body function instead.
# It returns the list of arguments that are specialized into the body function,
# the remaining arguments (that are of array type), their types, and
# the specialized body function.
function specialize(state::IRState, args::Array{Any,1}, typs::Array{Type,1}, bodyf::Function)
    local j = 0
    local len = length(typs)
    local idx = Array(Int, len)
    local args_ = Array(Any, len)
    local nonarrays = Array(Any, 0)
    local repldict = Dict{SymGen, Any}()
    for i = 1:len
        local typ = typs[i]
        if isarray(typ) || isbitarray(typ)
            j = j + 1
            typs[j] = typ
            args_[j] = args[i]
            idx[j] = i
        else
            if isa(args[i], GenSym) # cannot put GenSym into lambda! Add a temp variable to do it
                tmpv = addFreshLocalVariable(string(args[i]), getType(args[i], state.linfo), ISASSIGNED | ISASSIGNEDONCE, state.linfo)
                emitStmt(state, mk_expr(tmpv.typ, :(=), tmpv.name, args[i]))
                repldict[args[i]] = tmpv
                args[i] = tmpv
            end
            push!(nonarrays, args[i])
        end
    end
    function mkFun(plinfo, params)
        local myArgs = copy(args)
        assert(length(params)==j)
        local i
        for i=1:j
            myArgs[idx[i]] = params[i]
        end
        replaceExprWithDict(bodyf(myArgs), repldict)
    end
    return (nonarrays, args_[1:j], typs[1:j], mkFun)
end

function typeOfOpr(state :: IRState, x)
    CompilerTools.LivenessAnalysis.typeOfOpr(x, state.linfo)
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

# :lambda expression
# (:lambda, {param, meta@{localvars, types, freevars}, body})
function from_lambda(state, env, expr::Expr)
    local env_ = nextEnv(env)
    local head = expr.head
    local ast  = expr.args
    local typ  = expr.typ
    assert(length(ast) == 3)
    linfo = lambdaExprToLambdaInfo(expr) 
    local body  = ast[3]
    assert(isa(body, Expr) && is(body.head, :body))
    local state_ = newState(linfo, Dict{Union{Symbol,Int},Any}(), state)
    body = from_expr(state_, env_, body)
    # fix return type
    typ = body.typ
    dprintln(env,"from_lambda: body=", body)
    dprintln(env,"from_lambda: linfo=", linfo)
    return lambdaInfoToLambdaExpr(linfo, body)
end

# sequence of expressions {expr, ...}
# unlike from_body, from_exprs do not emit the input expressions
# as statements to the state, while still allowing side effects
# of emitting statements durinfg the translation of these expressions.
function from_exprs(state::IRState, env::IREnv, ast::Array{Any,1})
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
    head = expr.head 
    @assert head==:mmap || head==:mmap! "Input to mmapRemoveDupArg!() must be :mmap or :mmap!"
    arr = expr.args[1]
    f = expr.args[2]
    posMap = Dict{SymGen, Int}()
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
        if haskey(posMap, s)
            hasDup = true
            indices[i] = posMap[s]
        else
            if isa(s,SymAllGen) 
                posMap[s] = n
            end
            push!(newarr, arr[i])
            push!(newinp, f.inputs[i])
            n += 1
        end
    end
    if (!hasDup) return expr end
    dprintln(3, "MMRD: expr was ", expr)
    dprintln(3, "MMRD:  ", newarr, newinp, indices)
    expr.args[1] = newarr
    expr.args[2] = DomainLambda(newinp, f.outputs,
    (linfo, args) -> begin
        dupargs = Array(Any, oldn)
        for i=1:oldn
            dupargs[i] = args[indices[i]]
        end
        f.genBody(linfo, dupargs)
    end, f.linfo)
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
    assert(isa(lhs, Symbol) || isa(lhs, GenSym))
    rhs = from_expr(state, env_, rhs)
    dprintln(env, "from_assignment lhs=", lhs, " typ=", typ)
    # turn x = mmap((x,...), f) into x = mmap!((x,...), f)
    if isa(rhs, Expr) && is(rhs.head, :mmap) && length(rhs.args[1]) > 0 &&
        (is(lhs, rhs.args[1][1]) || (isa(rhs.args[1][1], SymbolNode) && is(lhs, rhs.args[1][1].name)))
        rhs.head = :mmap!
        # NOTE that we keep LHS to avoid a bug (see issue #...)
        typ = getType(lhs, state.linfo)
        lhs = addGenSym(typ, state.linfo) 
    end
    updateDef(state, lhs, rhs)
    # TODO: handle indirections like x = y so that x gets y's definition instead of just y.
    return mk_expr(typ, head, lhs, rhs)
end

function from_call(state::IRState, env::IREnv, expr::Expr)
    local env_ = nextEnv(env)
    local head = expr.head
    local ast = expr.args
    local typ = expr.typ
    @assert length(ast) >= 1 "call args cannot be empty"
    local fun  = ast[1]
    local args = ast[2:end]
    dprintln(env,"from_call: fun=", fun, " typeof(fun)=", typeof(fun), " args=",args, " typ=", typ)
    fun = from_expr(state, env_, fun)
    dprintln(env,"from_call: new fun=", fun)
    (fun_, args_) = normalize_callname(state, env, fun, args)
    result = translate_call(state, env, typ, :call, fun, args, fun_, args_)
    result
end

# turn Exprs in args into variable names, and put their definition into state
# anything of void type in the argument is omitted in return value.
function normalize_args(state::IRState, env::IREnv, args::Array{Any,1})
    in_args::Array{Any,1} = from_exprs(state, env, args)
    local out_args = Array(Any,length(in_args))
    j = 0
    for i = 1:length(in_args)
        local arg = in_args[i]
        if isa(arg, Expr) && arg.typ == Void
            # do not produce new assignment for Void values
            dprintln(3, "normalize_args got Void args[", i, "] = ", arg)
            emitStmt(state, arg)
        elseif isa(arg, Expr) || isa(arg, LambdaStaticData)
            typ = isa(arg, Expr) ? arg.typ : Any
            newVar = addGenSym(typ, state.linfo)
            # set flag [is assigned once][is const][is assigned by inner function][is assigned][is captured]
            updateDef(state, newVar, arg)
            emitStmt(state, mk_expr(typ, :(=), newVar, arg))
            j = j + 1
            out_args[j] = newVar
        else
            j = j + 1
            out_args[j] = in_args[i]
        end
    end
    return out_args[1:j]
end

# Fix Julia inconsistencies in call before we pattern match
function normalize_callname(state::IRState, env, fun::GlobalRef, args)
    return normalize_callname(state, env, fun.name, args)
end

function normalize_callname(state::IRState, env, fun::Symbol, args)
    if is(fun, :broadcast!)
        dst = lookupConstDefForArg(state, args[2])
        if isa(dst, Expr) && is(dst.head, :call) && isa(dst.args[1], TopNode) &&
            is(dst.args[1].name, :ccall) && isa(dst.args[2], QuoteNode) &&
            is(dst.args[2].value, :jl_new_array)
            # now we are sure destination array is new
            fun   = args[1]
            args  = args[3:end]
            if isa(fun, GlobalRef)
                fun = fun.name
            end
            if isa(fun, Symbol)
            elseif isa(fun, SymbolNode)
                fun = lookupConstDef(state, fun.name)
            else
                error("DomainIR: cannot handle broadcast! with function ", fun)
            end
        elseif isa(dst, Expr) && is(dst.head, :call) && isa(dst.args[1], DataType) &&
            is(dst.args[1].name, BitArray.name) 
            # destination array is a new bitarray
            fun   = args[1]
            args  = args[3:end]
            if isa(fun, Symbol) || isa(fun, GenSym) || isa(fun, SymbolNode)
                # fun could be a variable 
                fun = get(state.defs, fun, nothing)
            end
            if isa(fun, GlobalRef)
                func = eval(fun)  # should give back a function
                assert(isa(func, Function))
                fun = fun.name
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
    return (fun, args)
end

function normalize_callname(state::IRState, env, fun::TopNode, args)
    fun = fun.name
    if is(fun, :ccall)
        callee = lookupConstDefForArg(state, args[1])
        if isa(callee, QuoteNode) && in(callee.value, allocCalls)
            local realArgs = Any[]
            atype = args[4]
            elemtyp = elmTypOf(atype)
            push!(realArgs, elemtyp)
            for i = 6:2:length(args)
                if istupletyp(typeOfOpr(state, args[i]))
                    def = lookupConstDefForArg(state, args[i])
                    if isa(def, Expr) && is(def.head, :call) && is(def.args[1], TopNode(:tuple))
                        for j = 1:length(def.args) - 1
                            push!(realArgs, def.args[j + 1])
                        end
                        break
                    else
                        return (fun, args)
                    end
                end
                push!(realArgs, args[i])
            end
            fun  = :alloc
            args = realArgs
        end
    end
    return (fun, args)
end

function normalize_callname(state::IRState, env, fun :: ANY, args)
    return (fun, args)
end

# if a definition of arr is getindex(a, ...), return select(a, ranges(...))
# otherwise return arr unchanged.
function inline_select(env, state, arr)
    range_extra = Any[]
    if isa(arr, SymbolNode) 
        # TODO: this requires safety check. Local lookups are only correct if free variables in the definition have not changed.
        def = lookupConstDef(state, arr.name)
        if !isa(def, Void)  
            if isa(def, Expr) && is(def.head, :call) 
                target_arr = arr
                if is(def.args[1], :getindex) || (isa(def.args[1], GlobalRef) && is(def.args[1].name, :getindex))
                    target_arr = def.args[2]
                    range_extra = def.args[3:end]
                elseif def.args[1] == TopNode(:_getindex!) # getindex gets desugared!
                    error("we cannot handle TopNode(_getindex!) because it is effectful and hence will persist until J2C time")
                end
                dprintln(env, "inline-select: target_arr = ", target_arr, " range = ", range_extra)
                if length(range_extra) > 0
                    # if all ranges are int, then it is not a selection
                    if any(Bool[ismask(state,r) for r in range_extra])
                        ranges = mk_ranges([rangeToMask(state, range_extra[i], mk_arraysize(arr, i)) for i in 1:length(range_extra)]...)
                      dprintln(env, "inline-select: converted to ranges = ", ranges)
                      arr = mk_select(target_arr, ranges)
                    else
                      dprintln(env, "inline-select: skipped")
                    end
                end
            end
        end
    end
    return arr
end

# translate a function call to domain IR if it matches
function translate_call(state, env, typ, head, oldfun, oldargs, fun::Symbol, args::Array{Any,1})
    local env_ = nextEnv(env)
    expr = nothing
    dprintln(env, "translate_call fun=", fun, "::", typeof(fun), " args=", args, " typ=", typ)
    # new mainline Julia puts functions in Main module but PSE expects the symbol only
    #if isa(fun, GlobalRef) && fun.mod == Main
    #   fun = fun.name
    # end

    dprintln(env, "verifyMapOps -> ", verifyMapOps(state, fun, args))
    if verifyMapOps(state, fun, args) && (isarray(typ) || isbitarray(typ)) 
        return translate_call_map(state,env_,typ, fun, args)
    elseif is(fun, :cartesianarray)
        return translate_call_cartesianarray(state,env_,typ, args)
    elseif is(fun, :runStencil)
        return translate_call_runstencil(state,env_,args)
    elseif is(fun, :parallel_for)
        return translate_call_parallel_for(state,env_,args)
    elseif is(fun, :copy)
        args = normalize_args(state, env_, args[1:1])
        dprintln(env,"got copy, args=", args)
        expr = mk_copy(args[1])
        expr.typ = typ
    elseif in(fun, topOpsTypeFix) && is(typ, Any) && length(args) > 0
        typ = translate_call_typefix(state, env, typ, fun, args) 
    elseif is(fun, :getfield) && length(args) == 2 && isa(args[1], GenSym) && 
        (args[2] == QuoteNode(:stop) || args[2] == QuoteNode(:start) || args[2] == QuoteNode(:step))
        # Shortcut range object access
        rTyp = getType(args[1], state.linfo)
        rExpr = lookupConstDefForArg(state, args[1])
        if isrange(rTyp) && isa(rExpr, Expr)
            (start, step, final) = from_range(rExpr)
            fname = args[2].value
            if is(fname, :stop) 
                return final
            elseif is(fname, :start)
                return start
            else
                return step
            end
        end
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
            elemTyp = typExp
        elseif isa(typExp, GlobalRef)
            elemTyp = eval(typExp)
        else
            error("Expect QuoteNode or DataType, but got typExp = ", typExp)
        end
        assert(isa(elemTyp, DataType))
        expr = mk_alloc(elemTyp, args)
        expr.typ = typ
    elseif is(fun, :broadcast_shape)
        dprintln(env, "got ", fun)
        args = normalize_args(state, env_, args)
        expr = mk_expr(typ, :assertEqShape, args...)
    elseif is(fun, :checkbounds)
        dprintln(env, "got ", fun, " args = ", args)
        if length(args) == 2
            return translate_call_checkbounds(state,env_,args) 
        end
    elseif is(fun, :sitofp) || is(fun, :fpext) # typefix hack!
        typ = isa(args[1], Type) ? args[1] : eval(args[1]) 
    elseif is(fun, :getindex) || is(fun, :setindex!) 
        expr = translate_call_getsetindex(state,env_,typ,fun,args)
    elseif is(fun, :assign_bool_scalar_1d!) || # args = (array, scalar_value, bitarray)
        is(fun, :assign_bool_vector_1d!)    # args = (array, getindex_bool_1d(array, bitarray), bitarray)
        return translate_call_assign_bool(state,env_,typ,fun, args) 
    elseif is(fun, :fill!)
        return translate_call_fill!(state, env_, typ, args)
    elseif is(fun, :_getindex!) # see if we can turn getindex! back into getindex
        if isa(args[1], Expr) && args[1].head == :call && args[1].args[1] == TopNode(:ccall) && 
            (args[1].args[2] == :jl_new_array ||
            (isa(args[1].args[2], QuoteNode) && args[1].args[2].value == :jl_new_array))
            expr = mk_expr(typ, :call, :getindex, args[2:end]...)
        end
    elseif haskey(reduceOps, fun)
        args = normalize_args(state, env_, args)
        if length(args)==1
            return translate_call_reduce(state, env_, typ, fun, args)
        end
    elseif in(fun, ignoreSet)
    else
        args_typ = map(x -> typeOfOpr(state, x), args)
        dprintln(3,"args = ", args, " args_typ = ", args_typ)
        if !is(env.cur_module, nothing) && isdefined(env.cur_module, fun) && !isdefined(Base, fun) # only handle functions in Main module
            dprintln(env,"function to offload: ", fun, " methods=", methods(getfield(env.cur_module, fun)))
            _accelerate(getfield(env.cur_module, fun), tuple(args_typ...))
            oldfun = GlobalRef(env.cur_module, fun)
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
                return GlobalRef(m, args[2].value) 
            end
        else
            dprintln(env,"function call not translated: ", fun, ", typeof(fun)=", typeof(fun), " head = ", head, " oldfun = ", oldfun, ", args typ=", args_typ)
        end
    end

    if isa(expr, Void)
        if !is(fun, :ccall)
            if is(fun, :box) && isa(oldargs[2], Expr) # fix the type of arg[2] to be arg[1]
              oldargs[2].typ = typ
            end
            oldargs = normalize_args(state, env_, oldargs)
        end
        expr = mk_expr(typ, head, oldfun, oldargs...)
    end
    return expr
end

function translate_call_typefix(state, env, typ, fun, args::Array{Any,1})
    dprintln(env, " args = ", args, " type(args[1]) = ", typeof(args[1]))
    local typ1    
    if is(fun, :cat_t)
        typ1 = isa(args[2], GlobalRef) ? eval(args[2]) : args[2]
        @assert (isa(typ1, DataType)) "expect second argument to cat_t to be a type"
        dim1 = args[1]
        @assert (isa(dim1, Int)) "expect first argument to cat_t to be constant"
        typ1 = Array{typ1, dim1}
    else
        a1 = args[1]
        if typeof(a1) == GlobalRef
            a1 = eval(a1)
        end
        typ1 = typeOfOpr(state, a1)
        if is(fun, :fptrunc)
            if     is(a1, Float32) typ1 = Float32
            elseif is(a1, Float64) typ1 = Float64
            else throw(string("unknown target type for fptrunc: ", typ1, " args[1] = ", args[1]))
            end
        elseif is(fun, :fpsiround)
            if     is(a1, Float32) typ1 = Int32
            elseif is(a1, Float64) typ1 = Int64
                #        if is(typ1, Float32) typ1 = Int32
                #        elseif is(typ1, Float64) typ1 = Int64
            else throw(string("unknown target type for fpsiround: ", typ1, " args[1] = ", args[1]))
            end
        elseif searchindex(string(fun), "_int") > 0
            typ1 = Int
        end
    end
    dprintln(env,"fix type ", typ, " => ", typ1)
    return typ1
end

function translate_call_getsetindex(state, env, typ, fun, args::Array{Any,1})
    dprintln(env, "got getindex or setindex!")
    args = normalize_args(state, env, args)
    arr = args[1]
    arrTyp = getType(arr, state.linfo)
    dprintln(env, "arrTyp = ", arrTyp)
    if isrange(arrTyp) && length(args) == 2
        # shortcut range indexing if we can
        rExpr = lookupConstDefForArg(state, args[1])
        if isa(rExpr, Expr) && typ == Int 
            # only handle range of Int type 
            (start, step, final) = from_range(rExpr)
            return mk_expr(typ, :call, :add_int, start, mk_expr(typ, :call, :mul_int, mk_expr(typ, :call, :sub_int, args[2], 1), step))
        end
    elseif isarray(arrTyp) || isbitarray(arrTyp)
        ranges = is(fun, :getindex) ? args[2:end] : args[3:end]
        atyp = typeOfOpr(state, arr)
        expr = nothing
        dprintln(env, "ranges = ", ranges)
        if any(Bool[ ismask(state, range) for range in ranges])
            dprintln(env, "args is ", args)
            dprintln(env, "ranges is ", ranges)
            #newsize = addGenSym(Int, state.linfo)
            #newlhs = addGenSym(typ, state.linfo)
            etyp = elmTypOf(atyp)
            ranges = mk_ranges([rangeToMask(state, ranges[i], mk_arraysize(arr, i)) for i in 1:length(ranges)]...)
            if is(fun, :getindex) 
                expr = mk_mmap([mk_select(arr, ranges)], DomainLambda(Type[etyp], Type[etyp], (linfo, as) -> [Expr(:tuple, as...)], LambdaInfo())) 
            else
                args = Any[mk_select(arr, ranges), args[2]]
                typs = Type[atyp, typeOfOpr(state, args[2])]
                (nonarrays, args, typs, f) = specialize(state, args, typs, as -> [Expr(:tuple, as[2])])
                elmtyps = Type[ (isarray(t) || isbitarray(t)) ? elmTypOf(t) : t for t in typs ]
                linfo = LambdaInfo()
                for i=1:length(nonarrays)
                    # At this point, they are either symbol nodes, or constants
                    if isa(nonarrays[i], SymbolNode)
                        addEscapingVariable(nonarrays[i].name, nonarrays[i].typ, 0, linfo)
                    end
                end
                expr = mk_mmap!(args, DomainLambda(elmtyps, Type[etyp], f, linfo))
            end
            expr.typ = typ
        end
        return expr
    end
    return mk_expr(typ, :call, fun, args...)
end

function translate_call_map(state, env, typ, fun, args::Array{Any,1})
    # TODO: check for unboxed array type
    args = normalize_args(state, env, args)
    etyp = elmTypOf(typ) 
    if is(fun, :-) && length(args) == 1
        fun = :negate
    end
    typs = Type[ typeOfOpr(state, arg) for arg in args ]
    elmtyps = Type[ (isarray(t) || isbitarray(t)) ? elmTypOf(t) : t for t in typs ]
    opr, reorder = specializeOp(mapOps[fun], elmtyps)
    typs = reorder(typs)
    args = reorder(args)
    dprintln(env,"from_lambda: before specialize, opr=", opr, " args=", args, " typs=", typs)
    (nonarrays, args, typs, f) = specialize(state, args, typs, 
    as -> [Expr(:tuple, mk_expr(etyp, :call, opr, as...))])
    dprintln(env,"from_lambda: after specialize, typs=", typs)
    elmtyps = Type[ (isarray(t) || isbitarray(t)) ? elmTypOf(t) : t for t in typs ]
    # calculate escaping variables
    linfo = LambdaInfo()
    for i=1:length(nonarrays)
        # At this point, they are either symbol nodes, or constants
        if isa(nonarrays[i], SymbolNode)
            addEscapingVariable(nonarrays[i].name, nonarrays[i].typ, 0, linfo)
        end
    end
    domF = DomainLambda(elmtyps, [etyp], f, linfo)
    for i = 1:length(args)
        arg_ = inline_select(env, state, args[i])
        if arg_ != args[i] && i != 1 && length(args) > 1
            error("Selector array must be the only array argument to mmap: ", args)
        end
        args[i] = arg_
    end
    expr = mmapRemoveDupArg!(mk_mmap(args, domF))
    expr.typ = typ
    return expr
end

function translate_call_checkbounds(state, env, args::Array{Any,1})
    args = normalize_args(state, env, args)
    typ = typeOfOpr(state, args[1])
    local expr::Expr
    if isarray(typ)
        typ_second_arg = typeOfOpr(state, args[2])
        if isarray(typ_second_arg) || isbitarray(typ_second_arg)
            expr = mk_expr(Bool, :assert, mk_expr(Bool, :call, GlobalRef(Base, :(===)), mk_expr(Int64, :call, GlobalRef(Base,:arraylen), args[1]), mk_expr(Int64, :call, GlobalRef(Base,:arraylen), args[2])))
        else
            dprintln(0, args[2], " typ_second_arg = ", typ_second_arg)
            error("Unhandled bound in checkbounds: ", args[2])
        end
    elseif isinttyp(typ)
        if isinttyp(typeOfOpr(state, args[2]))
            expr = mk_expr(Bool, :assert, mk_expr(Bool, :call, TopNode(:sle_int), convert(typ, 1), args[2]),
            mk_expr(Bool, :call, TopNode(:sle_int), args[2], args[1]))
        elseif isa(args[2], SymbolNode) && (isunitrange(args[2].typ) || issteprange(args[2].typ))
            def = lookupConstDefForArg(state, args[2])
            (start, step, final) = from_range(def)
            expr = mk_expr(Bool, :assert, mk_expr(Bool, :call, TopNode(:sle_int), convert(typ, 1), start),
            mk_expr(Bool, :call, TopNode(:sle_int), final, args[1]))
        else
            error("Unhandled bound in checkbounds: ", args[2])
        end
    else
        error("Unhandled bound in checkbounds: ", args[1])
    end
    return expr
end

# this is legacy v0.3 call
function translate_call_assign_bool(state, env, typ, fun, args::Array{Any,1})
    etyp = elmTypOf(typ)
    args = normalize_args(state, env, args)
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
    typs = Type[typeOfOpr(state, a) for a in args]
    (nonarrays, args, typs, f) = specialize(state, args, typs, 
    as -> [ Expr(:tuple, mk_expr(etyp, :call, TopNode(:select_value), as[3], as[2], as[1])) ])
    elmtyps = Type[ (isarray(t) || isbitarray(t)) ? elmTypOf(t) : t for t in typs ]
    linfo = LambdaInfo()
    for i=1:length(nonarrays)
        # At this point, they are either symbol nodes, or constants
        if isa(nonarrays[i], SymbolNode)
            addEscapingVariable(nonarrays[i].name, nonarrays[i].typ, 0, linfo)
        end
    end
    domF = DomainLambda(elmtyps, Type[etyp], f, linfo)
    expr = mmapRemoveDupArg!(mk_mmap!(args, domF))
    expr.typ = typ
    return expr
end

function translate_call_runstencil(state, env, args::Array{Any,1})
    # we handle the following runStencil form:
    #  runStencil(kernelFunc, buf1, buf2, ..., [iterations], [border], buffers)
    # where kernelFunc takes the same number of bufs as inputs
    # and returns the same number of them, but in different order,
    # as a rotation specification for iterative stencils.
    dprintln(env,"got runStencil, args=", args)
    # need to retrieve stencil kernel lambda from inits, since it is already moved out.
    local nargs = length(args)
    args = normalize_args(state, env, args)
    @assert nargs >= 3 "runstencil needs at least a function, and two buffers"
    local iterations = 1               # default
    local borderExp = nothing          # default
    local kernelExp = args[1]
    local bufs = Any[]
    local bufstyp = Any[]
    local i
    for i = 2:nargs
        oprTyp = typeOfOpr(state, args[i])
        if isarray(oprTyp) || isbitarray(oprTyp)
            push!(bufs, args[i])
            push!(bufstyp, oprTyp)
        else
            break
        end
    end
    if i == nargs
        if is(typeOfOpr(state, args[i]), Int)
            iterations = args[i]
        else
            borderExp = args[i]
        end
    elseif i + 1 <= nargs
        iterations = args[i]
        borderExp = args[i+1]
    end
    assert(isa(kernelExp, SymbolNode) || isa(kernelExp, GenSym))
    kernelExp = lookupConstDefForArg(state, kernelExp)
    dprintln(env, "stencil kernelExp = ", kernelExp)
    dprintln(env, "stencil bufstyp = ", to_tuple_type(tuple(bufstyp...)))
    assert(isa(kernelExp, LambdaStaticData))
    # TODO: better infer type here
    (ast, ety) = lambdaTypeinf(kernelExp, tuple(bufstyp...))
    #etys = isa(ety, Tuple) ? Type [ t for t in ety ] : Type[ ety ]
    dprintln(env, "type inferred AST = ", ast)
    kernelExp = from_expr(state, env, ast)
    if is(borderExp, nothing)
        borderExp = QuoteNode(:oob_skip)
    else
        borderExp = lookupConstDefForArg(state, borderExp)
    end
    dprintln(env, "bufs = ", bufs, " kernelExp = ", kernelExp, " borderExp=", borderExp, " :: ", typeof(borderExp))
    local stat, kernelF
    stat, kernelF = mkStencilLambda(state, bufs, kernelExp, borderExp)
    dprintln(env, "stat = ", stat, " kernelF = ", kernelF)
    expr = mk_stencil!(stat, iterations, bufs, kernelF)
    #typ = length(bufs) > 2 ? tuple(kernelF.outputs...) : kernelF.outputs[1] 
    # force typ to be Void, which means stencil doesn't return anything
    typ = Void
    expr.typ = typ
    return expr
end


function translate_call_cartesianarray(state, env, typ, args::Array{Any,1})
    # equivalent to creating an array first, then map! with indices.
    dprintln(env, "got cartesianarray args=", args)
    # need to retrieve map lambda from inits, since it is already moved out.
    nargs = length(args)
    args = normalize_args(state, env, args)
    assert(nargs >= 3) # needs at least a function, one or more types, and a dimension tuple
    local dimExp = args[end]     # last argument is the dimension tuple
    assert(isa(dimExp, SymbolNode) || isa(dimExp, GenSym))
    dimExp = lookupConstDefForArg(state, dimExp)
    dprintln(env, "dimExp = ", dimExp, " head = ", dimExp.head, " args = ", dimExp.args)
    assert(isa(dimExp, Expr) && is(dimExp.head, :call) && is(dimExp.args[1], TopNode(:tuple)))
    dimExp = dimExp.args[2:end]
    ndim = length(dimExp)   # num of dimensions
    argstyp = Any[ Int for i in 1:ndim ] 
    local mapExp = args[1]     # first argument is the lambda
    #println("mapExp ", mapExp)
    #dump(mapExp,1000)
    if isa(mapExp, GlobalRef) && (mapExp.mod == Main  || mapExp.mod == ParallelAccelerator)
        mapExp = mapExp.name
    end
    if isa(mapExp, Symbol) && !is(env.cur_module, nothing) && (isdefined(env.cur_module, mapExp) || isdefined(ParallelAccelerator, mapExp)) && !isdefined(Base, mapExp) # only handle functions in current or Main module

        if(isdefined(ParallelAccelerator, mapExp))
            m = methods(getfield(ParallelAccelerator, mapExp), tuple(argstyp...))
        else
            m = methods(getfield(env.cur_module, mapExp), tuple(argstyp...))
        end
        dprintln(env,"function for cartesianarray: ", mapExp, " methods=", m, " argstyp=", argstyp)
        assert(length(m) > 0)
        mapExp = m[1].func.code
    elseif isa(mapExp, SymbolNode) || isa(mapExp, GenSym)
        mapExp = lookupConstDefForArg(state, mapExp)
    end
    @assert isa(mapExp, LambdaStaticData) "mapExp is not LambdaStaticData"*dump(mapExp)
    # call typeinf since Julia doesn't do it for us
    # and we can figure out the element type from mapExp's return type
    (ast, ety) = lambdaTypeinf(mapExp, tuple(argstyp...))
    # Element type is specified as an argument to cartesianarray
    # This allows us to cast the return type, but inference still needs to be
    # called on the mapExp ast.
    # These types are sometimes GlobalRefs, and must be evaled into Type
    etys = [ isa(t, GlobalRef) ? eval(t) : t for t in args[2:end-1] ]
    dprintln(env, "etys = ", etys)
    @assert all([ isa(t, DataType) for t in etys ]) "cartesianarray expects static type parameters, but got "*dump(etys) 
    ast = from_expr("anonymous", env.cur_module, ast)
    # dprintln(env, "ast = ", ast)
    # create tmp arrays to store results
    arrtyps = Type[ Array{t, ndim} for t in etys ]
    tmpNodes = Array(GenSym, length(arrtyps))
    # allocate the tmp array
    for i = 1:length(arrtyps)
        arrdef = type_expr(arrtyps[i], mk_alloc(etys[i], dimExp))
        tmparr = addGenSym(arrtyps[i], state.linfo)
        updateDef(state, tmparr, arrdef)
        emitStmt(state, mk_expr(arrtyps[i], :(=), tmparr, arrdef))
        tmpNodes[i] = tmparr
    end
    # produce a DomainLambda
    body = ast.args[3]
    params = [ if isa(x, Expr) x.args[1] else x end for x in ast.args[1] ]
    # dprintln(env, "params = ", params)
    #locals = metaToVarDef(ast.args[2][2])
    #escapes = metaToVarDef(ast.args[2][3])
    linfo = lambdaExprToLambdaInfo(ast)
    assert(isa(body, Expr) && is(body.head, :body))
    # fix the return in body
    lastExp = body.args[end]
    assert(isa(lastExp, Expr) && is(lastExp.head, :return))
    # dprintln(env, "fixReturn: lastExp = ", lastExp)
    args1 = lastExp.args[1]
    args1_typ = CompilerTools.LivenessAnalysis.typeOfOpr(args1, linfo)
    # lastExp may be returning a tuple
    if istupletyp(args1_typ)
        # create tmp variables to store results
        local tvar = args1
        local typs = args1_typ.parameters
        local nvar = length(typs)
        local retNodes = GenSym[ addGenSym(t, linfo) for t in typs ]
        local retExprs = Array(Expr, length(retNodes))
        for i in 1:length(retNodes)
            n = retNodes[i]
            t = typs[i]
            retExprs[i] = mk_expr(typ, :(=), n, mk_expr(t, :call, GlobalRef(Base, :getfield), tvar, i))
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
    function bodyF(plinfo, args)
        #bt = backtrace() ;
        #s = sprint(io->Base.show_backtrace(io, bt))
        #dprintln(3, "bodyF backtrace ")
        #dprintln(3, s)
        ldict = CompilerTools.LambdaHandling.mergeLambdaInfo(plinfo, linfo)
        #dprintln(2,"cartesianarray body = ", body, " type = ", typeof(body))
        ldict = merge(ldict, Dict{SymGen,Any}(zip(params, args[1+length(etys):end])))
        # dprintln(2,"cartesianarray idict = ", ldict)
        ret = replaceExprWithDict(body, ldict).args
        #dprintln(2,"cartesianarray ret = ", ret)
        ret
    end
    domF = DomainLambda(vcat(etys, argstyp), etys, bodyF, linfo)
    expr = mk_mmap!(tmpNodes, domF, true)
    expr.typ = length(arrtyps) == 1 ? arrtyps[1] : to_tuple_type(tuple(arrtyps...))
    return expr
end


function translate_call_reduce(state, env, typ, fun::Symbol, args::Array{Any,1})
    arr = args[1]
    # element type is the same as typ
    etyp = is(typ, Any) ? elmTypOf(typeOfOpr(state, arr)) : typ;
    neutral = (reduceNeutrals[fun])(etyp)
    fun = reduceOps[fun]
    typs = Type[ etyp for arg in args] # just use etyp for input element types
    opr, reorder = specializeOp(fun, typs)
    # ignore reorder since it is always id function
    f(linfo,as) = [Expr(:tuple, mk_expr(etyp, :call, opr, as...))]
    domF = DomainLambda([etyp, etyp], [etyp], f, LambdaInfo())
    # turn reduce(z, getindex(a, ...), f) into reduce(z, select(a, ranges(...)), f)
    arr = inline_select(env, state, arr)
    expr = mk_reduce(convert(etyp, neutral), arr, domF)
    expr.typ = etyp
    return expr
end

function translate_call_fill!(state, env, typ, args::Array{Any,1})
    args = normalize_args(state, env, args)
    @assert length(args)==2 "fill! should have 2 arguments"
    arr = args[1]
    ival = args[2]
    typs = Type[typeOfOpr(state, arr)]
    linfo = LambdaInfo()
    if isa(ival, GenSym)
        tmpv = addFreshLocalVariable(string(ival), getType(ival, state.linfo), ISASSIGNED | ISASSIGNEDONCE, state.linfo)
        emitStmt(state, mk_expr(tmpv.typ, :(=), tmpv.name, ival))
        ival = tmpv
    end
    if isa(ival, SymbolNode)
        def = getVarDef(ival.name, state.linfo)
        flag = def == nothing ? 0 : def.desc
        addEscapingVariable(ival.name, ival.typ, flag, linfo)
    end
    f(linfo,as) = [ Expr(:tuple, ival) ]
    domF = DomainLambda(typs, typs, f, linfo)
    expr = mmapRemoveDupArg!(mk_mmap!([arr], domF))
    expr.typ = typ
    return expr
end

function translate_call_parallel_for(state, env, args::Array{Any,1})
    (ast, ety) = lambdaTypeinf(args[1], (Int, ))
    ast = from_expr("anonymous", env.cur_module, ast)
    loopvars = [ if isa(x, Expr) x.args[1] else x end for x in ast.args[1] ]
    etys = [Int for _ in length(loopvars)]
    body = ast.args[3]
    ranges = args[2:end]
    linfo = lambdaExprToLambdaInfo(ast)
    assert(isa(body, Expr) && is(body.head, :body))
    lastExp = body.args[end]
    assert(isa(lastExp, Expr) && is(lastExp.head, :return))
    # Remove return statement
    pop!(body.args)
    function bodyF(plinfo, args)
        ldict = CompilerTools.LambdaHandling.mergeLambdaInfo(plinfo, linfo)
        # ldict = merge(ldict, Dict{SymGen,Any}(zip(loopvars, etys)))
        ret = replaceExprWithDict(body, ldict).args
        ret
    end
    domF = DomainLambda(etys, etys, bodyF, linfo)
    return mk_parallel_for(loopvars, ranges, domF)
end

# translate a function call to domain IR if it matches
function translate_call(state, env, typ, head, oldfun, oldargs, fun::GlobalRef, args)
    local env_ = nextEnv(env)
    expr = nothing
    dprintln(env, "translate_call fun=", fun, "::", typeof(fun), " args=", args, " typ=", typ)
    # new mainline Julia puts functions in Main module but PSE expects the symbol only
    #if isa(fun, GlobalRef) && fun.mod == Main
    #   fun = fun.name
    # end
    if is(fun.mod, Base.Math)
        # NOTE: we simply bypass all math functions for now
        dprintln(env,"by pass math function ", fun, ", typ=", typ)
        # Fix return type of math functions
        if is(typ, Any) && length(args) > 0
            dprintln(env,"fix type for ", expr, " from ", typ, " => ", args[1].typ)
            typ = args[1].typ
        end
        #    elseif is(fun.mod, Base) && is(fun.name, :arraysize)
        #     args = normalize_args(state, env_, args)
        #    dprintln(env,"got arraysize, args=", args)
        #   expr = mk_arraysize(args...)
        #    expr.typ = typ
    elseif isdefined(fun.mod, fun.name)
        args_typ = map(x -> typeOfOpr(state, x), args)
        gf = getfield(fun.mod, fun.name)
        if isgeneric(gf)
            dprintln(env,"function to offload: ", fun, " methods=", methods(gf))
            _accelerate(gf, tuple(args_typ...))
        else
            dprintln(env,"function ", fun, " not offloaded since it isn't generic.")
        end
    else
        dprintln(env,"function call not translated: ", fun, ", and is not found!")
    end
    if isa(expr, Void)
        oldargs = normalize_args(state, env_, oldargs)
        expr = mk_expr(typ, head, oldfun, oldargs...)
    end
    return expr
end

function translate_call(state, env, typ, head, oldfun, oldargs, fun::ANY, args)
    dprintln(3,"unrecognized fun type ",fun, " args ", args)
    oldargs = normalize_args(state, env, oldargs)
    expr = mk_expr(typ, head, oldfun, oldargs...)
end

function from_return(state, env, expr)
    local env_ = nextEnv(env)
    local head = expr.head
    local typ  = expr.typ
    local args = normalize_args(state, env, expr.args)
    # fix return type, the assumption is there is either 0 or 1 args.
    typ = length(args) > 0 ? typeOfOpr(state, args[1]) : Void
    return mk_expr(typ, head, args...)
end

function from_expr(function_name::AbstractString, cur_module :: Module, ast :: Expr)
    dprintln(2,"DomainIR translation function = ", function_name, " on:")
    dprintln(2,ast)
    ast = from_expr(emptyState(), newEnv(cur_module), ast)
    dprintln(2,"DomainIR translation returns:")
    dprintln(2,ast)
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::LambdaStaticData)
    dprintln(env, "from_expr: LambdaStaticData inferred = ", ast.inferred)
    if !ast.inferred
        # we return this unmodified since we want the caller to
        # type check it before conversion.
        return ast
    else
        ast = uncompressed_ast(ast)
        # (tree, ty)=Base.typeinf(ast, argstyp, ())
    end
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::Union{SymbolNode,Symbol})
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
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::Expr)
    local asttyp = typeof(ast)
    dprint(env,"from_expr: ", asttyp)
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
    elseif is(head, :type_goto)
        # skip?
    elseif is(head, :gotoifnot)
        # specific check to take care of artifact from comprhension-to-cartesianarray translation
        # gotoifnot (Base.slt_int(1,0)) label ===> got label
        if length(args) == 2 && isa(args[1], Expr) && is(args[1].head, :call) && 
            is(args[1].args[1], GlobalRef(Base, :slt_int)) && args[1].args[2] == 1 && args[1].args[3] == 0
            dprintln(env, "Match gotoifnot shortcut!")
            return GotoNode(args[2])
        else # translate arguments
          ast.args = normalize_args(state, env, args)
        end
        # ?
    elseif is(head, :meta)
        # skip
    elseif is(head, :static_typeof)
        typ = getType(args[1], state.linfo)
        return typ  
    else
        throw(string("from_expr: unknown Expr head :", head))
    end
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::ANY)
    dprintln(2, " not handled ", ast)
    return ast
end

type DirWalk
    callback
    cbdata
end

function AstWalkCallback(x :: ANY, dw :: DirWalk, top_level_number, is_top_level, read)
    dprintln(3,"DomainIR.AstWalkCallback ", x)
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    dprintln(3,"DomainIR.AstWalkCallback ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
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
                args[1][i] = AstWalker.AstWalk(input_arrays[i], AstWalkCallback, dw)
            end
            args[2] = AstWalker.AstWalk(args[2], AstWalkCallback, dw)
            return x
        elseif head == :reduce
            assert(length(args) == 3)
            for i = 1:3
                args[i] = AstWalker.AstWalk(args[i], AstWalkCallback, dw)
            end
            return x
        elseif head == :select 
            # it is always in the form of select(arr, mask), where range can itself be ranges(range(...), ...))
            assert(length(args) == 2)
            args[1] = AstWalker.AstWalk(args[1], AstWalkCallback, dw)
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
                    ranges[i].args[j] = AstWalker.AstWalk(ranges[i].args[j], AstWalkCallback, dw)
                end
            end
            return x
        elseif head == :stencil!
            assert(length(args) == 4)
            args[2] = AstWalker.AstWalk(args[2], AstWalkCallback, dw)
            for i in 1:length(args[3]) # buffer array
                args[3][i] = AstWalker.AstWalk(args[3][i], AstWalkCallback, dw)
            end
            return x
        elseif head == :parallel_for
            map!((a) -> AstWalker.AstWalk(a, AstWalkCallback, dw), args[1])
            map!((a) -> AstWalker.AstWalk(a, AstWalkCallback, dw), args[2])
            args[3] = AstWalker.AstWalk(args[3], AstWalkCallback, dw)
            return x
        elseif head == :assertEqShape
            assert(length(args) == 2)
            for i = 1:length(args)
                args[i] = AstWalker.AstWalk(args[i], AstWalkCallback, dw)
            end
            return x
        elseif head == :assert 
            for i = 1:length(args)
                AstWalker.AstWalk(args[i], AstWalkCallback, dw)
            end
            return x
        elseif head == :select
            for i = 1:length(args)
                args[i] = AstWalker.AstWalk(args[i], AstWalkCallback, dw)
            end
            return x
        elseif head == :range
            for i = 1:length(args)
                args[i] = AstWalker.AstWalk(args[i], AstWalkCallback, dw)
            end
            return x
        end
        x = Expr(head, args...)
        x.typ = typ
    elseif asttyp == DomainLambda
        dprintln(3,"DomainIR.AstWalkCallback for DomainLambda", x)
        return x
        end

        return CompilerTools.AstWalker.ASTWALK_RECURSE
        end

        function AstWalk(ast :: ANY, callback, cbdata :: ANY)
        dprintln(3,"DomainIR.AstWalk ", ast)
        dw = DirWalk(callback, cbdata)
        AstWalker.AstWalk(ast, AstWalkCallback, dw)
        end

        function dir_live_cb(ast :: ANY, cbdata :: ANY)
        dprintln(4,"dir_live_cb ")
        asttyp = typeof(ast)
        if asttyp == Expr
        head = ast.head
        args = ast.args
        if head == :mmap
        expr_to_process = Any[]
        assert(isa(args[2], DomainLambda))
        dl = args[2]

        assert(length(args) == 2)
        input_arrays = args[1]
        for i = 1:length(input_arrays)
            push!(expr_to_process, input_arrays[i])
        end
        for (v, d) in dl.linfo.escaping_defs
            push!(expr_to_process, v)
        end 

        dprintln(3, ":mmap ", expr_to_process)
        return expr_to_process
    elseif head == :mmap!
        expr_to_process = Any[]
        assert(isa(args[2], DomainLambda))
        dl = args[2]

        assert(length(args) >= 2)
        input_arrays = args[1]
        for i = 1:length(input_arrays)
            if i <= length(dl.outputs)
                # We need both a read followed by a write.
                push!(expr_to_process, input_arrays[i])
                push!(expr_to_process, Expr(symbol('='), input_arrays[i], 1))
            else
                # Need to make input_arrays[1] written?
                push!(expr_to_process, input_arrays[i])
            end
        end
        for (v, d) in dl.linfo.escaping_defs
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
        for (v, d) in dl.linfo.escaping_defs
            push!(expr_to_process, v)
        end

        dprintln(3, ":reduce ", expr_to_process)
        return expr_to_process
    elseif head == :stencil!
        expr_to_process = Any[]

        sbufs = args[3]
        for i = 1:length(sbufs)
            # sbufs both read and possibly written
            push!(expr_to_process, sbufs[i])
            push!(expr_to_process, Expr(symbol('='), sbufs[i], 1))
        end

        dl = args[4]
        assert(isa(dl, DomainLambda))
        for (v, d) in dl.linfo.escaping_defs
            push!(expr_to_process, v)
        end

        dprintln(3, ":stencil! ", expr_to_process)
        return expr_to_process
    elseif head == :parallel_for
        expr_to_process = Any[]

        assert(length(args) == 3)
        loopvars = args[1]
        ranges = args[2]
        escaping_defs = args[3].linfo.escaping_defs
        push!(expr_to_process, loopvars)
        append!(expr_to_process, ranges)
        for (v, d) in escaping_defs
            push!(expr_to_process, v)
        end

        dprintln(3, ":parallel_for ", expr_to_process)
        return expr_to_process
    elseif head == :assertEqShape
        assert(length(args) == 2)
        #dprintln(3,"liveness: assertEqShape ", args[1], " ", args[2], " ", typeof(args[1]), " ", typeof(args[2]))
        expr_to_process = Any[]
        push!(expr_to_process, symbol_or_gensym(args[1]))
        push!(expr_to_process, symbol_or_gensym(args[2]))
        return expr_to_process
    elseif head == :assert
        expr_to_process = Any[]
        for i = 1:length(args)
            push!(expr_to_process, args[i])
        end
        return expr_to_process
    elseif head == :select
        expr_to_process = Any[]
        for i = 1:length(args)
            push!(expr_to_process, args[i])
        end
        return expr_to_process
    elseif head == :range
        expr_to_process = Any[]
        for i = 1:length(args)
            push!(expr_to_process, args[i])
        end
        return expr_to_process
    end
elseif asttyp == KernelStat
    return Any[]
end
return nothing
end

function symbol_or_gensym(x)
    xtyp = typeof(x)
    if xtyp == Symbol
        return x
    elseif xtyp == SymbolNode
        return x.name
    elseif xtyp == GenSym
        return x
    else
        throw(string("Don't know how to convert ", x, " of type ", xtyp, " to a symbol or a GenSym"))
    end
end

function dir_alias_cb(ast, state, cbdata)
    dprintln(4,"dir_alias_cb ")
    asttyp = typeof(ast)
    if asttyp == Expr
        head = ast.head
        args = ast.args
        if head == :mmap
            # TODO: inspect the lambda body to rule out assignment?
            return AliasAnalysis.next_node(state)
        elseif head == :mmap!
            dl :: DomainLambda = args[2]
            # n_outputs = length(dl.outputs)
            # assert(n_outputs == 1) 
            # FIXME: fix it in case of multiple return!
            tmp = args[1][1]
            if isa(tmp, Expr) && is(tmp.head, :select) # selecting a range
                tmp = tmp.args[1] 
            end 
            toSymGen(x) = isa(x, SymbolNode) ? x.name : x
            return AliasAnalysis.lookup(state, toSymGen(tmp))
        elseif head == :reduce
            # TODO: inspect the lambda body to rule out assignment?
            return AliasAnalysis.NotArray
        elseif head == :stencil!
            # args is a list of PIRParForAst nodes.
            assert(length(args) > 0)
            krnStat = args[1]
            iterations = args[2]
            bufs = args[3]
            dprintln(3, "AA: rotateNum = ", krnStat.rotateNum, " out of ", length(bufs), " input bufs")
            if !((isa(iterations, Number) && iterations == 1) || (krnStat.rotateNum == 0))
                # when iterations > 1, and we have buffer rotation, need to set alias Unknown for all rotated buffers
                for i = 1:min(krnStat.rotateNum, length(bufs))
                    v = bufs[i]
                    if isa(v, SymbolNode)
                        AliasAnalysis.update_unknown(state, v.name)
                    end
                end
            end
            return Any[] # empty array of Expr
        elseif head == :parallel_for
            # TODO: What should we do here?
            return AliasAnalysis.NotArray
        elseif head == :assertEqShape
            return AliasAnalysis.NotArray
        elseif head == :assert
            return AliasAnalysis.NotArray 
        elseif head == :select
            return AliasAnalysis.next_node(state) 
        elseif head == :ranges
            return AliasAnalysis.NotArray
        elseif is(head, :tomask)
            toSymGen(x) = isa(x, SymbolNode) ? x.name : x
            return AliasAnalysis.lookup(state, toSymGen(args[1]))
        elseif is(head, :arraysize)
            return AliasAnalysis.NotArray
        elseif is(head, :tuple)
            return AliasAnalysis.NotArray
        elseif is(head, :alloc)
            return AliasAnalysis.next_node(state)
        elseif is(head, :copy)
            return AliasAnalysis.next_node(state)
        elseif is(head, :call) || is(head, :call1)
            local fun  = ast.args[1]
            local args = ast.args[2:end]
            (fun_, args) = DomainIR.normalize_callname(DomainIR.emptyState(), DomainIR.newEnv(nothing), fun, args)
            dprintln(2, "AA from_call: normalized fun=", fun_)
            if(haskey(DomainIR.mapOps, fun_))
                return AliasAnalysis.NotArray
            elseif is(fun_, :alloc)
                return AliasAnalysis.next_node(state)
            elseif is(fun_, :fill!)
                return args[1]
            end
        end

        return nothing
    end
end

end
