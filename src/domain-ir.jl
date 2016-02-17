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

#using Debug

import CompilerTools.DebugMsg
DebugMsg.init()

import CompilerTools.AstWalker
using CompilerTools
using CompilerTools.LivenessAnalysis
using CompilerTools.LambdaHandling
using CompilerTools.Helper
using Core.Inference: to_tuple_type
using Base.uncompressed_ast
using CompilerTools.AliasAnalysis

import ..H5SizeArr_t
import ..SizeArr_t

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


mk_eltype(arr) = Expr(:eltype, arr)
mk_ndim(arr) = Expr(:ndim, arr)
mk_length(arr) = Expr(:length, arr)
#mk_arraysize(arr, d) = Expr(:arraysize, arr, d)
mk_arraysize(arr, dim) = TypedExpr(Int64, :call, TopNode(:arraysize), arr, dim)
mk_sizes(arr) = Expr(:sizes, arr)
mk_strides(arr) = Expr(:strides, arr)
mk_alloc(typ, s) = Expr(:alloc, typ, s)
mk_call(fun,args) = Expr(:call, fun, args...)
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
mk_reduce(zero, arr, f, dim) = Expr(:reduce, zero, arr, f, dim) # reduce across a single dimension
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

export DomainLambda, KernelStat, AstWalk, arraySwap, lambdaSwapArg, isarray, isbitarray

# A representation for anonymous lambda used in domainIR.
#   inputs:  types of input tuple
#   outputs: types of output tuple
#   genBody: (LambdaVarInfo, Array{Any,1}) -> Array{Expr, 1}
#   escapes: escaping variables in the body
#
# So the downstream can just call genBody, and pass it
# the downstream's LambdaVarInfo and an array of parameters,
# and it will return an expression (with :body head) that 
# represents the loop body. The input LambdaVarInfo, if
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
    linfo   :: LambdaVarInfo

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
    linfo  :: LambdaVarInfo
    defs   :: Dict{Union{Symbol,Int}, Any}  # stores local definition of LHS = RHS
    boxtyps:: Dict{Union{Symbol,Int}, Any}  # finer types for those have Box type
    stmts  :: Array{Any, 1}
    parent :: Union{Void, IRState}
    data_source_counter::Int64 # a unique counter for data sources in program
end

emptyState() = IRState(LambdaVarInfo(), Dict{Union{Symbol,Int},Any}(), Dict{Union{Symbol,Int},Any}(), Any[], nothing, 0)
newState(linfo, defs, state::IRState) = IRState(linfo, defs, Dict{Union{Symbol,Int},Any}(), Any[], state, state.data_source_counter)

"""
Update the type of a variable.
"""
function updateTyp(state::IRState, s::SymAllGen, typ)
    updateType(state.linfo, s, typ)
end

function updateBoxType(state::IRState, s::SymAllGen, typ)
    x = isa(s, SymbolNode) ? s.name : s
    y = isa(s, GenSym) ? s.id : x
    state.boxtyps[y] = typ
end

function getBoxType(state::IRState, s::SymAllGen)
    x = isa(s, SymbolNode) ? s.name : s
    y = isa(s, GenSym) ? s.id : x
    state.boxtyps[y] 
end

"""
Update the definition of a variable.
"""
function updateDef(state::IRState, s::SymAllGen, rhs)
    s = isa(s, SymbolNode) ? s.name : s
    #@dprintln(3, "updateDef: s = ", s, " rhs = ", rhs, " typeof s = ", typeof(s))
    @assert ((isa(s, GenSym) && isLocalGenSym(s, state.linfo)) ||
    (isa(s, Symbol) && isLocalVariable(s, state.linfo)) ||
    (isa(s, Symbol) && isInputParameter(s, state.linfo))) state.linfo
    s = isa(s, GenSym) ? s.id : s
    state.defs[s] = rhs
end

"""
Look up a definition of a variable.
Return nothing If none is found.
"""
function lookupDef(state::IRState, s::SymAllGen)
    s = isa(s, SymbolNode) ? s.name : (isa(s, GenSym) ? s.id : s)
    get(state.defs, s, nothing)
end

function lookupDef(state, s)
    return nothing
end

"""
Look up a definition of a variable only when it is const or assigned once.
Return nothing If none is found.
"""
function lookupConstDef(state::IRState, s::SymAllGen)
    def = lookupDef(state, s)
    # we assume all GenSym is assigned once 
    desc = isa(s, SymbolNode) ? getDesc(s.name, state.linfo) : (ISASSIGNEDONCE | ISASSIGNED)
    if !is(def, nothing) && ((desc & (ISASSIGNEDONCE | ISCONST)) != 0 || typeOfOpr(state, s) <: Function)
        return def
    end
    return nothing
end

"""
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

function lookupConstDefForArg(state::Void, s::Any)
    return nothing
end

"""
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
    @dprintln(2,"emit stmt: ", stmt)
    if isa(stmt, Expr) && is(stmt.head, :(=)) && stmt.typ == Box
        @dprintln(2, "skip Box assigment")
    else
        push!(state.stmts, stmt)
    end
end

type IREnv
    cur_module  :: Module
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

const compareOpSet = Set{Symbol}(API.comparison_map_operators)
const mapOps = Dict{Symbol,Symbol}(zip(mapSym, mapVal))
# symbols that when lifted up to array level should be changed.
const liftOps = Dict{Symbol,Symbol}(zip(Symbol[:<=, :>=, :<, :(==), :>, :+,:-,:*,:/], Symbol[:.<=, :.>=, :.<, :.==, :.>, :.+, :.-, :.*, :./]))

# legacy v0.3
# const topOpsTypeFix = Set{Symbol}([:not_int, :and_int, :or_int, :neg_int, :add_int, :mul_int, :sub_int, :neg_float, :mul_float, :add_float, :sub_float, :div_float, :box, :fptrunc, :fpsiround, :checked_sadd, :checked_ssub, :rint_llvm, :floor_llvm, :ceil_llvm, :abs_float, :cat_t, :srem_int])

const opsSym = Symbol[:negate, :+, :-, :*, :/, :(==), :!=, :<, :<=]
const opsSymSet = Set{Symbol}(opsSym)
const floatOps = Dict{Symbol,Symbol}(zip(opsSym, [:neg_float, :add_float, :sub_float, :mul_float, :div_float, :eq_float, :ne_float, :lt_float, :le_float]))
const sintOps  = Dict{Symbol,Symbol}(zip(opsSym, [:neg_int, :add_int, :sub_int, :mul_int, :sdiv_int, :eq_int, :ne_int, :slt_int, :sle_int]))

const reduceSym = Symbol[:sum, :prod, :maximum, :minimum, :any, :all]
const reduceVal = Symbol[:+, :*, :max, :min, :|, :&]
const reduceFun = Function[zero, one, typemin, typemax, x->false, x->true]
const reduceOps = Dict{Symbol,Symbol}(zip(reduceSym,reduceVal))
const reduceNeutrals = Dict{Symbol,Function}(zip(reduceSym,reduceFun))

const ignoreSym = Symbol[:box]
const ignoreSet = Set{Symbol}(ignoreSym)

const allocCalls = Set{Symbol}([:jl_alloc_array_1d, :jl_alloc_array_2d, :jl_alloc_array_3d, :jl_new_array])

const afoldTyps = Type[Base.AddFun, Base.MulFun, Base.AndFun, Base.OrFun]
const afoldOprs = Symbol[:+, :*, :&, :|]
const afoldlDict = Dict{Type,Symbol}(zip(afoldTyps, afoldOprs))

# some part of the code still requires this
unique_id = 0
function addFreshLocalVariable(s::AbstractString, t::Any, desc, linfo::LambdaVarInfo)
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


function istupletyp(typ::DataType)
    is(typ.name, Tuple.name)
end

function istupletyp(typ::ANY)
    false
end

function isarray(typ::DataType)
    is(typ.name, Array.name)
end

function isarray(typ::ANY)
    return false
end

function isbitarray(typ::DataType)
    is(typ.name, BitArray.name) ||
    (isarray(typ) && is(eltype(typ), Bool))
end

function isbitarray(typ::ANY)
    false
end

function isUnitRange(typ::DataType)
    is(typ.name, UnitRange.name)
end

function isUnitRange(typ::ANY)
    return false
end

function isStepRange(typ::DataType)
    is(typ.name, StepRange.name)
end

function isStepRange(typ::ANY)
    return false
end

function isrange(typ)
    isUnitRange(typ) || isStepRange(typ)
end

function ismask(state, r::SymbolNode)
    return isrange(r.typ) || isbitarray(r.typ)
end

function ismask(state, r::SymGen)
    typ = getType(r, state.linfo)
    return isrange(typ) || isbitarray(typ)
end

function ismask(state, r::GlobalRef)
    return r.name==:(:)
end

function ismask(state, r::Any)
    typ = typeOfOpr(state, r)
    return isrange(typ) || isbitarray(typ)
end

# never used!
#=
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
=#

function from_range(rhs::Expr)
    start = 1
    step = 1
    final = 1
    if is(rhs.head, :new) && isUnitRange(rhs.args[1]) &&
        isa(rhs.args[3], Expr) && is(rhs.args[3].head, :call) &&
        isa(rhs.args[3].args[1], Expr) && is(rhs.args[3].args[1].head, :call) &&
        is(rhs.args[3].args[1].args[1], TopNode(:getfield)) &&
        is(rhs.args[3].args[1].args[2], GlobalRef(Base, :Intrinsics)) &&
        is(rhs.args[3].args[1].args[3], QuoteNode(:select_value))
        # only look at final value in select_value of UnitRange
        start = rhs.args[2]
        step  = 1 # FIXME: could be wrong here!
        final = rhs.args[3].args[3]
    elseif is(rhs.head, :new) && isStepRange(rhs.args[1])
        assert(length(rhs.args) == 4)
        start = rhs.args[2]
        step  = rhs.args[3]
        final = rhs.args[4]
    else
        error("expect Expr(:new, UnitRange, ...) or Expr(:new, StepRange, ...) but got ", rhs)
    end
    return (start, step, final)
end

function from_range(rhs::ANY)
    error("expect Expr(:new, UnitRange, ...) or Expr(:new, StepRange, ...) but got ", rhs)
    return (1,1,1)
end

function rangeToMask(state, r::Int, arraysize)
    # return mk_range(state, r, r, 1)
    # scalar is not treated as a range in Julia
    return r
end

function rangeToMask(state, r::SymAllGen, arraysize)
    typ = getType(r, state.linfo)
    if isbitarray(typ)
        mk_tomask(r)
    elseif isUnitRange(typ)
        r = lookupConstDefForArg(state, r)
        (start, step, final) = from_range(r)
        mk_range(state, start, step, final)
    elseif isIntType(typ) 
        #mk_range(state, r, convert(typ, 1), r)
        r
    else
        error("Unhandled range object: ", r)
    end
end

function rangeToMask(state, r::GlobalRef, arraysize)
    # FIXME: why doesn't this assert work?
    #@assert (r.mod!=Main || r.name!=symbol(":")) "unhandled GlobalRef range"
    if r.name==symbol(":")
        return mk_range(state, 1, 1, arraysize)
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
            if isArrayType(typ)
                n = n + 1
            end
        end
        return (n == 1)
    end     
end

function specializeOp(opr::Symbol, argstyp)
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
    @dprintln(2, "specializeOp opsSymSet[", opr, "] = ", in(opr, opsSymSet), " typ=", typ)
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
    local body = f.genBody(LambdaVarInfo(), syms) # use a dummy LambdaVarInfo for show purpose
    print(io, "(")
    show(io, symNodes)
    print(io, ";")
    #show(io, f.linfo)
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
        if isArrayType(typ)
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

"""
get elem type T from an Array{T} type
"""
function elmTypOf(x::DataType)
    @assert isarray(x) || isbitarray(x) "expect Array type"
    return eltype(x)
end

function elmTypOf(x::Expr)
    if x.head == :call && x.args[1] == TopNode(:apply_type) && x.args[2] == :Array
        return x.args[3] # element type
    else
        error("expect Array type, but got ", x)
    end
    return DataType
end

# A simple integer arithmetic simplifier that does three things:
# 1. inline constant definitions.
# 2. normalize into sums, with variables and constants.
# 3. statically evaluate arithmetic operations on constants. 
# Note that no sharing is preserved due to flattening, so use it with caution.
function simplify(state, expr::Expr, default::Any)
    if isAddExpr(expr)
        x = simplify(state, expr.args[2])
        y = simplify(state, expr.args[3])
        add(x, y)
    elseif isSubExpr(expr)
        x = simplify(state, expr.args[2])
        y = simplify(state, expr.args[3])
        add(x, neg(y))
    elseif isMulExpr(expr)
        x = simplify(state, expr.args[2])
        y = simplify(state, expr.args[3])
        mul(x, y)
    elseif isNegExpr(expr)
        neg(simplify(state, expr.args[2]))
    elseif isBoxExpr(expr)
        simplify(state, expr.args[3])
    else
        default
    end
end

function simplify(state, expr::Expr)
    simplify(state, expr, expr)
end 

function simplify(state, expr::SymbolNode)
    def = lookupConstDefForArg(state, expr)
    is(def, nothing) ? expr : (isa(def, Expr) ? simplify(state, def, expr) : def)
end

function simplify(state, expr::GenSym)
    def = lookupConstDefForArg(state, expr)
    is(def, nothing) ? expr : (isa(def, Expr) ? simplify(state, def, expr) : def)
end

function simplify(state, expr::Array)
    [ simplify(state, e) for e in expr ]
end

function simplify(state, expr)
    return expr
end

isTopNodeOrGlobalRef(x,s) = is(x, TopNode(s)) || is(x, GlobalRef(Core.Intrinsics, s))
add_expr(x,y) = y == 0 ? x : mk_expr(Int, :call, TopNode(:add_int), x, y)
sub_expr(x,y) = y == 0 ? x : mk_expr(Int, :call, TopNode(:sub_int), x, y)
mul_expr(x,y) = y == 0 ? 0 : (y == 1 ? x : mk_expr(Int, :call, TopNode(:mul_int), x, y))
sdiv_int_expr(x,y) = y == 1 ? x : mk_epr(Int, :call, TopNode(:sdiv_int), x, y)
neg_expr(x)   = mk_expr(Int, :call, TopNode(:neg_int), x)
isBoxExpr(x::Expr) = is(x.head, :call) && isTopNodeOrGlobalRef(x.args[1], :box)
isNegExpr(x::Expr) = is(x.head, :call) && isTopNodeOrGlobalRef(x.args[1], :neg_int) 
isAddExpr(x::Expr) = is(x.head, :call) && (isTopNodeOrGlobalRef(x.args[1], :add_int) || isTopNodeOrGlobalRef(x.args[1], :checked_sadd) || isTopNodeOrGlobalRef(x.args[1], :checked_sadd_int))
isSubExpr(x::Expr) = is(x.head, :call) && (isTopNodeOrGlobalRef(x.args[1], :sub_int) || isTopNodeOrGlobalRef(x.args[1], :checked_ssub) || isTopNodeOrGlobalRef(x.args[1], :checked_ssub_int))
isMulExpr(x::Expr) = is(x.head, :call) && (isTopNodeOrGlobalRef(x.args[1], :mul_int) || isTopNodeOrGlobalRef(x.args[1], :checked_smul) || isTopNodeOrGlobalRef(x.args[1], :checked_smul_int))
isAddExprInt(x::Expr) = isAddExpr(x) && isa(x.args[3], Int)
isMulExprInt(x::Expr) = isMulExpr(x) && isa(x.args[3], Int)
isAddExpr(x::ANY) = false
isSubExpr(x::ANY) = false
sub(x, y) = add(x, neg(y))
add(x::Int,  y::Int) = x + y
add(x::Int,  y::Expr)= add(y, x)
add(x::Int,  y)      = add(y, x)
add(x::Expr, y::Int) = isAddExprInt(x) ? add(x.args[2], x.args[3] + y) : add_expr(x, y)
add(x::Expr, y::Expr)= isAddExprInt(x) ? add(add(x.args[2], y), x.args[3]) : add_expr(x, y)
add(x::Expr, y)      = isAddExprInt(x) ? add(add(x.args[2], y), x.args[3]) : add_expr(x, y)
add(x,       y::Expr)= add(y, x)
add(x,       y)      = add_expr(x, y)
neg(x::Int)          = -x
neg(x::Expr)         = isNegExpr(x) ? x.args[2] : 
                       (isAddExpr(x) ? add(neg(x.args[2]), neg(x.args[3])) :
                       (isMulExpr(x) ? mul(x.args[2], neg(x.args[3])) : neg_expr(x)))
neg(x)               = neg_expr(x)
mul(x::Int,  y::Int) = x * y
mul(x::Int,  y::Expr)= mul(y, x)
mul(x::Int,  y)      = mul(y, x)
mul(x::Expr, y::Int) = isMulExprInt(x) ? mul(x.args[2], x.args[3] * y) : 
                       (isAddExpr(x) ? add(mul(x.args[2], y), mul(x.args[3], y)) : mul_expr(x, y))
mul(x::Expr, y::Expr)= isMulExprInt(x) ? mul(mul(x.args[2], y), x.args[3]) : 
                       (isAddExpr(x) ? add(mul(x.args[2], y), mul(x.args[3], y)) : mul_expr(x, y))
mul(x::Expr, y)      = isMulExprInt(x) ? mul(mul(x.args[2], y), x.args[3]) : 
                       (isAddExpr(x) ? add(mul(x.args[2], y), mul(x.args[3], y)) : mul_expr(x, y))
mul(x,       y::Expr)= mul(y, x)
mul(x,       y)      = mul_expr(x, y)

# simplify expressions passed to alloc and range.
mk_alloc(state, typ, s) = mk_alloc(typ, simplify(state, s))
mk_range(state, start, step, final) = mk_range(simplify(state, start), simplify(state, step), simplify(state, final))

"""
 :lambda expression
 (:lambda, {param, meta@{localvars, types, freevars}, body})
"""
function from_lambda(state, env, expr::Expr, closure = nothing)
    local env_ = nextEnv(env)
    local head = expr.head
    local ast  = expr.args
    local typ  = expr.typ
    assert(length(ast) == 3)
    linfo = lambdaExprToLambdaVarInfo(expr) 
    local body  = ast[3]
    assert(isa(body, Expr) && is(body.head, :body))
    defs = Dict{Union{Symbol,Int},Any}()
    if !is(closure, nothing)
        # Julia 0.5 feature, closure refers to the #self# argument
        if isa(closure, Expr)
            def = closure
        elseif isa(closure, SymAllGen)
            def = lookupDef(state, closure) # somehow closure variables are not Const defs
            dprintln(env, "closure ", closure, " = ", def) 
        else 
            def = nothing  # error(string("Unhandled closure: ", closure))
        end
        if isa(def, Expr) && def.head == :new
            ctyp = def.args[1]
            if isa(ctyp, GlobalRef) 
                ctyp = getfield(ctyp.mod, ctyp.name)
            end
            @assert (isa(ctyp, DataType) && ctyp <: Function) "closure type " * ctyp * " is not a function"
            fnames = fieldnames(ctyp)
            dprintln(env, "fieldnames of closure = ", fnames)
            args = def.args[2:end]
            @assert (length(fnames) == length(args)) "Mismatch: closure fields " * fnames * "  with " * args
            dprintln(env, "args = ", args)
            #defs[symbol("#self#")] = Dict(zip(fnames, args))
            for (p, q) in zip(fnames, args)
                qtyp = typeOfOpr(state, q)
                # if q has a Box type, we lookup its definition (due to setfield!) instead
                qtyp = is(qtyp, Box) ? getBoxType(state, q) : qtyp
                dprintln(env, "field ", p, " has type ", qtyp)
                addEscapingVariable(p, qtyp, 0, linfo)
            end
        end
    end
    dprintln(env,"from_lambda: linfo=", linfo)
    local state_ = newState(linfo, defs, state)
    body = from_expr(state_, env_, body)
    # fix return type
    typ = body.typ
    dprintln(env,"from_lambda: body=", body)
    dprintln(env,"from_lambda: linfo=", linfo)
    # fix Box types
    #for (k, t) in state_.boxtyps
    #    updateType(linfo, k, t)
    #end
    return LambdaVarInfoToLambdaExpr(linfo, body)
end

"""
 sequence of expressions {expr, ...}
 unlike from_body, from_exprs do not emit the input expressions
 as statements to the state, while still allowing side effects
 of emitting statements durinfg the translation of these expressions.
"""
function from_exprs(state::IRState, env::IREnv, ast::Array{Any,1})
    local env_ = nextEnv(env)
    local len  = length(ast)
    local body = Array(Any, len)
    for i = 1:len
        body[i] = from_expr(state, env_, ast[i])
    end
    return body
end

"""
 :body expression (:body, {expr, ... })
 Unlike from_exprs, from_body treats every expr in the body
 as separate statements, and emit them (after translation)
 to the state one by one.
"""
function from_body(state, env, expr::Expr)
    local env_ = nextEnv(env)
    # So long as :body immediate nests with :lambda, we do not
    # need to create new state.
    local head = expr.head
    local body = expr.args
    local typ  = expr.typ
    for i = 1:length(body)
        stmt = from_expr(state, env_, body[i])
        emitStmt(state, stmt)
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

function mmapRemoveDupArg!(expr::Expr)
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
    @dprintln(3, "MMRD: expr was ", expr)
    @dprintln(3, "MMRD:  ", newarr, newinp, indices)
    expr.args[1] = newarr
    expr.args[2] = DomainLambda(newinp, f.outputs,
    (linfo, args) -> begin
        dupargs = Array(Any, oldn)
        for i=1:oldn
            dupargs[i] = args[indices[i]]
        end
        f.genBody(linfo, dupargs)
    end, f.linfo)
    @dprintln(3, "MMRD: expr becomes ", expr)
    return expr
end

function pattern_match_hps_dist_calls(state, env, lhs::SymGen, rhs::Expr)
    # example of data source call: 
    # :((top(typeassert))((top(convert))(Array{Float64,1},(ParallelAccelerator.API.__hps_data_source_HDF5)("/labels","./test.hdf5")),Array{Float64,1})::Array{Float64,1})
    if rhs.head==:call && length(rhs.args)>=2 && isCall(rhs.args[2])
        in_call = rhs.args[2]
        if length(in_call.args)>=3 && isCall(in_call.args[3]) 
            inner_call = in_call.args[3]
            if isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hps_data_source_HDF5
                dprintln(env,"data source found ", inner_call)
                hdf5_var = inner_call.args[2]
                hdf5_file = inner_call.args[3]
                # update counter and get data source number
                state.data_source_counter += 1
                dsrc_num = state.data_source_counter
                dsrc_id_var = addGenSym(Int64, state.linfo)
                updateDef(state, dsrc_id_var, dsrc_num)
                emitStmt(state, mk_expr(Int64, :(=), dsrc_id_var, dsrc_num))
                # get array type
                arr_typ = getType(lhs, state.linfo)
                dims = ndims(arr_typ)
                elem_typ = eltype(arr_typ)
                # generate open call
                # lhs is dummy argument so ParallelIR wouldn't reorder
                open_call = mk_call(:__hps_data_source_HDF5_open, [dsrc_id_var, hdf5_var, hdf5_file, lhs])
                emitStmt(state, open_call)
                # generate array size call
                # arr_size_var = addGenSym(Tuple, state.linfo)
                # assume 1D for now
                arr_size_var = addGenSym(H5SizeArr_t, state.linfo)
                size_call = mk_call(:__hps_data_source_HDF5_size, [dsrc_id_var, lhs])
                updateDef(state, arr_size_var, size_call)
                emitStmt(state, mk_expr(arr_size_var, :(=), arr_size_var, size_call))
                # generate array allocation
                size_expr = Any[]
                for i in dims:-1:1
                    size_i = addGenSym(Int64, state.linfo)
                    size_i_call = mk_call(:__hps_get_H5_dim_size, [arr_size_var, i])
                    updateDef(state, size_i, size_i_call)
                    emitStmt(state, mk_expr(Int64, :(=), size_i, size_i_call))
                    push!(size_expr, size_i)
                end
                arrdef = type_expr(arr_typ, mk_alloc(state, elem_typ, size_expr))
                updateDef(state, lhs, arrdef)
                emitStmt(state, mk_expr(arr_typ, :(=), lhs, arrdef))
                # generate read call
                read_call = mk_call(:__hps_data_source_HDF5_read, [dsrc_id_var, lhs])
                return read_call
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hps_data_source_TXT
                dprintln(env,"data source found ", inner_call)
                txt_file = inner_call.args[2]
                # update counter and get data source number
                state.data_source_counter += 1
                dsrc_num = state.data_source_counter
                dsrc_id_var = addGenSym(Int64, state.linfo)
                updateDef(state, dsrc_id_var, dsrc_num)
                emitStmt(state, mk_expr(Int64, :(=), dsrc_id_var, dsrc_num))
                # get array type
                arr_typ = getType(lhs, state.linfo)
                dims = ndims(arr_typ)
                elem_typ = eltype(arr_typ)
                # generate open call
                # lhs is dummy argument so ParallelIR wouldn't reorder
                open_call = mk_call(:__hps_data_source_TXT_open, [dsrc_id_var, txt_file, lhs])
                emitStmt(state, open_call)
                # generate array size call
                # arr_size_var = addGenSym(Tuple, state.linfo)
                arr_size_var = addGenSym(SizeArr_t, state.linfo)
                size_call = mk_call(:__hps_data_source_TXT_size, [dsrc_id_var, lhs])
                updateDef(state, arr_size_var, size_call)
                emitStmt(state, mk_expr(arr_size_var, :(=), arr_size_var, size_call))
                # generate array allocation
                size_expr = Any[]
                for i in dims:-1:1
                    size_i = addGenSym(Int64, state.linfo)
                    size_i_call = mk_call(:__hps_get_TXT_dim_size, [arr_size_var, i])
                    updateDef(state, size_i, size_i_call)
                    emitStmt(state, mk_expr(Int64, :(=), size_i, size_i_call))
                    push!(size_expr, size_i)
                end
                arrdef = type_expr(arr_typ, mk_alloc(state, elem_typ, size_expr))
                updateDef(state, lhs, arrdef)
                emitStmt(state, mk_expr(arr_typ, :(=), lhs, arrdef))
                # generate read call
                read_call = mk_call(:__hps_data_source_TXT_read, [dsrc_id_var, lhs])
                return read_call
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hps_kmeans
                dprintln(env,"kmeans found ", inner_call)
                lib_call = mk_call(:__hps_kmeans, [lhs,inner_call.args[2], inner_call.args[3]])
                return lib_call 
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hps_LinearRegression
                dprintln(env,"LinearRegression found ", inner_call)
                lib_call = mk_call(:__hps_LinearRegression, [lhs,inner_call.args[2], inner_call.args[3]])
                return lib_call 
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hps_NaiveBayes
                dprintln(env,"NaiveBayes found ", inner_call)
                lib_call = mk_call(:__hps_NaiveBayes, [lhs,inner_call.args[2], inner_call.args[3], inner_call.args[4]])
                return lib_call 
            end
        end
    end
    
    return Expr(:not_matched)
end

function pattern_match_hps_dist_calls(state, env, lhs::Any, rhs::Any)
    return Expr(:not_matched)
end

# :(=) assignment (:(=), lhs, rhs)
function from_assignment(state, env, expr::Expr)
    local env_ = nextEnv(env)
    local head = expr.head
    local ast  = expr.args
    local typ  = expr.typ
    @assert length(ast)==2 "DomainIR: assignment nodes should have two arguments."
    local lhs = ast[1]
    local rhs = ast[2]
    lhs = toSymGen(lhs)
    
    # pattern match distributed calls that need domain-ir translation
    matched = pattern_match_hps_dist_calls(state, env, lhs, rhs)
    # matched is an expression, :not_matched head is used if not matched 
    if matched.head!=:not_matched
        return matched
    end
    
    rhs = from_expr(state, env_, rhs)
    rhstyp = typeOfOpr(state, rhs)
    dprintln(env, "from_assignment lhs=", lhs, " typ=", typ, " rhs.typ=", rhstyp)
    if typ != rhstyp && rhstyp != Any
        updateTyp(state, lhs, rhstyp)
        typ = rhstyp
    end
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
    dprintln(env,"normalized callname: ", fun_)
    result = translate_call(state, env, typ, :call, fun, args, fun_, args_)
    result
end

function translate_call(state, env, typ::DataType, head, oldfun::ANY, oldargs, fun::GlobalRef, args)
    translate_call_globalref(state, env, typ, head, oldfun, oldargs, fun, args)
end

function translate_call(state, env, typ::DataType, head, oldfun::ANY, oldargs, fun::Symbol, args)
    translate_call_symbol(state, env, typ, head, oldfun, oldargs, fun, args)
end

function translate_call(state, env, typ::DataType, head, oldfun::ANY, oldargs, fun::ANY, args)
    @dprintln(3,"unrecognized fun type ", fun, " type ", typeof(fun), " args ", args)
    oldargs = normalize_args(state, env, oldargs)
    mk_expr(typ, head, oldfun, oldargs...)
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
            @dprintln(3, "normalize_args got Void args[", i, "] = ", arg)
            emitStmt(state, arg)
        elseif isa(arg, Expr) || isa(arg, LambdaStaticData)
            typ = isa(arg, Expr) ? arg.typ : Any
            dprintln(env, "addGenSym with typ ", typ)
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
    fun = Base.resolve(fun, force=true)
    if is(fun.mod, API) || is(fun.mod, API.Stencil)
      return normalize_callname(state, env, fun.name, args)
    elseif is(fun.mod, Base.Random) && (is(fun.name, :rand!) || is(fun.name, :randn!))
        if is(fun.name, :rand!) 
            # splice!(args,3)
        end
        return (fun.name, args)
    else
        return (fun, args)
    end
end

# legacy v0.3
#=
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
                func = getfield(fun.mod, fun.name)  # should give back a function
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
=#

function normalize_callname(state::IRState, env, fun::TopNode, args)
    fun = fun.name
    if is(fun, :ccall)
        callee = lookupConstDefForArg(state, args[1])
        if isa(callee, QuoteNode) && in(callee.value, allocCalls)
            local realArgs = Any[]
            atype = args[4]
            elemtyp = elmTypOf(atype)
            push!(realArgs, elemtyp)
            dprintln(env, "normalize_callname: fun = :ccall args = ", args)
            for i = 6:2:length(args)
                if istupletyp(typeOfOpr(state, args[i]))
                    dprintln(env, "found tuple arg: ", args[i])
                    def = lookupConstDefForArg(state, args[i])
                    dprintln(env, "definition: ", def)
                    if isa(def, Expr) && is(def.head, :call) && (is(def.args[1], GlobalRef(Base, :tuple)) || is(def.args[1], TopNode(:tuple)))
                        dprintln(env, "definition is inlined")
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
            dprintln(env, "alloc call becomes: ", fun, " ", args)
        end
    end
    return (fun, args)
end

function normalize_callname(state::IRState, env, fun :: SymbolNode, args)
    def = lookupConstDefForArg(state, fun)
    if !is(def, nothing) && !isa(def, Expr)
        normalize_callname(state, env, def, args)
    else 
        return (fun, args)
    end
end

function normalize_callname(state::IRState, env, fun :: ANY, args)
    return (fun, args)
end

# if a definition of arr is getindex(a, ...), return select(a, ranges(...))
# otherwise return arr unchanged.
function inline_select(env, state, arr::SymAllGen)
    range_extra = Any[]

    # TODO: this requires safety check. Local lookups are only correct if free variables in the definition have not changed.
    def = lookupConstDef(state, arr)
    dprintln(env, "inline_select: arr = ", arr, " def = ", def)
    if !isa(def, Void)  
        if isa(def, Expr) 
            if is(def.head, :call) 
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
            elseif is(def.head, :select)
                arr = def
            end
        end
    end
    return arr
end

function inline_select(env, state, arr::ANY)
    return arr
end

function getElemTypeFromAllocExp(typExp::QuoteNode)
    return typExp.value
end

function getElemTypeFromAllocExp(typExp::DataType)
    return typExp
end

function getElemTypeFromAllocExp(typExp::GlobalRef)
    return getfield(typExp.mod, typExp.name)
end

function translate_call_alloc(state, env_, typ, typ_arg, args_in)
    typExp::Union{QuoteNode,DataType,GlobalRef}
    typExp = lookupConstDefForArg(state, typ_arg)
    elemTyp::DataType = getElemTypeFromAllocExp(typExp)
    args = normalize_args(state, env_, args_in)

    expr::Expr = mk_alloc(state, elemTyp, args)
    expr.typ = typ
    return expr
end

function translate_call_rangeshortcut(state, arg1::GenSym, arg2::QuoteNode)
    local ret::Union{SymAllGen,Expr,Int}
    ret = Expr(:null)
    if arg2.value==:stop || arg2.value==:start || arg2.value==:step
        rTyp = getType(arg1, state.linfo)
        rExpr = lookupConstDefForArg(state, arg1)
        if isrange(rTyp) && isa(rExpr, Expr)
            (start, step, final) = from_range(rExpr)
            fname = arg2.value
            if is(fname, :stop) 
                ret = final
            elseif is(fname, :start)
                ret = start
            else
                ret = step
            end
        end
    end
    #return Expr(:null)
    return ret
end

function translate_call_rangeshortcut(state, arg1::ANY, arg2::ANY)
    return Expr(:null)
    #return nothing
end

"""
A hack to avoid eval() since it affects type inference significantly
"""
function eval_dataType(typ::GlobalRef)
    return getfield(typ.mod, typ.name)
end

function eval_dataType(typ::DataType)
    return typ
end

"""
 translate a function call to domain IR if it matches Symbol.
 things handled as Symbols are: 
    those captured in ParallelAPI, or
    those that can be TopNode, or
    those from Core.Intrinsics, or
    those that are DomainIR specific, such as :alloc, or
    a few exceptions.
"""
function translate_call_symbol(state, env, typ::DataType, head, oldfun::ANY, oldargs, fun::Symbol, args::Array{Any,1})
    local env_ = nextEnv(env)
    local expr::Expr
    expr = Expr(:null)
    #expr = nothing
    dprintln(env, "translate_call fun=", fun, "::", typeof(fun), " args=", args, " typ=", typ)
    # new mainline Julia puts functions in Main module but PSE expects the symbol only
    #if isa(fun, GlobalRef) && fun.mod == Main
    #   fun = fun.name
    # end

    dprintln(env, "verifyMapOps -> ", verifyMapOps(state, fun, args))
    if verifyMapOps(state, fun, args) 
        args = normalize_args(state, env_, args)
        # checking first argument is only an estimate in case checking typ fails
        if typ == Any && length(args) > 0 
            atyp = typeOfOpr(state, args[1])
            if isArrayType(atyp)
                # try to fix return typ, because all map operators returns the same type as its first argument
                # except for comparisons
                typ = in(fun, compareOpSet) ? Bool : atyp
            end
        end
        if isArrayType(typ)
            return translate_call_mapop(state,env_,typ, fun, args)
        end
    end
    
    if is(fun, :map)
        return translate_call_map(state, env_, typ, args)
    elseif is(fun, :map!)
        return translate_call_map!(state, env_, typ, args)
    elseif is(fun, :reduce)
        return translate_call_reduce(state, env_, typ, args)
    elseif is(fun, :cartesianarray)
        return translate_call_cartesianarray(state, env_, typ, args)
    elseif is(fun, :runStencil)
        return translate_call_runstencil(state, env_, args)
    elseif is(fun, :parallel_for)
        return translate_call_parallel_for(state, env_, args)
# legacy v0.3
#    elseif in(fun, topOpsTypeFix) && is(typ, Any) && length(args) > 0
#        typ = translate_call_typefix(state, env, typ, fun, args) 
    elseif haskey(reduceOps, fun)
        dprintln(env, "haskey reduceOps ", fun)
        return translate_call_reduceop(state, env_, typ, fun, args)
    elseif is(fun, :arraysize)
        args = normalize_args(state, env_, args)
        dprintln(env,"got arraysize, args=", args)
        arr_size_expr::Expr = mk_arraysize(args...)
        arr_size_expr.typ = typ
        return arr_size_expr
    elseif is(fun, :alloc) || is(fun, :Array)
        return translate_call_alloc(state, env_, typ, args[1], args[2:end])
    elseif is(fun, :sitofp) # typefix hack!
        typ = args[1]
    elseif is(fun, :fpext) # typefix hack!
        #println("TYPEFIX ",fun," ",args)
        # a hack to avoid eval
        # typ = eval(args[1])
        typ = eval_dataType(args[1]) 
    elseif is(fun, :getindex) || is(fun, :setindex!) 
        expr = translate_call_getsetindex(state,env_,typ,fun,args)
    elseif is(fun, :getfield) && length(args) == 2
        # Shortcut range object access
        range_out::Union{SymAllGen,Expr,Int} = translate_call_rangeshortcut(state, args[1],args[2])
        if range_out!=Expr(:null)
            return range_out
        end
    elseif in(fun, ignoreSet)
    else
        dprintln(env,"function call not translated: ", fun, ", typeof(fun)=Symbol head = ", head, " oldfun = ", oldfun)
    end

    if expr.head==:null
    #if isa(expr, Void)
        if !is(fun, :ccall)
            if is(fun, :box) && isa(oldargs[2], Expr) # fix the type of arg[2] to be arg[1]
              oldargs[2].typ = typ
            end
            oldargs = normalize_args(state, env_, oldargs)
        end
        expr = mk_expr(typ, head, oldfun, oldargs...)
        if is(fun, :setfield!) && length(oldargs) == 3 && oldargs[2] == QuoteNode(:contents) #&& 
            #typeOfOpr(state, oldargs[1]) == Box
            # special handling for setting Box variables
            dprintln(env, "got setfield! with Box argument: ", oldargs)
            assert(isa(oldargs[1], SymbolNode))
            typ = typeOfOpr(state, oldargs[3])
            updateTyp(state, oldargs[1], typ)
            updateBoxType(state, oldargs[1], typ)
            # change setfield! to direct assignment
            expr = mk_expr(typ, :(=), oldargs[1].name, oldargs[3])
        elseif is(fun, :getfield) && length(oldargs) == 2 
            dprintln(env, "got getfield ", oldargs)
            if oldargs[2] == QuoteNode(:contents)
                return oldargs[1]
            elseif isa(oldargs[1], SymbolNode) && oldargs[1].name == symbol("#self#")
                fname = oldargs[2]
                assert(isa(fname, QuoteNode))
                dprintln(env, "lookup #self# closure field ", fname, " :: ", typeof(fname))
                esc = isEscapingVariable(fname.value, state.linfo) 
                dprintln(env, "isEscapingVariable = ", esc)
                ftyp = getType(fname.value, state.linfo)
                return SymbolNode(fname.value, ftyp) 
            end
        end
    else
    end
    return expr
end

# legacy v0.3
#=
function translate_call_typefix(state, env, typ, fun, args::Array{Any,1})
    dprintln(env, " args = ", args, " type(args[1]) = ", typeof(args[1]))
    local typ1    
    if is(fun, :cat_t)
        typ1 = isa(args[2], GlobalRef) ? getfield(args[2].mod, args[2].name) : args[2]
        @assert (isa(typ1, DataType)) "expect second argument to cat_t to be a type"
        dim1 = args[1]
        @assert (isa(dim1, Int)) "expect first argument to cat_t to be constant"
        typ1 = Array{typ1, dim1}
    else
        a1 = args[1]
        if typeof(a1) == GlobalRef
            a1 = getfield(a1.mod, a1.name)
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
=#

function translate_call_getsetindex(state, env, typ, fun::Symbol, args::Array{Any,1})
    dprintln(env, "got getindex or setindex!")
    args = normalize_args(state, env, args)
    arr = args[1]
    arrTyp = typeOfOpr(state, arr)
    dprintln(env, "arrTyp = ", arrTyp)
    if isrange(arrTyp) && length(args) == 2
        # shortcut range indexing if we can
        rExpr = lookupConstDefForArg(state, args[1])
        if isa(rExpr, Expr) && typ == Int 
            # only handle range of Int type 
            (start, step, final) = from_range(rExpr)
            return mk_expr(typ, :call, GlobalRef(Base, :add_int), start, mk_expr(typ, :call, GlobalRef(Base, :mul_int), mk_expr(typ, :call, GlobalRef(Base, :sub_int), args[2], 1), step))
        end
    elseif isArrayType(arrTyp)
      ranges = is(fun, :getindex) ? args[2:end] : args[3:end]
      expr = Expr(:null)
      dprintln(env, "ranges = ", ranges)
      try 
        if any(Bool[ ismask(state, range) for range in ranges])
            dprintln(env, "args is ", args)
            dprintln(env, "ranges is ", ranges)
            #newsize = addGenSym(Int, state.linfo)
            #newlhs = addGenSym(typ, state.linfo)
            etyp = elmTypOf(arrTyp)
            ranges = mk_ranges([rangeToMask(state, ranges[i], mk_arraysize(arr, i)) for i in 1:length(ranges)]...)
            dprintln(env, "ranges becomes ", ranges)
            if is(fun, :getindex) 
                # expr = mk_mmap([mk_select(arr, ranges)], DomainLambda(Type[etyp], Type[etyp], (linfo, as) -> [Expr(:tuple, as...)], LambdaVarInfo())) 
                expr = mk_select(arr, ranges)
                # TODO: need to calculate the correct result dimesion
                typ = arrTyp
            else
                args = Any[ inline_select(env, state, e) for e in Any[mk_select(arr, ranges), args[2]]]
                typs = Type[arrTyp, typeOfOpr(state, args[2])]
                (nonarrays, args, typs, f) = specialize(state, args, typs, as -> [Expr(:tuple, as[2])])
                elmtyps = Type[ isArrayType(t) ? elmTypOf(t) : t for t in typs ]
                linfo = LambdaVarInfo()
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
      catch err
        dprintln(env, "Exception caught during range conversion: ", err)
      end
    end
    return mk_expr(typ, :call, fun, args...)
end

# operator mapping over inputs
function translate_call_mapop(state, env, typ, fun::Symbol, args::Array{Any,1})
    # TODO: check for unboxed array type
    args = normalize_args(state, env, args)
    etyp = elmTypOf(typ) 
    if is(fun, :-) && length(args) == 1
        fun = :negate
    end
    typs = Type[ typeOfOpr(state, arg) for arg in args ]
    elmtyps = Type[ isArrayType(t) ? elmTypOf(t) : t for t in typs ]
    opr, reorder = specializeOp(mapOps[fun], elmtyps)
    typs = reorder(typs)
    args = reorder(args)
    dprintln(env,"translate_call_mapop: before specialize, opr=", opr, " args=", args, " typs=", typs)
    (nonarrays, args, typs, f) = specialize(state, args, typs, 
    as -> [Expr(:tuple, mk_expr(etyp, :call, opr, as...))])
    dprintln(env,"translate_call_mapop: after specialize, typs=", typs)
    elmtyps = Type[ isArrayType(t) ? elmTypOf(t) : t for t in typs ]
    # calculate escaping variables
    linfo = LambdaVarInfo()
    for i=1:length(nonarrays)
        # At this point, they are either symbol nodes, or constants
        if isa(nonarrays[i], SymbolNode)
            addEscapingVariable(nonarrays[i].name, nonarrays[i].typ, 0, linfo)
        end
    end
    domF = DomainLambda(elmtyps, [etyp], f, linfo)
    for i = 1:length(args)
        arg_ = inline_select(env, state, args[i])
        #if arg_ != args[i] && i != 1 && length(args) > 1
        #    error("Selector array must be the only array argument to mmap: ", args)
        #end
        args[i] = arg_
    end
    expr::Expr = endswith(string(fun), '!') ? mk_mmap!(args, domF) : mk_mmap(args, domF)
    expr = mmapRemoveDupArg!(expr)
    expr.typ = typ
    return expr
end

"""
Run type inference and domain process over the income function object.
Return the result AST with a modified return statement, namely, return
is changed to Expr(:tuple, retvals...)
"""
function get_ast_for_lambda(state, env, func::Union{LambdaStaticData,SymbolNode,Expr}, argstyp)
    if isa(func, SymbolNode) && func.typ <: Function
        # function/closure support is changed in julia 0.5
        lambda = func.typ #.name.primary
    elseif isa(func, Expr) && is(func.head, :new)
        lambda = func.args[1]
        if isa(lambda, GlobalRef)
            lambda = getfield(lambda.mod, lambda.name)
        end
    else
        lambda = func
    end
    (ast, aty) = lambdaTypeinf(lambda, tuple(argstyp...))
    dprintln(env, "type inferred AST = ", ast)
    dprintln(env, "aty = ", aty)
    # recursively process through domain IR with new state and env
    ast = from_lambda(state, env, ast, func)
    body::Expr = ast.args[3]
    linfo = lambdaExprToLambdaVarInfo(ast)
    lastExp::Expr = body.args[end]
    assert(is(lastExp.head, :return))
    args1_typ::DataType = Void
    if length(lastExp.args) > 0 
        args1 = lastExp.args[1]
        if isa(args1, SymbolNode) || isa(args1, GenSym)
            args1_typ = getType(args1, linfo)
            dprintln(env, "lastExp=", lastExp, " args1=", args1, " typ=", args1_typ)
        end
    end
    # modify the last return statement if it's a tuple
    if istupletyp(args1_typ)
        # take a shortcut if the second last statement is the tuple creation
        exp = body.args[end-1]
        if exp.head == :(=) && exp.args[1] == args1 && isa(exp.args[2], Expr) &&
           exp.args[2].head == :call && exp.args[2].args[1] == TopNode(:tuple)
            dprintln(env, "second last is tuple assignment, we'll take shortcut")
            pop!(body.args)
            exp.head = :tuple
            exp.args = exp.args[2].args[2:end]
        else
            # create tmp variables to store results
            tvar = args1
            typs::SimpleVector = args1_typ.parameters
            nvar = length(typs)
            retNodes = GenSym[ addGenSym(t, linfo) for t in typs ]
            retExprs = Array(Expr, length(retNodes))
            for i in 1:length(retNodes)
                n = retNodes[i]
                t = typs[i]
                retExprs[i] = mk_expr(t, :(=), n, mk_expr(t, :call, GlobalRef(Base, :getfield), tvar, i))
            end
            lastExp.head = retExprs[1].head
            lastExp.args = retExprs[1].args
            lastExp.typ  = retExprs[1].typ
            for i = 2:length(retExprs)
                push!(body.args, retExprs[i])
            end
            push!(body.args, mk_expr(typs, :tuple, retNodes...))
            ast = LambdaVarInfoToLambdaExpr(linfo, body)
        end
    else
        lastExp.head = :tuple
    end
    if aty == Any
        aty = args1_typ
    end
    dprintln(env, "aty becomes ", aty)
    lastExp.typ = aty
    body.typ = aty
    return ast, aty
end

"""
Lookup a function object for the given argument (variable),
infer its type and return the result ast together with return type.
"""
function get_lambda_for_arg(state, env, func::SymNodeGen, argstyp)
    dprintln(env, "get_lambda_for_arg: lookup ", func)
    lambda = lookupConstDefForArg(state, func)
    dprintln(env, "get_lambda_for_arg: got ", lambda)
    get_ast_for_lambda(state, env, lambda, argstyp)
end

function get_lambda_for_arg(state, env, func::GlobalRef, argstyp)
    #if(isdefined(ParallelAccelerator, func))
    #    m = methods(getfield(ParallelAccelerator, func), tuple(argstyp...))
    #else
    m = methods(getfield(func.mod, func.name), tuple(argstyp...))
    dprintln(env,"get_lambda_for_arg: ", func, " methods=", m, " argstyp=", argstyp)
    assert(length(m) > 0)
    get_ast_for_lambda(state, env, m[1].func.code, argstyp)
end

#map with a generic function 
function translate_call_map(state, env, typ, args::Array{Any,1})
    # equivalent to creating an array first, then map! with indices.
    dprintln(env, "got map args=", args)
    # need to retrieve map lambda from inits, since it is already moved out.
    nargs = length(args)
    args = normalize_args(state, env, args)
    @assert (nargs >= 2) "Expect 2 or more arguments to map, but got " * string(args[2:end])
    argtyps = DataType[ typeOfOpr(state, arg) for arg in args[2:end] ]
    dprintln(env, "argtyps =", argtyps)
    inptyps = DataType[ isArrayType(t) ? elmTypOf(t) : t for t in argtyps ]
    (ast, ety) = get_lambda_for_arg(state, env, args[1], inptyps) 
    etys = istupletyp(ety) ? [ety.parameters...] : DataType[ety]
    dprintln(env, "etys = ", etys)
    # assume return dimension is the same as the first array argument
    rdim = ndims(argtyps[1])
    rtys = DataType[ Array{t, rdim} for t in etys ]
    linfo = lambdaExprToLambdaVarInfo(ast)
    params = getParamsNoSelf(linfo)
    body = ast.args[3]
    bodyF = (plinfo, args) -> begin
        ldict = CompilerTools.LambdaHandling.mergeLambdaVarInfo(plinfo, linfo)
        ldict = merge(ldict, Dict{SymGen,Any}(zip(params, args)))
        ret = replaceExprWithDict(body, ldict).args
        ret
    end
    domF = DomainLambda(inptyps, etys, bodyF, linfo)
    expr::Expr = mk_mmap(args[2:end], domF)
    expr.typ = length(rtys) == 1 ? rtys[1] : to_tuple_type(tuple(rtys...))
    return expr
end

# translate map! with a generic function 
# Julia's Base.map! has a slightly different semantics than our internal mmap!.
# First of all, it doesn't allow multiple desntiation array. Secondly,
# when there is more than 1 array argument, the first argument is treated as 
# destination only, and the rest are inputs.
function translate_call_map!(state, env, typ, args::Array{Any,1})
    # equivalent to creating an array first, then map! with indices.
    dprintln(env, "got map! args=", args)
    # need to retrieve map lambda from inits, since it is already moved out.
    nargs = length(args)
    args = normalize_args(state, env, args)
    @assert (nargs >= 2) "Expect 2 or more arguments to map!, but got " * string(args)
    argtyps = DataType[ typeOfOpr(state, arg) for arg in args[2:end] ]
    dprintln(env, "argtyps =", argtyps)
    inptyps = DataType[ isArrayType(t) ? elmTypOf(t) : t for t in argtyps ]
    (ast, ety) = get_lambda_for_arg(state, env, args[1], nargs == 2 ? inptyps : inptyps[2:end]) 
    etys = istupletyp(ety) ? [ety.parameters...] : DataType[ety]
    dprintln(env, "etys = ", etys)
    # assume return dimension is the same as the first array argument
    rdim = ndims(argtyps[1])
    rtys = DataType[ Array{t, rdim} for t in etys ]
    linfo = lambdaExprToLambdaVarInfo(ast)
    params = getParamsNoSelf(linfo)
    body = ast.args[3]
    bodyF = (plinfo, args) -> begin
        ldict = CompilerTools.LambdaHandling.mergeLambdaVarInfo(plinfo, linfo)
        # if more than one arguments, the first argument (from destination array) is ignored
        if length(args) > 1
            args = args[2:end]
        end
        ldict = merge(ldict, Dict{SymGen,Any}(zip(params, args)))
        ret = replaceExprWithDict(body, ldict).args
        ret
    end
    domF = DomainLambda(inptyps, etys, bodyF, linfo)
    expr::Expr = mk_mmap!(args[2:end], domF) 
    expr.typ = length(rtys) == 1 ? rtys[1] : to_tuple_type(tuple(rtys...))
    return expr
end

# legacy v0.3
#=
function translate_call_checkbounds(state, env, args::Array{Any,1})
    args = normalize_args(state, env, args)
    typ = typeOfOpr(state, args[1])
    local expr::Expr
    if isarray(typ)
        typ_second_arg = typeOfOpr(state, args[2])
        if isarray(typ_second_arg) || isbitarray(typ_second_arg)
            expr = mk_expr(Bool, :assert, mk_expr(Bool, :call, GlobalRef(Base, :(===)), mk_expr(Int64, :call, GlobalRef(Base,:arraylen), args[1]), mk_expr(Int64, :call, GlobalRef(Base,:arraylen), args[2])))
        else
            @dprintln(0, args[2], " typ_second_arg = ", typ_second_arg)
            error("Unhandled bound in checkbounds: ", args[2])
        end
    elseif isIntType(typ)
        if isIntType(typeOfOpr(state, args[2]))
            expr = mk_expr(Bool, :assert, mk_expr(Bool, :call, TopNode(:sle_int), convert(typ, 1), args[2]),
            mk_expr(Bool, :call, TopNode(:sle_int), args[2], args[1]))
        elseif isa(args[2], SymbolNode) && (isUnitRange(args[2].typ) || isStepRange(args[2].typ))
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
=#

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
    local kernelExp_var::SymNodeGen = args[1]
    local bufs = Any[]
    local bufstyp = Any[]
    local i
    for i = 2:nargs
        oprTyp = typeOfOpr(state, args[i])
        if isArrayType(oprTyp)
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
    if is(borderExp, nothing)
        borderExp = QuoteNode(:oob_skip)
    else
        borderExp = lookupConstDefForArg(state, borderExp)
    end
    dprintln(env, "stencil bufstyp = ", to_tuple_type(tuple(bufstyp...)))
    # assert(isa(kernelExp, SymbolNode) || isa(kernelExp, GenSym))
    (kernelExp, ety) = get_lambda_for_arg(state, env, kernelExp_var, tuple(bufstyp...))
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
    local dimExp_var::SymNodeGen = args[end]     # last argument is the dimension tuple
    
    dimExp_e::Expr = lookupConstDefForArg(state, dimExp_var)
    dprintln(env, "dimExp = ", dimExp_e, " head = ", dimExp_e.head, " args = ", dimExp_e.args)
    assert(is(dimExp_e.head, :call) && is(dimExp_e.args[1], TopNode(:tuple)))
    dimExp = dimExp_e.args[2:end]
    ndim = length(dimExp)   # num of dimensions
    argstyp = Any[ Int for i in 1:ndim ] 
    
    (ast, ety) = get_lambda_for_arg(state, env, args[1], argstyp)     # first argument is the lambda
    etys = istupletyp(ety) ? [ety.parameters...] : DataType[ety]
    dprintln(env, "etys = ", etys)
    @assert all([ isa(t, DataType) for t in etys ]) "cartesianarray expects static type parameters, but got "*dump(etys) 
    # create tmp arrays to store results
    arrtyps = Type[ Array{t, ndim} for t in etys ]
    dprintln(env, "arrtyps = ", arrtyps)
    tmpNodes = Array(Any, length(arrtyps))
    # allocate the tmp array
    for i = 1:length(arrtyps)
        arrdef = type_expr(arrtyps[i], mk_alloc(state, etys[i], dimExp))
        tmparr = addGenSym(arrtyps[i], state.linfo)
        updateDef(state, tmparr, arrdef)
        emitStmt(state, mk_expr(arrtyps[i], :(=), tmparr, arrdef))
        tmpNodes[i] = tmparr
    end
    # produce a DomainLambda
    body::Expr = ast.args[3]
    linfo = lambdaExprToLambdaVarInfo(ast)
    params = getParamsNoSelf(linfo)
    bodyF = (plinfo, args) -> begin
        #bt = backtrace() ;
        #s = sprint(io->Base.show_backtrace(io, bt))
        #@dprintln(3, "bodyF backtrace ")
        #@dprintln(3, s)
        ldict = CompilerTools.LambdaHandling.mergeLambdaVarInfo(plinfo, linfo)
        #@dprintln(2,"cartesianarray body = ", body, " type = ", typeof(body))
        ldict = merge(ldict, Dict{SymGen,Any}(zip(params, args[1+length(etys):end])))
        # @dprintln(2,"cartesianarray idict = ", ldict)
        ret = replaceExprWithDict(body, ldict).args
        #@dprintln(2,"cartesianarray ret = ", ret)
        ret
    end
    domF = DomainLambda(vcat(etys, argstyp), etys, bodyF, linfo)
    expr::Expr = mk_mmap!(tmpNodes, domF, true)
    expr.typ = length(arrtyps) == 1 ? arrtyps[1] : to_tuple_type(tuple(arrtyps...))
    dprintln(env, "cartesianarray return type = ", expr.typ)
    return expr
end

function translate_call_reduce(state, env, typ, args::Array{Any,1})
    nargs = length(args)
    @assert (nargs >= 3) "expect at least 3 arguments to reduce, but got " * args
    args = normalize_args(state, env, args)
    dprintln(env, "translate_call_reduce: args = ", args)
    fun = args[1]
    neutralelt = args[2] 
    arr = args[3]
    # element type is the same as typ
    arrtyp = typeOfOpr(state, arr)
    etyp = elmTypOf(arrtyp) 
    ntyp = typeOfOpr(state, neutralelt)
    @assert (ntyp == etyp) "expect neutral value " * neutralelt * " to be " * elt * " type but got " * ntyp
    # infer the type of the given lambda
    inptyps = DataType[etyp, etyp]
    (ast, ety) = get_lambda_for_arg(state, env, fun, inptyps)
    @assert (ety == etyp) "expect return type of reduce function to be " * etyp * " but got " * ety
    linfo = lambdaExprToLambdaVarInfo(ast)
    params = getParamsNoSelf(linfo)
    body = ast.args[3]
    bodyF = (plinfo, args) -> begin
        ldict = CompilerTools.LambdaHandling.mergeLambdaVarInfo(plinfo, linfo)
        ldict = merge(ldict, Dict{SymGen,Any}(zip(params, args)))
        ret = replaceExprWithDict(body, ldict).args
        ret
    end
    red_dim = []
    neutral = neutralelt
    outtyp = etyp
    domF = DomainLambda([outtyp, outtyp], [outtyp], bodyF, linfo)
    # turn reduce(z, getindex(a, ...), f) into reduce(z, select(a, ranges(...)), f)
    arr = inline_select(env, state, arr)
    expr = mk_reduce(neutral, arr, domF)
    expr.typ = outtyp
    return expr
end

function translate_call_reduceop(state, env, typ, fun::Symbol, args::Array{Any,1})
    args = normalize_args(state, env, args)
    dprintln(env, "translate_call_reduceop: args = ", args)
    arr = args[1]
    if length(args) > 2
        error("DomainIR: expect only 1 or 2 arguments to reduction function ", fun, ", but got ", args)
    end
    # element type is the same as typ
    arrtyp = typeOfOpr(state, arr)
    etyp = elmTypOf(arrtyp) 
    neutralelt = convert(etyp, (reduceNeutrals[fun])(etyp))
    fun = reduceOps[fun]
    if length(args) == 2
        etyp    = arrtyp.parameters[1]
        num_dim = arrtyp.parameters[2]
        red_dim = [args[2]]
        dimExp = Any[ mk_expr(arrtyp, :call, TopNode(:select_value), 
                         mk_expr(Bool, :call, TopNode(:eq_int), red_dim[1], dim), 
                         1, mk_arraysize(arr, dim)) for dim = 1:num_dim ]
        neutral = DomainLambda([arrtyp], [],  
                  (linfo,lhs)->Any[ Expr(:(=), isa(lhs[1], SymbolNode) ? lhs[1].name : lhs[1], mk_alloc(state, etyp, dimExp)),
                                    mk_mmap!(lhs, DomainLambda([etyp], [etyp], (linfo, x) -> [Expr(:tuple, neutralelt)], LambdaVarInfo())),
                                    Expr(:tuple) ], LambdaVarInfo())
        opr, reorder = specializeOp(fun, [etyp])
        # ignore reorder since it is always id function
        f = (linfo, as) -> [Expr(:tuple, mk_mmap!(as, DomainLambda([etyp, etyp], [etyp], 
                                             (linfo,as)->[Expr(:tuple, mk_expr(etyp, :call, opr, as...))], LambdaVarInfo())))]
        outtyp = arrtyp
    else
        red_dim = []
        neutral = neutralelt
        opr, reorder = specializeOp(fun, [etyp])
        # ignore reorder since it is always id function
        f = (linfo,as) -> [Expr(:tuple, mk_expr(etyp, :call, opr, as...))]
        outtyp = etyp
    end
    domF = DomainLambda([outtyp, outtyp], [outtyp], f, LambdaVarInfo())
    # turn reduce(z, getindex(a, ...), f) into reduce(z, select(a, ranges(...)), f)
    arr = inline_select(env, state, arr)
    expr = mk_reduce(neutral, arr, domF, red_dim...)
    expr.typ = outtyp
    return expr
end

function translate_call_fill!(state, env, typ, args::Array{Any,1})
    args = normalize_args(state, env, args)
    @assert length(args)==2 "fill! should have 2 arguments"
    arr = args[1]
    ival = args[2]
    typs = Type[typeOfOpr(state, arr)]
    linfo = LambdaVarInfo()
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
    linfo = lambdaExprToLambdaVarInfo(ast)
    assert(isa(body, Expr) && is(body.head, :body))
    # Remove return statement
    pop!(body.args)
    bodyF = (plinfo, args) -> begin
        ldict = CompilerTools.LambdaHandling.mergeLambdaVarInfo(plinfo, linfo)
        # ldict = merge(ldict, Dict{SymGen,Any}(zip(loopvars, etys)))
        ret = replaceExprWithDict(body, ldict).args
        ret
    end
    domF = DomainLambda(etys, etys, bodyF, linfo)
    return mk_parallel_for(loopvars, ranges, domF)
end

# translate a function call to domain IR if it matches GlobalRef.
function translate_call_globalref(state, env, typ::DataType, head, oldfun::ANY, oldargs, fun::GlobalRef, args)
    local env_ = nextEnv(env)
    expr = nothing
    dprintln(env, "translate_call fun ", fun, "::", typeof(fun), " args=", args, " typ=", typ)
    # new mainline Julia puts functions in Main module but PSE expects the symbol only
    #if isa(fun, GlobalRef) && fun.mod == Main
    #   fun = fun.name
    # end
    if is(fun.mod, Core.Intrinsics) || (is(fun.mod, Core) && 
       (is(fun.name, :Array) || is(fun.name, :arraysize) || is(fun.name, :getfield)))
        expr = translate_call_symbol(state, env, typ, head, fun, oldargs, fun.name, args)
    elseif is(fun.mod, Core) && is(fun.name, :convert)
        # fix type of convert
        args = normalize_args(state, env_, args)
        if isa(args[1], Type)
            typ = args[1]
        end
    elseif is(fun.mod, Base) 
        if is(fun.name, :afoldl) && haskey(afoldlDict, typeOfOpr(state, args[1]))
            opr, reorder = specializeOp(afoldlDict[typeOfOpr(state, args[1])], [typ, typ])
            dprintln(env, "afoldl operator detected = ", args[1], " opr = ", opr)
            expr = Base.afoldl((x,y)->mk_expr(typ, :call, opr, reorder([x, y])...), args[2:end]...)
            dprintln(env, "translated expr = ", expr)
        elseif is(fun.name, :copy)
            args = normalize_args(state, env_, args[1:1])
            dprintln(env,"got copy, args=", args)
            expr = mk_copy(args[1])
            expr.typ = typ
    # legacy v0.3
    #=
        elseif is(fun.name, :checkbounds)
            dprintln(env, "got ", fun.name, " args = ", args)
            if length(args) == 2
                expr = translate_call_checkbounds(state,env_,args) 
            end
    =#
        elseif is(fun.name, :getindex) || is(fun.name, :setindex!) # not a domain operator, but still, sometimes need to shortcut it
            expr = translate_call_getsetindex(state,env_,typ,fun.name,args)
    # legacy v0.3
    #    elseif is(fun.name, :assign_bool_scalar_1d!) || # args = (array, scalar_value, bitarray)
    #           is(fun.name, :assign_bool_vector_1d!)    # args = (array, getindex_bool_1d(array, bitarray), bitarray)
    #        expr = translate_call_assign_bool(state,env_,typ,fun.name, args) 
        elseif is(fun.name, :fill!)
            return translate_call_fill!(state, env_, typ, args)
        elseif is(fun.name, :_getindex!) # see if we can turn getindex! back into getindex
            if isa(args[1], Expr) && args[1].head == :call && args[1].args[1] == TopNode(:ccall) && 
                (args[1].args[2] == :jl_new_array ||
                (isa(args[1].args[2], QuoteNode) && args[1].args[2].value == :jl_new_array))
                expr = mk_expr(typ, :call, :getindex, args[2:end]...)
            end
        elseif fun.name==:println || fun.name==:print # fix type for println
            typ = Void
        end
    elseif is(fun.mod, Base.Broadcast)
        if is(fun.name, :broadcast_shape)
            dprintln(env, "got ", fun.name)
            args = normalize_args(state, env_, args)
            expr = mk_expr(typ, :assertEqShape, args...)
        end
    elseif is(fun.mod, Base.Random) #skip, let cgen handle it
    elseif is(fun.mod, Base.LinAlg) || is(fun.mod, Base.LinAlg.BLAS) #skip, let cgen handle it
    elseif is(fun.mod, Base.Math)
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
        if isa(gf, Function) && !is(fun.mod, Core) # fun != GlobalRef(Core, :(===))
            dprintln(env,"function to offload: ", fun, " methods=", methods(gf))
            _accelerate(gf, tuple(args_typ...))
        else
            dprintln(env,"function ", fun, " not offloaded.")
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

function from_return(state, env, expr)
    local env_ = nextEnv(env)
    local head = expr.head
    local typ  = expr.typ
    local args = normalize_args(state, env, expr.args)
    # fix return type, the assumption is there is either 0 or 1 args.
    typ = length(args) > 0 ? typeOfOpr(state, args[1]) : Void
    return mk_expr(typ, head, args...)
end

function from_expr_tiebreak(state::IRState, env::IREnv, ast :: Expr)
    asts::Array{Any,1} = [ast]
    res::Array{Any,1} = [from_expr(state, env, ast) for ast in asts ]
    return res[1]
end

function from_expr(function_name::AbstractString, cur_module :: Module, ast :: Expr)
    @dprintln(2,"DomainIR translation function = ", function_name, " on:")
    @dprintln(2,ast)
    res = from_expr_tiebreak(emptyState(), newEnv(cur_module), ast) 
    @dprintln(2,"DomainIR translation returns:")
    @dprintln(2,res)
    return res
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

function from_expr(state::IRState, env::IREnv, ast::GlobalRef)
    if ccall(:jl_is_const, Int32, (Any, Any), ast.mod, ast.name) == 1
        def = getfield(ast.mod, ast.name)
        if isbits(def) && !isa(def, IntrinsicFunction) && !isa(def, Function)
            return def
        end
    end
    @dprintln(2, " not handled ", ast)
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::Union{SymbolNode,Symbol})
    name = isa(ast, SymbolNode) ? ast.name : ast
    # if it is global const, we replace it with its const value
    def = lookupDefInAllScopes(state, name)
    if is(def, nothing) && isdefined(env.cur_module, name) && ccall(:jl_is_const, Int32, (Any, Any), env.cur_module, name) == 1
        def = getfield(env.cur_module, name)
        if isbits(def) && !isa(def, IntrinsicFunction) && !isa(def, Function)
            return def
        end
    end
    typ = typeOfOpr(state, ast)
    if isa(ast, SymbolNode) && ast.typ != typ
        @dprintln(2, " SymbolNode ", ast, " updates its type to ", typ)
        return SymbolNode(ast.name, typ)
    end
    @dprintln(2, " not handled ", ast)
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::Expr)
    dprint(env,"from_expr: Expr")
    local head = ast.head
    local args = ast.args
    local typ  = ast.typ
    @dprintln(2, " :", head)
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
    # legacy v0.3
    #elseif is(head, :call1)
    #    return from_call(state, env, ast)
        # TODO?: tuple
    # legacy v0.3
    # :method is not expected here
    #=
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
    =#
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
        throw(string("ParallelAccelerator.DomainIR.from_expr: unknown Expr head :", head))
    end
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::LabelNode)
    # clear defs for every basic block.
    state.defs = Dict{Union{Symbol,Int}, Any}()
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::ANY)
    @dprintln(2, " not handled ", ast)
    return ast
end

type DirWalk
    callback
    cbdata
end

function AstWalkCallback(x :: Expr, dw :: DirWalk, top_level_number, is_top_level, read)
    @dprintln(3,"DomainIR.AstWalkCallback ", x)
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(3,"DomainIR.AstWalkCallback ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end

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
        assert(length(args) == 3 || length(args) == 4)
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
            # @dprintln(3, "ranges[i] = ", ranges[i], " ", typeof(ranges[i]))
            if ((isa(ranges[i], Expr) && (ranges[i].head == :range || ranges[i].head == :tomask)))
                for j = 1:length(ranges[i].args)
                    ranges[i].args[j] = AstWalker.AstWalk(ranges[i].args[j], AstWalkCallback, dw)
                end
            else
                assert(isa(ranges[i], Integer) || isa(ranges[i], SymbolNode) || isa(ranges[i], GenSym))
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


    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function AstWalkCallback(x :: DomainLambda, dw :: DirWalk, top_level_number, is_top_level, read)
    @dprintln(3,"DomainIR.AstWalkCallback ", x)
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(3,"DomainIR.AstWalkCallback ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end
    @dprintln(3,"DomainIR.AstWalkCallback for DomainLambda", x)
    return x
end

function AstWalkCallback(x :: ANY, dw :: DirWalk, top_level_number, is_top_level, read)
    @dprintln(3,"DomainIR.AstWalkCallback ", x)
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(3,"DomainIR.AstWalkCallback ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end
    
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function AstWalk(ast :: ANY, callback, cbdata :: ANY)
    @dprintln(3,"DomainIR.AstWalk ", ast)
    dw = DirWalk(callback, cbdata)
    AstWalker.AstWalk(ast, AstWalkCallback, dw)
end

function dir_live_cb(ast :: Expr, cbdata :: ANY)
    @dprintln(4,"dir_live_cb ")

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

        @dprintln(3, ":mmap ", expr_to_process)
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

        @dprintln(3, ":mmap! ", expr_to_process)
        return expr_to_process
    elseif head == :reduce
        expr_to_process = Any[]

        assert(length(args) == 3 || length(args) == 4)
        zero_val = args[1]
        input_array = args[2]
        dl = args[3]
        if !isa(zero_val, DomainLambda) 
            push!(expr_to_process, zero_val)
        end
        push!(expr_to_process, input_array)
        assert(isa(dl, DomainLambda))
        for (v, d) in dl.linfo.escaping_defs
            push!(expr_to_process, v)
        end

        @dprintln(3, ":reduce ", expr_to_process)
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

        @dprintln(3, ":stencil! ", expr_to_process)
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

        @dprintln(3, ":parallel_for ", expr_to_process)
        return expr_to_process
    elseif head == :assertEqShape
        assert(length(args) == 2)
        #@dprintln(3,"liveness: assertEqShape ", args[1], " ", args[2], " ", typeof(args[1]), " ", typeof(args[2]))
        expr_to_process = Any[]
        push!(expr_to_process, toSymGen(args[1]))
        push!(expr_to_process, toSymGen(args[2]))
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
        # arrayref only add read access
    elseif head == :call
        if args[1]==TopNode(:arrayref) || args[1]==TopNode(:arraysize)
            expr_to_process = Any[]
            for i = 2:length(args)
                push!(expr_to_process, args[i])
            end
            return expr_to_process
        end
    end

    return nothing
end

function dir_live_cb(ast :: KernelStat, cbdata :: ANY)
    @dprintln(4,"dir_live_cb ")
    return Any[]
end

function dir_live_cb(ast :: ANY, cbdata :: ANY)
    @dprintln(4,"dir_live_cb ")
    return nothing
end

function dir_alias_cb(ast::Expr, state, cbdata)
    @dprintln(4,"dir_alias_cb ")

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
        toSymGen = x -> isa(x, SymbolNode) ? x.name : x
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
        @dprintln(3, "AA: rotateNum = ", krnStat.rotateNum, " out of ", length(bufs), " input bufs")
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
        toSymGen = x -> isa(x, SymbolNode) ? x.name : x
        return AliasAnalysis.lookup(state, toSymGen(args[1]))
    elseif is(head, :arraysize)
        return AliasAnalysis.NotArray
    elseif is(head, :tuple)
        return AliasAnalysis.NotArray
    elseif is(head, :alloc)
        return AliasAnalysis.next_node(state)
    elseif is(head, :copy)
        return AliasAnalysis.next_node(state)
    end

    return nothing

end

function dir_alias_cb(ast::ANY, state, cbdata)
    @dprintln(4,"dir_alias_cb ")
end

end
