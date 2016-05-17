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
using Core: Box, IntrinsicFunction

import ..H5SizeArr_t
import ..SizeArr_t
import .._accelerate
import ..show_backtrace

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
mk_arraysize(arr, dim) = TypedExpr(Int64, :call, GlobalRef(Base, :arraysize), arr, dim)
mk_sizes(arr) = Expr(:sizes, arr)
mk_strides(arr) = Expr(:strides, arr)
mk_alloc(typ, s) = Expr(:alloc, typ, s)
mk_call(fun,args) = Expr(:call, fun, args...)
mk_copy(arr) = Expr(:call, GlobalRef(Base, :copy), arr)
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

export DomainLambda, KernelStat, AstWalk, arraySwap, lambdaSwapArg, isbitmask

"""
A representation for anonymous lambda used in domainIR.
   inputs:  types of input tuple
   outputs: types of output tuple
   lambda:  AST of the lambda, with :lambda as head
   linfo:   LambdaVariableInfo of the lambda
"""
type DomainLambda
    body    :: Expr
    inputs  :: Array{Type, 1}
    outputs :: Array{Type, 1}
    linfo   :: LambdaVarInfo

    function DomainLambda(li::LambdaVarInfo, body::Expr)
        @assert (body.head == :body) "Expects Expr(:body, ...) but got " * string(body)
        @dprintln(3, "create DomainLambda with body = ", body, " and linfo = ", li)
        inps = Type[getType(x, li) for x in getInputParameters(li)]
        rtyp = getReturnType(li)
        outs = (rtyp == nothing) ? [] : (isTupleType(rtyp) ? Type[t for t in rtyp.parameters] : Type[rtyp])
        new(body, inps, outs, li)
    end

    function DomainLambda(ast)
        li, body = lambdaToLambdaVarInfo(ast) 
        DomainLambda(li, body)
    end

    """
    Create a DomainLambda from simple expression (those with no local variables) by the
    input types, output types, and a body function that maps from parameters 
    to an Array of Exprs. Note that the linfo passed in is not the linfo for the DomainLambda, 
    but rather the linfo in which variable types can be looked up.
    """
    function DomainLambda(inps::Array{Type,1}, outs::Array{Type,1}, f::Function, linfo::LambdaVarInfo)
        params = [ gensym("x") for t in inps ]
        li = LambdaVarInfo()
        paramS = Array(Any, length(params))
        for i in 1:length(params)
            addLocalVariable(params[i], inps[i], 0, li)
            paramS[i] = toRHSVar(params[i], inps[i], li)
        end
        setInputParameters(params, li)
        body = f(paramS)
        @dprintln(3, "DomainLambda will have body = ", body)
        nouts = length(outs)
        setReturnType(nouts > 1 ? Tuple{outs...} : outs[1], li)
        bodyExpr = getBody(body, getReturnType(li))
        vars = countVariables(bodyExpr)
        @dprintln(3, "countVariables = ", vars)
        for v in vars
            if !isLocalVariable(v, li) 
                v = lookupVariableName(v, linfo)
                if v != Symbol("#self#")
                    addEscapingVariable(v, getType(v, linfo), getDesc(v, linfo), li)
                end
            end
        end
        # always try to fix the body's return statement to be :tuple
        lastExpr = body[end]
        @assert (isa(lastExpr, Expr))
        if lastExpr.head == :tuple || lastExpr.head == :return
            lastExpr.head = :tuple
            @assert (length(lastExpr.args) == nouts) "Expect last statement to return " * string(nouts) * " results of type " * string(outs) * ", but got " * string(lastExpr)
        else 
            # assume lastExpr to be the single return value
            body[end] = Expr(:tuple, lastExpr)
        end
        body[end].typ = li.return_type
        #ast_body = Expr(:body)
        #ast_body.args = body
        #ast_body.typ = li.return_type
        dprintln(3, "li = ", li)
        DomainLambda(li, bodyExpr)
    end

end

# We need to extend several functions to handle DomainLambda specifically
import Base.show
import CompilerTools.Helper.isfunctionhead
import CompilerTools.LambdaHandling: lambdaToLambdaVarInfo, getBody

function show(io::IO, f::DomainLambda)
    show(io, getInputParameters(f.linfo))
    show(io, " -> ")
    show(io, "[")
    show(io, [lookupVariableName(x, f.linfo) for x in getLocalVariables(f.linfo)])
    show(io, "]")
    show(io, f.body)
end

isfunctionhead(x::DomainLambda) = true
getBody(x::DomainLambda) = x.body
function lambdaToLambdaVarInfo(x::DomainLambda)
    return x.linfo, x.body
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
    linfo = LambdaVarInfo(f.linfo)
    params = arraySwap(getInputParameters(linfo), i, j)
    setInputParameters(params, linfo)
    return DomainLambda(linfo, f.body)
end

type IRState
    linfo  :: LambdaVarInfo
    defs   :: Dict{LHSVar, Any}  # stores local definition of LHS = RHS
    escDict :: Dict{Symbol, RHSVar} # mapping function closure fieldname to escaping variable
    boxtyps:: Dict{LHSVar, Any}  # finer types for those have Box type
    stmts  :: Array{Any, 1}
    parent :: Union{Void, IRState}
end

emptyState() = IRState(LambdaVarInfo(), Dict{LHSVar,Any}(), Dict{Symbol,RHSVar}(), Dict{LHSVar,Any}(), Any[], nothing)
newState(linfo, defs, escDict, state::IRState) = IRState(linfo, defs, escDict, Dict{LHSVar,Any}(), Any[], state)

"""
Update the type of a variable.
"""
function updateTyp(state::IRState, s, typ)
    setType(s, typ, state.linfo)
end

function updateBoxType(state::IRState, s::RHSVar, typ)
    state.boxtyps[toLHSVar(s)] = typ
end

function getBoxType(state::IRState, s::RHSVar)
    state.boxtyps[toLHSVar(s)] 
end

"""
Update the definition of a variable.
"""
function updateDefInternal(state::IRState, s::LHSVar, rhs)
    #@dprintln(3, "updateDef: s = ", s, " rhs = ", rhs, " typeof s = ", typeof(s))
    # This assert is Julia version dependent.
#    @assert ((isa(s, GenSym) && isLocalGenSym(s, state.linfo)) ||
#    (isa(s, Symbol) && isLocalVariable(s, state.linfo)) ||
#    (isa(s, Symbol) && isInputParameter(s, state.linfo)) ||
#    (isa(s, Symbol) && isEscapingVariable(s, state.linfo))) state.linfo
    state.defs[s] = rhs
end

function updateDef(state::IRState, s::RHSVar, rhs)
    s = toLHSVar(s)
    updateDefInternal(state, s, rhs)
end

function updateDef(state::IRState, s::LHSVar, rhs)
    updateDefInternal(state, s, rhs)
end

"""
Delete a definition of a variable from current state.
"""
function deleteDef(state::IRState, s::RHSVar)
    delete!(state.defs, toLHSVar(s))
end


"""
Look up a definition of a variable.
Return nothing If none is found.
"""
function lookupDef(state::IRState, s::RHSVar)
    get(state.defs, toLHSVar(s), nothing)
end

function lookupDef(state, s)
    return nothing
end

"""
Look up a definition of a variable only when it is const or assigned once.
Return nothing If none is found.
"""
function lookupConstDef(state::IRState, s::RHSVar)
    s = toLHSVar(s)
    def = lookupDef(state, s)
    # we assume all GenSym is assigned once 
    desc = getDesc(s, state.linfo) 
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
    while isa(s, RHSVar)
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
function lookupDefInAllScopes(state::IRState, s::RHSVar)
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

# functions that domain-ir should ignore, like ones generated by HPAT
funcIgnoreList = []

const mapSym = vcat(Symbol[:negate], API.unary_map_operators, API.binary_map_operators)

const mapVal = Symbol[ begin s = string(x); startswith(s, '.') ? Symbol(s[2:end]) : x end for x in mapSym]

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

if VERSION >= v"0.5.0-dev+3875"
const afoldTyps = Type[getfield(Base, Symbol(string("#",s))) for s in ["+","*","&","|"]]
else
const afoldTyps = Type[getfield(Base, s) for s in [:AddFun, :MulFun, :AndFun, :OrFun]]
end
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
        name = Symbol(string(s, "##", unique_id))
        unique = !isLocalVariable(name, linfo)
    end
    addLocalVariable(name, t, desc, linfo)
    return toRHSVar(name, t, linfo)
end

include("domain-ir-stencil.jl")

function isbitmask(typ::DataType)
    isBitArrayType(typ) || (isArrayType(typ) && is(eltype(typ), Bool))
end

function isbitmask(typ::ANY)
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

function ismask(state, r::RHSVar)
    typ = getType(toLHSVar(r), state.linfo)
    return isrange(typ) || isbitmask(typ)
end

function ismask(state, r::GlobalRef)
    return r.name==:(:)
end

function ismask(state, r::Any)
    typ = typeOfOpr(state, r)
    return isrange(typ) || isbitmask(typ)
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
        ((isa(rhs.args[3].args[1], GlobalRef) && 
          rhs.args[3].args[1] == GlobalRef(Base, :select_value)) ||
         (isa(rhs.args[3].args[1], Expr) && is(rhs.args[3].args[1].head, :call) &&
          isBaseFunc(rhs.args[3].args[1].args[1], :getfield) &&
          is(rhs.args[3].args[1].args[2], GlobalRef(Base, :Intrinsics)) &&
          is(rhs.args[3].args[1].args[3], QuoteNode(:select_value))))
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

function rangeToMask(state, r::RHSVar, arraysize)
    typ = getType(r, state.linfo)
    if isbitmask(typ)
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
    if r.name==:(:)
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
    reorder = x -> x        # default no reorder
    cast = (ty, x) -> x     # defualt no cast
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
                if opr == :/ # special case for division
                    opr = GlobalRef(Base, floatOps[opr])
                    cast = (ty, x) -> Expr(:call, GlobalRef(Base, :sitofp), ty, x)
                else
                    opr = GlobalRef(Base, sintOps[opr])
                end
            elseif is(typ, Float32) || is(typ, Float64)
                opr = GlobalRef(Base, floatOps[opr])
            end
        catch err
            error(string("Cannot specialize operator ", opr, " to type ", typ))
        end
    end
    if isa(opr, Symbol)
        opr = GlobalRef(Base, opr)
    end
    return opr, reorder, cast
end

# Specialize non-array arguments into the body function as either constants
# or escaping variables. Arguments of array type are treated as parameters to
# the body function instead.
# It returns the list of arguments that are specialized into the body function,
# the remaining arguments (that are of array type), their types, and
# the specialized body function.
function specialize(state::IRState, args::Array{Any,1}, typs::Array{Type,1}, f::DomainLambda)
    local j = 0
    local len = length(typs)
    local idx = Array(Int, len)
    local args_ = Array(Any, len)
    local nonarrays = Array(Any, 0)
    local old_inps = f.inputs
    local new_inps = Array(Any, 0)
    local old_params = getInputParameters(f.linfo)
    local new_params = Array(Symbol, 0)
    #local pre_body = Array(Any, 0)
    local repl_dict = Dict{LHSVar,Any}()
    @dprintln(2, "specialize typs = ", typs)
    @dprintln(2, "specialize old_params = ", old_params)
    for i = 1:len
        local typ = typs[i]
        if isArrayType(typ)
            j = j + 1
            typs[j] = typ
            args_[j] = args[i]
            push!(new_inps, old_inps[i])
            push!(new_params, old_params[i])
            idx[j] = i
        else
            if isa(args[i], GenSym) # cannot put GenSym into lambda! Add a temp variable to do it
                typ = getType(args[i], state.linfo)
                tmpv = addFreshLocalVariable(string(args[i]), typ, ISCAPTURED | ISASSIGNED | ISASSIGNEDONCE, state.linfo)
                emitStmt(state, mk_expr(tmpv.typ, :(=), tmpv, args[i]))
                args[i] = tmpv
            end
            if isa(args[i], TypedVar)
                tmpv = lookupVariableName(args[i], state.linfo)
                typ = getType(tmpv, state.linfo)
                desc = getDesc(tmpv, state.linfo)
                addEscapingVariable(tmpv, typ, desc, f.linfo)
            end
            #push!(pre_body, Expr(:(=), old_params[i], args[i]))
            repl_dict[toLHSVar(old_params[i], f.linfo)] = args[i]
            push!(nonarrays, args[i])
        end
    end
    dprintln(3, "repl_dict = ", repl_dict)
    #body = vcat(pre_body, f.lambda.args[3].args)
    body = f.body
    if !isempty(repl_dict)
        body = replaceExprWithDict!(body, repl_dict)
    end
    #body_expr = Expr(:body)
    #body_expr.args = body
    #body_expr.typ = getReturnType(f.linfo)
    setInputParameters(new_params, f.linfo)
    f.inputs = new_inps
    f.body = body
    return (nonarrays, args_[1:j], f.inputs, f)
end

function typeOfOpr(state :: IRState, x)
    CompilerTools.LivenessAnalysis.typeOfOpr(x, state.linfo)
end

"""
get elem type T from an Array{T} type
"""
function elmTypOf(x::DataType)
    @assert isArrayType(x) "expect Array type"
    return eltype(x)
end

function elmTypOf(x::Expr)
    if x.head == :call && isBaseFunc(x.args[1], :apply_type) && x.args[2] == :Array
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

function simplify(state, expr::RHSVar)
    def = lookupConstDefForArg(state, expr)
    is(def, nothing) ? expr : (isa(def, Expr) ? simplify(state, def, expr) : def)
end

function simplify(state, expr::Array)
    [ simplify(state, e) for e in expr ]
end

function simplify(state, expr)
    return expr
end

isTopNodeOrGlobalRef(x::Union{TopNode,GlobalRef},s) = is(x, TopNode(s)) || is(Base.resolve(x), GlobalRef(Core.Intrinsics, s))
isTopNodeOrGlobalRef(x,s) = false
add_expr(x,y) = y == 0 ? x : mk_expr(Int, :call, GlobalRef(Base, :add_int), x, y)
sub_expr(x,y) = y == 0 ? x : mk_expr(Int, :call, GlobalRef(Base, :sub_int), x, y)
mul_expr(x,y) = y == 0 ? 0 : (y == 1 ? x : mk_expr(Int, :call, GlobalRef(Base, :mul_int), x, y))
sdiv_int_expr(x,y) = y == 1 ? x : mk_epr(Int, :call, GlobalRef(Base, :sdiv_int), x, y)
neg_expr(x)   = mk_expr(Int, :call, GlobalRef(Base, :neg_int), x)
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
function from_lambda(state, env, expr, closure = nothing)
    local env_ = nextEnv(env)
    linfo, body = lambdaToLambdaVarInfo(expr) 
    @dprintln(2,"from_lambda typeof(body) = ", typeof(body))
    @dprintln(3,"expr = ", expr)
    @dprintln(3,"body = ", body)
    assert(isa(body, Expr) && is(body.head, :body))
    defs = Dict{LHSVar,Any}()
    escDict = Dict{Symbol,RHSVar}()
    if !is(closure, nothing)
        # Julia 0.5 feature, closure refers to the #self# argument
        if isa(closure, Expr)
            def = closure
        elseif isa(closure, RHSVar)
            def = lookupDef(state, closure) # somehow closure variables are not Const defs
            dprintln(env, "closure ", closure, " = ", def) 
        else 
            def = nothing  # error(string("Unhandled closure: ", closure))
        end
        if isa(def, Expr) && def.head == :new
            dprintln(env, "closure def = ", def)
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
                qtyp = is(qtyp, Box) ? getBoxType(state, q) : qtyp
                dprintln(env, "field ", p, " has type ", qtyp)
                if isa(q, GenSym)  # tempvariable must be renamed to a named variable
                    newq = addLocalVariable(gensym(string(q)), qtyp, getDesc(q, state.linfo), state.linfo)
                    emitStmt(state, TypedExpr(qtyp, :(=), toLHSVar(newq, state.linfo), q))
                    q = newq
                end
                # if q has a Box type, we lookup its definition (due to setfield!) instead
                qname = lookupVariableName(q, state.linfo)
                dprintln(env, "closure variable in parent = ", qname)
                escDict[p] = addEscapingVariable(qname, qtyp, 0, linfo)
            end
        end
    end
    dprintln(env,"from_lambda: linfo=", linfo)
    dprintln(env,"from_lambda: escDict=", escDict)
    local state_ = newState(linfo, defs, escDict, state)
    body = from_expr(state_, env_, body)
    # fix return type
    typ = body.typ
    dprintln(env,"from_lambda: body=", body)
    dprintln(env,"from_lambda: linfo=", linfo)
    # fix Box types
    #for (k, t) in state_.boxtyps
    #    updateType(linfo, k, t)
    #end
    return (linfo, body)
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
    setReturnType(typ, state.linfo)
    return mk_expr(typ, head, state.stmts...)
end

function mmapRemoveDupArg!(state, expr::Expr)
    head = expr.head 
    @assert head==:mmap || head==:mmap! "Input to mmapRemoveDupArg!() must be :mmap or :mmap!"
    arr = expr.args[1]
    f = expr.args[2]
    linfo = f.linfo
    posMap = Dict{LHSVar, Int}()
    hasDup = false
    n = 1
    old_inps = f.inputs
    new_inps = Array(Type, 0)
    old_params = getInputParameters(linfo)
    new_params = Array(Symbol, 0)
    pre_body = Array(Any, 0)
    new_arr = Array(Any, 0)
    oldn = length(arr)
    for i = 1:oldn
        s = isa(arr[i], RHSVar) ? toLHSVar(arr[i]) : arr[i]
        if haskey(posMap, s)
            hasDup = true
            v = old_params[posMap[s]] 
            t = getType(v, linfo)
            push!(pre_body, Expr(:(=), toLHSVar(old_params[i], linfo), toRHSVar(v,t,linfo)))
        else
            if isa(s,LHSVar) 
                posMap[s] = n
            end
            push!(new_arr, arr[i])
            push!(new_params, old_params[i])
            push!(new_inps, old_inps[i])
            n += 1
        end
    end
    if (!hasDup) return expr end
    dprintln(3, "MMRD: expr was ", expr)
    dprintln(3, "MMRD:  ", new_arr, new_inps, new_params)
    body = f.body
    body.args = vcat(pre_body, body.args)
    f.inputs = new_inps
    setInputParameters(new_params, linfo)
    expr.args[1] = new_arr
    expr.args[2] = f
    dprintln(3, "MMRD: expr becomes ", expr)
    return expr
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
    lhs = toLHSVar(lhs)
    
    rhs = from_expr(state, env_, rhs)
    rhstyp = typeOfOpr(state, rhs)
    lhstyp = typeOfOpr(state, lhs)
    dprintln(env, "from_assignment lhs=", lhs, " typ=", typ, " rhs.typ=", rhstyp)
    if typ != rhstyp && rhstyp != Any && rhstyp != Tuple{}
        #if rhstyp != lhstyp
            updateTyp(state, lhs, rhstyp)
        #end
        typ = rhstyp
    end
    # turn x = mmap((x,...), f) into x = mmap!((x,...), f)
    if isa(rhs, Expr) && is(rhs.head, :mmap) && length(rhs.args[1]) > 0 &&
        (isa(rhs.args[1][1], RHSVar) && lhs == toLHSVar(rhs.args[1][1]))
        rhs.head = :mmap!
        # NOTE that we keep LHS to avoid a bug (see issue #...)
        typ = getType(lhs, state.linfo)
        lhs = addTempVariable(typ, state.linfo) 
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
    local fun  = lookupConstDefForArg(state, ast[1])
    local args = ast[2:end]
    dprintln(env,"from_call: fun=", fun, " typeof(fun)=", typeof(fun), " args=",args, " typ=", typ)
    if in(fun, funcIgnoreList)
        dprintln(env,"from_call: fun=", fun, " in ignore list")
        return expr
    end
    fun = from_expr(state, env_, fun)
    dprintln(env,"from_call: new fun=", fun)
    (fun_, args_) = normalize_callname(state, env, fun, args)
    dprintln(env,"normalized callname: ", fun_)
    result = translate_call(state, env, typ, :call, fun, args, fun_, args_)
    result
end

function translate_call(state, env, typ, head, oldfun::ANY, oldargs, fun::GlobalRef, args)
    translate_call_globalref(state, env, typ, head, oldfun, oldargs, fun, args)
end

function translate_call(state, env, typ, head, oldfun::ANY, oldargs, fun::Symbol, args)
    translate_call_symbol(state, env, typ, head, oldfun, oldargs, fun, args)
end

function translate_call(state, env, typ, head, oldfun::ANY, oldargs, fun::ANY, args)
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
        elseif isa(arg, Expr) || isa(arg, LambdaInfo)
            typ = isa(arg, Expr) ? arg.typ : Any
            dprintln(env, "addTempVariable with typ ", typ)
            newVar = addTempVariable(typ, state.linfo)
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
        if isa(dst, Expr) && is(dst.head, :call) && isBaseFunc(dst.args[1], :ccall) && 
           isa(dst.args[2], QuoteNode) && is(dst.args[2].value, :jl_new_array)
            # now we are sure destination array is new
            fun   = args[1]
            args  = args[3:end]
            if isa(fun, GlobalRef)
                fun = fun.name
            end
            if isa(fun, Symbol)
            elseif isa(fun, TypedVar)
                fun = lookupConstDef(state, fun)
            else
                error("DomainIR: cannot handle broadcast! with function ", fun)
            end
        elseif isa(dst, Expr) && is(dst.head, :call) && isa(dst.args[1], DataType) &&
            isBitArrayType(dst.args[1])
            # destination array is a new bitarray
            fun   = args[1]
            args  = args[3:end]
            if isa(fun, RHSVar)
                # fun could be a variable 
                fun = get(state.defs, toLHSVar(fun), nothing)
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
                if isTupleType(typeOfOpr(state, args[i]))
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

function normalize_callname(state::IRState, env, fun :: TypedVar, args)
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
function inline_select(env, state, arr::RHSVar)
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
                elseif isBaseFunc(def.args[1], :_getindex!) # getindex gets desugared!
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

function translate_call_copy!(state, env, args)
    nargs = length(args)
    @assert (nargs == 2 || nargs == 4) "Expect either 2 or 4 argument to copy!, but got " * string(args)
    args = normalize_args(state, env, nargs == 2 ? args : Any[args[2], args[4]])
    dprintln(env,"got copy!, args=", args)
    argtyp = typeOfOpr(state, args[1])
    if isArrayType(argtyp)
        eltyp = eltype(argtyp)
        expr = mk_mmap!(args, DomainLambda(Type[eltyp,eltyp], Type[eltyp], params->Any[Expr(:tuple, params[2])], state.linfo))
        dprintln(env, "turn copy! into mmap! ", expr)
    else
        warn("cannot handle copy! with arguments ", args)
        #expr = mk_copy(args[1])
    end
    expr.typ = argtyp
    return expr
end

function translate_call_copy(state, env, args)
    @assert (length(args) == 1) "Expect only one argument to copy, but got " * string(args)
    args = normalize_args(state, env, args[1:1])
    dprintln(env,"got copy, args=", args)
    argtyp = typeOfOpr(state, args[1])
    if isArrayType(argtyp)
        eltyp = eltype(argtyp)
        expr = mk_mmap(args, DomainLambda(Type[eltyp],Type[eltyp],params->Any[Expr(:tuple, params[1])], state.linfo))
        dprintln(env, "turn copy into mmap ", expr)
    else
        expr = mk_copy(args[1])
    end
    expr.typ = argtyp
    return expr
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
    local ret::Union{RHSVar,Expr,Int}
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
function translate_call_symbol(state, env, typ, head, oldfun::ANY, oldargs, fun::Symbol, args::Array{Any,1})
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
        else
            return mk_expr(typ, :call, oldfun, args...)
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
    elseif is(fun, :cartesianmapreduce)
        return translate_call_cartesianmapreduce(state, env_, typ, args)
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
    elseif is(fun, :copy)
        return translate_call_copy(state, env, args)
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
        range_out::Union{RHSVar,Expr,Int} = translate_call_rangeshortcut(state, args[1],args[2])
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
            #assert(isa(oldargs[1], TypedVar))
            typ = typeOfOpr(state, oldargs[3])
            updateTyp(state, oldargs[1], typ)
            updateBoxType(state, oldargs[1], typ)
            # change setfield! to direct assignment
            expr = mk_expr(typ, :(=), oldargs[1], oldargs[3])
        elseif is(fun, :getfield) && length(oldargs) == 2 
            dprintln(env, "got getfield ", oldargs)
            if oldargs[2] == QuoteNode(:contents)
                return oldargs[1]
            elseif isa(oldargs[1], RHSVar) && lookupVariableName(oldargs[1], state.linfo) == Symbol("#self#")
                fname = oldargs[2]
                assert(isa(fname, QuoteNode))
                fname = fname.value
                dprintln(env, "lookup #self# closure field ", fname, " :: ", typeof(fname))
                @assert (haskey(state.escDict, fname)) "missing escaping variable mapping for field " * string(fname)
                escVar = state.escDict[fname]    
                dprintln(env, "matched escaping variable = ", escVar)
                return escVar
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
                expr = mk_select(arr, ranges)
                # TODO: need to calculate the correct result dimesion
                typ = arrTyp
            else
                args = Any[inline_select(env, state, e) for e in Any[mk_select(arr, ranges), args[2]]]
                var = args[2]
                vtyp = typeOfOpr(state, var)
                if isArrayType(vtyp) # set to array value
                    # TODO: assert that vtyp must be equal to etyp here, or do a cast?
                    f = DomainLambda(Type[etyp, etyp], Type[etyp], params->Any[Expr(:tuple, params[2])], state.linfo)
                else # set to scalar value
                    if isa(var, GenSym)
                       tmpv = addFreshLocalVariable(string(args[i]), typ, ISCAPTURED | ISASSIGNED | ISASSIGNEDONCE, state.linfo)
                       emitStmt(state, mk_expr(tmpv.typ, :(=), tmpv, args[i]))
                       var = tmpv
                    elseif isa(var, RHSVar)
                       var = toLHSVar(var)
                    end
                    pop!(args)
                    f = DomainLambda(Type[etyp], Type[etyp], params->Any[Expr(:tuple, var)], state.linfo)
                end
                expr = mk_mmap!(args, f)
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
    opr, reorder, cast = specializeOp(mapOps[fun], elmtyps)
    elmtyps = reorder(elmtyps)
    typs = reorder(typs)
    args = reorder(args)
    dprintln(env,"translate_call_mapop: before specialize, opr=", opr, " args=", args, " typs=", typs)
    f = DomainLambda(elmtyps, Type[etyp], params->Any[Expr(:tuple, mk_expr(etyp, :call, opr, params...))], state.linfo)
    (nonarrays, args, typs, f) = specialize(state, args, typs, f)
    dprintln(env,"translate_call_mapop: after specialize, typs=", typs)
    for i = 1:length(args)
        arg_ = inline_select(env, state, args[i])
        #if arg_ != args[i] && i != 1 && length(args) > 1
        #    error("Selector array must be the only array argument to mmap: ", args)
        #end
        args[i] = arg_
    end
    expr::Expr = endswith(string(fun), '!') ? mk_mmap!(args, f) : mk_mmap(args, f)
    expr = mmapRemoveDupArg!(state, expr)
    expr.typ = typ
    return expr
end

"""
Run type inference and domain process over the income function object.
Return the result AST with a modified return statement, namely, return
is changed to Expr(:tuple, retvals...)
"""
function get_ast_for_lambda(state, env, func::Union{LambdaInfo,TypedVar,Expr}, argstyp)
    if isa(func, TypedVar) && func.typ <: Function
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
    dprintln(env, "typeof(lambda) = ", typeof(lambda))
    (ast, aty) = lambdaTypeinf(lambda, tuple(argstyp...))
    dprintln(env, "type inferred AST = ", ast)
    dprintln(env, "aty = ", aty)
    # recursively process through domain IR with new state and env
    (linfo, body) = from_lambda(state, env, ast, func)
    params = getInputParameters(linfo)
    dprintln(env, "params = ", params)
    lastExp::Expr = body.args[end]
    assert(is(lastExp.head, :return))
    args1_typ::DataType = Void
    if length(lastExp.args) > 0 
        args1 = lastExp.args[1]
        if isa(args1, RHSVar)
            args1_typ = getType(args1, linfo)
            dprintln(env, "lastExp=", lastExp, " args1=", args1, " typ=", args1_typ)
        end
    end
    # modify the last return statement if it's a tuple
    if isTupleType(args1_typ)
        # take a shortcut if the second last statement is the tuple creation
        exp = body.args[end-1]
        if isa(exp, Expr) && exp.head == :(=) && exp.args[1] == args1 && isa(exp.args[2], Expr) &&
           exp.args[2].head == :call && isBaseFunc(exp.args[2].args[1], :tuple)
            dprintln(env, "second last is tuple assignment, we'll take shortcut")
            pop!(body.args)
            exp.head = :tuple
            exp.args = exp.args[2].args[2:end]
        else
            # create tmp variables to store results
            tvar = args1
            typs::SimpleVector = args1_typ.parameters
            nvar = length(typs)
            retNodes = GenSym[ addTempVariable(t, linfo) for t in typs ]
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
    setReturnType(aty, linfo)
    return body, linfo
end

"""
Lookup a function object for the given argument (variable),
infer its type and return the result ast together with return type.
"""
function get_lambda_for_arg(state, env, func::RHSVar, argstyp)
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
    (body, linfo) = get_lambda_for_arg(state, env, args[1], inptyps) 
    ety = getReturnType(linfo)
    etys = isTupleType(ety) ? [ety.parameters...] : DataType[ety]
    dprintln(env, "etys = ", etys)
    # assume return dimension is the same as the first array argument
    rdim = ndims(argtyps[1])
    rtys = DataType[ Array{t, rdim} for t in etys ]
    domF = DomainLambda(linfo, body)
    expr::Expr = mk_mmap(args[2:end], domF)
    expr.typ = length(rtys) == 1 ? rtys[1] : to_tuple_type(tuple(rtys...))
    return expr
end

# translate map! with a generic function 
# Julia's Base.map! has a slightly different semantics than our internal mmap!.
# First of all, it doesn't allow multiple desntiation arrays. Secondly,
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
    (body, linfo) = get_lambda_for_arg(state, env, args[1], nargs == 2 ? inptyps : inptyps[2:end]) 
    ety = getReturnType(linfo)
    etys = isTupleType(ety) ? [ety.parameters...] : DataType[ety]
    dprintln(env, "etys = ", etys)
    # assume return dimension is the same as the first array argument
    rdim = ndims(argtyps[1])
    rtys = DataType[ Array{t, rdim} for t in etys ]
    if nargs > 2 # more than just destiniation array in arguments
        # insert dummy argument to hold values from destination since DomainLambda for mmap! needs it
        v = gensym("dest")
        setInputParameters(vcat(v, getInputParameters(linfo)), linfo)
        addLocalVariable(v, ety, 0, linfo)
    end
    domF = DomainLambda(linfo, body)
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
    if isArrayType(typ)
        typ_second_arg = typeOfOpr(state, args[2])
        if isArrayType(typ_second_arg) || isbitmask(typ_second_arg)
            expr = mk_expr(Bool, :assert, mk_expr(Bool, :call, GlobalRef(Base, :(===)), mk_expr(Int64, :call, GlobalRef(Base,:arraylen), args[1]), mk_expr(Int64, :call, GlobalRef(Base,:arraylen), args[2])))
        else
            @dprintln(0, args[2], " typ_second_arg = ", typ_second_arg)
            error("Unhandled bound in checkbounds: ", args[2])
        end
    elseif isIntType(typ)
        if isIntType(typeOfOpr(state, args[2]))
            expr = mk_expr(Bool, :assert, mk_expr(Bool, :call, GlobalRef(Base, :sle_int), convert(typ, 1), args[2]),
            mk_expr(Bool, :call, GlobalRef(Base, :sle_int), args[2], args[1]))
        elseif isa(args[2], TypedVar) && (isUnitRange(args[2].typ) || isStepRange(args[2].typ))
            def = lookupConstDefForArg(state, args[2])
            (start, step, final) = from_range(def)
            expr = mk_expr(Bool, :assert, mk_expr(Bool, :call, GlobalRef(Base, :sle_int), convert(typ, 1), start),
            mk_expr(Bool, :call, GlobalRef(Base, :sle_int), final, args[1]))
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
    local kernelExp_var::RHSVar = args[1]
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
    (body, linfo) = get_lambda_for_arg(state, env, kernelExp_var, tuple(bufstyp...))
    ety = getReturnType(linfo)
    dprintln(env, "bufs = ", bufs, " body = ", body, " borderExp=", borderExp, " :: ", typeof(borderExp))
    local stat, kernelF
    stat, kernelF = mkStencilLambda(state, bufs, body, linfo, borderExp)
    dprintln(env, "stat = ", stat, " kernelF = ", kernelF)
    expr = mk_stencil!(stat, iterations, bufs, kernelF)
    #typ = length(bufs) > 2 ? tuple(kernelF.outputs...) : kernelF.outputs[1] 
    # force typ to be Void, which means stencil doesn't return anything
    typ = Void
    expr.typ = typ
    return expr
end

# translate API.cartesianmapreduce call.
function translate_call_cartesianmapreduce(state, env, typ, args::Array{Any,1})
    # equivalent to creating an array first, then map! with indices.
    dprintln(env, "got cartesianmapreduce args=", args)
    # need to retrieve map lambda from inits, since it is already moved out.
    nargs = length(args)
    assert(nargs >= 2) # needs at least a function, and a dimension tuple
    args_ = normalize_args(state, env, args[1:2])
    local dimExp_var::RHSVar = args_[2]     # last argument is the dimension tuple
    
    dimExp_e::Expr = lookupConstDefForArg(state, dimExp_var)
    dprintln(env, "dimExp = ", dimExp_e, " head = ", dimExp_e.head, " args = ", dimExp_e.args)
    assert(is(dimExp_e.head, :call) && isBaseFunc(dimExp_e.args[1], :tuple))
    dimExp = dimExp_e.args[2:end]
    ndim = length(dimExp)   # num of dimensions
    argstyp = Any[ Int for i in 1:ndim ] 
    
    (body, linfo) = get_lambda_for_arg(state, env, args_[1], argstyp)     # first argument is the lambda
    ety = getReturnType(linfo)
    #etys = isTupleType(ety) ? [ety.parameters...] : DataType[ety]
    dprintln(env, "ety = ", ety)
    #@assert all([ isa(t, DataType) for t in etys ]) "cartesianarray expects static type parameters, but got "*dump(etys) 
    # create tmp arrays to store results
    #arrtyps = Type[ Array{t, ndim} for t in etys ]
    #dprintln(env, "arrtyps = ", arrtyps)
    #tmpNodes = Array(Any, length(arrtyps))
    # allocate the tmp array
    #for i = 1:length(arrtyps)
    #    arrdef = type_expr(arrtyps[i], mk_alloc(state, etys[i], dimExp))
    #    tmparr = addGenSym(arrtyps[i], state.linfo)
    #    updateDef(state, tmparr, arrdef)
    #    emitStmt(state, mk_expr(arrtyps[i], :(=), tmparr, arrdef))
    #    tmpNodes[i] = tmparr
    #end
    # produce a DomainLambda
    domF = DomainLambda(linfo, body)
    params = getInputParameters(domF.linfo)
    expr::Expr = mk_parallel_for(params, dimExp, domF)
    for i=3:nargs # we have reduction here!
        tup = lookupConstDefForArg(state, args[i])
        @assert (isa(tup, Expr) && is(tup.head, :call) &&
                 isBaseFunc(tup.args[1], :tuple)) "Expect reduction arguments to cartesianmapreduce to be tuples, but got " * string(tup)
        redfunc = lookupConstDefForArg(state, tup.args[2])
        redvar = tup.args[3] # lookupConstDefForArg(state, tup.args[3])
        redvar = toLHSVar(redvar)
        rvtyp = typeOfOpr(state, redvar)
        dprintln(env, "redvar = ", redvar, " type = ", rvtyp, " redfunc = ", redfunc)
        #if (isa(redvar, GenSym))
        #  show_backtrace()
        #end
        @assert (!isa(redvar, GenSym)) "Unexpected GenSym  at the position of the reduction variable, " * string(redvar)
        nlinfo = LambdaVarInfo()
        redvarname = lookupVariableName(redvar, state.linfo)
        dprintln(env, "redvarname = ", redvarname)
        nvar = addEscapingVariable(redvarname, rvtyp, 0, nlinfo)
        nval = replaceExprWithDict!(translate_call_copy(state, env, Any[toRHSVar(redvar, rvtyp, state.linfo)]), Dict{LHSVar,Any}(Pair(redvar, nvar)), AstWalk)
        redparam = gensym(redvarname)
        redparamvar = addLocalVariable(redparam, rvtyp, ISASSIGNED | ISASSIGNEDONCE, nlinfo)
        setInputParameters(Symbol[redparam], nlinfo)
        neutral = DomainLambda(nlinfo, Expr(:body, Expr(:(=), redparamvar, nval), Expr(:tuple, toRHSVar(redparam, rvtyp, nlinfo))))
        (body, linfo) = get_ast_for_lambda(state, env, redfunc, DataType[rvtyp]) # this function expects only one argument
        redty = getReturnType(linfo)
        # Julia 0.4 gives Any type for expressions like s += x, so we skip the check below
        # @assert (redty == rvtyp) "Expect reduction function to return type " * string(rvtyp) * " but got " * string(redty)
        # redast only expects one parameter, so we must add redvar to the input parameter of redast, 
        # and remove it from escaping variable
        dprintln(env, "translate_call_cartesianmapreduce linfo = ", linfo)
        if isEscapingVariable(redvar, linfo) 
          unsetEscapingVariable(redvar, linfo)
        end
        setInputParameters(vcat(redvarname, getInputParameters(linfo)), linfo)
        #addLocalVariable(redvar, getType(redvar, linfo), getDesc(redvar, linfo), linfo)
        reduceF = DomainLambda(linfo, body)
        push!(expr.args, (redvar, neutral, reduceF)) 
    end
    #expr.typ = length(arrtyps) == 1 ? arrtyps[1] : to_tuple_type(tuple(arrtyps...))
    #dprintln(env, "cartesianarray return type = ", expr.typ)
    return expr
end

# Translate API.cartesianarray call.
function translate_call_cartesianarray(state, env, typ, args::Array{Any,1})
    # equivalent to creating an array first, then map! with indices.
    dprintln(env, "got cartesianarray args=", args)
    # need to retrieve map lambda from inits, since it is already moved out.
    nargs = length(args)
    args = normalize_args(state, env, args)
    assert(nargs >= 3) # needs at least a function, one or more types, and a dimension tuple
    local dimExp_var::RHSVar = args[end]     # last argument is the dimension tuple
    
    dimExp_e::Expr = lookupConstDefForArg(state, dimExp_var)
    dprintln(env, "dimExp = ", dimExp_e, " head = ", dimExp_e.head, " args = ", dimExp_e.args)
    assert(is(dimExp_e.head, :call) && isBaseFunc(dimExp_e.args[1], :tuple))
    dimExp = dimExp_e.args[2:end]
    ndim = length(dimExp)   # num of dimensions
    argstyp = Any[ Int for i in 1:ndim ] 
    
    (body, linfo) = get_lambda_for_arg(state, env, args[1], argstyp)     # first argument is the lambda
    ety = getReturnType(linfo)
    etys = isTupleType(ety) ? [ety.parameters...] : DataType[ety]
    dprintln(env, "etys = ", etys)
    @assert all([ isa(t, DataType) for t in etys ]) "cartesianarray expects static type parameters, but got "*dump(etys) 
    # create tmp arrays to store results
    arrtyps = Type[ Array{t, ndim} for t in etys ]
    dprintln(env, "arrtyps = ", arrtyps)
    tmpNodes = Array(Any, length(arrtyps))
    # allocate the tmp array
    for i = 1:length(arrtyps)
        arrdef = type_expr(arrtyps[i], mk_alloc(state, etys[i], dimExp))
        tmparr = toLHSVar(addTempVariable(arrtyps[i], state.linfo))
        updateDef(state, tmparr, arrdef)
        emitStmt(state, mk_expr(arrtyps[i], :(=), tmparr, arrdef))
        tmpNodes[i] = tmparr
    end
    # produce a DomainLambda
    params = getInputParameters(linfo)
    dprintln(env, "params = ", params)
    dummy_params = Array(Symbol, length(etys))
    for i in 1:length(etys)
        dummy_params[i] = gensym(string("x",i))
        addLocalVariable(dummy_params[i], etys[i], 0, linfo)
    end
    setInputParameters(vcat(dummy_params, params), linfo)
    domF = DomainLambda(linfo, body)
    expr::Expr = mk_mmap!(tmpNodes, domF, true)
    expr.typ = length(arrtyps) == 1 ? arrtyps[1] : to_tuple_type(tuple(arrtyps...))
    dprintln(env, "cartesianarray return type = ", expr.typ)
    return expr
end

# Translate API.reduce call, i.e., reduction over input arrays
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
    (body, linfo) = get_lambda_for_arg(state, env, fun, inptyps)
    ety = getReturnType(linfo)
    @assert (ety == etyp) "expect return type of reduce function to be " * etyp * " but got " * ety
    red_dim = []
    neutral = neutralelt
    outtyp = etyp
    domF = DomainLambda(linfo, body)
    # turn reduce(z, getindex(a, ...), f) into reduce(z, select(a, ranges(...)), f)
    arr = inline_select(env, state, arr)
    expr = mk_reduce(neutral, arr, domF)
    expr.typ = outtyp
    return expr
end

# reduction over input arrays using predefined operators.
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
        sizeVars = Array(TypedVar, num_dim)
        linfo = LambdaVarInfo()
        for i = 1:num_dim
            sizeVars[i] = addFreshLocalVariable(string("red_dim_size"), Int, ISASSIGNED | ISASSIGNEDONCE, state.linfo)
            dimExp = mk_expr(arrtyp, :call, GlobalRef(Base, :select_value), 
                         mk_expr(Bool, :call, GlobalRef(Base, :eq_int), red_dim[1], i), 
                         1, mk_arraysize(arr, i)) 
            emitStmt(state, mk_expr(Int, :(=), sizeVars[i], dimExp))
            addEscapingVariable(lookupVariableName(sizeVars[i], state.linfo), Int, 0, linfo)
        end
        redparam = gensym("redvar")
        rednode = toRHSVar(redparam, arrtyp, state.linfo)
        addLocalVariable(redparam, arrtyp, ISASSIGNED | ISASSIGNEDONCE, linfo)
        setInputParameters(Symbol[redparam], linfo)
        neutral_body = Expr(:body, 
                           Expr(:(=), redparam, mk_alloc(state, etyp, sizeVars)),
                           mk_mmap!([rednode], DomainLambda(Type[etyp], Type[etyp], params->Any[Expr(:tuple, neutralelt)], linfo)),
                           Expr(:tuple, rednode))
        neutral = DomainLambda(linfo, neutral_body)
        outtyp = arrtyp
        opr, reorder, cast = specializeOp(fun, [etyp])
        # ignore reorder and cast since they are always id function
        params = Symbol[ gensym(s) for s in [:x, :y]]
        linfo = LambdaVarInfo()
        for i in 1:length(params)
            addLocalVariable(params[i], outtyp, 0, linfo) 
        end
        setInputParameters(params, linfo)
        params = [ toRHSVar(x, outtyp, linfo) for x in params ]
        setReturnType(outtyp, linfo)
        inner_dl = DomainLambda(Type[etyp, etyp], Type[etyp], params->Any[Expr(:tuple, mk_expr(etyp, :call, opr, params...))], LambdaVarInfo())
        inner_expr = mk_mmap!(params, inner_dl)
        inner_expr.typ = outtyp 
        f = DomainLambda(linfo, Expr(:body, mk_expr(outtyp, :tuple, inner_expr)))
    else
        red_dim = []
        neutral = neutralelt
        outtyp = etyp
        opr, reorder, cast = specializeOp(fun, [etyp])
        # ignore reorder and cast since they are always id function
        f = DomainLambda(Type[outtyp, outtyp], Type[outtyp], 
                params->Any[Expr(:tuple, mk_expr(etyp, :call, opr, params...))], state.linfo)
    end
    # turn reduce(z, getindex(a, ...), f) into reduce(z, select(a, ranges(...)), f)
    arr = inline_select(env, state, arr)
    expr = mk_reduce(neutral, arr, f, red_dim...)
    expr.typ = outtyp
    return expr
end

function translate_call_fill!(state, env, typ, args::Array{Any,1})
    args = normalize_args(state, env, args)
    @assert length(args)==2 "fill! should have 2 arguments"
    arr = args[1]
    ival = args[2]
    typs = Type[typeOfOpr(state, arr)]
    if isa(ival, GenSym)
        tmpv = addFreshLocalVariable(string(ival), getType(ival, state.linfo), ISASSIGNED | ISASSIGNEDONCE, state.linfo)
        emitStmt(state, mk_expr(tmpv.typ, :(=), tmpv, ival))
        ival = tmpv
    end
    domF = DomainLambda(typs, typs, params->Any[Expr(:tuple, ival)], state.linfo)
    expr = mmapRemoveDupArg!(state, mk_mmap!([arr], domF))
    expr.typ = typ
    return expr
end

function translate_call_parallel_for(state, env, args::Array{Any,1})
    (lambda, ety) = lambdaTypeinf(args[1], (Int, ))
    ast = from_expr("anonymous", env.cur_module, lambda)
    loopvars = [ if isa(x, Expr) x.args[1] else x end for x in ast.args[1] ]
    etys = [Int for _ in length(loopvars)]
    body = ast.args[3]
    ranges = args[2:end]
    assert(isa(body, Expr) && is(body.head, :body))
    lastExp = body.args[end]
    assert(isa(lastExp, Expr) && is(lastExp.head, :return))
    # Replace return statement
    body.args[end] = Expr(:tuple)
    domF = DomainLambda(ast)
    return mk_parallel_for(loopvars, ranges, domF)
end

# translate a function call to domain IR if it matches GlobalRef.
function translate_call_globalref(state, env, typ, head, oldfun::ANY, oldargs, fun::GlobalRef, args)
    local env_ = nextEnv(env)
    expr = nothing
    dprintln(env, "translate_call globalref ", fun, "::", typeof(fun), " args=", args, " typ=", typ)
    # new mainline Julia puts functions in Main module but PSE expects the symbol only
    #if isa(fun, GlobalRef) && fun.mod == Main
    #   fun = fun.name
    # end
    if is(fun.mod, Core.Intrinsics) || (is(fun.mod, Core) && 
       (is(fun.name, :Array) || is(fun.name, :arraysize) || is(fun.name, :getfield)))
        expr = translate_call_symbol(state, env, typ, head, fun, oldargs, fun.name, args)
    elseif (is(fun.mod, Core) || is(fun.mod, Base)) && is(fun.name, :convert)
        # fix type of convert
        args = normalize_args(state, env_, args)
        if isa(args[1], Type)
            typ = args[1]
        end
    elseif is(fun.mod, Base) 
        if is(fun.name, :afoldl) && haskey(afoldlDict, typeOfOpr(state, args[1]))
            opr, reorder, cast = specializeOp(afoldlDict[typeOfOpr(state, args[1])], [typ, typ])
            # ignore reorder and cast since they are always id function
            dprintln(env, "afoldl operator detected = ", args[1], " opr = ", opr)
            expr = Base.afoldl((x,y)->mk_expr(typ, :call, opr, reorder([x, y])...), args[2:end]...)
            dprintln(env, "translated expr = ", expr)
        elseif is(fun.name, :copy!)
            expr = translate_call_copy!(state, env, args)
        elseif is(fun.name, :copy)
            expr = translate_call_copy(state, env, args)
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
            if isa(args[1], Expr) && args[1].head == :call && isBaseFunc(args[1].args[1], :ccall) && 
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

function from_root(state::IRState, env::IREnv, ast)
    assert(isfunctionhead(ast))
    from_lambda(state, env, ast)
end

function from_expr_tiebreak(state::IRState, env::IREnv, ast)
    asts::Array{Any,1} = [ast]
    res::Array{Any,1} = [from_root(state, env, ast) for ast in asts ]
    return res[1]
end

"""
Entry point of DomainIR optimization pass.
"""
function from_expr(function_name::AbstractString, cur_module :: Module, ast)
    assert(isfunctionhead(ast))
    linfo, body = from_expr_tiebreak(emptyState(), newEnv(cur_module), ast) 
    return linfo, body
end

function from_expr(state::IRState, env::IREnv, ast::LambdaInfo)
    dprintln(env, "from_expr: LambdaInfo inferred = ", ast.inferred)
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

function from_expr(state::IRState, env::IREnv, ast::Union{Symbol,TypedVar})
    # if it is global const, we replace it with its const value
    def = lookupDefInAllScopes(state, ast)
    name = lookupVariableName(ast, state.linfo)
    if is(def, nothing) && isdefined(env.cur_module, name) && ccall(:jl_is_const, Int32, (Any, Any), env.cur_module, name) == 1
        def = getfield(env.cur_module, name)
        if isbits(def) && !isa(def, IntrinsicFunction) && !isa(def, Function)
            return def
        end
    end
    typ = typeOfOpr(state, ast)
    if isa(ast, TypedVar) && ast.typ != typ
        @dprintln(2, " Variable ", ast, " gets updated type ", typ)
        return toRHSVar(lookupVariableName(ast, state.linfo), typ, state.linfo)
    elseif isa(ast, Symbol)
        @dprintln(2, " Symbol ", ast, " gets a type ", typ)
        return toRHSVar(ast, typ, state.linfo)
    end
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::Expr)
    dprint(env,"from_expr: Expr")
    local head = ast.head
    local args = ast.args
    local typ  = ast.typ
    @dprintln(2, " :", head)
    if is(head, :lambda)
        (linfo, body) = from_lambda(state, env, ast)
        return LambdaVarInfoToLambda(linfo, body.args)
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
        # skip or remove?
        return nothing
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
    elseif is(head, :inbounds)
        # skip
    elseif is(head, :meta)
        # skip
    elseif is(head, :static_parameter)
        # skip
    elseif is(head, :static_typeof)
        typ = getType(args[1], state.linfo)
        return typ  
    elseif is(head, :alloc)
        # other packages like HPAT can generate :alloc
    else
        throw(string("ParallelAccelerator.DomainIR.from_expr: unknown Expr head :", head))
    end
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::LabelNode)
    # clear defs for every basic block.
    state.defs = Dict{LHSVar, Any}()
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
    @dprintln(4,"DomainIR.AstWalkCallback ", x)
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(4,"DomainIR.AstWalkCallback ret = ", ret)
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
                assert(isa(ranges[i], Integer) || isa(ranges[i], RHSVar))
                ranges[i] = AstWalker.AstWalk(ranges[i], AstWalkCallback, dw)
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
    @dprintln(4,"DomainIR.AstWalkCallback DomainLambda ", x)
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(4,"DomainIR.AstWalkCallback ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end
    @dprintln(4,"DomainIR.AstWalkCallback for DomainLambda", x)
    return x
end

function AstWalkCallback(x :: ANY, dw :: DirWalk, top_level_number, is_top_level, read)
    @dprintln(4,"DomainIR.AstWalkCallback ", x)
    ret = dw.callback(x, dw.cbdata, top_level_number, is_top_level, read)
    @dprintln(4,"DomainIR.AstWalkCallback ret = ", ret)
    if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
        return ret
    end
    
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function AstWalk(ast :: ANY, callback, cbdata :: ANY)
    @dprintln(4,"DomainIR.AstWalk ", ast)
    dw = DirWalk(callback, cbdata)
    AstWalker.AstWalk(ast, AstWalkCallback, dw)
end

function dir_live_cb(ast :: Expr, cbdata :: ANY)
    @dprintln(4,"dir_live_cb ", ast)

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
        for v in getEscapingVariables(dl.linfo)
            push!(expr_to_process, v)
        end 

        @dprintln(4, ":mmap ", expr_to_process)
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
                push!(expr_to_process, Expr(:(=), input_arrays[i], 1))
            else
                # Need to make input_arrays[1] written?
                push!(expr_to_process, input_arrays[i])
            end
        end
        for v in getEscapingVariables(dl.linfo)
            push!(expr_to_process, v)
        end 

        @dprintln(4, ":mmap! ", expr_to_process)
        return expr_to_process
    elseif head == :reduce
        expr_to_process = Any[]

        assert(length(args) == 3 || length(args) == 4)
        zero_val = args[1]
        input_array = args[2]
        dl = args[3]
        if isa(zero_val, DomainLambda) 
            for v in getEscapingVariables(zero_val.linfo)
                push!(expr_to_process, v)
            end
        else
            push!(expr_to_process, zero_val)
        end
        push!(expr_to_process, input_array)
        assert(isa(dl, DomainLambda))
        for v in getEscapingVariables(dl.linfo)
            push!(expr_to_process, v)
        end

        @dprintln(4, ":reduce ", expr_to_process)
        return expr_to_process
    elseif head == :stencil!
        expr_to_process = Any[]

        sbufs = args[3]
        for i = 1:length(sbufs)
            # sbufs both read and possibly written
            push!(expr_to_process, sbufs[i])
            push!(expr_to_process, Expr(:(=), sbufs[i], 1))
        end

        dl = args[4]
        assert(isa(dl, DomainLambda))
        for v in getEscapingVariables(dl.linfo)
            push!(expr_to_process, v)
        end

        @dprintln(4, ":stencil! ", expr_to_process)
        return expr_to_process
    elseif head == :parallel_for
        expr_to_process = Any[]

        assert(length(args) >= 3)
        loopvars = args[1]
        ranges = args[2]
        push!(expr_to_process, loopvars)
        append!(expr_to_process, ranges)
        for v in getEscapingVariables(args[3].linfo)
            push!(expr_to_process, v)
            desc = getDesc(v, args[3].linfo)
            if (0 != desc & (ISASSIGNED | ISASSIGNEDONCE)) 
                push!(expr_to_process, Expr(:(=), v, 1))
            end
        end
        for i = 4:length(args)
            (rv, nf, rf) = args[i]
            push!(expr_to_process, rv)
            for v in getEscapingVariables(nf.linfo)
                desc = getDesc(v, nf.linfo)
                push!(expr_to_process, v)
                if (0 != desc & (ISASSIGNED | ISASSIGNEDONCE)) 
                    push!(expr_to_process, Expr(:(=), v, 1))
                end
            end
            for v in getEscapingVariables(rf.linfo)
                desc = getDesc(v, rf.linfo)
                push!(expr_to_process, v)
                if (0 != desc & (ISASSIGNED | ISASSIGNEDONCE)) 
                    push!(expr_to_process, Expr(:(=), v, 1))
                end
            end
        end
        @dprintln(4, ":parallel_for ", expr_to_process)
        return expr_to_process
    elseif head == :assertEqShape
        assert(length(args) == 2)
        #@dprintln(3,"liveness: assertEqShape ", args[1], " ", args[2], " ", typeof(args[1]), " ", typeof(args[2]))
        expr_to_process = Any[]
        push!(expr_to_process, toLHSVar(args[1]))
        push!(expr_to_process, toLHSVar(args[2]))
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
        if isBaseFunc(args[1], :arrayref) || isBaseFunc(args[1], :arraysize)
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
        return AliasAnalysis.lookup(state, toLHSVar(tmp))
    elseif head == :reduce
        # TODO: inspect the lambda body to rule out assignment?
        return AliasAnalysis.NotArray
    elseif head == :stencil!
        # args is a list of PIRParForAst nodes.
        assert(length(args) > 0)
        krnStat = args[1]
        iterations = args[2]
        bufs = args[3]
        @dprintln(4, "AA: rotateNum = ", krnStat.rotateNum, " out of ", length(bufs), " input bufs")
        if !((isa(iterations, Number) && iterations == 1) || (krnStat.rotateNum == 0))
            # when iterations > 1, and we have buffer rotation, need to set alias Unknown for all rotated buffers
            for i = 1:min(krnStat.rotateNum, length(bufs))
                v = bufs[i]
                if isa(v, RHSVar)
                    AliasAnalysis.update_unknown(state, toLHSVar(v))
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
        return AliasAnalysis.lookup(state, toLHSVar(args[1]))
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
