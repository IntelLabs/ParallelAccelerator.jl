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
using Base.FastMath

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
mk_arraysize(state, arr, dim) = TypedExpr(Int64, :call, GlobalRef(Base, :arraysize), arr, simplify(state, dim))
mk_sizes(arr) = Expr(:sizes, arr)
mk_strides(arr) = Expr(:strides, arr)
mk_alloc(typ, s) = Expr(:alloc, typ, s)
mk_call(fun,args) = Expr(:call, fun, args...)
mk_copy(arr) = Expr(:call, GlobalRef(Base, :copy), arr)
mk_generate(range, f) = Expr(:generate, range, f)
mk_reshape(arr, shape) = Expr(:reshape, arr, shape)
mk_backpermute(arr, f) = Expr(:backpermute, arr, f)
mk_arrayref(arr, idx) = Expr(:arrayref, arr, idx...)
mk_arrayset(arr, idx, v) = Expr(:arrayset, arr, idx..., v)
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
        paramS = Array{Any}(length(params))
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
                    addToEscapingVariable(v, li, linfo)
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
    for x in getLocalVariables(f.linfo)
        show(io, (lookupVariableName(x, f.linfo),toLHSVar(x),getType(x, f.linfo)))
    end
    show(io, ";")
    for x in getEscapingVariables(f.linfo)
        show(io, lookupVariableName(x, f.linfo))
        show(io, ",")
    end
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

function addToEscapingVariable(v, typ, inner_linfo, outer_linfo)
    desc = getDesc(v, outer_linfo)
    name = lookupVariableName(v, outer_linfo)
    rhs = addEscapingVariable(name, typ, desc & (~ (ISCAPTURED | ISASSIGNEDONCE)), inner_linfo)
    if (desc & ISASSIGNED == ISASSIGNED) && (desc & ISASSIGNEDONCE != ISASSIGNEDONCE)
        setDesc(v, desc | ISCAPTURED | ISASSIGNEDBYINNERFUNCTION, outer_linfo)
    else
        setDesc(v, desc | ISCAPTURED, outer_linfo)
    end
    return rhs
end

function addToEscapingVariable(v, inner_linfo, outer_linfo)
    typ = getType(v, outer_linfo)
    addToEscapingVariable(v, typ, inner_linfo, outer_linfo)
end

# Make a variable captured by inner lambda
# If the variable is a GenSym or already exists in inner lambda, first create a local variable for it.
function makeCaptured(state, var::RHSVar, inner_linfo=nothing)
    lhs = toLHSVar(var)
    if isa(lhs, GenSym) || (inner_linfo != nothing && isVariableDefined(lookupVariableName(lhs, state.linfo), inner_linfo))
    # cannot put GenSym into lambda! Add a temp variable to do it
       typ = getType(lhs, state.linfo)
       tmpv = addFreshLocalVariable(string(lhs), typ, ISCAPTURED | ISASSIGNED | ISASSIGNEDONCE, state.linfo)
       emitStmt(state, mk_expr(typ, :(=), toLHSVar(tmpv), var))
       return tmpv
    else
       return var
    end
end

function makeCaptured(state, val, inner_linfo=nothing)
   typ = typeof(val)
   tmpv = addFreshLocalVariable(string("tmp"), typ, ISCAPTURED | ISASSIGNED | ISASSIGNEDONCE, state.linfo)
   emitStmt(state, mk_expr(typ, :(=), toLHSVar(tmpv), val))
   return tmpv
end

type IRState
    linfo   :: LambdaVarInfo
    defs    :: Dict{LHSVar, Any} # stores local definition of LHS = RHS
    escDict :: Dict{Symbol, Any} # mapping function closure fieldname to escaping variable
    boxtyps :: Dict{LHSVar, Any} # finer types for those have Box type
    nonNegs :: Set{LHSVar}       # remember single-assigned variables which are non-negative
    stmts   :: Array{Any, 1}
    parent  :: Union{Void, IRState}
end

emptyState() = IRState(LambdaVarInfo(), Dict{LHSVar,Any}(), Dict{Symbol,RHSVar}(), Dict{LHSVar,Any}(), Set{LHSVar}(), Any[], nothing)
newState(linfo, defs, escDict, state::IRState) = IRState(linfo, defs, escDict, Dict{LHSVar,Any}(), Set{LHSVar}(), Any[], state)

"""
Update the type of a variable.
"""
function updateTyp(state::IRState, s, typ)
    @dprintln(3, "updateTyp ", s, " to type ", typ)
    setType(s, typ, state.linfo)
end

function updateBoxType(state::IRState, s::RHSVar, typ)
    state.boxtyps[toLHSVar(s, state.linfo)] = typ
end

function getBoxType(state::IRState, s::RHSVar)
    v = toLHSVar(s, state.linfo)
    t = getType(s, state.linfo)
    @dprintln(3, "getBoxType accessing ", v, " of ", t, " in cached boxtyps = ", state.boxtyps)
    get(state.boxtyps, v, Any)
end

isNonNeg(state::IRState, s::RHSVar) = in(toLHSVar(s), state.nonNegs)
isNonNeg(state::IRState, s::Integer) = s >= 0
isNonNeg(state::IRState, s::Expr) = (isCall(s) || isInvoke(s)) && isBaseFunc(getCallFunction(s), :arraysize)
isNonNeg(state::IRState, s::Any) = false

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
    if isNonNeg(state, rhs) && (getDesc(s, state.linfo) & (ISASSIGNEDONCE | ISCONST) != 0)
        push!(state.nonNegs, s)
    end
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
Delete all definitions for non-constant variables
"""
function deleteNonConstDefs(state)
    defs = Dict{LHSVar, Any}()
    for (v, d) in state.defs
        if (getDesc(v, state.linfo) & (ISASSIGNEDONCE | ISCONST) != 0) && (!isa(d, Expr) || d.head == :new)
            defs[v] = d
        end
    end
    for (v, d) in defs
        if isa(d, RHSVar)
            x = toLHSVar(d)
            if !haskey(defs, x)
                defs[v] = nothing
            end
        end
    end
    state.defs = defs
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
    if !(def === nothing) && ((desc & (ISASSIGNEDONCE | ISCONST)) != 0 || typeOfOpr(state, s) <: Function)
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
    (s === nothing) ? s1 : s
end

"""
Look up a definition of a variable recursively until the RHS is no-longer just a variable.
Return the last lhs (i.e. a variable) if found, or the input variable itself otherwise.
"""
function lookupConstVarForArg(state::IRState, s::Any)
    s1 = s
    while isa(s, RHSVar)
        s1 = s
        s = lookupConstDef(state, s1)
    end
    s1
end

function lookupConstDefForArg(state::Void, s::Any)
    return nothing
end

"""
Look up a definition of a variable throughout nested states until a definition is found.
Return nothing If none is found.
"""
function lookupDefInAllScopes(state::IRState, s::Union{Symbol,RHSVar})
    if isa(s, Symbol)
        @dprintln(2,"lookupDefInAllScopes s = ", s)
    end
    def = lookupDef(state, s)
    if def === nothing && !(state.parent === nothing)
        return lookupDefInAllScopes(state.parent, s)
    else
        return def
    end
end

function emitStmt(state::IRState, stmt)
    @dprintln(2,"emit stmt: ", stmt)
    if isa(stmt, Expr) && stmt.head === :(=) && stmt.typ == Box
        @dprintln(2, "skip Box assigment")
    elseif stmt != nothing
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


# hooks for other systems like HPAT:

# functions that domain-ir should ignore, like ones generated by HPAT
funcIgnoreList = []
# Expr heads domain-ir should ignore, like ones generated by HPAT
exprHeadIgnoreList = Symbol[]
externalCallback = nothing
externalLiveCB = nothing
externalAliasCB = nothing

function setExternalCallback(cb::Function)
    global externalCallback = cb
end

function setExternalLiveCB(cb::Function)
    global externalLiveCB = cb
end

function setExternalAliasCB(cb::Function)
    global externalAliasCB = cb
end

const mapSym = vcat(API.unary_map_operators, API.binary_map_operators)
const mapOpr = map(x -> API.rename_if_needed(x), mapSym)

const mapVal = Symbol[ begin s = string(x); startswith(s, '.') ? Symbol(s[2:end]) : x end for x in mapSym]

# * / are not point wise. it becomes point wise only when one argument is scalar.
const pointWiseOps = setdiff(Set{Symbol}(mapOpr), Set{Symbol}([:pa_api_mul, :pa_api_div]))

const compareOpSet = Set{Symbol}(API.comparison_map_operators)
const mapOps = Dict{Symbol,Symbol}(zip(mapOpr, mapVal))
# legacy v0.3
# symbols that when lifted up to array level should be changed.
# const liftOps = Dict{Symbol,Symbol}(zip(Symbol[:<=, :>=, :<, :(==), :>, :+,:-,:*,:/], Symbol[:.<=, :.>=, :.<, :.==, :.>, :.+, :.-, :.*, :./]))
# const topOpsTypeFix = Set{Symbol}([:not_int, :and_int, :or_int, :neg_int, :add_int, :mul_int, :sub_int, :neg_float, :mul_float, :add_float, :sub_float, :div_float, :box, :fptrunc, :fpsiround, :checked_sadd, :checked_ssub, :rint_llvm, :floor_llvm, :ceil_llvm, :abs_float, :cat_t, :srem_int])

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
        name = Symbol(string(s, "_domain_ir_fresh_local_variable_ASDFQWER_", unique_id))
        unique = !isLocalVariable(name, linfo)
    end
    addLocalVariable(name, t, desc, linfo)
    return toRHSVar(name, t, linfo)
end

include("domain-ir-stencil.jl")

function isbitmask(typ::DataType)
    isBitArrayType(typ) || (isArrayType(typ) && eltype(typ) === Bool)
end

function isbitmask(typ::ANY)
    false
end

function isUnitRange(typ::DataType)
    typ <: UnitRange
end

function isUnitRange(typ::ANY)
    return false
end

function isStepRange(typ::DataType)
    typ <: StepRange
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

function ismask(state, r::Colon)
    return true
end

function ismask(state, r::Any)
    typ = typeOfOpr(state, r)
    return isrange(typ) || isbitmask(typ)
end

function from_range(rhs::Expr)
    start = 1
    step = 1
    final = 1
    if rhs.head === :new && isUnitRange(rhs.args[1]) &&
        isa(rhs.args[3], Expr) && rhs.args[3].head === :call &&
        ((isa(rhs.args[3].args[1], GlobalRef) &&
          rhs.args[3].args[1] == GlobalRef(Base, :select_value)) ||
         (isa(rhs.args[3].args[1], Expr) && rhs.args[3].args[1].head === :call &&
          isBaseFunc(rhs.args[3].args[1].args[1], :getfield) &&
          rhs.args[3].args[1].args[2] === GlobalRef(Base, :Intrinsics) &&
          rhs.args[3].args[1].args[3] === QuoteNode(:select_value)))
        # only look at final value in select_value of UnitRange
        start = rhs.args[2]
        step  = 1 # FIXME: could be wrong here!
        final = rhs.args[3].args[3]
    elseif rhs.head === :new && isUnitRange(rhs.args[1]) &&
        (isa(rhs.args[2],LHSVar) || isa(rhs.args[2],Number)) && (isa(rhs.args[3],LHSVar) || isa(rhs.args[3],Number))
        start = rhs.args[2]
        step  = 1
        final = rhs.args[3]
    elseif rhs.head === :new && isStepRange(rhs.args[1])
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

function rangeToMask(state, r::Colon, arraysize)
    return mk_range(state, 1, 1, arraysize)
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

# Specialize non-array arguments into the body function as either constants
# or escaping variables. Arguments of array type are treated as parameters to
# the body function instead.
# It returns the list of arguments that are specialized into the body function,
# the remaining arguments (that are of array type), their types, and
# the specialized body function.
function specialize(state::IRState, args::Array{Any,1}, typs::Array{Type,1}, f::DomainLambda)
    local j = 0
    local len = length(typs)
    local idx = Array{Int}(len)
    local args_ = Array{Any}(len)
    local nonarrays = Array{Any}(0)
    local old_inps = f.inputs
    local new_inps = Array{Any}(0)
    local old_params = getInputParameters(f.linfo)
    local new_params = Array{Symbol}(0)
    #local pre_body = Array{Any}(0)
    local repl_dict = Dict{LHSVar,Any}()
    @dprintln(2, "specialize typs = ", typs)
    @dprintln(2, "specialize args = ", args)
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
        elseif isa(args[i], Number) # constant should be substituted directly
            repl_dict[toLHSVar(old_params[i], f.linfo)] = args[i]
            push!(nonarrays, args[i])
        else
            args[i] = makeCaptured(state, args[i], f.linfo)
            if isa(args[i], Union{LHSVar,RHSVar})
                tmpv = lookupVariableName(args[i], state.linfo)
                args[i] = addToEscapingVariable(tmpv, f.linfo, state.linfo)
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
        body = replaceExprWithDict!(body, repl_dict, f.linfo)
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

function typeOfOpr(linfo :: LambdaVarInfo, x)
    CompilerTools.LivenessAnalysis.typeOfOpr(x, linfo)
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
    elseif isCondExpr(expr)
        f = getCondFunc(expr.args[1], x -> isNonNeg(state, x))
        x = simplify(state, expr.args[2])
        y = simplify(state, expr.args[3])
        f(x, y)
    elseif isSelectExpr(expr)
        cond = simplify(state, expr.args[2])
        tval = simplify(state, expr.args[3])
        fval = simplify(state, expr.args[4])
        if cond == true
            tval
        elseif cond == false
            fval
        else
            # rewrite (1 <= x ? x : 0) to x when x >= 0
            if (isCall(cond) || isInvoke(cond))
                opr = getCallFunction(cond)
                args = getCallArguments(cond)
                if (isTopNodeOrGlobalRef(opr, :sle_int) || isTopNodeOrGlobalRef(opr, :ule_int)) &&
                   args[1] == 1 && isNonNeg(state, args[2]) && tval == args[2] && fval == 0
                   return tval
                end
            end
            mk_expr(expr.typ, :call, expr.args[1], cond, tval, fval)
        end
    else
        default
    end
end

function simplify(state, expr::Expr)
    expr_ = simplify(state, expr, expr)
@dprintln(2, "simplify ", expr, " to ", expr_)
    return expr_
end

function simplify(state, expr::RHSVar)
    def = lookupConstDefForArg(state, expr)
@dprintln(2, "lookup ", expr, " to be ", def)
    expr_ = (def === nothing) ? expr : (isa(def, Expr) ? simplify(state, def, expr) : def)
@dprintln(2, "simplify ", expr, " to ", expr_)
    return expr_
end

function simplify(state, expr::Array)
    [ simplify(state, e) for e in expr ]
end

function simplify(state, expr)
    return expr
end

isTopNodeOrGlobalRef(x::Union{TopNode,GlobalRef},s) = isa(x, TopNode) ? (x === TopNode(s)) : (Base.resolve(x) === GlobalRef(Core.Intrinsics, s))
isTopNodeOrGlobalRef(x,s) = false
if VERSION >= v"0.6.0-pre"
function box_ty(ty, x::Expr)
    x.typ = ty
    return x
end
else
function box_ty(ty, x::Expr)
  if ty == Bool
    x.typ = ty
    return x
  end
  @assert (x.head == :call)
  @assert (length(x.args) >= 2)
  opr = x.args[1]
  @assert (isa(opr, GlobalRef))
  real_opr = Base.resolve(opr, force = true)
  if real_opr.mod == Core.Intrinsics
    x.args[1] = real_opr
    mk_expr(ty, :call, GlobalRef(Base, :box), ty, x)
  else
    x.typ = ty
    return x
  end
end
end
box_ty(ty, x) = x

box_int(x) = box_ty(Int, x)
add_expr(x,y) = y == 0 ? x : box_int(Expr(:call, GlobalRef(Base, :add_int), x, y))
sub_expr(x,y) = y == 0 ? x : box_int(Expr(:call, GlobalRef(Base, :sub_int), x, y))
mul_expr(x,y) = y == 0 ? 0 : (y == 1 ? x : box_int(Expr(:call, GlobalRef(Base, :mul_int), x, y)))
sdiv_int_expr(x,y) = y == 1 ? x : box_int(Expr(:call, GlobalRef(Base, :sdiv_int), x, y))
neg_expr(x)   = box_int(Expr(:call, GlobalRef(Base, :neg_int), x))
isBoxExpr(x::Expr) = (x.head === :call) && isTopNodeOrGlobalRef(x.args[1], :box)
isNegExpr(x::Expr) = (x.head === :call) && isTopNodeOrGlobalRef(x.args[1], :neg_int)
isAddExpr(x::Expr) = (x.head === :call) && (isTopNodeOrGlobalRef(x.args[1], :add_int) || isTopNodeOrGlobalRef(x.args[1], :checked_sadd) || isTopNodeOrGlobalRef(x.args[1], :checked_sadd_int))
isSubExpr(x::Expr) = (x.head === :call) && (isTopNodeOrGlobalRef(x.args[1], :sub_int) || isTopNodeOrGlobalRef(x.args[1], :checked_ssub) || isTopNodeOrGlobalRef(x.args[1], :checked_ssub_int))
isMulExpr(x::Expr) = (x.head === :call) && (isTopNodeOrGlobalRef(x.args[1], :mul_int) || isTopNodeOrGlobalRef(x.args[1], :checked_smul) || isTopNodeOrGlobalRef(x.args[1], :checked_smul_int))
isAddExprInt(x::Expr) = isAddExpr(x) && isa(x.args[3], Int)
isMulExprInt(x::Expr) = isMulExpr(x) && isa(x.args[3], Int)
isAddExpr(x::ANY) = false
isSubExpr(x::ANY) = false
isCondExpr(x::Expr) = (x.head === :call) && (isTopNodeOrGlobalRef(x.args[1], :sle_int) || isTopNodeOrGlobalRef(x.args[1], :ule_int) ||
                                            isTopNodeOrGlobalRef(x.args[1], :ne_int) || isTopNodeOrGlobalRef(x.args[1], :eq_int) ||
                                            isTopNodeOrGlobalRef(x.args[1], :slt_int) || isTopNodeOrGlobalRef(x.args[1], :ult_int))
isSelectExpr(x::Expr) = (x.head === :call) && isTopNodeOrGlobalRef(x.args[1], :select_value)
sub(x, y) = add(x, neg(y))
add(x::Int,  y::Int) = x + y
add(x::Int,  y::Expr)= add(y, x)
add(x::Int,  y)      = add(y, x)
add(x::Expr, y::Int) = isBoxExpr(x) ? add(x.args[3], y) : (isAddExprInt(x) ? add(x.args[2], x.args[3] + y) : add_expr(x, y))
add(x::Expr, y::Expr)= isBoxExpr(x) ? add(x.args[3], y) : (isBoxExpr(y) ? add(x, y.args[3]) : (isAddExprInt(x) ? add(add(x.args[2], y), x.args[3]) : add_expr(x, y)))
add(x::Expr, y)      = isBoxExpr(x) ? add(x.args[3], y) : (isAddExprInt(x) ? add(add(x.args[2], y), x.args[3]) : add_expr(x, y))
add(x,       y::Expr)= add(y, x)
add(x,       y)      = add_expr(x, y)
neg(x::Int)          = -x
neg(x::Expr)         = isBoxExpr(x) ? neg(x.args[3]) : (isNegExpr(x) ? x.args[2] :
                       (isAddExpr(x) ? add(neg(x.args[2]), neg(x.args[3])) :
                       (isMulExpr(x) ? mul(x.args[2], neg(x.args[3])) : neg_expr(x))))
neg(x)               = neg_expr(x)
mul(x::Int,  y::Int) = x * y
mul(x::Int,  y::Expr)= mul(y, x)
mul(x::Int,  y)      = mul(y, x)
mul(x::Expr, y::Int) = isBoxExpr(x) ? mul(x.args[3], y) : (isMulExprInt(x) ? mul(x.args[2], x.args[3] * y) :
                       (isAddExpr(x) ? add(mul(x.args[2], y), mul(x.args[3], y)) : mul_expr(x, y)))
mul(x::Expr, y::Expr)= isBoxExpr(x) ? mul(x.args[3], y) : (isBoxExpr(y) ? mul(x, y.args[3]) : (isMulExprInt(x) ? mul(mul(x.args[2], y), x.args[3]) :
                       (isAddExpr(x) ? add(mul(x.args[2], y), mul(x.args[3], y)) : mul_expr(x, y))))
mul(x::Expr, y)      = isBoxExpr(x) ? mul(x.args[3], y) : (isMulExprInt(x) ? mul(mul(x.args[2], y), x.args[3]) :
                       (isAddExpr(x) ? add(mul(x.args[2], y), mul(x.args[3], y)) : mul_expr(x, y)))
mul(x,       y::Expr)= mul(y, x)
mul(x,       y)      = mul_expr(x, y)

# convert a conditional to a Julia function for static evaluation
function getCondFunc(opr, isNonNeg)
  if isTopNodeOrGlobalRef(opr, :sle_int) || isTopNodeOrGlobalRef(opr, :ule_int)
     (x, y) -> mk_le_expr(opr, x, y, isNonNeg)
  elseif isTopNodeOrGlobalRef(opr, :slt_int) || isTopNodeOrGlobalRef(opr, :ult_int)
     (x, y) -> mk_lt_expr(opr, x, y, isNonNeg)
  elseif isTopNodeOrGlobalRef(opr, :ne_int)
     (x, y) -> mk_ne_expr(opr, x, y)
  elseif isTopNodeOrGlobalRef(opr, :eq_int)
     (x, y) -> mk_eq_expr(opr, x, y)
  else
     (x, y) -> mk_expr(Bool, :call, opr, x, y)
  end
end

function mk_le_expr(opr, x, y, isNonNeg)
    if isa(x, Integer) && isa(y, Integer)
        x <= y
    elseif isa(x, Integer) && x <= 1 && isNonNeg(y)
        true
    elseif isa(y, Integer) && y <= 0 && isNonNeg(x)
        false
    else
        mk_expr(Bool, :call, opr, x, y)
    end
end

function mk_lt_expr(opr, x, y, isNonNeg)
    if isa(x, Integer) && isa(y, Integer)
        x < y
    elseif isa(x, Integer) && x <= 0 && isNonNeg(y)
        true
    elseif isa(y, Integer) && y <= 1 && isNonNeg(x)
        false
    else
        mk_expr(Bool, :call, opr, x, y)
    end
end

mk_ne_expr(opr, x, y) = x != y ?  true : mk_expr(Bool, :call, opr, x, y)
mk_eq_expr(opr, x, y) = x == y ?  true : mk_expr(Bool, :call, opr, x, y)


# simplify expressions passed to alloc and range.
mk_alloc(state, typ, s) = mk_alloc(typ, simplify(state, s))
mk_range(state, start, step, final) = mk_range(simplify(state, start), simplify(state, step), simplify(state, final))

"""
 :lambda expression
 (:lambda, {param, meta@{localvars, types, freevars}, body})
"""
function from_lambda(state, env, expr, closure = nothing)
    @dprintln(3,"from_lambda expr = ", expr, " type = ", typeof(expr))
    local env_ = nextEnv(env)
    linfo, body = lambdaToLambdaVarInfo(expr)
    cfg = CompilerTools.CFGs.from_lambda(body)
    body = getBody(CompilerTools.CFGs.createFunctionBody(cfg), getReturnType(linfo))
    @dprintln(2,"from_lambda typeof(body) = ", typeof(body))

    assert(isa(body, Expr) && (body.head === :body))
    defs = Dict{LHSVar,Any}()
    escDict = Dict{Symbol,Any}()
    if !(closure === nothing)
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
                qtyp = (qtyp === Box) ? getBoxType(state, q) : qtyp
                dprintln(env, "field ", p, " has type ", qtyp)
                if isa(q, RHSVar)
                    q = makeCaptured(state, q, linfo)
                    # if q has a Box type, we lookup its definition (due to setfield!) instead
                    qname = lookupVariableName(q, state.linfo)
                    dprintln(env, "closure variable in parent = ", qname)
                    escDict[p] = addToEscapingVariable(qname, qtyp, linfo, state.linfo)
                else
                    if isa(q, GlobalRef) && isdefined(q.mod, q.name)
                        q = getfield(q.mod, q.name)
                    end
                    dprintln(env, "closure enclosed a constant? ", q)
                    escDict[p] = q
                end
            end
        end
    end
    dprintln(env,"from_lambda: linfo=", linfo)
    dprintln(env,"from_lambda: escDict=", escDict)
    local state_ = newState(linfo, defs, escDict, state)
    # first we rewrite gotoifnot with constant condition
    for i in 1:length(body.args)
       s = body.args[i]
       if isa(s, Expr) && (s.head === :gotoifnot) && simplify(state_, s.args[1]) == false
          body.args[i] = GotoNode(s.args[2])
       end
    end
    # then we get rid of empty basic blocks!
    lives = CompilerTools.LivenessAnalysis.from_lambda(linfo, body, dir_live_cb, linfo)
    body = CompilerTools.LambdaHandling.getBody(CompilerTools.CFGs.createFunctionBody(lives.cfg), CompilerTools.LambdaHandling.getReturnType(linfo))
    @dprintln(3,"body = ", body)
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
    local body = Array{Any}(len)
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
    # typ = getReturnType(state.linfo)
    newtyp = Any
    n = length(state.stmts)
    while n > 0
        last_exp = state.stmts[n]
        if isa(last_exp, LabelNode)
            n = n - 1
        elseif isa(last_exp, Expr) && last_exp.head == :return
            newtyp = state.stmts[n].typ
            break
        else
            break
        end
    end
    if newtyp <: typ
      setReturnType(newtyp, state.linfo)
      typ = newtyp
    end
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
    new_inps = Array{Type}(0)
    old_params = getInputParameters(linfo)
    new_params = Array{Symbol}(0)
    pre_body = Array{Any}(0)
    new_arr = Array{Any}(0)
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

    # The following is unsafe unless x is not aliased, AND x is not parameter.
    # turn x = mmap((x,...), f) into x = mmap!((x,...), f)
    # if isa(rhs, Expr) && (rhs.head === :mmap) && length(rhs.args[1]) > 0 &&
    #     (isa(rhs.args[1][1], RHSVar) && lhs == toLHSVar(rhs.args[1][1]))
    #     rhs.head = :mmap!
    #     # NOTE that we keep LHS to avoid a bug (see issue #...)
    #     typ = getType(lhs, state.linfo)
    #     lhs = addTempVariable(typ, state.linfo)
    # end
    updateDef(state, lhs, rhs)
    return mk_expr(typ, head, lhs, rhs)
end

function from_foreigncall(state::IRState, env::IREnv, expr::Expr)
    local env_ = nextEnv(env)

    expr.args[2:end] = normalize_args(state, env, expr.args[2:end])
    return expr
end

function from_call(state::IRState, env::IREnv, expr::Expr)
    local env_ = nextEnv(env)
    local fun = getCallFunction(expr)
    local args = getCallArguments(expr)
    local typ = expr.typ
    if expr.head == :invoke
        # change all :invoke to :call, since :invoke doesn't pass inference
        expr.head = :call
    end
    expr.args = [fun; args]
    if in(fun, funcIgnoreList)
        dprintln(env,"from_call: fun=", fun, " in ignore list")
        return expr
    end
    fun = lookupConstDefForArg(state, fun)
    dprintln(env,"from_call: fun=", fun, " typeof(fun)=", typeof(fun), " args=",args, " typ=", typ)
    fun = from_expr(state, env_, fun)
    dprintln(env,"from_call: new fun=", fun)
    (fun_, args_) = normalize_callname(state, env, fun, args)
    dprintln(env,"normalized callname: ", fun_)
    result = translate_call(state, env, typ, :call, fun, args, fun_, args_)
    if result == nothing
        # do not normalize :ccall
        args = fun_ == :ccall ? args : normalize_args(state, env, args)
        if fun_ == GlobalRef(Core.Intrinsics, :box)
            args[2] = lookupConstDefForArg(state, args[2])
        end
        expr.args = expr.head == :invoke ? [expr.args[1]; fun; args] : [fun; args]
        expr
    else
        result
    end
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

if VERSION >= v"0.6.0-pre"
import CompilerTools.LambdaHandling.LambdaInfo
end

# turn Exprs in args into variable names, and put their definition into state
# anything of void type in the argument is omitted in return value.
function normalize_args(state::IRState, env::IREnv, args::Array{Any,1})
    in_args::Array{Any,1} = from_exprs(state, env, args)
    local out_args = Array{Any}(length(in_args))
    j = 0
    for i = 1:length(in_args)
        local arg = in_args[i]
        @dprintln(3, "normalize_args arg ", i, " is ", arg)
        if isa(arg, Expr)
            @dprintln(3, "is Expr with typ = ", arg.typ)
        end
        if isa(arg, Expr) && arg.typ == Any
            j = j + 1
            out_args[j] = in_args[i]
        elseif isa(arg, Expr) && arg.typ == Void
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
    if (fun.mod === API) || (fun.mod === API.Stencil)
        return normalize_callname(state, env, fun.name, args)
    elseif (fun.mod === Base.Random) && ((fun.name === :rand!) || (fun.name === :randn!))
        return (fun.name, args)
    elseif (fun.mod === Base) && (fun.name === :getindex) || (fun.name === :setindex!) || (fun.name === :_getindex!)
        return (fun.name, args)
    elseif (fun === GlobalRef(Core.Intrinsics, :ccall))
        return normalize_callname(state, env, fun.name, args)
    else
        return (fun, args)
    end
end

function normalize_callname(state::IRState, env, fun::TopNode, args)
    normalize_callname(state, env, fun.name, args)
end

function normalize_callname(state::IRState, env, fun::Symbol, args)
    if (fun === :ccall)
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
                    if isa(def, Expr) && (def.head === :call) && (isBaseFunc(def.args[1], :tuple) || (def.args[1] === TopNode(:tuple)))
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
    if !(def === nothing) && !isa(def, Expr)
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
            if (def.head === :call)
                target_arr = arr
                if (def.args[1] === :getindex) || (isa(def.args[1], GlobalRef) && (def.args[1].name === :getindex))
                    target_arr = def.args[2]
                    range_extra = def.args[3:end]
                elseif isBaseFunc(def.args[1], :_getindex!) # getindex gets desugared!
                    error("we cannot handle TopNode(_getindex!) because it is effectful and hence will persist until J2C time")
                end
                dprintln(env, "inline-select: target_arr = ", target_arr, " range = ", range_extra)
                if length(range_extra) > 0
                    # if all ranges are int, then it is not a selection
                    if any(Bool[ismask(state,r) for r in range_extra])
                        ranges = mk_ranges([rangeToMask(state, range_extra[i], mk_arraysize(state, arr, i)) for i in 1:length(range_extra)]...)
                      dprintln(env, "inline-select: converted to ranges = ", ranges)
                      arr = mk_select(target_arr, ranges)
                    else
                      dprintln(env, "inline-select: skipped")
                    end
                end
            elseif (def.head === :select)
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
    idx_to = nothing
    if nargs == 2
       args = normalize_args(state, env, args)
    elseif nargs == 4
       args = normalize_args(state, env, args[[2,4]])
    elseif nargs == 5
       args = normalize_args(state, env, args)
       idx_to = args[2]
       idx_from = args[4]
       copy_len = args[5]
       args = args[[1,3]]
       dprintln(env,"got copy!, idx_to=", idx_to, " idx_from=", idx_from, " copy_len=", copy_len)
    else
       error("Expect either 2 or 4 or 5 arguments to copy!, but got " * string(args))
    end
    #args = normalize_args(state, env, nargs == 2 ? args : Any[args[2], args[4]])
    dprintln(env,"got copy!, args=", args)
    argtyp1 = typeOfOpr(state, args[1])
    argtyp2 = typeOfOpr(state, args[2])
    if isArrayType(argtyp1) && isArrayType(argtyp2)
        eltyp1 = eltype(argtyp1)
        eltyp2 = eltype(argtyp2)
        if eltyp1 == eltyp2
            if idx_to == nothing
              expr = mk_mmap!(args, DomainLambda(Type[eltyp1,eltyp2], Type[eltyp1], params->Any[Expr(:tuple, params[2])], state.linfo))
            else # range copy
              to_range = mk_range(state, idx_to, 1, copy_len)
              from_range = mk_range(state, idx_from, 1, copy_len)
              ranges = Any[from_range, to_range]
              args = Any[inline_select(env, state, mk_select(args[1], mk_ranges(to_range))),
                         inline_select(env, state, mk_select(args[2], mk_ranges(from_range)))]
              expr = mk_mmap!(args, DomainLambda(Type[eltyp1,eltyp2], Type[eltyp1], params->Any[Expr(:tuple, params[2])], state.linfo))
            end
            dprintln(env, "turn copy! into mmap! ", expr)
        else
            warn("cannot handle non-matching types ", (eltyp1, eltyp2), " for copy! ", args)
        end
    else
        warn("cannot handle copy! with arguments ", args)
        #expr = mk_copy(args[1])
    end
    expr.typ = argtyp1
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
    local typExp::Union{QuoteNode,DataType,GlobalRef}
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
            if (fname === :stop)
                ret = final
            elseif (fname === :start)
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
        if isArrayType(typ) || (isa(typ, Union) && any(Bool[isArrayType(x) for x in typ.types]))
            return translate_call_mapop(state,env_,typ, fun, args)
        else
            return mk_expr(typ, :call, oldfun, args...)
        end
    end

    if (fun === :map)
        return translate_call_map(state, env_, typ, args)
    elseif (fun === :map!)
        return translate_call_map!(state, env_, typ, args)
    elseif (fun === :reduce)
        return translate_call_reduce(state, env_, typ, args)
    elseif (fun === :cartesianarray)
        return translate_call_cartesianarray(state, env_, typ, args)
    elseif (fun === :cartesianmapreduce)
        return translate_call_cartesianmapreduce(state, env_, typ, args)
    elseif (fun === :broadcast)
        return translate_call_broadcast(state, env_, typ, args)
    elseif (fun === :runStencil)
        return translate_call_runstencil(state, env_, args)
    elseif (fun === :parallel_for)
        return translate_call_parallel_for(state, env_, args)
# legacy v0.3
#    elseif in(fun, topOpsTypeFix) && (typ === Any) && length(args) > 0
#        typ = translate_call_typefix(state, env, typ, fun, args)
    elseif haskey(reduceOps, fun)
        dprintln(env, "haskey reduceOps ", fun)
        return translate_call_reduceop(state, env_, typ, fun, args)
    elseif (fun === :arraysize)
        args = normalize_args(state, env_, args)
        dprintln(env,"got arraysize, args=", args)
        arr = lookupConstVarForArg(state, args[1])
        arr_size_expr::Expr = mk_arraysize(state, arr, args[2:end]...)
        arr_size_expr.typ = typ
        return arr_size_expr
    elseif (fun === :alloc) || (fun === :Array)
        return translate_call_alloc(state, env_, typ, args[1], args[2:end])
    elseif (fun === :copy)
        return translate_call_copy(state, env, args)
    elseif (fun === :sitofp) # typefix hack!
        typ = args[1]
    elseif (fun === :fpext) # typefix hack!
        #println("TYPEFIX ",fun," ",args)
        # a hack to avoid eval
        # typ = eval(args[1])
        typ = eval_dataType(args[1])
    elseif (fun === :getindex) || (fun === :setindex!)
        expr = translate_call_getsetindex(state,env_,typ,fun,args)
    elseif (fun === :getfield) && length(args) == 2
        # convert UpperTriangular variable coming from chol to regular matrix to simplify analysis (lasso example)
        # UpperTriangular has extra index bounds checks but we don't fully check bounds now
        if isa(args[1],LHSVar) && args[2]==QuoteNode(:data) && getType(args[1],state.linfo)<:UpperTriangular
            new_type = getType(args[1],state.linfo).parameters[2]
            updateTyp(state, args[1], new_type)
            dprintln(env, "UpperTriangular replaced with ", new_type)
        end
        if isa(args[1],LHSVar) && args[2]==QuoteNode(:data) && getType(args[1],state.linfo)<:Array
            dprintln(env, "replacing getfield :data")
            return args[1]
        end
        # fix checksquare (lasso example)
        if args[1]==GlobalRef(Base,:LinAlg) && args[2]==QuoteNode(:checksquare)
            dprintln(env, "fixing checksquare")
            return GlobalRef(Base,:checksquare)
        end
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
        if !(fun === :ccall)
            if (fun === :box) && isa(oldargs[2], Expr) # fix the type of arg[2] to be arg[1]
              oldargs[2].typ = typ
            end
        end
        if (fun === :setfield!) && length(oldargs) == 3 && oldargs[2] == QuoteNode(:contents) #&&
            #typeOfOpr(state, oldargs[1]) == Box
            # special handling for setting Box variables
            oldargs = normalize_args(state, env_, oldargs)
            dprintln(env, "got setfield! with Box argument: ", oldargs)
            #assert(isa(oldargs[1], TypedVar))
            typ = typeOfOpr(state, oldargs[3])
            updateTyp(state, oldargs[1], typ)
            updateBoxType(state, oldargs[1], typ)
            # change setfield! to direct assignment
            return mk_expr(typ, :(=), toLHSVar(oldargs[1]), oldargs[3])
        elseif (fun === :getfield) && length(oldargs) == 2
            oldargs = normalize_args(state, env_, oldargs)
            dprintln(env, "got getfield ", oldargs)
            if oldargs[2] == QuoteNode(:contents)
                return toRHSVar(oldargs[1], typ, state.linfo)
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
        return nothing
    end
    return expr
end

# legacy v0.3
#=
function translate_call_typefix(state, env, typ, fun, args::Array{Any,1})
    dprintln(env, " args = ", args, " type(args[1]) = ", typeof(args[1]))
    local typ1
    if (fun === :cat_t)
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
        if (fun === :fptrunc)
            if     (a1 === Float32) typ1 = Float32
            elseif (a1 === Float64) typ1 = Float64
            else throw(string("unknown target type for fptrunc: ", typ1, " args[1] = ", args[1]))
            end
        elseif (fun === :fpsiround)
            if     (a1 === Float32) typ1 = Int32
            elseif (a1 === Float64) typ1 = Int64
                #        if (typ1 === Float32) typ1 = Int32
                #        elseif (typ1 === Float64) typ1 = Int64
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
            return add_expr(start, mul_expr(sub_expr(args[2], 1), step))
        end
    elseif isArrayType(arrTyp)
      ranges = (fun === :getindex) ? args[2:end] : args[3:end]
      expr = Expr(:null)
      dprintln(env, "ranges = ", ranges)
      try
        if any(Bool[ ismask(state, range) for range in ranges])
            dprintln(env, "args is ", args)
            dprintln(env, "ranges is ", ranges)
            #newsize = addGenSym(Int, state.linfo)
            #newlhs = addGenSym(typ, state.linfo)
            etyp = elmTypOf(arrTyp)
            ranges = mk_ranges([rangeToMask(state, ranges[i], mk_arraysize(state, arr, i)) for i in 1:length(ranges)]...)
            dprintln(env, "ranges becomes ", ranges)
            if (fun === :getindex)
                expr = mk_select(arr, ranges)
                # TODO: need to calculate the correct result dimesion
                typ = (typ == Any) ? arrTyp : typ
            else
                args = Any[inline_select(env, state, e) for e in Any[mk_select(arr, ranges), args[2]]]
                var = lookupConstDefForArg(state, args[2])
                var = isa(var, Expr) ? args[2] : var
                vtyp = typeOfOpr(state, var)
                if isArrayType(vtyp) # set to array value
                    evtyp = elmTypOf(vtyp)
                    # f = DomainLambda(Type[etyp, evtyp], Type[etyp], params->Any[Expr(:tuple, params[2])], state.linfo)
                    (body, linfo) = get_lambda_for_arg(state, env, GlobalRef(Base, :convert), Type[Type{etyp}, evtyp])
                    lhs = addFreshLocalVariable(string("ignored"), etyp, 0, linfo)
                    lhsname = CompilerTools.LambdaHandling.getVarDef(lhs, linfo).name
                    params = CompilerTools.LambdaHandling.getInputParameters(linfo)
                    CompilerTools.LambdaHandling.setInputParameters(Symbol[lhsname, params[2]], linfo)
                    body.args = [ mk_expr(Type{etyp}, :(=), lookupLHSVarByName(params[1], linfo), etyp);
                                  body.args...]
                    f = DomainLambda(linfo, body)
                else # set to scalar value
                    pop!(args)
                    # f = DomainLambda(Type[etyp], Type[etyp], params->Any[Expr(:tuple, var)], state.linfo)
                    (body, linfo) = get_lambda_for_arg(state, env, GlobalRef(Base, :convert), Type[Type{etyp}, vtyp])
                    lhs = addFreshLocalVariable(string("ignored"), etyp, 0, linfo)
                    lhsname = CompilerTools.LambdaHandling.getVarDef(lhs, linfo).name
                    params = CompilerTools.LambdaHandling.getInputParametersAsLHSVar(linfo)
                    if isa(var, RHSVar)
                      var = makeCaptured(state, var)
                      var = toLHSVar(var)
                      rhs = addToEscapingVariable(var, linfo, state.linfo)
                    else
                      rhs = var
                    end
                    CompilerTools.LambdaHandling.setInputParameters(Symbol[lhsname], linfo)
                    body.args = [ mk_expr(Type{etyp}, :(=), params[1], etyp);
                                  mk_expr(vtyp, :(=), params[2], rhs);
                                  body.args...]
                    f = DomainLambda(linfo, body)
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
    dprintln(env,"translate_call_mapop: ", fun, " ", args)
    # TODO: check for unboxed array type
    args = normalize_args(state, env, args)
    #etyp = elmTypOf(typ)
    #if (fun === :-) && length(args) == 1
    #    fun = :negate
    #end
    typs = Type[ typeOfOpr(state, arg) for arg in args ]
    elmtyps = Type[ isArrayType(t) ? elmTypOf(t) : t for t in typs ]
    scalar_op = mapOps[fun]
    if haskey(Base.FastMath.fast_op, scalar_op)
        dprintln(env,"translate_call_mapop: found op ", scalar_op, " in fastmath.")
        scalar_op = Base.FastMath.fast_op[scalar_op]
        opr = GlobalRef(Base.FastMath, scalar_op)
    else
        opr = GlobalRef(Base, scalar_op)
    end
    dprintln(env,"translate_call_mapop: before specialize, opr=", opr, " args=", args, " typs=", typs)
    # f = DomainLambda(elmtyps, Type[etyp], params->Any[Expr(:tuple, box_ty(etyp, Expr(:call, opr, params...)))], state.linfo)
    # (body, linfo) = get_lambda_for_arg(state, env, opr, elmtyps)
    func_args = [ Symbol(string("x", i)) for i = 1:length(elmtyps) ]
    func = eval(Expr(:(->), Expr(:tuple, func_args...), Expr(:call, opr, func_args...)))
    (body, linfo) = get_ast_for_lambda(state, env, func, elmtyps)
    if isa(typ, Union)
        dim = 1
        for x in typ.types
            if isArrayType(x)
                dim = x.parameters[2]
            end
        end
        @dprintln(3, "Found result type ", typ)
        typ = Array{body.typ, dim}
        @dprintln(3, "New result type ", typ)
    end
    f = DomainLambda(linfo, body)
    (nonarrays, args, typs, f) = specialize(state, args, typs, f)
    dprintln(env,"translate_call_mapop: after specialize, typs=", typs)
    for i = 1:length(args)
        # need to deepcopy because sharing Expr will cause renaming trouble
        arg_ = deepcopy(inline_select(env, state, args[i]))
        #if arg_ != args[i] && i != 1 && length(args) > 1
        #    error("Selector array must be the only array argument to mmap: ", args)
        #end
        args[i] = arg_
    end
    expr::Expr = endswith(string(fun), '!') ? mk_mmap!(args, f) : mk_mmap(args, f)
    expr = mmapRemoveDupArg!(state, expr)
    if typ <: BitArray
        typ = Array{Bool, typ.parameters[1]}
    end
    expr.typ = typ
    return expr
end

"""
Run type inference and domain process over the income function object.
Return the result AST with a modified return statement, namely, return
is changed to Expr(:tuple, retvals...)
"""
function get_ast_for_lambda(state, env, func::Union{Function,LambdaInfo,TypedVar,Expr}, argstyp)
    dprintln(env, "get_ast_for_lambda func = ", func, " type = ", typeof(func), " argstyp = ", argstyp)
    if isa(func, TypedVar) && func.typ <: Function
        dprintln(env, "func is TypedVar and func.typ is Function")
        # function/closure support is changed in julia 0.5
        lambda = func.typ #.name.primary
    elseif isa(func, Expr) && (func.head === :new)
        dprintln(env, "func is Expr and func.head is :new")
        lambda = func.args[1]
        if isa(lambda, GlobalRef)
            lambda = getfield(lambda.mod, lambda.name)
        end
    else
        dprintln(env, "lambda = func")
        lambda = func
    end
    dprintln(env, "typeof(lambda) = ", typeof(lambda))
    (ast, aty) = lambdaTypeinf(lambda, tuple(argstyp...))
    dprintln(env, "type inferred AST = ", ast)
    dprintln(env, "aty = ", aty)
    # recursively process through domain IR with new state and env
    (linfo, body) = from_lambda(state, env, ast, func)
    params = getInputParameters(linfo)
    dprintln(env, "type inferred AST linfo = ", linfo, " body = ", body)
    dprintln(env, "params = ", params)
    # Check for multiple return statements
    max_label = 0
    rtys = Any[]
    for expr in body.args
        if isa(expr, Expr) && (expr.head === :return)
            rty = Void
            if length(expr.args) > 0
                rty = typeOfOpr(linfo, expr.args[1])
            end
            push!(rtys, rty)
        elseif isa(expr, LabelNode)
            max_label = max(max_label, expr.label)
        end
    end
    @assert (length(rtys) > 0) "cannot find a return statement in body"
    if aty == Any
        # TODO: take a Union of all rtys
        aty = rtys[1]
        dprintln(env, "aty becomes ", aty)
    end
    ret_var = nothing
    lastExp = body.args[end]
    dprintln(env, "rtys = ", rtys, " lastExp = ", lastExp)
    # Turn multiple return into a single return
    if length(rtys) > 1 || !(isa(lastExp, Expr) && (lastExp.head == :return))
        dprintln(env, "Transforming multiple returns into a single return.")
        ret_var = addTempVariable(aty, linfo)
        max_label = max_label + 1
        new_body = Any[]
        for expr in body.args
            if isa(expr, Expr) && (expr.head === :return)
                push!(new_body, mk_expr(expr.typ, :(=), ret_var, expr.args[1]))
                push!(new_body, GotoNode(max_label))
            else
                push!(new_body, expr)
            end
        end
        push!(new_body, LabelNode(max_label))
        push!(new_body, mk_expr(aty, :return, ret_var))
        body.args = new_body
        lastExp = body.args[end]
        dprintln(env, "expanded body with one return is ", body)
    elseif isa(lastExp, Expr) && (lastExp.head === :return) && length(lastExp.args) > 0
        dprintln(env, "A single return at the end.")
        if isa(lastExp.args[1], RHSVar)
            ret_var = toLHSVar(lastExp.args[1])
        elseif isa(lastExp.args[1], Expr)
            ret_var = addTempVariable(aty, linfo)
            body.args[end] = mk_expr(aty, :(=), ret_var, lastExp.args[1])
            lastExp = mk_expr(aty, :return, ret_var)
            push!(body.args, lastExp)
        end
        dprintln(env, "cur body with single return is ", body)
    end
    dprintln(env, "lastExp = ", lastExp)
    # modify the last return statement if it's a tuple
    if isTupleType(aty)
        @assert (ret_var != nothing) "cannot find return value in lastExp = " * string(lastExp)
        dprintln(env, "isTupleType(aty) is true.")
        # take a shortcut if the second last statement is the tuple creation
        exp = body.args[end-1]
        if length(rtys) == 1 && isa(exp, Expr) && exp.head == :(=) && exp.args[1] == ret_var && isa(exp.args[2], Expr) &&
           exp.args[2].head == :call && isBaseFunc(exp.args[2].args[1], :tuple)
            dprintln(env, "second last is tuple assignment, we'll take shortcut")
            pop!(body.args)
            exp.head = :tuple
            exp.args = exp.args[2].args[2:end]
        else
            # create tmp variables to store results
            typs::SimpleVector = aty.parameters
            nvar = length(typs)
            retNodes = GenSym[ addTempVariable(t, linfo) for t in typs ]
            retExprs = Array{Expr}(length(retNodes))
            for i in 1:length(retNodes)
                n = retNodes[i]
                t = typs[i]
                retExprs[i] = mk_expr(t, :(=), n, mk_expr(t, :call, GlobalRef(Base, :getfield), ret_var, i))
            end
            body.args[end] = retExprs[1]
            for i = 2:length(retExprs)
                push!(body.args, retExprs[i])
            end
            push!(body.args, mk_expr(typs, :tuple, retNodes...))
        end
    else
        dprintln(env, "isTupleType(aty) is false.")
        lastExp.head = :tuple
    end
    body.typ = aty
    setReturnType(aty, linfo)
    dprintln(env, "End of get_ast_for_lambda is ", body)
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
    # m = code_typed(getfield(func.mod, func.name), tuple(argstyp...))
    #dprintln(env,"get_lambda_for_arg: ", func, " methods=", m, " argstyp=", argstyp)
    #assert(length(m) > 0)
    gf = getfield(func.mod, func.name)
    dprintln(env, "get_lambda_for_arg GlobalRef: func = ", func, " getfield = ", gf, " type = ", typeof(gf))
    get_ast_for_lambda(state, env, gf, argstyp)
end

# broadcast support for Julia's semantics:
# 1. the input dimensions may not match, output becomes the max of them
# 2. for each dimension d, size(output, d) = size(A, d), if there exists one A in inputs, where size(A, d) > 1, otherwise size(output, d) = 1
# 3. the value in output becomes a fold(op, A[i,j,k], B[i,j,k], ...)
function translate_call_broadcast(state, env, typ, args::Array{Any,1})
    # equivalent to creating an array first, then map! with indices.
    dprintln(env, "got broadcast args=", args)
    # need to retrieve map lambda from inits, since it is already moved out.
    nargs = length(args)
    args = normalize_args(state, env, args)
    @assert (nargs >= 3) "Expect 3 or more arguments to broadcast, but got " * string(args[1:end])
    argtyps = DataType[ typeOfOpr(state, arg) for arg in args[2:end] ]
    dprintln(env, "argtyps =", argtyps)
    inptyps = DataType[ isArrayType(t) ? elmTypOf(t) : t for t in argtyps ]
    (body, linfo) = get_lambda_for_arg(state, env, args[1], inptyps)
    ety = getReturnType(linfo)
    etys = isTupleType(ety) ? [ety.parameters...] : DataType[ety]
    dprintln(env, "etys = ", etys)
    # return dimension is max of all input dimensions
    rdim = maximum([ ndims(atyp) for atyp in argtyps ])
    size_var = Array{RHSVar}(nargs - 1, rdim)
    size_var_inner = Array{RHSVar}(nargs - 1, rdim)
    rtyp = Array{ety, rdim}
    rvar = addFreshLocalVariable(string("out_arr"), rtyp, ISASSIGNED | ISASSIGNEDONCE, state.linfo)
    rsizes = Array{RHSVar}(rdim)
    for j = 1:rdim
        sizes = Any[]
        for i = 1:(nargs - 1)
            inp = args[i + 1]
            inptyp = inptyps[i]
            if inptyp == argtyps[i] # scalar
                push!(sizes, 1)
            else
                size_var[i, j] = addFreshLocalVariable(string("s_", i, "_", j), Int, ISASSIGNED | ISASSIGNEDONCE, state.linfo)
                # size_var will be used by inner lambda
                size_var_inner[i, j] = addToEscapingVariable(toLHSVar(size_var[i, j]), Int, linfo, state.linfo)
                emitStmt(state, mk_expr(Int, :(=), toLHSVar(size_var[i,j]), mk_expr(Int, :call, GlobalRef(Base, :size), inp, j)))
                push!(sizes, size_var[i, j])
            end
        end
        rsizes[j] = addFreshLocalVariable(string("s_", j), Int, ISASSIGNED | ISASSIGNEDONCE, state.linfo)
        emitStmt(state, mk_expr(Int, :(=), toLHSVar(rsizes[j]), mk_expr(Int, :call, GlobalRef(Base, :max), sizes...)))
    end
    emitStmt(state, mk_expr(rtyp, :(=), toLHSVar(rvar), mk_alloc(state, ety, rsizes)))
    pre_body = Any[]
    idx_params = RHSVar[ addFreshLocalVariable(string("idx_", j), Int, ISASSIGNED | ISASSIGNEDONCE, linfo) for j = 1:rdim ]
    idx_syms = Symbol[ lookupVariableName(p, linfo) for p in idx_params ]
    dummy_sym = lookupVariableName(addFreshLocalVariable("dummy", ety, 0, linfo), linfo)
    old_params = getInputParameters(linfo)
    setInputParameters(vcat(dummy_sym, idx_syms), linfo)
    # make the expression that calculates output element value
    for i = 1:(nargs - 1)
        inp = makeCaptured(state, args[i + 1], linfo)
        inp = addToEscapingVariable(toLHSVar(inp, state.linfo), linfo, state.linfo)
        inptyp = inptyps[i]
        if inptyp == argtyps[i] # scalar
            push!(pre_body, mk_expr(inptyp, :(=), toLHSVar(old_params[i], linfo), inp))
        else
            inp_dim = ndims(argtyps[i])
            inp_idx = [ mk_expr(Int, :call, GlobalRef(Base, :min), idx_params[j], size_var_inner[i, j]) for j = 1:inp_dim ]
            #inp_idx = [ add_expr(1, mk_expr(Int, :call, GlobalRef(Base, :rem), sub_expr(idx_params[j], 1), size_var_inner[i, j])) for j = 1:inp_dim ]
            push!(pre_body, mk_expr(inptyp, :(=), toLHSVar(old_params[i], linfo), mk_expr(inptyp, :call, GlobalRef(Base, :unsafe_arrayref), inp, inp_idx...)))
        end
    end
    body.args = vcat(pre_body, body.args)
    domF = DomainLambda(linfo, body)
    expr::Expr = mk_mmap!([rvar], domF, true)
    expr.typ = rtyp
    return expr
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
        if (typeOfOpr(state, args[i]) === Int)
            iterations = args[i]
        else
            borderExp = args[i]
        end
    elseif i + 1 <= nargs
        iterations = args[i]
        borderExp = args[i+1]
    end
    if (borderExp === nothing)
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
    assert((dimExp_e.head === :call) && isBaseFunc(dimExp_e.args[1], :tuple))
    dimExp = dimExp_e.args[2:end]
    ndim = length(dimExp)   # num of dimensions
    argstyp = Any[ Int for i in 1:ndim ]

    lambdaDef = args_[1]
    #lambdaDef::Expr = lookupConstDefForArg(state, args_[1])
    dprintln(env, "lambdaDef = ", lambdaDef)
    (body, linfo) = get_lambda_for_arg(state, env, lambdaDef, argstyp)     # first argument is the lambda
    #(body, linfo) = get_lambda_for_arg(state, env, args_[1], argstyp)     # first argument is the lambda
    ety = getReturnType(linfo)
    #etys = isTupleType(ety) ? [ety.parameters...] : DataType[ety]
    dprintln(env, "ety = ", ety)
    #@assert all([ isa(t, DataType) for t in etys ]) "cartesianarray expects static type parameters, but got "*dump(etys)
    # create tmp arrays to store results
    #arrtyps = Type[ Array{t, ndim} for t in etys ]
    #dprintln(env, "arrtyps = ", arrtyps)
    #tmpNodes = Array{Any}(length(arrtyps))
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
        @assert (isa(tup, Expr) && (tup.head === :call) &&
                 isBaseFunc(tup.args[1], :tuple)) "Expect reduction arguments to cartesianmapreduce to be tuples, but got " * string(tup)
        redfunc = lookupConstDefForArg(state, tup.args[2])
        redvar = lookupConstVarForArg(state, tup.args[3]) # tup.args[3]
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
        nvar = addToEscapingVariable(redvarname, rvtyp, nlinfo, state.linfo)
        nval = replaceExprWithDict!(translate_call_copy(state, env, Any[toRHSVar(redvar, rvtyp, state.linfo)]), Dict{LHSVar,Any}(Pair(redvar, nvar)), state.linfo, AstWalk)
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
    assert((dimExp_e.head === :call) && isBaseFunc(dimExp_e.args[1], :tuple))
    dimExp = Any[simplify(state, x) for x in dimExp_e.args[2:end]]
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
    tmpNodes = Array{Any}(length(arrtyps))
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
    dummy_params = Array{Symbol}(length(etys))
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
        sizeVars = Array{TypedVar}(num_dim)
        linfo = LambdaVarInfo()
        for i = 1:num_dim
            sizeVars[i] = addFreshLocalVariable(string("red_dim_size"), Int, ISASSIGNED | ISASSIGNEDONCE, state.linfo)
            dimExp = mk_expr(arrtyp, :call, GlobalRef(Base, :select_value),
                         mk_expr(Bool, :call, GlobalRef(Base, :eq_int), red_dim[1], i),
                         1, mk_arraysize(state, arr, i))
            emitStmt(state, mk_expr(Int, :(=), sizeVars[i], dimExp))
            sizeVars[i] = addToEscapingVariable(lookupVariableName(sizeVars[i], state.linfo), Int, linfo, state.linfo)
        end
        redparam = gensym("redvar")
        rednode = addLocalVariable(redparam, arrtyp, ISASSIGNED | ISASSIGNEDONCE, linfo)
        setInputParameters(Symbol[redparam], linfo)
        neutral_body = Expr(:body,
                           Expr(:(=), rednode, mk_alloc(state, etyp, sizeVars)),
                           mk_mmap!([rednode], DomainLambda(Type[etyp], Type[etyp], params->Any[Expr(:tuple, neutralelt)], LambdaVarInfo())),
                           Expr(:tuple, rednode))
        neutral = DomainLambda(linfo, neutral_body)
        outtyp = arrtyp
        opr = GlobalRef(Base, fun)
        params = Symbol[ gensym(s) for s in [:x, :y]]
        linfo = LambdaVarInfo()
        for i in 1:length(params)
            addLocalVariable(params[i], outtyp, 0, linfo)
        end
        setInputParameters(params, linfo)
        params = [ toRHSVar(x, outtyp, linfo) for x in params ]
        setReturnType(outtyp, linfo)
        #(inner_body, inner_linfo) = get_lambda_for_arg(state, env, opr, [etyp, etyp])
        (inner_body, inner_linfo) = make_fake_reduce_lambda(state, env, opr, etyp)
        inner_dl = DomainLambda(inner_linfo, inner_body)
        # inner_dl = DomainLambda(Type[etyp, etyp], Type[etyp], params->Any[Expr(:tuple, box_ty(etyp, Expr(:call, opr, params...)))], LambdaVarInfo())
        inner_expr = mk_mmap!(params, inner_dl)
        inner_expr.typ = outtyp
        f = DomainLambda(linfo, Expr(:body, mk_expr(outtyp, :tuple, inner_expr)))
    else
        red_dim = []
        neutral = neutralelt
        outtyp = etyp
        opr = GlobalRef(Base, fun)
        #(inner_body, inner_linfo) = get_lambda_for_arg(state, env, opr, [etyp, etyp])
        (inner_body, inner_linfo) = make_fake_reduce_lambda(state, env, opr, etyp)
        f = DomainLambda(inner_linfo, inner_body)
    end
    # turn reduce(z, getindex(a, ...), f) into reduce(z, select(a, ranges(...)), f)
    arr = inline_select(env, state, arr)
    expr = mk_reduce(neutral, arr, f, red_dim...)
    expr.typ = outtyp
    return expr
end

"""
Create a fake lambda for reduction functions so that the backend can replace
with high performance implementation (e.g. std::min, OpenMP reductions in CGen).
The required information is lost with capturing and inlining the operation lambda.
"""
function make_fake_reduce_lambda(state, env, opr::GlobalRef, etyp::Type)
    linfo = LambdaVarInfo()
    setInputParameters([:x,:y], linfo)
    setReturnType(etyp, linfo)
    addLocalVariable(:x,etyp,0,linfo)
    addLocalVariable(:y,etyp,0,linfo)
    body = mk_expr(etyp,:body, mk_expr(etyp, :tuple,
      mk_expr(etyp,:call, opr, toLHSVar(:x,linfo), toLHSVar(:y,linfo))))
    return (body, linfo)
end

function translate_call_fill!(state, env, typ, args::Array{Any,1})
    args = normalize_args(state, env, args)
    @assert length(args)==2 "fill! should have 2 arguments"
    arr = args[1]
    ival = lookupConstDefForArg(state, args[2])
    ival = isa(ival, Expr) ? args[2] : ival
    ityp = typeOfOpr(state, ival)
    atyp = typeOfOpr(state, arr)
    etyp = elmTypOf(atyp)
    #domF = DomainLambda(typs, typs, params->Any[Expr(:tuple, ival)], state.linfo)
    (body, linfo) = get_lambda_for_arg(state, env, GlobalRef(Base, :convert), Type[Type{etyp}, ityp])
    lhs = addFreshLocalVariable(string("ignored"), etyp, 0, linfo)
    lhsname = CompilerTools.LambdaHandling.getVarDef(lhs, linfo).name
    params = CompilerTools.LambdaHandling.getInputParametersAsLHSVar(linfo)
    if isa(ival, RHSVar)
        ival = makeCaptured(state, ival)
        rhs = addToEscapingVariable(ival, linfo, state.linfo)
    else
        rhs = ival
    end
    CompilerTools.LambdaHandling.setInputParameters(Symbol[lhsname], linfo)
    body.args = [ mk_expr(Type{etyp}, :(=), params[1], etyp);
                  mk_expr(ityp, :(=), params[2], rhs);
                  body.args...]
    domF = DomainLambda(linfo, body)
    expr = mmapRemoveDupArg!(state, mk_mmap!([arr], domF))
    expr.typ = typ
    return expr
end

function translate_call_parallel_for(state, env, args::Array{Any,1})
    (lambda, ety) = lambdaTypeinf(args[1], (Int, ))
    ast = from_expr(env.cur_module, lambda)
    loopvars = [ if isa(x, Expr) x.args[1] else x end for x in ast.args[1] ]
    etys = [Int for _ in length(loopvars)]
    body = ast.args[3]
    ranges = args[2:end]
    assert(isa(body, Expr) && (body.head === :body))
    lastExp = body.args[end]
    assert(isa(lastExp, Expr) && (lastExp.head === :return))
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
    if (fun.mod === Core.Intrinsics) || ((fun.mod === Core) &&
       ((fun.name === :Array) || (fun.name === :arraysize) || (fun.name === :getfield) || (fun.name === :setfield!)))
        expr = translate_call_symbol(state, env, typ, head, fun, oldargs, fun.name, args)
    elseif ((fun.mod === Core) || (fun.mod === Base)) && ((fun.name === :typeassert) || (fun.name === :isa))
        # remove all typeassert
        dprintln(env, "got typeassert args = ", args)
        args = normalize_args(state, env_, args)
        expr = (fun.name === :isa) ? (getType(args[1], state.linfo) == args[2]) : args[1]
        typ = typeOfOpr(state, expr)
        if typ == Any
            typ = lookupConstDefForArg(state, args[2])
            dprintln(env, "got type =", typ)
            if isa(typ, Type)
                updateTyp(state, expr, typ)
            else
                dprintln(env, " skip updateTyp for typ = ", typ)
            end
        end
        dprintln(env, "typ = ", typ, " expr = ", expr)
        # @assert (typ == args[2]) "typeassert finds mismatch " *string(expr)* " and " *string(args[2])
    elseif ((fun.mod === Core) || (fun.mod === Base)) && (fun.name === :convert)
        # fix type of convert
        args = normalize_args(state, env_, args)
        if isa(args[1], Type)
            typ = args[1]
        end
    elseif (fun.mod === Base)
        if (fun.name === :afoldl) && haskey(afoldlDict, typeOfOpr(state, args[1]))
            opr = GlobalRef(Base, afoldlDict[typeOfOpr(state, args[1])])
            dprintln(env, "afoldl operator detected = ", args[1], " opr = ", opr)
            expr = Base.afoldl((x,y)->box_ty(typ, Expr(:call, opr, [x, y]...)), args[2:end]...)
            dprintln(env, "translated expr = ", expr)
        elseif (fun.name === :copy!)
            expr = translate_call_copy!(state, env, args)
        elseif (fun.name === :copy)
            expr = translate_call_copy(state, env, args)
    # legacy v0.3
    #=
        elseif (fun.name === :checkbounds)
            dprintln(env, "got ", fun.name, " args = ", args)
            if length(args) == 2
                expr = translate_call_checkbounds(state,env_,args)
            end
    =#
        elseif (fun.name === :getindex) || (fun.name === :setindex!) # not a domain operator, but still, sometimes need to shortcut it
            expr = translate_call_getsetindex(state,env_,typ,fun.name,args)
    # legacy v0.3
    #    elseif (fun.name === :assign_bool_scalar_1d!) || # args = (array, scalar_value, bitarray)
    #           (fun.name === :assign_bool_vector_1d!)    # args = (array, getindex_bool_1d(array, bitarray), bitarray)
    #        expr = translate_call_assign_bool(state,env_,typ,fun.name, args)
        elseif (fun.name === :fill!)
            return translate_call_fill!(state, env_, typ, args)
        elseif (fun.name === :_getindex!) # see if we can turn getindex! back into getindex
            if isa(args[1], Expr) && args[1].head == :call && isBaseFunc(args[1].args[1], :ccall) &&
                (args[1].args[2] == :jl_new_array ||
                (isa(args[1].args[2], QuoteNode) && args[1].args[2].value == :jl_new_array))
                expr = mk_expr(typ, :call, :getindex, args[2:end]...)
            end
        elseif fun.name==:println || fun.name==:print # fix type for println
            typ = Void
            expr = mk_expr(typ, head, oldfun, oldargs...)
        elseif fun.name==:typed_hcat
            # convert typed_hcat to regular hcat
            new_fun = GlobalRef(Base,:hcat)
            # ignore 1st arg which is type
            new_args = normalize_args(state, env, args[2:end])
            expr = mk_expr(typ, head, new_fun, new_args...)
        elseif fun.name==:typed_vcat
            # convert typed_hcat to regular hcat
            new_fun = GlobalRef(Base,:vcat)
            # ignore 1st arg which is type
            new_args = normalize_args(state, env, args[2:end])
            expr = mk_expr(typ, head, new_fun, new_args...)
        elseif fun.name==:ctranspose && !(typ.parameters[1]<:Complex)
            # Julia doesn't translate ctranspose to regular transpose for
            # UpperTriangular type so we fix it here
            new_fun = GlobalRef(Base,:transpose)
            new_args = normalize_args(state, env, args)
            dprintln(env, "ctranspose replaced with transpose ")
            expr = mk_expr(typ, head, new_fun, new_args...)
        elseif fun.name==:checksquare
            # remove checksquare (lasso example)
            return Expr(:meta)
        end
    elseif (fun.mod === Base.Broadcast)
        if (fun.name === :broadcast_shape)
            dprintln(env, "got ", fun.name)
            args = normalize_args(state, env_, args)
            expr = mk_expr(typ, :assertEqShape, args...)
        end
    elseif (fun.mod === Base.Random) #skip, let cgen handle it
    elseif (fun.mod === Base.LinAlg) || (fun.mod === Base.LinAlg.BLAS) || (fun.mod === Base.LinAlg.LAPACK) #skip, let cgen handle it
    elseif (fun.mod === Base.Math)
        # NOTE: we simply bypass all math functions for now
        dprintln(env,"by pass math function ", fun, ", typ=", typ)
        # Fix return type of math functions
        if (typ === Any) && length(args) > 0
            dprintln(env,"fix type for ", expr, " from ", typ, " => ", args[1].typ)
            typ = args[1].typ
        end
        #    elseif (fun.mod === Base) && (fun.name === :arraysize)
        #     args = normalize_args(state, env_, args)
        #    dprintln(env,"got arraysize, args=", args)
        #   expr = mk_arraysize(args...)
        #    expr.typ = typ
    elseif (fun.mod === API.Lib.NoInline)
        oldfun = Base.resolve(GlobalRef(Base, fun.name))
        dprintln(env,"Translate function from API.Lib back to Base: ", oldfun)
        oldargs = normalize_args(state, env_, oldargs)
        expr = mk_expr(typ, head, oldfun, oldargs...)
    elseif isdefined(fun.mod, fun.name)
        gf = getfield(fun.mod, fun.name)
        if isa(gf, Function) && !(fun.mod === Core) # fun != GlobalRef(Core, :(===))
            dprintln(env,"function to offload: ", fun, " methods=", methods(gf))
            args = normalize_args(state, env_, args)
            args_typ = map(x -> typeOfOpr(state, x), args)
            _accelerate(gf, tuple(args_typ...))
            expr = mk_expr(typ, head, oldfun, args...)
        else
            dprintln(env,"function ", fun, " not offloaded.")
        end
    else
        dprintln(env,"function call not translated: ", fun, ", and is not found!")
    end
    #if isa(expr, Void)
    #    oldargs = normalize_args(state, env_, oldargs)
    #    expr = mk_expr(typ, head, oldfun, oldargs...)
    #end
    return expr
end

function from_return(state, env, expr)
    local env_ = nextEnv(env)
    local head = expr.head
    local typ  = expr.typ
    local args = normalize_args(state, env, expr.args)
    if length(args) == 0
        args=Any[nothing]
    end
    assert(length(args) == 1)
    # fix return type, the assumption is there is either 0 or 1 args.
    typ = typeOfOpr(state, args[1])
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
function from_expr(cur_module :: Module, ast)
    @dprintln(3, "Entry point from_expr ", typeof(ast), " ", ast)
    assert(isfunctionhead(ast))
    linfo, body = from_expr_tiebreak(emptyState(), newEnv(cur_module), ast)
    return linfo, body
end

function from_expr(state::IRState, env::IREnv, ast::LambdaInfo)
    #dprintln(env, "from_expr: LambdaInfo inferred = ", ast.inferred)
    #if !ast.inferred
        # we return this unmodified since we want the caller to
        # type check it before conversion.
    #    return ast
    #else
    #    ast = uncompressed_ast(ast)
        # (tree, ty)=Base.typeinf(ast, argstyp, ())
    #end
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
    return Base.resolve(ast, force=true)
end

function from_expr(state::IRState, env::IREnv, ast::Union{Symbol,TypedVar})
    # if it is global const, we replace it with its const value
    def = lookupDefInAllScopes(state, ast)
    name = lookupVariableName(ast, state.linfo)
    if (def === nothing) && isdefined(env.cur_module, name) && ccall(:jl_is_const, Int32, (Any, Any), env.cur_module, name) == 1
        def = getfield(env.cur_module, name)
        if isbits(def) && !isa(def, IntrinsicFunction) && !isa(def, Function)
            return def
        end
    end
    typ = typeOfOpr(state, toLHSVar(ast))
    if isa(ast, TypedVar) && ast.typ != typ && typ != Void
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
    if (head === :lambda)
        (linfo, body) = from_lambda(state, env, ast)
        return LambdaVarInfoToLambda(linfo, body.args, ParallelAccelerator.DomainIR.AstWalk)
    elseif (head === :body)
        return from_body(state, env, ast)
    elseif (head === :(=))
        return from_assignment(state, env, ast)
    elseif (head === :return)
        return from_return(state, env, ast)
    elseif (head === :call) || (head === :invoke)
        return from_call(state, env, ast)
    elseif (head === :foreigncall)
        return from_foreigncall(state, env, ast)
        # TODO: catch domain IR result here
    # legacy v0.3
    #elseif (head === :call1)
    #    return from_call(state, env, ast)
        # TODO?: tuple
    # legacy v0.3
    # :method is not expected here
    #=
    elseif (head === :method)
        # change it to assignment
        ast.head = :(=)
        n = length(args)
        ast.args = Array{Any}(n-1)
        ast.args[1] = args[1]
        for i = 3:n
            ast.args[i-1] = args[i]
        end
        return from_assignment(state, env, ast)
    =#
    elseif (head === :line)
        # skip
    elseif (head === :new)
        return TypedExpr(typ, :new, args[1], normalize_args(state, env, args[2:end])...)
    elseif (head === :boundscheck)
        # skip or remove?
        return nothing
    elseif (head === :type_goto)
        # skip?
    elseif (head === :gotoifnot)
        # specific check to take care of artifact from comprhension-to-cartesianarray translation
        # gotoifnot (Base.slt_int(1,0)) label ===> got label
        if length(args) == 2 && isa(args[1], Expr) && (args[1].head === :call) &&
            (args[1].args[1] === GlobalRef(Base, :slt_int)) && args[1].args[2] == 1 && args[1].args[3] == 0
            dprintln(env, "Match gotoifnot shortcut!")
            return GotoNode(args[2])
        else # translate arguments
          ast.args = normalize_args(state, env, args)
          if ast.args[1] == true
            return nothing
          elseif ast.args[1] == false
            return GotoNode(args[2])
          end
        end
        # ?
    elseif (head === :inbounds)
        # skip
    elseif (head === :meta)
        # skip
    elseif (head === :llvmcall)
        # skip
    elseif (head === :simdloop)
        # skip
    elseif (head === :static_parameter)
        p = args[1]
        @assert (isa(p, Int)) "Expect constant Int argument to :static_parameter, but got " * string(ast)
        ast = getStaticParameterValue(p, state.linfo)
    elseif (head === :static_typeof)
        typ = getType(args[1], state.linfo)
        return typ
    elseif head in exprHeadIgnoreList
        # other packages like HPAT can generate new nodes like :alloc, :join
    else
        throw(string("ParallelAccelerator.DomainIR.from_expr: unknown Expr head :", head))
    end
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::LabelNode)
    # clear defs for every basic block.
    deleteNonConstDefs(state)
    return ast
end

function from_expr(state::IRState, env::IREnv, ast::ANY)
    @dprintln(2, " not handled ", ast, " :: ", typeof(ast))
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

    if externalCallback!=nothing
        ret = externalCallback(x,dw)
        @dprintln(4,"External callback ret = ", ret)
        if ret != CompilerTools.AstWalker.ASTWALK_RECURSE
            return ret
        end
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
        map!((a) -> AstWalker.AstWalk(a, AstWalkCallback, dw), args[1], args[1])
        map!((a) -> AstWalker.AstWalk(a, AstWalkCallback, dw), args[2], args[2])
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

    if externalLiveCB!=nothing
        ret = externalLiveCB(ast)
        @dprintln(4,"External live callback ret = ", ret)
        if ret != nothing
            return ret
        end
    end

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
#    elseif head == :call || head == :invoke
#        fun = getCallFunction(ast)
#        if isBaseFunc(fun, :arrayref) || isBaseFunc(fun, :arraysize)
#            expr_to_process = Any[]
#            args = getCallArguments(ast)
#            for i = 2:length(args)
#                push!(expr_to_process, args[i])
#            end
#            return expr_to_process
#        end
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

    if externalAliasCB!=nothing
        ret = externalAliasCB(ast)
        @dprintln(4,"External alias callback ret = ", ret)
        if ret != nothing
            return ret
        end
    end

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
        if isa(tmp, Expr) && (tmp.head === :select) # selecting a range
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
        if !((isa(iterations, Integer) && iterations == 1) || (krnStat.rotateNum == 0))
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
    elseif (head === :tomask)
        return AliasAnalysis.lookup(state, toLHSVar(args[1]))
    elseif (head === :arraysize)
        return AliasAnalysis.NotArray
    elseif (head === :tuple)
        return AliasAnalysis.NotArray
    elseif (head === :alloc)
        return AliasAnalysis.next_node(state)
    elseif (head === :copy)
        return AliasAnalysis.next_node(state)
    end

    return nothing

end

function dir_alias_cb(ast::ANY, state, cbdata)
    @dprintln(4,"dir_alias_cb ")
end

end
