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

baremodule API

using Base
#import Base: call

if VERSION >= v"0.6.0-pre"
using SpecialFunctions
end

eval(x) = Core.eval(API, x)

@noinline function setindex!{T}(A::DenseArray{T}, args...)
  Base.setindex!(A, args...)
end

@inline function setindex!(A, args...)
  Base.setindex!(A, args...)
end

@noinline function getindex{T}(A::DenseArray{T}, args...)
  Base.getindex(A, args...)
end

@inline function getindex{T}(A::Type{T}, X...)
  T[ X[i] for i=1:length(X) ]
end

@inline function getindex(A, args...)
  Base.getindex(A, args...)
end

@noinline function reshape{T}(A::DenseArray{T}, args)
  Base.reshape(A, args...)
end

# inline reshape to version with dims as tuple for easier handling in compiler
@inline function reshape{T}(A::DenseArray{T}, args...)
  reshape(A, args)
end

# Unary operators/functions
const unary_map_operators_normal = Symbol[
    :-, :+, :acos, :acosh, :angle, :asin, :asinh, :atan, :atanh, :cbrt,
    :cis, :cos, :cosh, :exp10, :exp2, :exp, :expm1, :lgamma,
    :log10, :log1p, :log2, :log, :sin, :sinh, :sqrt, :tan, :tanh,
    :abs, :isnan]

const special_functions = Symbol[:erf]

const unary_map_operators = vcat(unary_map_operators_normal, special_functions)

const reduce_operators = Symbol[:sum, :prod, :minimum, :maximum, :any, :all]

const unary_operators = vcat(unary_map_operators, reduce_operators, Symbol[:copy])

# Binary operators/functions
const comparison_map_operators = Symbol[ :.>, :.<, :.<=, :.>=, :.== ]

const binary_map_operators = vcat(comparison_map_operators, Symbol[ :*, :/,
    :-, :+, :.+, :.-, :.*, :./, :.\, :.%, :.<<, :.>>, :.^,
    :div, :mod, :rem, :&, :|, :$, :min, :max])

const binary_operators = binary_map_operators

const elem_ops = Symbol[ :.-, :.+, :.*, :./, :.>, :.<, :.<=, :.>=, :.==, :.\, :.%, :.<<, :.>>, :.^ ]
const elem_bare_ops = Symbol[ :-, :+, :*, :/, :>, :<, :<=, :>=, :(==), :\, :%, :<<, :>>, :^ ]
const elem_map = Dict{Symbol,Symbol}(zip(elem_ops,elem_bare_ops))

const rename_op_from = Symbol[ :-, :+, :*, :/, :.-, :.+, :.*, :./, :.>, :.<, :.<=, :.>=, :.==, :.\, :.%, :.<<, :.>>, :.^, :&, :|, :$ ]
const rename_op_to = Symbol[ :pa_api_sub, :pa_api_add, :pa_api_mul, :pa_api_div, :pa_api_elem_sub, :pa_api_elem_add, :pa_api_elem_mul, :pa_api_elem_div, :pa_api_elem_gt, :pa_api_elem_lt, :pa_api_elem_lte, :pa_api_elem_gte, :pa_api_elem_equal, :pa_api_elem_back, :pa_api_elem_mod, :pa_api_elem_lshift, :pa_api_elem_rshift, :pa_api_elem_carat, :pa_api_ampersand, :pa_api_pipe, :pa_api_dollar ]
const rename_forward = Dict{Symbol,Symbol}(zip(rename_op_from,rename_op_to))
const rename_back = Dict{Symbol,Symbol}(zip(rename_op_to,rename_op_from))

@inline sum(A::DenseArray{Bool}) = sum(pa_api_elem_mul(1,A))
@inline sum(A::DenseArray{Bool}, x::Int) = sum(pa_api_elem_mul(1,A), x)

function rename_if_needed(f)
    if haskey(rename_forward, f)
        return rename_forward[f]
    else
        return f
    end
end

function rename_back_if_needed(f)
    if haskey(rename_back, f)
        return rename_back[f]
    else
        return f
    end
end

# reduction across a dimension
for f in reduce_operators
    nf = rename_if_needed(f)
    @eval begin
        @noinline function ($nf){T<:Number}(A::DenseArray{T}, x::Int)
            (Base.$f)(A, x)
        end
    end
end

for f in unary_operators
    nf = rename_if_needed(f)
    @eval begin
        @noinline function ($nf){T<:Number}(A::DenseArray{T})
            (Base.$f)(A)
        end
    end
    if !(in(f, binary_operators))
        @eval begin
            @inline function ($nf)(A...)
                (Base.$f)(A...)
            end
        end
    end
end

for f in unary_map_operators_normal
    nf = rename_if_needed(f)
    @eval begin
        @noinline function ($nf){T<:Number}(A::T)
            (Base.$f)(A)
        end
    end
end

for f in special_functions
    nf = rename_if_needed(f)
    if VERSION >= v"0.6.0-pre"
        @eval begin
            @noinline function ($nf){T<:Number}(A::T)
                (SpecialFunctions.$f)(A)
            end
        end
    else
        @eval begin
            @noinline function ($nf){T<:Number}(A::T)
                (Base.$f)(A)
            end
        end
    end
end

for f in binary_operators
    nf = rename_if_needed(f)
    bare_op = f
    if haskey(elem_map, bare_op)
        bare_op = elem_map[bare_op]
    end

    @eval begin
        @noinline function ($nf){T1<:Number,T2<:Number}(A::T1, B::DenseArray{T2})
            (Base.$f)(A, B)
        end
    end

    if bare_op == f
		@eval begin
			@noinline function ($nf){T1<:Number,T2<:Number}(B::DenseArray{T1}, A::T2)
				(Base.$f)(B, A)
			end
		end
    else
		@eval begin
			@noinline function ($nf){T1<:Number,T2<:Number}(B::DenseArray{T1}, A::T2)
				broadcast($bare_op, B, A)
			end
		end
    end

    if f != :* && f != :/
        if bare_op == f
			@eval begin
				@noinline function ($nf){T1<:Number,T2<:Number}(A::DenseArray{T1}, B::DenseArray{T2})
					(Base.$f)(A, B)
				end
			end
        else
			@eval begin
				@noinline function ($nf){T1<:Number,T2<:Number}(A::DenseArray{T1}, B::DenseArray{T2})
					broadcast($bare_op, A, B)
				end
			end
        end
    end

    @eval begin
        @inline function ($nf)(args...)
            (Base.$f)(args...)
        end
    end
end

@noinline function cartesianmapreduce(body, ndims, reductions...)
  # reductions are ignored in sequential semantics
  for (redFunc, redVar) in reductions
      # redVar holds initial value that has to be neutral
      vs = deepcopy(redVar)
      @assert (vs == redFunc(redVar)) ("initial value " * string(vs) * " not idempotent with respect to the reduction function")
  end
  for I in CartesianRange(ndims)
    body(I.I...)
  end
end

if VERSION >= v"0.5.0-dev+5306"
@inline function to_tuple_type(t)
  if t <: Tuple
    t
  else
    Tuple{t}
  end
end

@inline function cartesianarray(body, L::Int)
   if 1<0
     x=body(L)
   end
   cartesianarray(body, to_tuple_type(typeof(x)), (L,))
end

@inline function cartesianarray(body, L::Int, M::Int)
   if 1<0
     x=body(L, M)
   end
   cartesianarray(body, to_tuple_type(typeof(x)), (L,M,))
end

@inline function cartesianarray(body, L::Int, M::Int, N::Int)
   if 1<0
     x=body(L, M, N)
   end
   cartesianarray(body, to_tuple_type(typeof(x)), (L,M,N))
end

end

function cartesianarray{T}(body, ::Type{Tuple{T}}, ndims)
  a = Array{T}(ndims...)
  for I in CartesianRange(ndims)
    a[I.I...] = body(I.I...)
  end
  a
end

function cartesianarray{T1,T2}(body, ::Type{Tuple{T1, T2}}, ndims)
  a = Array{T1}(ndims...)
  b = Array{T2}(ndims...)
  for I in CartesianRange(ndims)
    (u, v) = body(I.I...)
    a[I.I...] = u
    b[I.I...] = v
  end
  return a, b
end

function cartesianarray{T1,T2,T3}(body, ::Type{Tuple{T1, T2, T3}}, ndims)
  a = Array{T1}(ndims...)
  b = Array{T2}(ndims...)
  c = Array{T3}(ndims...)
  for I in CartesianRange(ndims)
    (u, v, w) = body(I.I...)
    a[I.I...] = u
    b[I.I...] = v
    c[I.I...] = w
  end
  return a, b, c
end

function parallel_for(loopvar, range, body)
  throw("Not Implemented")
end

@noinline function map{T<:Number}(f::Function, a::DenseArray{T})
    Base.map(f, a)
end

@noinline function map{T<:Number}(f::Function, a::DenseArray{T}, b...)
    Base.map(f, a, b...)
end

@noinline function map{T<:Array}(f::Function, a::Array{T})
    Base.map(f, a)
end

@noinline function map{T<:Array}(f::Function, a::Array{T}, b...)
    Base.map(f, a, b...)
end

@noinline function map(f, a::Range)
    Base.map(f, a)
end

@noinline function map(f, a::Range, b...)
    Base.map(f, a, b...)
end

@inline function map(f, a...)
    Base.map(f, a...)
end

@noinline function map!{T<:Number}(f::Function, a::DenseArray{T})
    Base.map!(f, a)
end

@noinline function map!{T<:Number}(f::Function, a::DenseArray{T}, b...)
    Base.map!(f, a, b...)
end

@inline function map!(f, a...)
    Base.map!(f, a...)
end

@noinline function reduce{T<:Number}(f::Function, v::T, a::DenseArray{T})
    Base.reduce(f, v, a)
end

@noinline function reduce{T<:Number}(f::Function, v::T, a::DenseArray{T}, b...)
    Base.reduce(f, v, a, b...)
end

@inline function reduce(f, a...)
    Base.reduce(f, a...)
end

@noinline function broadcast{T<:Number}(f::Function, a::DenseArray{T}, b...)
    Base.broadcast(f, a, b...)
end

@noinline function broadcast{T<:Number}(f::Function, a::T, b...)
    Base.broadcast(f, a, b...)
end

@inline function broadcast(f, a...)
    Base.broadcast(f, a...)
end

operators = Set(vcat(unary_operators, binary_operators,
    Symbol[:map, :map!, :reduce, :broadcast, :setindex!, :getindex, :reshape]))

for opr in operators
    @eval export $opr
end

include("api-stencil.jl")

@noinline function pointer(args...)
    Base.pointer(args...)
end

macro par(args...)
    na = length(args)
    @assert (na > 0) "Expect a for loop as argument to @par"
    redVars = Array{Symbol}(na-1)
    redOps = Array{Symbol}(na-1)
    loop = args[end]
    if !isa(loop,Expr) || !(loop.head === :for)
        error("malformed @par loop")
    end
    if !isa(loop.args[1], Expr)
        error("maltformed for loop")
    end
    for i = 1:na-1
        @assert (isa(args[i], Expr) && args[i].head == :call) "Expect @par reduction in the form of var(op), but got " * string(args[i])
        v = args[i].args[1]
        op = args[i].args[2]
        #println("got reduction variable ", v, " with function ", op)
        @assert (isa(v, Symbol)) "Expect reduction variable to be symbols, but got " * string(v)
        @assert (isa(op, Symbol)) "Expect reduction operator to be symbols, but got " * string(op)
        redVars[i] = v
        redOps[i] = op
    end
    if loop.args[1].head == :block
        loopheads = loop.args[1].args
    else
        loopheads = Any[loop.args[1]]
    end
    # Here is a no-op, real implementation is in Capture module, and will run as an OptFramework pass
    esc(loop)
end

import .Stencil.runStencil
export @par, cartesianmapreduce, cartesianarray, parallel_for, runStencil, pointer

include("api-lib.jl")
importall .Lib

function enableLib()
  global operators
  oprs = delete!(Set(names(Lib)), :Lib)
  union!(operators, oprs)
  for opr in oprs
    @eval export $opr
  end
end

function disableLib()
  global operators
  setdiff!(operators, delete!(Set(names(Lib)), :Lib))
end

enableLib()

include("api-capture.jl")

end
