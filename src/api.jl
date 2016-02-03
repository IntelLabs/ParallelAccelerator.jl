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
import Base: call

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

@inline function getindex(A::DataType, args...)
  A[ x for x in args ]
end

@inline function getindex(A, args...) 
  Base.getindex(A, args...)
end

# Unary operators/functions
const unary_map_operators = Symbol[
    :-, :+, :acos, :acosh, :angle, :asin, :asinh, :atan, :atanh, :cbrt,
    :cis, :cos, :cosh, :exp10, :exp2, :exp, :expm1, :lgamma,
    :log10, :log1p, :log2, :log, :sin, :sinh, :sqrt, :tan, :tanh, 
    :abs, :erf, :isnan, :rand!, :randn!]

const reduce_operators = Symbol[:sum, :prod, :minimum, :maximum, :any, :all]

const unary_operators = vcat(unary_map_operators, reduce_operators, Symbol[:copy])

@inline sum(A::DenseArray{Bool}) = sum(1 .* A)
@inline sum(A::DenseArray{Bool}, x::Int) = sum(1 .* A, x)

@noinline function pointer(args...)
    Base.pointer(args...)
end

# reduction across a dimension
for f in reduce_operators
    @eval begin
        @noinline function ($f){T<:Number}(A::DenseArray{T}, x::Int)
            (Base.$f)(A, x)
        end
    end
end

for f in unary_operators
    @eval begin
        @noinline function ($f){T<:Number}(A::DenseArray{T})
            (Base.$f)(A)
        end
        @inline function ($f)(A...)
            (Base.$f)(A...)
        end
    end
end

for f in unary_map_operators
    @eval begin
        @noinline function ($f){T<:Number}(A::T)
            (Base.$f)(A)
        end
    end
end

# Binary operators/functions
const binary_map_operators = Symbol[ :*, :/,
    :-, :+, :.+, :.-, :.*, :./, :.\, :.%, :.>, :.<, :.<=, :.>=, :.==, :.<<, :.>>, :.^, 
    :div, :mod, :rem, :&, :|, :$, :min, :max]

const binary_operators = binary_map_operators

for f in binary_operators
    @eval begin
        @noinline function ($f){T1<:Number,T2<:Number}(A::T1, B::DenseArray{T2})
            (Base.$f)(A, B)
        end
        @noinline function ($f){T1<:Number,T2<:Number}(B::DenseArray{T1}, A::T2)
            (Base.$f)(B, A)
        end
    end
    if f != :* && f != :/
        @eval @noinline function ($f){T1<:Number,T2<:Number}(A::DenseArray{T1}, B::DenseArray{T2})
            (Base.$f)(A, B)
        end
    end
    if f != :- && f != :+
        @eval begin
            @inline function ($f)(args...)
                (Base.$f)(args...)
            end
        end
    end
end

function cartesianarray(body, T, ndims)
  a = Array(T, ndims...)
  for I in CartesianRange(ndims)
    a[I.I...] = body(I.I...)
  end
  a
end

function cartesianarray(body, T1, T2, ndims)
  a = Array(T1, ndims...)
  b = Array(T2, ndims...)
  for I in CartesianRange(ndims)
    (u, v) = body(I.I...)
    a[I.I...] = u
    b[I.I...] = v
  end
  return a, b
end

function cartesianarray(body, T1, T2, T3, ndims)
  a = Array(T1, ndims...)
  b = Array(T2, ndims...)
  c = Array(T3, ndims...)
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

const operators = Set(vcat(unary_operators, binary_operators, Symbol[:setindex!, :getindex,:__hps_data_source_HDF5,:__hps_kmeans, :__hps_LinearRegression, :__hps_NaiveBayes]))

for opr in operators
  @eval export $opr
end

include("api-capture.jl")
include("api-stencil.jl")

import .Stencil.runStencil

export cartesianarray, parallel_for, runStencil

end

