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

baremodule Lib

using Base
import Base: (==), copy
#import Base: call, (==), copy

using ..cartesianmapreduce, ..cartesianarray, ..sum, ..(.*), ..map, ..map!

type MT{T}
  val :: T
  idx :: Int
end

function ==(x::MT, y::MT)
  x.val == y.val && x.idx == y.idx
end

function copy(x::MT)
  MT(x.val, x.idx)
end

@inline function indmin{T<:Number}(A::DenseArray{T})
  m = MT(typemax(T), 1)
  cartesianmapreduce((length(A),),
    (x -> begin
            if m.val > x.val
              m.val = x.val
              m.idx = x.idx
            end
            return m
          end,
     m)) do i
    if A[i] < m.val
      m.val = A[i]
      m.idx = i
    end
    0
  end
  return m.idx
end

@inline function indmax{T<:Number}(A::DenseArray{T})
  m = MT(typemin(T), 1)
  cartesianmapreduce((length(A),),
    (x -> begin
            if m.val < x.val
              m.val = x.val
              m.idx = x.idx
            end
            return m
          end,
     m)) do i
    if A[i] > m.val
      m.val = A[i]
      m.idx = i
    end
    0
  end
  return m.idx
end

@inline function sumabs2(A::DenseArray)
  sum(A .* A)
end

@inline function diag(A::DenseMatrix)
  d::Int = min(size(A, 1), size(A, 2))
  cartesianarray(Tuple{eltype(A)}, (d,)) do i
     A[i, i]
  end
end

@inline function diagm(A::DenseVector)
  d::Int = size(A, 1)
  cartesianarray(Tuple{eltype(A)}, (d, d)) do i, j
    # the assignment below is a hack to avoid mutiple return in body
    v = i == j ? A[i] : zero(eltype(A))
  end
end

@inline function trace(A::DenseMatrix)
  sum(diag(A))
end

@inline function scale(A::DenseMatrix, b::DenseVector)
  m::Int, n::Int = size(A)
  cartesianarray(Tuple{eltype(A)}, (m, n)) do i, j
    A[i,j] * b[j]
  end
end

@inline function scale(b::DenseVector, A::DenseMatrix)
  m::Int, n::Int = size(A)
  cartesianarray(Tuple{eltype(A)}, (m, n)) do i, j
    b[i] * A[i,j]
  end
end

@inline function eye(m::Int, n::Int)
  cartesianarray(Tuple{Float64}, (m, n)) do i, j
    # the assignment below is a hack to avoid mutiple return in body
    v = i == j ? 1.0 : 0.0
  end
end

@inline function eye(m::Int)
  eye(m, m)
end

@inline function eye(A::DenseMatrix)
  eye(size(A, 1), size(A, 2))
end

@inline function repmat{T<:Number}(A::Array{T, 3}, m::Int, n::Int, l::Int)
  s::Int = size(A, 1)
  t::Int = size(A, 2)
  u::Int = size(A, 3)
  cartesianarray(Tuple{eltype(A)}, (m * s, n * t, l * u)) do i, j, k
    A[1 + rem(i - 1, s), 1 + rem(j - 1, t), 1 + rem(k - 1, u)]
  end
end

@inline function repmat(A::DenseMatrix, m::Int, n::Int, l::Int)
  s::Int = size(A, 1)
  t::Int = size(A, 2)
  cartesianarray(Tuple{eltype(A)}, (m * s, n * t, l)) do i, j, k
    A[1 + rem(i - 1, s), 1 + rem(j -1, t)]
  end
end

@inline function repmat(A::DenseVector, m::Int, n::Int, l::Int)
  s::Int = size(A, 1)
  cartesianarray(Tuple{eltype(A)}, (m * s, n, l)) do i, j, k
    A[1 + rem(i - 1, s)]
  end
end

@inline function repmat(A::DenseVector, m::Int, n::Int)
  s::Int = size(A, 1)
  cartesianarray(Tuple{eltype(A)}, (m * s, n)) do i, j
    A[1 + rem(i - 1, s)]
  end
end

@inline function repmat(A::DenseMatrix, m::Int, n::Int)
  s::Int, t::Int = size(A)
  cartesianarray(Tuple{eltype(A)}, (m * s, n * t)) do i, j
    A[1 + rem(i - 1, s), 1 + rem(j - 1, t)]
  end
end

@inline function repmat(A::DenseVector, m::Int)
  repmat(A, m, 1)
end

@inline function repmat(A::DenseMatrix, m::Int)
  repmat(A, m, 1)
end

baremodule NoInline

using Base
import Base: (==)
#import Base: call, (==)

@noinline function rand(args...)
  Base.Random.rand(args...)
end

@noinline function randn(args...)
  Base.Random.rand(args...)
end

@noinline function rand!(args...)
  Base.Random.rand(args...)
end

@noinline function randn!(args...)
  Base.Random.rand(args...)
end

end
import .NoInline

@inline function rand(dims::Int...)
  _pa_rand_gen_arr = Array{Float64}(dims...)
  map!(x -> NoInline.rand(Float64)::Float64, _pa_rand_gen_arr)
  return _pa_rand_gen_arr
end

@inline function randn(dims::Int...)
  _pa_rand_gen_arr = Array{Float64}(dims...)
  map!(x -> NoInline.randn()::Float64, _pa_rand_gen_arr)
  return _pa_rand_gen_arr
end

@inline function rand(t::Tuple)
  _pa_rand_gen_arr = Array{Float64}(t)
  map!(x -> NoInline.rand(Float64)::Float64, _pa_rand_gen_arr)
  return _pa_rand_gen_arr
end

@inline function randn(t::Tuple)
  _pa_rand_gen_arr = Array{Float64}(t)
  map!(x -> NoInline.randn()::Float64, _pa_rand_gen_arr)
  return _pa_rand_gen_arr
end

@inline function rand(T::Type{Float32}, d::Int, dims::Int...)
  _pa_rand_gen_arr = Array{Float32}(d, dims...)
  map!(x -> NoInline.rand(Float32)::Float32, _pa_rand_gen_arr)
  return _pa_rand_gen_arr
end

@inline function rand(T::Type, d::Int, dims::Int...)
  _pa_rand_gen_arr = Array{T}(d, dims...)
  map!(x -> NoInline.rand(T)::T, _pa_rand_gen_arr)
  return _pa_rand_gen_arr
end

#@inline function randn(T::Type, d::Int, dims::Int...)
#  _pa_rand_gen_arr = Array{T}(d, dims...)
#  map!(x -> NoInline.randn(T)::T, _pa_rand_gen_arr)
#  return _pa_rand_gen_arr
#end

@inline function rand(r::AbstractRNG, T::Type, d::Int, dims::Int...)
  _pa_rand_gen_arr = Array{T}(d, dims...)
  map!(x -> NoInline.rand(r, T)::T, _pa_rand_gen_arr)
  return _pa_rand_gen_arr
end

#@inline function randn(r::AbstractRNG, T::Type, d::Int, dims::Int...)
#  _pa_rand_gen_arr = Array{T}(d, dims...)
#  map!(x -> NoInline.randn(r, T)::T, _pa_rand_gen_arr)
#  return _pa_rand_gen_arr
#end

@inline function rand!{T}(_pa_rand_gen_arr::DenseArray{T})
  map!(x -> NoInline.rand(T)::T, _pa_rand_gen_arr)
end

#@inline function randn!{T}(_pa_rand_gen_arr::DenseArray{T})
#  map!(x -> NoInline.randn(T)::T, _pa_rand_gen_arr)
#end

@inline function rand!{T}(r::AbstractRNG, _pa_rand_gen_arr::DenseArray{T})
  map!(x -> NoInline.rand(r, T)::T, _pa_rand_gen_arr)
end

#@inline function randn!{T}(r::AbstractRNG, _pa_rand_gen_arr::DenseArray{T})
#  map!(x -> NoInline.randn(r, T)::T, _pa_rand_gen_arr)
#end

@inline function rand(args...)
  Base.Random.rand(args...)
end

@inline function randn(args...)
  Base.Random.rand(args...)
end

@inline function rand!(args...)
  Base.Random.rand(args...)
end

@inline function randn!(args...)
  Base.Random.rand(args...)
end

export indmin, indmax, sumabs2
export diag, diagm, trace, scale, eye, repmat, rand, randn, rand!, randn!

end
