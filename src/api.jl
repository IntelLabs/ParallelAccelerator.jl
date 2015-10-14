baremodule API

using Base
import Base: call

eval(x) = Core.eval(API, x)

@noinline function setindex!{T}(A::DenseArray{T}, args...) 
  Base.setindex!(A, args...)
end

function setindex!(A, args...) 
  Base.setindex!(A, args...)
end

@noinline function getindex{T}(A::DenseArray{T}, args...) 
  Base.getindex(A, args...)
end

function getindex(A, args...) 
  Base.getindex(A, args...)
end

# Unary operators/functions
const unary_operators = Symbol[
    :-, :+, :acos, :acosh, :angle, :asin, :asinh, :atan, :atanh, :cbrt,
    :cis, :cos, :cosh, :exp10, :exp2, :exp, :expm1, :lgamma,
    :log10, :log1p, :log2, :log, :sin, :sinh, :sqrt, :tan, :tanh, 
    :sum, :prod, :pointer]

for f in unary_operators
    @eval begin
        @noinline function ($f){T<:Number}(A::DenseArray{T})
            (Base.$f)(A)
        end
        function ($f)(A...)
            (Base.$f)(A...)
        end
    end
end

# Binary operators/functions
const binary_operators = Symbol[
    :-, :+, :.+, :.-, :.*, :./, :.\, :.%, :.<<, :.>>, :div, :mod, :rem, :&, :|, :$]

for f in binary_operators
    @eval begin
        @noinline function ($f){T}(A::T, B::DenseArray{T})
            (Base.$f)(A, B)
        end
        @noinline function ($f){T}(B::DenseArray{T}, A::T)
            (Base.$f)(B, A)
        end
        @noinline function ($f){T}(A::DenseArray{T}, B::DenseArray{T})
            (Base.$f)(A, B)
        end
        function ($f)(A, B)
            (Base.$f)(A, B)
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

const operators = vcat(unary_operators, binary_operators, Symbol[:setindex!, :getindex])

for opr in operators
  @eval export $opr
end

end

