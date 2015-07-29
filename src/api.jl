baremodule API

using Base

# export .+, .-, .*, ./, .\, .%, .<<, .>>, div, mod, rem, &, |, $, log10, sin, cos, exp
export cartesianarray, runStencil

eval(x) = Core.eval(API, x)

for f in (:.+, :.-, :.*, :./, :.\, :.%, :.<<, :.>>, :div, :mod, :rem, :&, :|, :$)
    @eval begin
        @noinline function ($f){T}(A::Number, B::StridedArray{T})
            (Base.$f)(A, B)
        end
        @noinline function ($f){T}(B::StridedArray{T}, A::Number)
            (Base.$f)(B, A)
        end
        @noinline function ($f){T}(A::StridedArray{T}, B::StridedArray{T})
            (Base.$f)(A, B)
        end
    end
end

for f in (:-, :log10, :sin, :cos, :exp)
    @eval begin
        @noinline function ($f){T<:Number}(A::AbstractArray{T})
            (Base.$f)(A)
        end
    end
end

function cartesianarray(body, T, ndims)
  a = Array(T, ndims...)
  cartesianmap((idx...) -> a[idx...] = body(idx...), ndims)
  a
end

function cartesianarray(body, T1, T2, ndims)
  a = Array(T1, ndims...)
  b = Array(T2, ndims...)
  cartesianmap(ndims) do idx...
      (u, v) = body(idx...)
      a[idx...] = u
      b[idx...] = v
  end
  return a, b
end

function cartesianarray(body, T1, T2, T3, ndims)
  a = Array(T1, ndims...)
  b = Array(T2, ndims...)
  c = Array(T3, ndims...)
  cartesianmap(ndims) do idx...
      (u, v, w) = body(idx...)
      a[idx...] = u
      b[idx...] = v
      c[idx...] = w
  end
  return a, b, c
end

function runStencil(inputs...)
  #  (func, buf, ..., iterations, border)
  arrs = Array[]
  assert(length(inputs) >= 3)
  kernelFunc = inputs[1]
  iterations = 1
  borderStyle = nothing
  for i = 2:length(inputs)
    typ = typeof(inputs[i])
    if isa(typ, DataType) && is(typ.name, Array.name)
      push!(arrs, inputs[i])
    elseif is(typ, Int)
      iterations = inputs[i]
    else
      borderStyle = inputs[i]
    end
  end
  narrs = length(arrs)
  # println("borderStyle: ", borderStyle)
  assert(narrs > 1)
  local sizes = [size(arrs[1])...]
  local n = length(sizes)
  local indices = Array(Int, n)
  local bufs = [ AbstractStencilArray(n, sizes, indices, arr, borderStyle) for arr in arrs ]
  for steps = 1 : iterations
    for i = 1:n
      indices[i] = 1
    end
    local done = false
    local ret = nothing
    while !done
      try
        ret = kernelFunc(bufs...)
      catch e
        #ignore
      end
      i=1
      while i<=n
        if indices[i] < sizes[i]
          indices[i] = indices[i] + 1
          break
        else
          indices[i] = 1
          if i < n
            i = i + 1
          else
            done = true
            break
          end
        end
      end
    end
    if isa(ret, Tuple) bufs = ret end
  end
  if isa(typeof(bufs), Tuple)
    return ntuple(length(bufs), i -> bufs[i].src)
  else
    return nothing
  end
end
end

