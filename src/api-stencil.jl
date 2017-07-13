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

module Stencil

import Base: getindex, setindex!
import CompilerTools
export runStencil, @runStencil

type AbstractStencilArray
  dimension :: Int
  sizes :: Array{Int, 1}
  baseIdx :: Array{Int, 1}
  src :: Any
  borderStyle :: Symbol
end

function getindex(asa::AbstractStencilArray, idx...)
  idx = [idx...]
  local n = length(idx)
  assert(n == asa.dimension)
  for i = 1:n
    idx[i] = idx[i] + asa.baseIdx[i]
    if (idx[i] < 1 || idx[i] > asa.sizes[i])
      if asa.borderStyle == :oob_src_zero
        return 0
      elseif asa.borderStyle == :oob_wraparound
        idx[i] = mod(idx[i] - 1, asa.sizes[i]) + 1
      else
        return NaN         # TODO: better ways to symbolize out-of-bounds
      end
    end
  end
  asa.src[idx...]
end

function setindex!(asa::AbstractStencilArray, value, idx...)
  idx = [idx...]
  local n = length(idx)
  assert(n == asa.dimension)
  for i = 1:n
    idx[i] = idx[i] + asa.baseIdx[i]
    if idx[i] < 1 || idx[i] > asa.sizes[i]
      throw(string("Index ", idx, " is out of bounds!"))
    end
  end
  if (value === NaN)
    if asa.borderStyle == :oob_dst_zero
      asa.src[idx...] = 0
    end
  else
    asa.src[idx...] = value
  end
end

"""
"runStencil" takes arguments in the form of "(kernel_function, A, B, C, ...,
iteration, border_style)" where "kernel_function" is a lambda that represent
stencil kernel, and "A", "B", "C", ... are arrays (of same dimension and size)
that will be traversed by the stencil, and they must match the number of
arguments of the stencil kernel. "iteration" and "border_style" are optional,
and if present, "iteration" is the number of steps to repeat the stencil
computation, and "border_style" is a symbol of the following:

    :oob_src_zero, returns zero from a read when it is out-of-bound. 

    :oob_dst_zero, writes zero to destination array in an assigment should any read in its right-hand-side become out-of-bound.

    :oob_wraparound, wraps around the index (with respect to source array dimension and sizes) of a read operation when it is out-of-bound.

    :oob_skip, skips write to destination array in an assignment should any read in its right-hand-side become out-of-bound.

The "kernel_function" should take a set of arrays as input, and in the function
body only index them with relative indices as if there is a cursor (at index 0)
traversing them. It may contain more than one assignment to any input arrays,
but such writes must always be indexed at 0 to guarantee write operation never
goes out-of-bound. Also care must be taken when the same array is both read
from and written into in the kernel function, as they'll result in
non-deterministic behavior when the stencil is parallelized by ParallelAccelerator. 

The "kernel_function" may optionally access scalar variables from outerscope,
but no write is permitted.  It may optinally contain a return statement, which
acts as a specification for buffer swapping when it is an interative stencil
loop (ISL). The number of arrays returned must match the input arrays.  If it
is not an ISL, it should always return nothing.

This function is a reference implementation of stencil in native Julia for two
purposes: to verify expected return result, and to make sure user code type
checks. It runs very very slow, so any real usage should go through ParallelAccelerator
optimizations.

"runStencil" always returns nothing.
"""
function runStencil(inputs...)
  #  (func, buf, ..., iterations, border)
  arrs = Array[]
  assert(length(inputs) >= 3)
  kernelFunc = inputs[1]
  iterations = 1
  borderStyle = nothing
  for i = 2:length(inputs)
    typ = typeof(inputs[i])
    if typ <: AbstractArray
      push!(arrs, inputs[i])
    elseif (typ === Int)
      iterations = inputs[i]
    else
      borderStyle = inputs[i]
    end
  end
  narrs = length(arrs)
  assert(narrs > 1)
  local sizes = [size(arrs[1])...]
  local n = length(sizes)
  local indices = Array{Int}(n)
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
  return nothing
end

type KernelStat
  dimension :: Int             # number of dimensions of the stencil
  shapeMax  :: Array{Int,1}    # max extent of the stencil for each dimension
  shapeMin  :: Array{Int,1}    # min extent of the stencil for each dimension
  idxSym    :: Array{Symbol,1} # (fresh) symbol of index variable 
  bufSym    :: Array{Symbol,1} # (fresh) symbol of buffer variables
  swapSym   :: Union{Void, Array{Symbol, 1}}
end

# Analyze a stencil kernel specified as a lambda expression.
# Return both the kernel stat, and the modified kernel LHS expression.
# Note that the input kernel is modified in-place.
function analyze_kernel(krn, esc)
  assert(krn.head == :(->))
  local stat = ()
  local params = krn.args[1]  # parameter of the kernel lambda, must be a tuple of symbols
  local bufs
  if isa(params, Expr) && (params.head === :tuple)
    bufs = Symbol[ arg for arg in params.args ] # all buffers, must be an array of symbols
  else isa(params, Symbol)
    bufs = Symbol[ params ]
  end
  local nbufs = length(bufs)
  bufSym = Symbol[ gensym(string("b", i)) for i=1:nbufs ]
  bufMap = Dict{Symbol,Symbol}(zip(bufs, bufSym))
  local expr = krn.args[2]
  function traverse(expr)     # traverse expr to find the places where arrSym is refernced
    assert(isa(expr, Expr))
    if (expr.head === :ref)
      if haskey(bufMap, expr.args[1]) # one of input bufers
        expr.args[1] = bufMap[expr.args[1]] # modify the reference to actual source array
        local dim = length(expr.args) - 1
        @assert (dim <= 10) "the dimension must be no greater than 10"
        if (stat === ())         # create stat if not already exists
          local idxSym = [ gensym(string("i", i)) for i in 1:dim ]
          stat = KernelStat(dim, zeros(dim), zeros(dim), idxSym, bufSym, nothing) 
        else                    # if stat already exists, check if the dimension matches
          @assert (dim == stat.dimension) "number of dimensions do not match"
        end
        for i = 1:dim           # update extents and index calculation in expr
          @assert (isa(expr.args[i+1], Int)) "array indices must be int literals"
          local idx = Int(expr.args[i+1])
          stat.shapeMax[i] = max(idx, stat.shapeMax[i])
          stat.shapeMin[i] = min(idx, stat.shapeMin[i])
          expr.args[i+1] = :($(stat.idxSym[i]) + $(expr.args[i+1]))
        end
      end
    elseif (expr.head === :return)
      swapSym = Symbol[]
      args = expr.args[1]
      if isa(args, Expr) && (args.head === :tuple)
        args = args.args
      elseif isa(args, Symbol)
        args = Symbol[args]
      end
      for s in args
        @assert (isa(s, Symbol)) "return value must be plain variable"
        i = 1
        while i <= nbufs
          if bufs[i] == s break end
          i += 1
        end
        @assert (i <= nbufs) "return value must be one of the kernel parameters"
        push!(swapSym, bufSym[i])
      end
      @assert (length(swapSym) == nbufs) "number of return values must match number parameters"
      stat.swapSym = swapSym
       # turn it into something harmless
      expr.head = :block
      expr.args = Any[]
    else                        # recurse when expr is not array ref 
      for i in 1:length(expr.args)
        e = expr.args[i]
        if isa(e, Expr)
          traverse(e)
        elseif isa(e, Symbol)
          if !(expr.head == :line || expr.head == :method) && !(expr.head == :call && i == 1) && !haskey(bufMap, e)
            expr.args[i] = esc(e)
          end
        end
      end
    end
  end
  traverse(expr)
  @assert (stat != ()) "Fail to analyze stencil kernel"
  return stat, expr, nbufs
end

# Helper function to create nested loops with an innermost body
function nestedLoop(dim, indices, iters, body)
  function mkLoop(i, body)
    if i > dim
      body
    else
        mkLoop(i+1, :(for $(indices[i]) = $(iters[i]); $(body); end))
    end
  end
  mkLoop(1, body)
end

# turn an array of Expr blocks into an a single block of Exprs
function liftQuote(exprs::Array{Expr,1})
  args = vcat([ (expr.head === :block) ? expr.args : expr for expr in exprs ]...)
  Expr(:block, args...)
end

function specializeBorder(borderSty, borderCheckF, modF, krnExpr)
  oob_dst_zero = borderSty == :oob_dst_zero
  oob_src_zero = borderSty == :oob_src_zero
  oob_wraparound = borderSty == :oob_wraparound
  function traverse(expr)
    if (expr.head === :(=)) && isa(expr.args[1], Expr) && (expr.args[1].head === :ref)
      lhs = expr.args[1]
      rhs = expr.args[2]
      if oob_dst_zero 
        rhs = 0
      elseif isa(rhs, Expr) 
        rhs = traverse(rhs)
      end
      return Expr(expr.head, lhs, rhs)
    elseif (expr.head === :ref)
      if oob_src_zero
        return Expr(:if, borderCheckF(expr.args[2:end]), expr, 0)
      elseif oob_wraparound
        return Expr(expr.head, expr.args[1], modF(expr.args[2:end])...)
      end
    end
    args = [ isa(e, Expr) ? traverse(e) : e for e in expr.args ]
    return Expr(expr.head, args...)
  end
  traverse(krnExpr)
end

# A macro version of runStencil that unfolds stencil call into nested sequential loops.
# Note that it has yet to handle border settings.
macro runStencil(krn, args...)
  translateStencil(krn, Any[x for x in args], esc)
end

function translateStencil(krn, args::Array, esc)
  local steps = 1
  local borderSty = :oob_skip
  stat, krnExpr, nbufs = analyze_kernel(krn, esc)
  @assert (length(args) >= nbufs) "Expect number of array arguments to runStencil to be no less than that of the kernel"
  # get the real buffer arguments
  bufs = args[1:nbufs]
  # get iteration and border style from args
  args = args[(nbufs+1):end]
  if length(args) == 2
    steps = args[1]
    borderSty = args[2]
  elseif length(args) == 1
    if isa(args[1], Expr) && args[1].head == :quote
      borderSty = args[1]
    else
      steps = args[1]
    end
  end
  # check border style
  if isa(borderSty, Expr)
    try 
      borderSty = eval(borderSty)
    catch e
      error("Expect border style to be :oob_src_zero, :oob_dst_zero, :oob_wraparound, or :oob_skip")
    end
  end
  @assert isa(borderSty, Symbol) "Expect border style to be :oob_src_zero, :oob_dst_zero, :oob_wraparound, or :oob_skip"
  # produce the expanded stencil loop
  local tmpSym  = [ gensym("tmp") for i = 1:nbufs ]
  local idxSym  = stat.idxSym
  local bufSym  = stat.bufSym
  local swapSym = stat.swapSym
  local sizeSym = [ gensym(string("len", i)) for i = 1:stat.dimension ]
  local stepSym = gensym("step")
  local bufInitExpr = liftQuote([ :($(bufSym[i]) = $(esc(bufs[i]))) for i = 1:nbufs ])
  local sizeTuple = Expr(:tuple, sizeSym...)
  local sizeInitExpr = :($(sizeTuple) = size($(bufSym[1])))
  # border region
  local innerIterExpr = [ :((1-$(stat.shapeMin[i])) : ($(sizeSym[i]) - $(stat.shapeMax[i]))) for i = 1:stat.dimension ]
  local borderIterExpr = [ :(1:$(sizeSym[i])) for i = 1:stat.dimension ]
  local innerCheckF(idx) = Expr(:call, GlobalRef(Base,:&), [ Expr(:call, GlobalRef(Base,:in), idx[i], innerIterExpr[i]) for i = 1:stat.dimension ]...)
  local borderCheckF(idx) = Expr(:call, GlobalRef(Base,:&), [ Expr(:call, GlobalRef(Base,:in), idx[i], borderIterExpr[i]) for i = 1:stat.dimension ]...)
  local modF(idx) = [ :((($(idx[i]) + $(sizeSym[i]) - 1) % $(sizeSym[i])) + 1) for i = 1:stat.dimension ]
  local borderKrnExpr = specializeBorder(borderSty, borderCheckF, modF, krnExpr)
  local borderExpr = borderSty == :oob_skip ? :() : 
                    nestedLoop(stat.dimension, idxSym, borderIterExpr, :(if $(innerCheckF(idxSym)) else $(borderKrnExpr) end))
  # inner region
  local loopExpr = nestedLoop(stat.dimension, stat.idxSym, innerIterExpr, krnExpr)
  local swapExpr = (swapSym === nothing) ? :() :
                     liftQuote(vcat([ :($(tmpSym[i]) = $(swapSym[i])) for i = 1:nbufs ],
                                    [ :($(bufSym[i]) = $(tmpSym[i])) for i = 1:nbufs ]))
  local expr = quote
    $(bufInitExpr)
    $(sizeInitExpr)
    for $(stepSym) = 1:$(esc(steps))
      $(borderExpr)
      @inbounds $(loopExpr)
      $(swapExpr)
    end
  end
  return expr
end

"""
This function is a AstWalker callback.
"""
function process_node(node, state, top_level_number, is_top_level, read)
  if !isa(node,Expr)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
  end
  if node.head == :call && node.args[1] == :runStencil
    return translateStencil(node.args[2], node.args[3:end], x -> x)
  else
    return CompilerTools.AstWalker.ASTWALK_RECURSE
  end
end


"""
Translate all comprehension in an AST into equivalent code that uses cartesianarray call.
"""
macro comprehend(ast)
  AstWalker.AstWalk(ast, process_node, nothing)
  Core.eval(current_module(), ast)
end



end

