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

type KernelStat
  dimension :: Int             # number of dimensions of the stencil
  shapeMax  :: Array{Int,1}    # max extent of the stencil for each dimension
  shapeMin  :: Array{Int,1}    # min extent of the stencil for each dimension
  bufSym    :: Array{GenSym,1} # (fresh) buffer symbols
  idxSym    :: Array{GenSym,1} # (fresh) symbol of the index variable into the source image
  strideSym :: Array{GenSym,1} # (fresh) symbol of the strides data of source image
  borderSty :: Symbol          # Symbol :oob_dst_zero, :oob_src_zero, :oob_skip, :oob_wraparound
  rotateNum :: Int             # number of buffers that is affected by rotation
  modified  :: Array{Int,1}    # which buffer is modified
end

supportedBorderStyle = Set([:oob_dst_zero, :oob_src_zero, :oob_skip, :oob_wraparound])

# Analyze a stencil kernel specified as a lambda expression.
# Return both the kernel stat, and the modified kernel LHS expression.
# Note that the input kernel is modified in-place.
# NOTE: currently only handle kernel specified as: a -> c * a[...] + ...
function analyze_kernel(state::IRState, bufTyps::Array{Type, 1}, krn::Expr, borderSty::Symbol)
  #assert(krn.head == symbol("->"))
  dprintln(3, "typeof krn = ", typeof(krn), " ", krn.head, " :: ", typeof(krn.head), " ", object_id(krn.head), " ", object_id(:lambda)) 
  assert(isa(krn, Expr))
  assert(is(krn.head, :lambda))
  local stat = ()
  # warn(string("krn.args[1]=", krn.args[1]))
  local arrSyms = krn.args[1] # parameter of the kernel lambda
  local narrs = length(arrSyms)
  local bufSyms = Array(GenSym, narrs)
  local arrSymDict = Dict{Symbol,GenSym}()
  for i in 1:length(arrSyms)
    if isa(arrSyms[i], Expr) && arrSyms[i].head == :(::) # Expr in the form (x :: t).
      arrSyms[i] = arrSyms[i].args[1]
    end
    assert(isa(arrSyms[i], Symbol))
    bufSyms[i] = addGenSym(bufTyps[i], state.linfo)
    arrSymDict[arrSyms[i]] = bufSyms[i]
  end
  local bufSymSet = Set{GenSym}(bufSyms)
  local expr = krn.args[3]
  assert(isa(expr, Expr) && expr.head == :body)
  # warn(string("got arrSymDict = ", arrSymDict))
  # traverse expr to fix types, and places where arrSyms is refernced
  function traverse(expr)
    # recurse
    # warn(string("expr=", expr, ", head=", expr.head, " args=", expr.args))
    for i = 1:length(expr.args)
      e = expr.args[i]
      if isa(e, Expr)
        traverse(e)
      elseif isa(e, Symbol) && haskey(arrSymDict, e)
        expr.args[i] = arrSymDict[e]
      elseif isa(e, SymbolNode) && haskey(arrSymDict, e.name)
        expr.args[i] = arrSymDict[e.name]
      end
    end
    # fix assignment and fill in variable type
    if is(expr.head, :(=))
      lhs = expr.args[1]
      if isa(lhs, SymbolNode) lhs = lhs.name end
      rhs = expr.args[2]
      typ = typeOfOpr(state, rhs)
      expr.typ = typ
      # warn(string("lhs=",lhs, " typ=",typ))
      updateDef(state, lhs, rhs)
    end
    # fix getindex and setindex!, note that their operand may already have been replaced by bufSyms, which are SymGen.
    if is(expr.head, :call) && isa(expr.args[1], GlobalRef) && 
       (is(expr.args[1].name, :getindex) || is(expr.args[1].name, :setindex!) ||
        is(expr.args[1].name, :arrayref) || is(expr.args[1].name, :arrayset)) && 
       isa(expr.args[2], GenSym) && in(expr.args[2], bufSymSet)
      local isGet = is(expr.args[1].name, :arrayref) || is(expr.args[1].name, :getindex)
      #(bufSym, bufTyp) = arrSymDict[expr.args[2].name] # modify the reference to actual source array
      bufSym = expr.args[2]
      elmTyp = elmTypOf(getType(bufSym, state.linfo))
      local idxOffset = isGet ? 2 : 3
      local dim = length(expr.args) - idxOffset
      assert(dim <= 10)      # arbitrary limit controlling the total num of dimensions
      if is(stat, ())         # create stat if not already exists
        local idxSym = [ addGenSym(Int, state.linfo) for i in 1:dim ]
        local strideSym = [ addGenSym(Int, state.linfo) for i in 1:dim ]
        stat = KernelStat(dim, zeros(dim), zeros(dim), bufSyms, idxSym, strideSym, borderSty, 0, Int[])
      else                    # if stat already exists, check if the dimension matches
        assert(dim == stat.dimension)
      end
      for i = 1:dim           # update extents and index calculation in expr
        local idx = Int(expr.args[idxOffset+i])
        stat.shapeMax[i] = max(idx, stat.shapeMax[i])
        stat.shapeMin[i] = min(idx, stat.shapeMin[i])
        expr.args[idxOffset+i] = mk_expr(Int, :call, TopNode(:add_int), stat.idxSym[i], idx)
      end
      # local idx1D = nDimTo1Dim(expr.args[(idxOffset+1):end], stat.strideSym)
      expr.args = isGet ? [ TopNode(:unsafe_arrayref), expr.args[2], expr.args[(idxOffset+1):end]...] :
                          [ TopNode(:unsafe_arrayset), expr.args[2], expr.args[3], expr.args[(idxOffset+1):end]... ]
      # fix numerical coercion when converting setindex! into unsafe_arrayset
      if is(expr.args[1].name, :unsafe_arrayset)
          if typeOfOpr(state, expr.args[3]) != elmTyp 
            expr.args[3] = mk_expr(elmTyp, :call, GlobalRef(Base, symbol(string(elmTyp))), expr.args[3])
          end
      end

      if !isGet
        v = expr.args[2]
        if isa(v, SymbolNode) v = v.name end
        for i = 1:length(stat.bufSym)
          if stat.bufSym[i] == v
            push!(stat.modified, i)
          end
        end
      end
    end
  end
  traverse(expr)
  # remove LineNumberNode, :line, and tuple assignment from expr
  #warn(string("kernel expr = ", expr))
  body = Array(Any, 0)
  for e in expr.args
    if isa(e, LineNumberNode)
    elseif isa(e, Expr) && is(e.head, :line)
    # Skip tuple assignments in stencil kernel since codegen can't handle it.
    # TODO: Ensure it is only safe when the LHS tuple is used in return statement.
    elseif isa(e, Expr) && is(e.head, :(=)) && istupletyp(e.typ)
    else
      push!(body, e)
    end
  end
  # fix return statments
  expr.args = body
  local lastExpr = expr.args[end]
  assert(lastExpr.head == :return)
  # default to returning nothing
  #warn(string("lastExpr=",lastExpr))
  rhs = nothing
  if (isa(lastExpr.args[1], SymbolNode) || isa(lastExpr.args[1], GenSym)) && istupletyp(getType(lastExpr.args[1], state.linfo))
    rhs = lookupDef(state, lastExpr.args[1])
    if rhs != nothing
      #warn(string("rhs = ", rhs))
      # real return arguments (from a tuple)
      args = rhs.args[2:end]
      # number of returned buffers must match number of inputs
      assert(length(args) == narrs)
      # and they must all be SymbolNodes or GenSyms
      for i = 1:narrs assert(isa(args[i], SymbolNode) || isa(args[i], GenSym)) end
      # convert last Expr to multi-return
      lastExpr.args = args
      stat.rotateNum = length(args)
    end
  end
  # if not returning buffers or buffers not found, return nothing
  if rhs == nothing 
    lastExpr.args = Any[ nothing ]
  end 
  krnExpr = expr
  assert(stat != ())            # check if stat is indeed created

  # The genBody function returns the kernel computation.
  # It is supposed to be part of DomainLambda, and will have
  # to make fresh local variables (idxSym, strideSym, bufSym, etc)
  # on every invocation. Thus, these names are passed in as arguments.
  function genBody(dict, idxSymNodes, strideSymNodes, bufSymNodes)
    # assert(length(stat.idxSym) == length(idxSymNodes))
    # assert(length(stat.strideSym) == length(strideSymNodes))
    # assert(2 == length(bufSymNodes))
    # we first rename all GenSym to parent
    dprintln(3, "in stencil genBody, dict = ", dict)
    # CompilerTools.LambdaHandling.replaceExprWithDict(krnExpr, dict)
    local idxDict = Dict{SymGen, Any}(zip(stat.idxSym, idxSymNodes))
    local strideDict = Dict{SymGen, Any}(zip(stat.strideSym, strideSymNodes))
    local bufDict = Dict{SymGen, Any}(zip(bufSyms, bufSymNodes))
    local ldict = merge(dict, bufDict, idxDict, strideDict)
    dprintln(3,"stencil genBody")
    dprintln(3,"idxDict = ", idxDict)
    dprintln(3,"strideDict = ", strideDict)
    dprintln(3,"bufDict = ", bufDict)
    dprintln(3,"ldict = ", ldict)
    dprintln(3,"krnExpr = ", krnExpr)
    # warn(string("\nreplaceWithDict ", idxDict, strideDict, bufDict))
    CompilerTools.LambdaHandling.replaceExprWithDict(krnExpr, ldict)
  end
  # Remove those with no definition from locals as a sanity cleanup.
  # Note that among those removed are the input arguments, but this
  # what we want since they are useful in kernel specification, but
  # they do not appear in kernel's DomainLambda.
  #for (v,d) in locals
  #  if is(d.rhs, nothing)
  #    delete!(locals, v)
  #  end
  #end
  # warn(string("return from analyze kernel: ", (stat, state.linfo, krnExpr)))
  return stat, genBody
end

# Helper function to join Symbols into Expr
function nDimTo1Dim(exprs, strides)
  f(e, stride) = mk_expr(Int, :call, TopNode(:mul_int), stride, mk_expr(Int, :call, TopNode(:sub_int), e, 1))
  g(r, e) = mk_expr(Int, :call, TopNode(:add_int), r, e)
  e = mapReduceWith(exprs, strides, f, g)
  return mk_expr(Int, :call, TopNode(:add_int), e, 1)
end

function mapReduceWith(as, bs, mapf, redf)
  local l = min(length(as), length(bs))
  assert(l > 1)
  local cs = [ mapf(as[i], bs[i]) for i=1:l ]
  local e = cs[1]
  for i = 2:l
      e = redf(e, cs[i])
  end
  return e
end

# Returns a KernelStat object and a DomainLambda after analyzing the input
# kernel specification that has the source in the form of a -> a[...] + ...
# It expects krn to be of Expr(:lambda, ...) type.
function mkStencilLambda(state_, bufs, kernelExp, borderExp)
  local linfo = lambdaExprToLambdaInfo(kernelExp)
  local typs = Type[ typeOfOpr(state_, a) for a in bufs ]
  local state = newState(linfo, Dict(), state_)
  local stat, genBody
  if !(isa(borderExp, QuoteNode))
    error("Border specification in runStencil can only be Symbols.")
  end
  borderSty = borderExp.value 
  if !in(borderSty, supportedBorderStyle)
    error("Expect stencil border style to be one of ", supportedBorderStyle, ", but got ", borderSty)
  end
  # warn(string(typeof(state), " ", "typs = ", typs, " :: ", typeof(typs), " ", typeof(kernelExp), " ", typeof(borderSty)))
  stat, genBody = analyze_kernel(state, typs, kernelExp, borderSty)
  # f expects input of either [indices, strides, buffers] or [symbols].
  # the latter is used in show method for DomainLambda
  function f(plinfo, inputs)
    local indices, strides
    # warn(string("sizeof(inputs)=",length(inputs)))
    if isa(inputs[1], Array) # real inputs
      indices = inputs[1]
      strides = inputs[2]
      bufs = inputs[3]
    else # mock up inputs from show
      indices = stat.idxSym
      strides = stat.strideSym
      bufs = inputs
    end
    dict = CompilerTools.LambdaHandling.mergeLambdaInfo(plinfo, linfo)
    body = genBody(dict, indices, strides, bufs)
  end
  return stat, DomainLambda(typs, typs, f, state.linfo)
end



