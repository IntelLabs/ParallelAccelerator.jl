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
  initialized :: Bool             # whether this object is initialized
  
  KernelStat() = new(0,Int[],Int[],GenSym[], GenSym[], GenSym[], :-, 0, Int[], false)
end

function set_kernel_stat(stat::KernelStat, dimension::Int, shapeMax::Array{Int,1}, 
                         shapeMin::Array{Int,1}, bufSym::Array{GenSym,1}, idxSym::Array{GenSym,1}, 
                         strideSym::Array{GenSym,1}, borderSty :: Symbol, rotateNum :: Int, modified::Array{Int,1})
    stat.dimension = dimension
    stat.shapeMax = shapeMax
    stat.shapeMin = shapeMin
    stat.bufSym = bufSym
    stat.idxSym = idxSym
    stat.strideSym = strideSym
    stat.borderSty = borderSty
    stat.rotateNum = rotateNum
    stat.modified = modified
    stat.initialized = true
end

supportedBorderStyle = Set([:oob_dst_zero, :oob_src_zero, :oob_skip, :oob_wraparound])

"""
 Analyze a stencil kernel specified as a lambda expression.
 Return both the kernel stat, and the modified kernel LHS expression.
 Note that the input kernel is modified in-place.
 NOTE: currently only handle kernel specified as: a -> c * a[...] + ...
"""
function analyze_kernel(state::IRState, bufTyps::Array{Type, 1}, krnBody::Expr, borderSty::Symbol)
  #assert(krn.head == Symbol("->"))
  @dprintln(3, "typeof krnBody = ", typeof(krnBody), " ", krnBody.head)
  assert(isa(krnBody, Expr))
  #assert((krn.head === :lambda))
  local stat = KernelStat()
  # warn(string("krn.args[1]=", krn.args[1]))
  local arrSyms::Array{Symbol,1} = getInputParameters(state.linfo)
  if length(arrSyms) > 0  && arrSyms[1] == Symbol("#self#")
    arrSyms = arrSyms[2:end]
  end
  local narrs = length(arrSyms)
  local bufSyms = Array{GenSym}(narrs)
  local arrSymDict = Dict{LHSVar,GenSym}()
  for i in 1:length(arrSyms)
    # this case is not possible since krn is a type inferred AST and not LambdaInfo
    # if isa(arrSyms[i], Expr) && arrSyms[i].head == :(::) # Expr in the form (x :: t).
    #  arrSyms[i] = arrSyms[i].args[1]
    #end
    # assert(isa(arrSyms[i], Symbol))
    bufSyms[i] = addTempVariable(bufTyps[i], state.linfo)
    arrSymDict[lookupLHSVarByName(arrSyms[i], state.linfo)] = bufSyms[i]
  end
  @dprintln(3, "bufSyms = ", bufSyms, " arrSymDict = ", arrSymDict)
  local expr::Expr = krnBody
  assert(expr.head == :body)
  # warn(string("got arrSymDict = ", arrSymDict))
  
  traverse(state, expr, bufSyms, arrSymDict, stat, borderSty)
  # remove LineNumberNode, :line, and tuple assignment from expr
  #warn(string("kernel expr = ", expr))
  body = Array{Any}(0)
  for e in expr.args
    if isa(e, LineNumberNode)
    elseif isa(e, Expr) && (e.head === :line)
    else
      push!(body, e)
    end
  end
  # fix return statments
  expr.args = body
  local lastExpr = expr.args[end]
  assert(lastExpr.head == :tuple)
  # default to returning nothing
  #warn(string("lastExpr=",lastExpr))
  if length(lastExpr.args) > 1
      assert(length(lastExpr.args) == narrs)
      stat.rotateNum = length(lastExpr.args)
  else
      lastExpr.args = Any[ nothing ]
  end 
  krnExpr = expr
  assert(stat.initialized)            # check if stat is indeed created

  # Remove those with no definition from locals as a sanity cleanup.
  # Note that among those removed are the input arguments, but this
  # what we want since they are useful in kernel specification, but
  # they do not appear in kernel's DomainLambda.
  #for (v,d) in locals
  #  if (d.rhs === nothing)
  #    delete!(locals, v)
  #  end
  #end
  # warn(string("return from analyze kernel: ", (stat, state.linfo, krnExpr)))
  return stat, krnExpr
end

"""
 traverse expr to fix types, and places where arrSyms is refernced
"""
function traverse(state, expr::Expr, bufSyms, arrSymDict, stat, borderSty)
  # recurse
  # warn(string("expr=", expr, ", head=", expr.head, " args=", expr.args))
  for i = 1:length(expr.args)
    e = expr.args[i]
    if isa(e, Expr)
      traverse(state, e, bufSyms, arrSymDict, stat, borderSty)
    elseif isa(e, RHSVar) 
      s = toLHSVar(e)
      if haskey(arrSymDict, s)
        expr.args[i] = arrSymDict[s]
      end
    end
  end
  # fix assignment and fill in variable type
  if (expr.head === :(=))
    lhs = toLHSVar(expr.args[1])
    rhs = expr.args[2]
    typ = typeOfOpr(state, rhs)
    expr.typ = typ
    # warn(string("lhs=",lhs, " typ=",typ))
    updateDef(state, lhs, rhs)
  end
  # fix getindex and setindex!, note that their operand may already have been replaced by bufSyms, which are LHSVar.
  if isCall(expr) || isInvoke(expr)
   fun = getCallFunction(expr)
   args = getCallArguments(expr)
   if isa(fun, GlobalRef) && 
     ((fun.name === :getindex) || (fun.name === :setindex!) || (fun.name === :arrayref) || (fun.name === :arrayset)) && 
     isa(args[1], RHSVar) && in(args[1], bufSyms)
    local isGet = (fun.name === :arrayref) || (fun.name === :getindex)
    #(bufSym, bufTyp) = arrSymDict[expr.args[2].name] # modify the reference to actual source array
    bufSym = args[1]
    elmTyp = elmTypOf(getType(bufSym, state.linfo))
    local idxOffset = isGet ? 1 : 2
    local dim = length(args) - idxOffset
    assert(dim <= 10)      # arbitrary limit controlling the total num of dimensions
    if !stat.initialized         # create stat if not already exists
      local idxSym = GenSym[ addTempVariable(Int, state.linfo) for i in 1:dim ]
      local strideSym = GenSym[ addTempVariable(Int, state.linfo) for i in 1:dim ]
      set_kernel_stat(stat, dim, zeros(Int,dim), zeros(Int,dim), bufSyms, idxSym, strideSym, borderSty, 0, Int[])
    else                    # if stat already exists, check if the dimension matches
      assert(dim == stat.dimension)
    end
    for i = 1:dim           # update extents and index calculation in expr
      local idx = Int(args[idxOffset+i])
      stat.shapeMax[i] = max(idx, stat.shapeMax[i])
      stat.shapeMin[i] = min(idx, stat.shapeMin[i])
      args[idxOffset+i] = add_expr(stat.idxSym[i], idx)
    end
    # local idx1D = nDimTo1Dim(expr.args[(idxOffset+1):end], stat.strideSym)
    expr.head = :call
    expr.args = isGet ? [ GlobalRef(Base, :unsafe_arrayref), args[1], args[(idxOffset+1):end]...] :
                        [ GlobalRef(Base, :unsafe_arrayset), args[1], args[2], args[(idxOffset+1):end]... ]
    # fix numerical coercion when converting setindex! into unsafe_arrayset
    if (expr.args[1].name === :unsafe_arrayset)
        if typeOfOpr(state, expr.args[3]) != elmTyp 
          expr.args[3] = mk_expr(elmTyp, :call, GlobalRef(Base, Symbol(string(elmTyp))), expr.args[3])
        end
    end

    if !isGet
      v = toLHSVar(expr.args[2])
      for i = 1:length(stat.bufSym)
        if stat.bufSym[i] == v
          push!(stat.modified, i)
        end
      end
    end
   end
  end
end

# Helper function to join Symbols into Expr
function nDimTo1Dim(exprs, strides)
  f(e, stride) = mul_expr(stride, box_int(sub_expr(e, 1)))
  g(r, e) = add_expr(r, e)
  e = mapReduceWith(exprs, strides, f, g)
  return add_expr(e, 1)
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

"""
 Returns a KernelStat object and a DomainLambda after analyzing the input
 kernel specification that has the source in the form of a -> a[...] + ...
 It expects krn to be of Expr(:lambda, ...) type.
 Border specification in runStencil can only be Symbols.
"""
function mkStencilLambda(state_, bufs, kernelBody, linfo, borderExp::QuoteNode)
  typs = Type[ typeOfOpr(state_, a) for a in bufs ]
  state = newState(linfo, Dict(), Dict(), state_)
  #if !(isa(borderExp, QuoteNode))
  #  error("Border specification in runStencil can only be Symbols.")
  #end
  borderSty = borderExp.value 
  if !in(borderSty, supportedBorderStyle)
    error("Expect stencil border style to be one of ", supportedBorderStyle, ", but got ", borderSty)
  end
  # warn(string(typeof(state), " ", "typs = ", typs, " :: ", typeof(typs), " ", typeof(kernelExp), " ", typeof(borderSty)))
  stat, krnBody = analyze_kernel(state, typs, kernelBody, borderSty)
  return stat, DomainLambda(linfo, krnBody)
end

function stencilGenBody(stat, linfo, body, idxSymNodes, strideSymNodes, bufSymNodes, plinfo, private_flag)
    dict = CompilerTools.LambdaHandling.mergeLambdaVarInfo(plinfo, linfo, private_flag) 
    @dprintln(3, "in stencilGenBody, dict = ", dict)
    # CompilerTools.LambdaHandling.replaceExprWithDict(krnExpr, dict)
    idxDict = Dict{LHSVar, Any}(zip(stat.idxSym, idxSymNodes))
    strideDict = Dict{LHSVar, Any}(zip(stat.strideSym, strideSymNodes))
    bufDict = Dict{LHSVar, Any}(zip(stat.bufSym, bufSymNodes))
    ldict = merge(dict, bufDict, idxDict, strideDict)
    krnExpr = body.args # body Expr array
    @dprintln(3,"idxDict = ", idxDict)
    @dprintln(3,"strideDict = ", strideDict)
    @dprintln(3,"bufDict = ", bufDict)
    @dprintln(3,"ldict = ", ldict)
    @dprintln(3,"krnExpr = ", krnExpr)
    # warn(string("\nreplaceWithDict ", idxDict, strideDict, bufDict))
    CompilerTools.LambdaHandling.replaceExprWithDict!(deepcopy(krnExpr), ldict, plinfo)
end
