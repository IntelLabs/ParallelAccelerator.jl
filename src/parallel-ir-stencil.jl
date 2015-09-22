import ..DomainIR

function relabel(exprs::Array{Any}, irState)
  labelDict = Dict{Int, Int}()
  for i = 1:length(exprs)
    expr = exprs[i]
    if isa(expr, LabelNode)
      new_label = next_label(irState)
      labelDict[expr.label] = new_label
    end
  end
  for i = 1:length(exprs)
    expr = exprs[i]
    if isa(expr, LabelNode)
      exprs[i] = LabelNode(labelDict[expr.label])
    elseif isa(expr, GotoNode)
      exprs[i] = GotoNode(labelDict[expr.label])
    elseif isa(expr, Expr) && is(expr.head, :gotoifnot)
      exprs[i] = TypedExpr(expr.typ, expr.head, expr.args[1], labelDict[expr.args[2]])
    end
  end
  return exprs
end

function mk_parfor_args_from_stencil(typ, head, args, irState)
  assert(length(args) == 4)
  local stat = args[1]
  local border = stat.borderSty
  local border_inloop = false
  local oob_dst_zero = false
  local oob_src_zero = false
  local oob_wraparound = false
  assert(isa(border, Symbol))
  oob_dst_zero = is(border, :oob_dst_zero)
  oob_src_zero = is(border, :oob_src_zero)
  oob_wraparound = is(border, :oob_wraparound)
  local iterations = args[2]
  # convert all SymbolNode in bufs to just Symbol
  local bufs = args[3]
  local kernelF :: DomainLambda = args[4] 
  local linfo = irState.lambdaInfo
  # println(stat)
  local buf = bufs[1] # assumes that first buffer has the same dimension as all other buffers
  local toSymGen(x) = isa(x, SymbolNode) ? x.name : x 
  main_length_correlation = getOrAddArrayCorrelation(toSymGen(bufs[1]), irState)
  local n = stat.dimension
  local sizeNodes = [ addGenSym(Int, linfo) for i = 1:n ]
  local sizeInitExpr = [ TypedExpr(Int, :(=), sizeNodes[i],
                                   TypedExpr(Int, :call, TopNode(:arraysize), buf, i))
                         for i in 1:n ]
  local strideNodes = [ addGenSym(Int, linfo) for s in stat.strideSym ]
  local strideInitExpr = Array(Any, n)
  strideInitExpr[1] = TypedExpr(Int, :(=), strideNodes[1], 1)
  # the following assumes contiguous column major layout for multi-dimensional arrays
  for i = 2:n
    strideInitExpr[i] = TypedExpr(Int, :(=), strideNodes[i],
                                  TypedExpr(Int, :call, TopNode(:mul_int), strideNodes[i-1], sizeNodes[i-1]))
  end
  local idxNodes = Any[ DomainIR.addFreshLocalVariable(string("i",s.id), Int, ISASSIGNED | ISASSIGNEDONCE, linfo) for s in stat.idxSym ]
  local loopNest = [ PIRLoopNest(idxNodes[i],
                                 border_inloop ? 1 : 1-stat.shapeMin[i],
                                 border_inloop ? sizeNodes[i] : TypedExpr(Int, :call, TopNode(:sub_int), sizeNodes[i], stat.shapeMax[i]),
                                 1)
                     for i in n:-1:1 ]
  local nbufs = length(bufs)
  tmpBufs = Array(SymbolNode, nbufs)
  for i = 1:length(bufs)
    elmTyp = DomainIR.elmTypOf(getType(bufs[i], linfo))
    arrTyp = Array{elmTyp, n}
    # TODO: enforce buffer type
    # bufs[i].typ = arrTyp
    tmpBufs[i] = DomainIR.addFreshLocalVariable("tmp", arrTyp, ISASSIGNED, linfo)

    this_correlation = getOrAddArrayCorrelation(toSymGen(bufs[i]), irState)
    if this_correlation != main_length_correlation
      merge_correlations(irState, main_length_correlation, this_correlation)
    end
  end
  local kernArgs = Array(Any, 3)
  kernArgs[1] = idxNodes
  kernArgs[2] = strideNodes
  kernArgs[3] = bufs
  dprintln(3,"kernArgs = ", kernArgs)
  bodyExpr = relabel(kernelF.genBody(linfo, kernArgs).args, irState)
  # rotate
  assert(is(bodyExpr[end].head, :return))
  dprintln(3,"bodyExpr = ")
  printBody(3, bodyExpr)
  local rotateExpr = Array(Any, 0)
  local revertExpr = Array(Any, 0)
  # warn(string("last return=", bodyExpr[end]))
  if bodyExpr[end].args[1] != nothing
    rets = bodyExpr[end].args
    dprintln(3,"bodyExpr[end].args = ", rets)
    assert(length(rets) == nbufs)
    for i = 1:nbufs
      push!(rotateExpr, TypedExpr(CompilerTools.LambdaHandling.getType(rets[i], kernelF.linfo), :(=), toSymGen(tmpBufs[i]), rets[i]))
      push!(revertExpr, TypedExpr(CompilerTools.LambdaHandling.getType(rets[i], kernelF.linfo), :(=), toSymGen(tmpBufs[i]), bufs[i]))
    end
    for i = 1:nbufs
      push!(rotateExpr, TypedExpr(CompilerTools.LambdaHandling.getType(tmpBufs[i], kernelF.linfo), :(=), toSymGen(bufs[i]), tmpBufs[i]))
      push!(revertExpr, TypedExpr(CompilerTools.LambdaHandling.getType(tmpBufs[i], kernelF.linfo), :(=), toSymGen(rets[i]), tmpBufs[i]))
    end
  end
  bodyExpr = bodyExpr[1:end-1]
  # border handling
  borderLabel = next_label(irState)
  afterBorderLabel = next_label(irState)
  #borderHead = [ Expr(:loophead, idxNodes[i].name, 1, sizeNodes[i]) for i = n:-1:1 ]
  #borderTail = [ Expr(:loopend, idxNodes[i].name) for i in 1:n ]
  lowerExprs = [ TypedExpr(Bool, :call, TopNode(:sle_int), 1-stat.shapeMin[i], idxNodes[i])
                 for i in 1:n ]
  upperExprs = [ TypedExpr(Bool, :call, TopNode(:sle_int), idxNodes[i],
                      TypedExpr(Int, :call, TopNode(:sub_int), sizeNodes[i], stat.shapeMax[i]))
                 for i in 1:n ]
  lowerGotos = [ Expr(:gotoifnot, e, borderLabel) for e in lowerExprs ]
  upperGotos = [ Expr(:gotoifnot, e, borderLabel) for e in upperExprs ]
  borderExpr = Any[ TypedExpr(Int, :(=), toSymGen(idxNodes[1]),
                      TypedExpr(Int, :call, TopNode(:sub_int), sizeNodes[1], stat.shapeMax[1])),
                    GotoNode(afterBorderLabel),
                    LabelNode(borderLabel),
                  ]
  if oob_dst_zero
    for expr in bodyExpr
      if is(expr.head, :call) && isa(expr.args[1], TopNode) && is(expr.args[1].name, :unsafe_arrayset)
        assert(isa(expr.args[2], SymbolNode))
dprintln(3, " elmTyp = ", DomainIR.elmTypOf(expr.args[2].typ), " :: ", typeof(DomainIR.elmTypOf(expr.args[2].typ)))
        zero = Base.convert(DomainIR.elmTypOf(expr.args[2].typ), 0)
        push!(borderExpr, TypedExpr(expr.typ, :call, expr.args[1], expr.args[2], zero, expr.args[4:end]...))
      end
    end
  elseif oob_src_zero
dprintln(3, "inside oob_src_zero")
    for expr in bodyExpr
      if isa(expr, Expr) && is(expr.head, :(=))
        lhs = expr.args[1]
        rhs = expr.args[2]
        if isa(rhs, Expr) && is(rhs.head, :call) && isa(rhs.args[1], TopNode) && is(rhs.args[1].name, :unsafe_arrayref)
          zero = Base.convert(rhs.typ, 0)
          expr = TypedExpr(expr.typ, :(=), lhs,
                  TypedExpr(rhs.typ, :call, TopNode(:safe_arrayref),
                      rhs.args[2], zero, rhs.args[3:end]...))
        end
      end
dprintln(3, "borderExpr pushed")
      push!(borderExpr, expr)
    end
  elseif oob_wraparound
    for expr in bodyExpr
      if isa(expr, Expr) && is(expr.head, :(=))
        lhs = expr.args[1]
        rhs = expr.args[2]
        if isa(rhs, Expr) && is(rhs.head, :call) && isa(rhs.args[1], TopNode) && is(rhs.args[1].name, :unsafe_arrayref)
          indices = Expr[ TypedExpr(Int, :call, TopNode(:select_value), 
                            TypedExpr(Bool, :call, TopNode(:sle_int), rhs.args[2+i], 0),
                            TypedExpr(Int, :call, TopNode(:add_int), rhs.args[2+i], sizeNodes[i]),
                            TypedExpr(Int, :call, TopNode(:select_value),
                              TypedExpr(Bool, :call, TopNode(:sle_int), 
                                TypedExpr(Int, :call, TopNode(:add_int), sizeNodes[i], 1), rhs.args[2+i]),
                              TypedExpr(Int, :call, TopNode(:sub_int), rhs.args[2+i], sizeNodes[i]),
                              rhs.args[2+i])) for i = 1:n ]
          expr = TypedExpr(expr.typ, :(=), lhs,
                  TypedExpr(rhs.typ, :call, TopNode(:unsafe_arrayref),
                      rhs.args[2], indices...))
        end
      end
      push!(borderExpr, expr)
    end
  else #FIXME: the following is to make a dummy node to avoid an IR bug
#    push!(borderExpr, TypedExpr(Int, :(=), idxNodes[1].name, idxNodes[1]))
  end
  push!(borderExpr, LabelNode(afterBorderLabel))
  # borderCond = [ borderHead, lowerGotos, upperGotos, borderExpr, borderTail ]
  borderCond = TypedExpr(Void, head,
        PIRParForAst(vcat(lowerGotos, upperGotos, deepcopy(borderExpr)),
        [],
        [ PIRLoopNest(idxNodes[i], 1, sizeNodes[i], 1) for i = n:-1:1 ],
        PIRReduction[],
        [], [], irState.top_level_number, get_unique_num(), false))

  stepNode = DomainIR.addFreshLocalVariable("step", Int, ISASSIGNED | ISASSIGNEDONCE, linfo)
  # Sequential loop for multi-iterations
  iterPre  = (isa(iterations, Number) && iterations == 1) ?
            Any[] : Any[ Expr(:loophead, stepNode.name, 1, iterations) ]
  iterPost = (isa(iterations, Number) && iterations == 1) ?
            Any[] : vcat(rotateExpr, Expr(:loopend, stepNode.name))
  preExpr = vcat(sizeInitExpr, strideInitExpr, iterPre, borderCond)
  postExpr = vcat(iterPost)
  expr = PIRParForAst(bodyExpr,
    [],
    loopNest,
    PIRReduction[],
    Any[ length(bufs) > 2 ? tuple(bufs...) : bufs[1] ],
    DomainOperation[ DomainOperation(:stencil!, args) ],
    irState.top_level_number,
    get_unique_num(), 
    false)
  return vcat(preExpr, TypedExpr(typ, head, expr), postExpr)
end
