import ..DomainIR.freshsym

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
  local border_inloop = false, oob_dst_zero = false, oob_src_zero = false
  if isa(border, QuoteNode)
    local b = border.value
    oob_dst_zero = is(b, :oob_dst_zero)
    oob_src_zero = is(b, :oob_src_zero)
    oob_wraparound = is(b, :oob_wraparound)
  end
  local iterations = args[2]
  local bufs = args[3]
  local kernelF = args[4]
  # println(stat)
  local buf = bufs[1] # assumes that first buffer has the same dimension as all other buffers
  main_length_correlation = getOrAddArrayCorrelation(bufs[1].name, irState)
  local n = stat.dimension
  local sizeNodes = [ SymbolNode(freshsym(string("len", string(i))), Int) for i = 1:n ]
  local sizeInitExpr = [ TypedExpr(Int, :(=), sizeNodes[i].name,
                                   TypedExpr(Int, :call, TopNode(:arraysize), buf, i))
                         for i in 1:n ]
  local strideNodes = [ SymbolNode(freshsym(string(s)), Int) for s in stat.strideSym ]
  local strideInitExpr = Array(Any, n)
  strideInitExpr[1] = TypedExpr(Int, :(=), strideNodes[1].name, 1)
  # the following assumes contiguous column major layout for multi-dimensional arrays
  for i = 2:n
    strideInitExpr[i] = TypedExpr(Int, :(=), strideNodes[i].name,
                                  TypedExpr(Int, :call, TopNode(:mul_int), strideNodes[i-1], sizeNodes[i-1]))
  end
  local idxNodes = [ SymbolNode(freshsym(string(s)), Int) for s in stat.idxSym ]
  local loopNest = [ PIRLoopNest(idxNodes[i],
                                 border_inloop ? 1 : 1-stat.shapeMin[i],
                                 border_inloop ? sizeNodes[i] : TypedExpr(Int, :call, TopNode(:sub_int), sizeNodes[i], stat.shapeMax[i]),
                                 1)
                     for i in n:-1:1 ]
  local nbufs = length(bufs)
  tmpBufs = Array(SymbolNode, nbufs)
  for i = 1:length(bufs)
    elmTyp = DomainIR.elmTypOf(bufs[i].typ)
    arrTyp = Array{elmTyp, n}
    bufs[i].typ = arrTyp
    tmpBufs[i] = SymbolNode(freshsym("tmp"), arrTyp)

    this_correlation = getOrAddArrayCorrelation(bufs[i].name, irState)
    if this_correlation != main_length_correlation
      merge_correlations(irState, main_length_correlation, this_correlation)
    end
  end
  local kernArgs = Array(Any, 3)
  kernArgs[1] = idxNodes
  kernArgs[2] = strideNodes
  kernArgs[3] = bufs
  bodyExpr = relabel(kernelF.genBody(kernArgs).args, irState)
  # rotate
  assert(is(bodyExpr[end].head, :return))
  local rotateExpr = Array(Any, 0)
  local revertExpr = Array(Any, 0)
  # warn(string("last return=", bodyExpr[end]))
  if bodyExpr[end].args[1] != nothing
    rets = bodyExpr[end].args
    assert(length(rets) == nbufs)
    for i = 1:nbufs
      push!(rotateExpr, TypedExpr(rets[i].typ, :(=), tmpBufs[i].name, rets[i]))
      push!(revertExpr, TypedExpr(rets[i].typ, :(=), tmpBufs[i].name, bufs[i]))
    end
    for i = 1:nbufs
      push!(rotateExpr, TypedExpr(tmpBufs[i].typ, :(=), bufs[i].name, tmpBufs[i]))
      push!(revertExpr, TypedExpr(tmpBufs[i].typ, :(=), rets[i].name, tmpBufs[i]))
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
  borderExpr = Any[ TypedExpr(Int, :(=), idxNodes[1].name,
                      TypedExpr(Int, :call, TopNode(:sub_int), sizeNodes[1], stat.shapeMax[1])),
                    GotoNode(afterBorderLabel),
                    LabelNode(borderLabel),
                  ]
  if oob_dst_zero
    for expr in bodyExpr
      if is(expr.head, :call) && isa(expr.args[1], TopNode) && is(expr.args[1].name, :unsafe_arrayset)
        assert(isa(expr.args[2], SymbolNode))
        zero = convert(DomainIR.elmTypOf(expr.args[2].typ), 0)
        push!(borderExpr, TypedExpr(expr.typ, :call, expr.args[1], expr.args[2], zero, expr.args[4:end]...))
      end
    end
  elseif oob_src_zero
    for expr in bodyExpr
      if isa(expr, Expr) && is(expr.head, :(=))
        lhs = expr.args[1]
        rhs = expr.args[2]
        if isa(rhs, Expr) && is(rhs.head, :call) && isa(rhs.args[1], TopNode) && is(rhs.args[1].name, :unsafe_arrayref)
          zero = convert(rhs.typ, 0)
          expr = TypedExpr(expr.typ, :(=), lhs,
                  TypedExpr(rhs.typ, :call, TopNode(:safe_arrayref),
                      rhs.args[2], zero, rhs.args[3:end]...))
        end
      end
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
        PIRParForAst([lowerGotos, upperGotos, borderExpr],
        [],
        [ PIRLoopNest(idxNodes[i], 1, sizeNodes[i], 1) for i = n:-1:1 ],
        PIRReduction[],
        [], [], irState.top_level_number, get_unique_num()))

  stepNode = SymbolNode(freshsym("step"), Int)
  # Sequential loop for multi-iterations
  iterPre  = (isa(iterations, Number) && iterations == 1) ?
            Any[] : [ Expr(:loophead, stepNode.name, 1, iterations) ]
  iterPost = (isa(iterations, Number) && iterations == 1) ?
            Any[] : [ rotateExpr, Expr(:loopend, stepNode.name) ]
  preExpr = [
      sizeInitExpr,
      strideInitExpr,
      iterPre,
      borderCond ]
  postExpr = [
      iterPost,
      # length(bufs) > 2 ? tuple(bufs...) : bufs[1]
    ]
  expr = PIRParForAst(relabel(bodyExpr, irState),
    [],
    loopNest,
    PIRReduction[],
    [ length(bufs) > 2 ? tuple(bufs...) : bufs[1] ],
    [DomainOperation(:stencil!, args)],
    irState.top_level_number,
    get_unique_num())
  for (v,d) in kernelF.locals
    addStateVar(irState, new_var(v, d.typ, d.flag | ISPRIVATEPARFORLOOP))
  end
  for v in [stepNode, idxNodes, sizeNodes, strideNodes]
    addStateVar(irState, new_var(v.name, v.typ, ISASSIGNEDONCE | ISASSIGNED))
  end
  for v in [tmpBufs]
    addStateVar(irState, new_var(v.name, v.typ, ISASSIGNED))
  end
  return [ preExpr, TypedExpr(typ, head, expr), postExpr ]
end
