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
    elseif isa(expr, Expr) && (expr.head === :gotoifnot)
      exprs[i] = TypedExpr(expr.typ, expr.head, expr.args[1], labelDict[expr.args[2]])
    end
  end
  return exprs
end

function simplifyBodyExpr(kernelF, state)
    LambdaVarInfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(kernelF)
    cfg = CompilerTools.CFGs.from_lambda(body)
    body = CompilerTools.LambdaHandling.getBody(CompilerTools.CFGs.createFunctionBody(cfg), CompilerTools.LambdaHandling.getReturnType(LambdaVarInfo))
    non_array_params = Set{LHSVar}()
    changed = true
    lives = computeLiveness(body, LambdaVarInfo)
    while changed
        @dprintln(1,"Removing statement with no dependencies from the AST with parameters")
        rnd_state = RemoveNoDepsState(lives, non_array_params)
        body = AstWalk(body, remove_no_deps, rnd_state)
        @dprintln(3,"body after no dep stmts removed = ", body)

        @dprintln(3,"top_level_no_deps = ", rnd_state.top_level_no_deps)

        @dprintln(1,"Adding statements with no dependencies to the start of the AST.")
        body = CompilerTools.LambdaHandling.prependStatements(body, rnd_state.top_level_no_deps)
        @dprintln(3,"body after no dep stmts re-inserted = ", body)

        @dprintln(1,"Re-starting liveness analysis.")
        lives = computeLiveness(body, LambdaVarInfo)
        @dprintln(1,"Finished liveness analysis.")

        changed = rnd_state.change
    end
    body = AstWalk(body, remove_dead, RemoveDeadState(lives,LambdaVarInfo))
    return LambdaVarInfo, body
end

function mk_parfor_args_from_stencil(typ, head, args, irState)
  assert(length(args) == 4)
  local stat = args[1]
  local border = stat.borderSty
  local border_inloop = false
  local oob_skip = false
  local oob_dst_zero = false
  local oob_src_zero = false
  local oob_wraparound = false
  assert(isa(border, Symbol))
  oob_skip = (border === :oob_skip)
  oob_dst_zero = (border === :oob_dst_zero)
  oob_src_zero = (border === :oob_src_zero)
  oob_wraparound = (border === :oob_wraparound)
  local iterations = args[2]
  # convert all TypedVar in bufs to just Symbol
  local bufs = args[3]
  local kernelF :: DomainLambda = args[4]
  local linfo = irState.LambdaVarInfo
  # println(stat)
  local buf = bufs[1] # assumes that first buffer has the same dimension as all other buffers
  main_length_correlation = getOrAddArrayCorrelation(toLHSVar(bufs[1]), irState)
  local n = stat.dimension
  local sizeNodes = [ addTempVariable(Int, linfo) for i = 1:n ]
  local sizeInitExpr = [ TypedExpr(Int, :(=), sizeNodes[i],
                                   TypedExpr(Int, :call, GlobalRef(Base, :arraysize), buf, i))
                         for i in 1:n ]
  local strideNodes = [ addTempVariable(Int, linfo) for s in stat.strideSym ]
  local strideInitExpr = Array{Any}(n)
  strideInitExpr[1] = TypedExpr(Int, :(=), strideNodes[1], 1)
  # the following assumes contiguous column major layout for multi-dimensional arrays
  for i = 2:n
    strideInitExpr[i] = TypedExpr(Int, :(=), strideNodes[i],
                                  DomainIR.mul_expr(strideNodes[i-1], sizeNodes[i-1]))
  end
  local idxNodes = Any[ DomainIR.addFreshLocalVariable(string("i",s.id), Int, ISASSIGNED | ISASSIGNEDONCE, linfo) for s in stat.idxSym ]
  local loopNest = [ PIRLoopNest(idxNodes[i],
                                 border_inloop ? 1 : 1-stat.shapeMin[i],
                                 border_inloop ? sizeNodes[i] : DomainIR.sub_expr(sizeNodes[i], stat.shapeMax[i]),
                                 1)
                     for i in n:-1:1 ]
  local nbufs = length(bufs)
  tmpBufs = Array{TypedVar}(nbufs)
  for i = 1:length(bufs)
    elmTyp = DomainIR.elmTypOf(getType(bufs[i], linfo))
    arrTyp = Array{elmTyp, n}
    # TODO: enforce buffer type
    # bufs[i].typ = arrTyp
    tmpBufs[i] = DomainIR.addFreshLocalVariable("tmp", arrTyp, ISASSIGNED, linfo)

    this_correlation = getOrAddArrayCorrelation(toLHSVar(bufs[i]), irState)
    if this_correlation != main_length_correlation
      merge_correlations(irState, main_length_correlation, this_correlation)
    end
  end
  @dprintln(3, "before simplifyBodyExpr body=", kernelF.body)
  bodyLinfo, bodyExpr = simplifyBodyExpr(kernelF, irState)
  @dprintln(3, "before simplifyBodyExpr body=", bodyExpr)
  bodyExpr = relabel(DomainIR.stencilGenBody(stat, bodyLinfo, bodyExpr, idxNodes, strideNodes, bufs, linfo, getLoopPrivateFlags()), irState)
  # rotate
  assert((bodyExpr[end].head === :tuple))
  @dprintln(3,"bodyExpr = ")
  printBody(3, bodyExpr)
  local rotateExpr = Array{Any}(0)
  local revertExpr = Array{Any}(0)
  # warn(string("last return=", bodyExpr[end]))
  if bodyExpr[end].args[1] != nothing
    rets = bodyExpr[end].args
    @dprintln(3,"bodyExpr[end].args = ", rets)
    assert(length(rets) == nbufs)
    for i = 1:nbufs
      push!(rotateExpr, TypedExpr(CompilerTools.LambdaHandling.getType(rets[i], linfo), :(=), toLHSVar(tmpBufs[i]), rets[i]))
      push!(revertExpr, TypedExpr(CompilerTools.LambdaHandling.getType(rets[i], linfo), :(=), toLHSVar(tmpBufs[i]), bufs[i]))
    end
    for i = 1:nbufs
      push!(rotateExpr, TypedExpr(CompilerTools.LambdaHandling.getType(tmpBufs[i], linfo), :(=), toLHSVar(bufs[i]), tmpBufs[i]))
      push!(revertExpr, TypedExpr(CompilerTools.LambdaHandling.getType(tmpBufs[i], linfo), :(=), toLHSVar(rets[i]), tmpBufs[i]))
    end
  end
  bodyExpr = bodyExpr[1:end-1]
  # border handling
  borderLabel = next_label(irState)
  afterBorderLabel = next_label(irState)
  lowerExprs = [ DomainIR.box_ty(Bool, Expr(:call, GlobalRef(Base, :sle_int), 1-stat.shapeMin[i], idxNodes[i]))
                 for i in 1:n ]
  upperExprs = [ DomainIR.box_ty(Bool, Expr(:call, GlobalRef(Base, :sle_int), idxNodes[i],
                      DomainIR.sub_expr(sizeNodes[i], stat.shapeMax[i])))
                 for i in 1:n ]
  lowerGotos = [ Expr(:gotoifnot, e, borderLabel) for e in lowerExprs ]
  upperGotos = [ Expr(:gotoifnot, e, borderLabel) for e in upperExprs ]
  borderHead = Any[ TypedExpr(Int, :(=), toLHSVar(idxNodes[1]),
                      DomainIR.sub_expr(sizeNodes[1], stat.shapeMax[1])),
                    GotoNode(afterBorderLabel),
                    LabelNode(borderLabel),
                  ]
  borderExpr = Any[]
  if oob_dst_zero
    for expr in bodyExpr
      expr = deepcopy(expr)
      if (expr.head === :call) && isBaseFunc(expr.args[1], :unsafe_arrayset)
        zero = Base.convert(DomainIR.elmTypOf(getType(expr.args[2], linfo)), 0)
        push!(borderExpr, TypedExpr(expr.typ, :call, expr.args[1], expr.args[2], zero, expr.args[4:end]...))
      end
    end
  elseif oob_src_zero
    for expr in bodyExpr
      expr = deepcopy(expr)
      if isa(expr, Expr) && (expr.head === :(=))
        lhs = expr.args[1]
        rhs = expr.args[2]
        if isa(rhs, Expr) && (rhs.head === :call) && isBaseFunc(rhs.args[1], :unsafe_arrayref)
          zero = Base.convert(rhs.typ, 0)
          expr = TypedExpr(expr.typ, :(=), lhs,
                  TypedExpr(rhs.typ, :call, GlobalRef(Base, :safe_arrayref),
                      rhs.args[2], zero, rhs.args[3:end]...))
        end
      end
      push!(borderExpr, expr)
    end
  elseif oob_wraparound
    for expr in bodyExpr
      expr = deepcopy(expr)
      if isa(expr, Expr) && (expr.head === :(=))
        lhs = expr.args[1]
        rhs = expr.args[2]
        if isa(rhs, Expr) && (rhs.head === :call) && isBaseFunc(rhs.args[1], :unsafe_arrayref)
          indices = Expr[ TypedExpr(Int, :call, GlobalRef(Base, :select_value),
                            DomainIR.box_ty(Bool, Expr(:call, GlobalRef(Base, :sle_int), deepcopy(rhs.args[2+i]), 0)),
                            DomainIR.add_expr(deepcopy(rhs.args[2+i]), sizeNodes[i]),
                            TypedExpr(Int, :call, GlobalRef(Base, :select_value),
                              DomainIR.box_ty(Bool, Expr(:call, GlobalRef(Base, :sle_int),
                                DomainIR.add_expr(sizeNodes[i], 1), deepcopy(rhs.args[2+i]))),
                              DomainIR.sub_expr(deepcopy(rhs.args[2+i]), sizeNodes[i]),
                              deepcopy(rhs.args[2+i]))) for i = 1:n ]
          expr = TypedExpr(expr.typ, :(=), lhs,
                  TypedExpr(rhs.typ, :call, GlobalRef(Base, :unsafe_arrayref),
                      rhs.args[2], indices...))
        end
      end
      push!(borderExpr, expr)
    end
  else #FIXME: the following is to make a dummy node to avoid an IR bug
#    push!(borderExpr, TypedExpr(Int, :(=), idxNodes[1].name, idxNodes[1]))
  end
  borderExpr = vcat(borderHead, relabel(borderExpr, irState), LabelNode(afterBorderLabel))
  # borderCond = [ borderHead, lowerGotos, upperGotos, borderExpr, borderTail ]
  borderCond = oob_skip ? Any[] : Any[TypedExpr(Void, head,
        PIRParForAst(InputInfo(toLHSVar(buf)),
            vcat(lowerGotos, upperGotos, borderExpr),
            [],
            [],
            [ PIRLoopNest(idxNodes[i], 1, sizeNodes[i], 1) for i = n:-1:1 ],
            PIRReduction[],
            [], [], irState.top_level_number, get_unique_num(), Set{LHSVar}(), Set{LHSVar}([toLHSVar(x) for x in bufs])))]

  stepNode = DomainIR.addFreshLocalVariable("step", Int, ISASSIGNED | ISASSIGNEDONCE, linfo)
  # Sequential loop for multi-iterations
  iterPre  = (isa(iterations, Number) && iterations == 1) ?
            Any[] : Any[ Expr(:loophead, stepNode, 1, iterations) ]
  iterPost = (isa(iterations, Number) && iterations == 1) ?
            Any[] : vcat(rotateExpr, Expr(:loopend, stepNode))
  preExpr = vcat(sizeInitExpr, strideInitExpr, iterPre, borderCond)
  postExpr = vcat(iterPost)
  expr = PIRParForAst(
    InputInfo(buf),
    bodyExpr,
    [],
    [],
    loopNest,
    PIRReduction[],
    Any[ length(bufs) > 2 ? tuple(bufs...) : bufs[1] ],
    DomainOperation[ DomainOperation(:stencil!, args) ],
    irState.top_level_number,
    get_unique_num(),
    Set{LHSVar}(),
    Set{LHSVar}([toLHSVar(x) for x in bufs]))
  return vcat(preExpr, TypedExpr(typ, head, expr), postExpr)
end
