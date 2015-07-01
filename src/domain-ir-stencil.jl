type KernelStat
  dimension :: Int             # number of dimensions of the stencil
  shapeMax  :: Array{Int,1}    # max extent of the stencil for each dimension
  shapeMin  :: Array{Int,1}    # min extent of the stencil for each dimension
  bufSym    :: Array{Symbol,1} # (fresh) buffer symbols
  idxSym    :: Array{Symbol,1} # (fresh) symbol of the index variable into the source image
  strideSym :: Array{Symbol,1} # (fresh) symbol of the strides data of source image
  borderSty :: Any             # QuoteNode :oob_dst_zero, :oob_src_zero, :oob_skip
  rotateNum :: Int             # number of buffers that is affected by rotation
  modified  :: Array{Int,1}    # which buffer is modified
end

# replace the symbols in an expression with those defined in
# the dictionary.
# The returned result may share part of the input expression,
# but the input is not changed.
function replaceWithDict(expr::Any, dict::Dict{Symbol, Any})
  function traverse(expr)       # traverse expr to find the places where arrSym is refernced
    if isa(expr, Symbol)
      if haskey(dict, expr)
        return dict[expr]
      end
      return expr
    elseif isa(expr, SymbolNode)
      if haskey(dict, expr.name)
        return dict[expr.name]
      end
      return expr
    elseif isa(expr, Array)
      Any[ traverse(e) for e in expr ]
    elseif isa(expr, Expr)
      local head = expr.head
      local args = copy(expr.args)
      local typ  = expr.typ
      for i = 1:length(args)
        args[i] = traverse(args[i])
      end
      return mk_expr(typ, expr.head, args...)
    else
      expr
    end
  end
  expr=traverse(expr)
  return expr
end

# Analyze a stencil kernel specified as a lambda expression.
# Return both the kernel stat, and the modified kernel LHS expression.
# Note that the input kernel is modified in-place.
# NOTE: currently only handle kernel specified as: a -> c * a[...] + ...
function analyze_kernel(bufTyps, krn, borderSty)
  #assert(krn.head == symbol("->"))
  assert(isa(krn, Expr) && krn.head == :lambda)
  local locals  = metaToVarDef(krn.args[2][2])
  local escapes = metaToVarDef(krn.args[2][3])
  local stat :: Union((),KernelStat) = ()
  # warn(string("krn.args[1]=", krn.args[1][1].head, ",", krn.args[1][1].args))
  local arrSyms = krn.args[1] # parameter of the kernel lambda
  local narrs = length(arrSyms)
  local bufSyms = Array(Symbol, narrs)
  local arrSymDict = Dict{Symbol,(Symbol, Type)}()
  for i in 1:length(arrSyms)
    if isa(arrSyms[i], Expr)
      arrSyms[i] = arrSyms[i].args[1]
    end
    assert(isa(arrSyms[i], Symbol))
    bufSyms[i] = freshsym("a")
    arrSymDict[arrSyms[i]] = (bufSyms[i], bufTyps[i])
  end
  local bufSymSet = Set{Symbol}(bufSyms)
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
        expr.args[i] = SymbolNode(arrSymDict[e]...)
      elseif isa(e, SymbolNode) && haskey(arrSymDict, e.name)
        expr.args[i] = SymbolNode(arrSymDict[e.name]...)
      end
    end
    # Julia doesn't infer type for lambda, so we have to hack it
    # fix symbol node
    for i = 1:length(expr.args)
      e = expr.args[i]
      if isa(e, Symbol)
        e = SymbolNode(e, Any)
      end
      if isa(e, SymbolNode) && is(e.typ, Any)
        name = e.name
        typ = Any
        if haskey(locals, name) typ = locals[name].typ end
        if haskey(escapes, name) typ = escapes[name].typ end
        # warn(string("fix symbolnode ", e, " to type ", typ))
        if !is(typ, Any)
          e.typ = typ
          expr.args[i] = e
        end
      end
    end
    # fix assignment and fill in variable type
    if is(expr.head, :(=))
      lhs = expr.args[1]
      if isa(lhs, SymbolNode) lhs = lhs.name end
      rhs = expr.args[2]
      typ = typeOfOpr(rhs)
      expr.typ = typ
      # warn(string("lhs=",lhs, " typ=",typ))
      if haskey(locals, lhs)
        # warn("in locals")
        def = locals[lhs]
        push!(locals, lhs, VarDef(typ, def.flag, rhs))
      elseif haskey(escapes, lhs)
        def = escapes[lhs]
        push!(escapes, lhs, VarDef(typ, def.flag, rhs))
      end
    # fix expr typ
    elseif is(expr.typ, Any) && is(expr.head, :call)
      # warn(string("in :call expr =",expr))
      typ = Any
      otherTyp = Any
      opr = expr.args[1]
      if in(opr, opsSymSet) || in(opr, topOpsTypeFix)
        for i = 2:length(expr.args)
          e = expr.args[i]
          # warn(string("inspect ", e, " typof(e)=", typeof(e)))
          if isa(e, SymbolNode) && is(typ, Any)
            typ = e.typ
          elseif is(otherTyp, Any)
            otherTyp = typeOfOpr(e)
          end
        end
      elseif is(opr, TopNode(:tuple))
        typ = tuple(map(typeOfOpr, expr.args[2:end])...)
      end
      typ = is(typ, Any) ? otherTyp : typ
      # warn(string("fix type from ", expr.typ, " to ", typ))
      expr.typ = typ
      if in(opr, opsSymSet)
        _args = expr.args[2:end]
        _typs = Type[ typ for arg in _args ] # use previuosly fixed typ of args
        opr, reorder = specializeOp(opr, _typs)
        _args = reorder(_args)
        expr.args = Any[opr,_args...]
        for i=2:length(expr.args)
          arg = expr.args[i]
          if isa(arg, Number) # auto convert number literals
            expr.args[i] = convert(typ, arg)
          end
        end
        if length(expr.args) > 3
          # more than 2 arguments, need to fold pairwise
          function fold(args)
            local n = length(args)
            local exp
            if n >= 2
              local m = div(n, 2)
              exp = Expr(:call, opr, fold(args[1:m]), fold(args[m+1:n]))
              exp.typ = typ
            else
              exp = args[1]
            end
            return exp
          end
          tmpExpr = fold(expr.args[2:end])
          expr.head = tmpExpr.head
          expr.args = tmpExpr.args
          expr.typ  = tmpExpr.typ
        end
      end
    end
    # fix getindex and setindex!
    if is(expr.head, :call) && (expr.args[1] == TopNode(:arrayref) || expr.args[1] == TopNode(:arrayset) || expr.args[1] == :getindex || expr.args[1] == :setindex!) &&
       isa(expr.args[2], SymbolNode) && in(expr.args[2].name, bufSymSet)
      local isGet = expr.args[1] == :getindex || expr.args[1] == TopNode(:arrayref)
      #(bufSym, bufTyp) = arrSymDict[expr.args[2].name] # modify the reference to actual source array
      bufSym = expr.args[2].name
      bufTyp = expr.args[2].typ
      elmTyp = elmTypOf(bufTyp)
      expr.args[2] = bufSym
      local idxOffset = isGet ? 2 : 3
      local dim = length(expr.args) - idxOffset
      assert (dim <= 10)      # arbitrary limit controlling the total num of dimensions
      if is(stat, ())         # create stat if not already exists
        local idxSym = [ freshsym(string(char(i+'h'))) for i in 1:dim ]
        local strideSym = [ freshsym(string("st", char(i+'h'))) for i in 1:dim ]
        stat = KernelStat(dim, zeros(dim), zeros(dim), bufSyms, idxSym, strideSym, borderSty, 0, Int[])
      else                    # if stat already exists, check if the dimension matches
        assert(dim == stat.dimension)
      end
      for i = 1:dim           # update extents and index calculation in expr
        local idx = int(expr.args[idxOffset+i])
        stat.shapeMax[i] = max(idx, stat.shapeMax[i])
        stat.shapeMin[i] = min(idx, stat.shapeMin[i])
        expr.args[idxOffset+i] = mk_expr(Int, :call, TopNode(:add_int), stat.idxSym[i], idx)
      end
      # local idx1D = nDimTo1Dim(expr.args[(idxOffset+1):end], stat.strideSym)
      expr.args = isGet ? [ TopNode(:unsafe_arrayref), expr.args[2], expr.args[(idxOffset+1):end]...] :
                          [ TopNode(:unsafe_arrayset), expr.args[2],
                          isa(expr.args[3], Symbol) ? SymbolNode(expr.args[3], elmTyp) : expr.args[3],
                          expr.args[(idxOffset+1):end]... ]
      if !isGet
	v = expr.args[2]
	if isa(v, SymbolNode) v = v.name end
	for i = 1:length(stat.bufSym)
	  if stat.bufSym[i] == v
	    push!(stat.modified, i)
	  end
	end
      end
      if is(expr.typ, Any)
        expr.typ = elmTyp
      end
      # warn(string("after :call expr=",expr))
    end
  end
  traverse(expr)
  # remove LineNumberNode, :line, and tuple assignment from expr
  #warn(string("kernel expr = ", expr))
  body = Array(Any, 0)
  for e in expr.args
    if isa(e, LineNumberNode)
    elseif isa(e, Expr) && is(e.head, :line)
    # TODO: the following needs to be tighten up
    elseif isa(e, Expr) && is(e.head, :(=)) && isa(e.typ, Tuple)
      # remember tuple assignment in locals, but don't emit it
      # delete!(locals, e.args[1])
    else
      push!(body, e)
    end
  end
  # fix return statments
  expr.args = body
  local lastExpr = expr.args[end]
  assert(lastExpr.head == :return)
  #warn(string("lastExpr=",lastExpr))
  if isa(lastExpr.args[1], SymbolNode) && isa(lastExpr.args[1].typ, Tuple) &&
     haskey(locals, lastExpr.args[1].name)
    rhs = locals[lastExpr.args[1].name].rhs
    #warn(string("rhs = ", rhs))
    # real return arguments (from a tuple)
    args = rhs.args[2:end]
    # number of returned buffers must match number of inputs
    assert(length(args) == narrs)
    # and they must all be SymbolNodess
    for i = 1:narrs assert(isa(args[i], SymbolNode)) end
    # convert last Expr to multi-return
    lastExpr.args = args
    stat.rotateNum = length(args)
  else
    # if not returning buffers, make it return nothing
    lastExpr.args = Any[nothing]
  end
  krnExpr = expr
  assert(stat != ())            # check if stat is indeed created
  # The genBody function returns the kernel computation.
  # It is supposed to be part of DomainLambda, and will have
  # to make fresh local variables (idxSym, strideSym, bufSym, etc)
  # on every invocation. Thus, these names are passed in as arguments.
  function genBody(idxSymNodes, strideSymNodes, bufSymNodes)
    # assert(length(stat.idxSym) == length(idxSymNodes))
    # assert(length(stat.strideSym) == length(strideSymNodes))
    # assert(2 == length(bufSymNodes))
    local idxDict = Dict{Symbol, Any}(zip(stat.idxSym, idxSymNodes))
    local strideDict = Dict{Symbol, Any}(zip(stat.strideSym, strideSymNodes))
    local bufDict = Dict{Symbol, Any}(zip(bufSyms, bufSymNodes))
    local dict = merge(bufDict, idxDict, strideDict)
    # warn(string("\nreplaceWithDict ", idxDict, strideDict, bufDict))
    replaceWithDict(krnExpr, dict)
  end
  # Remove those with no definition from locals as a sanity cleanup.
  # Note that among those removed are the input arguments, but this
  # what we want since they are useful in kernel specification, but
  # they do not appear in kernel's DomainLambda.
  for (v,d) in locals
    if is(d.rhs, nothing)
      delete!(locals, v)
    end
  end
  #warn("return from analyze kernel")
  return stat, locals, escapes, genBody
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
function mkStencilLambda(bufs, kernelExp, borderSty)
  local typs = Type[ typeOfOpr(a) for a in bufs ]
  local stat, genBody
  stat, locals, escapes, genBody = analyze_kernel(typs, kernelExp, borderSty)
  # f expects input of either [indices, strides, buffers] or [symbols].
  # the latter is used in show method for DomainLambda
  function f(inputs)
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
    genBody(indices, strides, bufs)
  end
  return stat, DomainLambda(typs, typs, f, locals, escapes)
end


# Below is a reference implementation of stencil function in
# native Julia for two purposes: to verify expected return result,
# and to make sure user code type checks.

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
  if is(value, NaN)
    if asa.borderStyle == :oob_dst_zero
      asa.src[idx...] = 0
    end
  else
    asa.src[idx...] = value
  end
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

