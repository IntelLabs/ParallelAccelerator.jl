# A basic alias analysis with limited scope:
#
# 1. Only objects pointed to by variables, but not elements of arrays or other structs
# 2. We only consider variables that are assigned once
# 3. No inter-precedural (i.e. function call) analysis
#
# The only useful result from this alias analysis is whether
# some variable definitely doesn't alias with anything else.
#
# We are NOT interested in the set of potential aliased variables.
#
# The algorithm is basically an abstract interpreter of Julia AST.

module AliasAnalysis

using Base.uncompressed_ast
using CompilerTools.LambdaHandling
using CompilerTools

#import CompilerTools


DEBUG_LVL=0

function set_debug_level(x)
    global DEBUG_LVL = x
end

# A debug print routine.
function dprint(level,msgs...)
    if(DEBUG_LVL >= level)
        print(msgs...)
    end
end

# A debug print routine.
function dprintln(level,msgs...)
    if(DEBUG_LVL >= level)
        println(msgs...)
    end
end

# state to keep track of variable values
const Unknown  = -1
const NotArray = 0

type State
  baseID :: Int
  locals :: Dict{SymGen, Int}
  revmap :: Dict{Int, Set{SymGen}}
  nest_level :: Int
  top_level_idx :: Int
  liveness :: CompilerTools.LivenessAnalysis.BlockLiveness
end

init_state(liveness) = State(0, Dict{SymGen,Int}(), Dict{Int, Set{SymGen}}(), 0, 0, liveness)

function increaseNestLevel(state)
state.nest_level = state.nest_level + 1
end

function decreaseNestLevel(state)
state.nest_level = state.nest_level -1
end

function next_node(state)
  local n = state.baseID + 1
  state.baseID = n
  return n
end

function update_node(state, v, w)
  if !haskey(state.locals, v)
    # new initialization
    state.locals[v] = w
    if haskey(state.revmap, w)
      push!(state.revmap[w], v)
    else
      state.revmap[w] = push!(Set{SymGen}(), v)
    end
  else
    # when a variable is initialized more than once, set to Unknown
    state.locals[v] = Unknown
    if haskey(state.revmap, w)
      for u in state.revmap[w]
        state.locals[u] = Unknown
      end
      pop!(state.revmap, w)
    end
  end
end

function update_unknown(state, v)
  state.locals[v] = Unknown
end

function update_notarray(state, v)
  state.locals[v] = NotArray
end

function update(state, v, w)
  if w > 0
    update_node(state, v, w)
  elseif w == NotArray
    update_notarray(state, v)
  else
    update_unknown(state, v)
  end
end

function toSymGen(x)
  if isa(x, SymbolNode)
    x.name
  elseif isa(x, GenSym) || isa(x, Symbol) 
    x
  else
    error("Expecting Symbol, SymbolNode, or GenSym, but got ", x)
  end
end

function lookup(state, v)
  if haskey(state.locals, v)
    state.locals[v]
  else
    Unknown
  end
end

# (:lambda, {param, meta@{localvars, types, freevars}, body})
function from_lambda(state, expr)
  local head = expr.head
  local ast  = expr.args
  local typ  = expr.typ
  assert(length(ast) == 3)
  local linfo = lambdaExprToLambdaInfo(expr)
  # very conservative handling by setting free variables to Unknown.
  # TODO: may want to simulate function call at call site to get
  #       more accurate information.
  for (v, vd) in linfo.escaping_defs
    update_unknown(state, v)
  end
  return NotArray
end

function from_exprs(state, ast, callback=not_handled, cbdata=nothing)
  local len  = length(ast)
  [ from_expr(state, exp, callback, cbdata) for exp in ast ]
end

function from_body(state, expr::Any, callback, cbdata)
  local exprs = expr.args
  local ret = NotArray       # default return
  for i = 1:length(exprs)
    if state.nest_level == 0
      state.top_level_idx = i
    end
    ret = from_expr(state, exprs[i], callback, cbdata)
  end
  return ret
end

function from_assignment(state, expr::Any, callback, cbdata)
  local head = expr.head
  local ast  = expr.args
  local typ  = expr.typ
  assert(length(ast) == 2)
  local lhs = ast[1]
  local rhs = ast[2]
  dprintln(2, "AA ", lhs, " = ", rhs)
  if isa(lhs, SymbolNode)
    lhs = lhs.name
  end
  assert(isa(lhs, Symbol) || isa(lhs, GenSym))
  if lookup(state, lhs) != NotArray
    rhs = from_expr(state, rhs, callback, cbdata)
    # if all vars that have rhs are not alive afterwards
    # then we can safely give v a fresh ID.
    if state.nest_level == 0
      tls = CompilerTools.LivenessAnalysis.find_top_number(state.top_level_idx, state.liveness)
      assert(tls != nothing)
      assert(CompilerTools.LivenessAnalysis.isDef(lhs, tls))
      if (haskey(state.revmap, rhs))
        dead = true
        for v in state.revmap[rhs]
          dead = dead && !in(v, tls.live_out)
        end
        if dead
          rhs = next_node(state)
        end
      end
    end
    dprintln(2, "AA update ", lhs, " <- ", rhs)
    update(state, lhs, rhs)
  end
end

function from_call(state, expr::Any)
  # The assumption here is that the program has already been translated
  # by DomainIR, and all args are either SymbolNode or Constant.
  local head = expr.head
  local ast = expr.args
  local typ = expr.typ
  assert(length(ast) >= 1)
  local fun  = ast[1]
  local args = ast[2:end]
  dprintln(2, "AA from_call: fun=", fun, " typeof(fun)=", typeof(fun), " args=",args, " typ=", typ)
  #fun = from_expr(state, fun)
  #dprintln(2, "AA from_call: new fun=", fun)
  if is(fun, :arrayref) || is(fun, :arrayset) || fun==TopNode(:arrayref) || fun==TopNode(:arrayset)
    # This is actually an conservative answer since arrayref might return
    # an array too, but we don't consider it as a case to handle.
    return NotArray
  else
    dprintln(2, "AA: unknown call ", fun)
    # For unknown calls, conservative assumption is that after
    # the call, its array type arguments might alias each other.
    for exp in args
      if isa(exp, SymbolNode)
        update_unknown(state, exp.name)
      elseif isa(exp, Symbol)
        update_unknown(state, exp)
      end
    end
    return Unknown
  end
end

function from_return(state, expr, callback, cbdata)
  local head = expr.head
  local typ  = expr.typ
  local args = from_exprs(state, expr.args, callback, cbdata)
  if length(args) == 1
    return args[1]
  else
    return Unknown
  end
end

function from_expr(state, ast, callback=not_handled, cbdata=nothing)
  if isa(ast, LambdaStaticData)
    ast = uncompressed_ast(ast)
  end

  # "nothing" output means couldn't be handled
  handled = callback(ast, state, cbdata)
  if isa(handled, Array)
    dprintln(3,"Processing expression from callback for ", ast) 
    dprintln(3,handled)
    return from_exprs(state, handled, callback, cbdata)
    # AST node replaced
  elseif isa(handled, Expr)
    ast = handled
  elseif isa(handled,Integer)
    return handled
  end


  local asttyp = typeof(ast)
  dprint(2, "AA from_expr: ", asttyp)
  if is(asttyp, Expr)
    local head = ast.head
    local args = ast.args
    local typ  = ast.typ
    dprintln(2, " --> ", head)
    if is(head, :lambda)
        return from_lambda(state, ast)
    elseif is(head, :body)
        return from_body(state, ast, callback, cbdata)
    elseif is(head, :(=))
        return from_assignment(state, ast, callback, cbdata)
    elseif is(head, :return)
        return from_return(state, ast, callback, cbdata)
    elseif is(head, :call)
        return from_call(state, ast)
        # TODO: catch domain IR result here
    elseif is(head, :call1)
      return from_call(state, ast)
    elseif is(head, :method)
        # skip
    elseif is(head, :line)
        # skip
    elseif is(head, :new)
        # skip
    elseif is(head, :boundscheck)
        # skip?
    elseif is(head, :type_goto)
        # skip?
    elseif is(head, :gotoifnot)
        # skip
    elseif is(head, :loophead)
        # skip
    elseif is(head, :loopend)
        # skip
    elseif is(head, :meta)
        # skip
    else
        throw(string("from_expr: unknown Expr head :", head))
    end
  elseif is(asttyp, SymbolNode)
    dprintln(2, " ", ast)
    return lookup(state, ast.name)
  else
    dprintln(2, " not handled ", ast)
  end
  return Unknown
end

function isarray(typ)
  isa(typ, DataType) && is(typ.name, Array.name)
end

function isbitarray(typ)
  isa(typ, DataType) && is(typ.name, BitArray.name)
end


function analyze_lambda_body(body :: Expr, lambdaInfo :: CompilerTools.LambdaHandling.LambdaInfo, liveness, callback, cbdata)
  local state = init_state(liveness)
  dprintln(2, "AA ", isa(body, Expr), " ", is(body.head, :body)) 
  # FIXME: surprisingly the first value printed above is false!
  for (v, vd) in lambdaInfo.var_defs
    if !(isarray(vd.typ) || isbitarray(vd.typ))
      update_notarray(state, v)
    end
  end
  for v in lambdaInfo.input_params
    vtyp = CompilerTools.LambdaHandling.getType(v, lambdaInfo)
    # Note we assume all input parameters do not aliasing each other,
    # which is a very strong assumption. This may require reconsideration.
    # Update: changed to assum nothing by default.
    if isarray(vtyp) || isbitarray(vtyp)
      #update_node(state, v, next_node(state))
      update_unknown(state, v)
    end
  end
  dprintln(2, "AA locals=", state.locals)
  from_expr(state, body, callback, cbdata)
  dprintln(2, "AA locals=", state.locals)
  local revmap = Dict{Int, SymGen}()
  local unique = Set{SymGen}()
  # keep only variables that have unique object IDs.
  # TODO: should consider liveness either here or during analysis,
  #       since its ok to alias dead vars.
  for (v, w) in state.locals
    if w > 0
      if haskey(revmap, w)
        delete!(unique, revmap[w])
      else
        push!(unique, v)
        revmap[w] = v
      end
    end
  end
  dprintln(2, "AA after alias analysis: ", unique)
  # return the set of variables that are confirmed to have no aliasing
  return unique
end

function analyze_lambda(expr :: Expr, liveness, callback, cbdata)
  lambdaInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(expr)
  analyze_lambda_body(CompilerTools.LambdaHandling.getBody(expr), lambdaInfo, liveness, callback, cbdata)
end

end

