module Comprehension

export @comprehend

import CompilerTools

@doc """
Translate an ast whose head is :comprehension into equivalent code that uses cartesianarray call.
"""
function comprehension_to_cartesianarray(ast)
  assert(ast.head == :comprehension)
  body = ast.args[1]
  ndim = length(ast.args) - 1
  params = Array(Symbol, ndim)
  indices = Array(Symbol, ndim)
  ranges = Array(Any, ndim)
  headers = Array(Any, ndim)
  for i = 1:ndim
    r = ast.args[i + 1]
    assert(r.head == :(=))
    indices[i] = r.args[1]
    params[i] = gensym(string(indices[i]))
    ranges[i] = r.args[2]
    headers[i] = Expr(:(=), indices[i], Expr(:call, GlobalRef(Base, :getindex), ranges[i], params[i]))
  end
  args = Expr(:tuple, params...)
  dims = Expr(:tuple, [ Expr(:call, GlobalRef(Base, :length), r) for r in ranges ]...)
  tmpret = gensym("tmp")
  tmpinits = [ :($idx = 1) for idx in indices ]
  typetest = :(local $tmpret; if 1<0 let $(tmpinits...); $tmpret=$body end end)
  ast = Expr(:call, GlobalRef(current_module(), :cartesianarray), 
                :($args -> let $(headers...); $body end), Expr(:static_typeof, tmpret), :($dims))
  Expr(:block, typetest, ast) 
end

@doc """
This function is a AstWalker callback.
"""
function process_node(node, state, top_level_number, is_top_level, read)
  if !isa(node,Expr)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
  end
  if node.head == :typed_comprehension
    typ = node.args[1]
    # Transform into untyped :comprehension because type will be passed as a
    # parameter the same way we handle untyped :comprehensions
    node.head = :comprehension
    node.args = node.args[2:end]
  end
  return (node.head == :comprehension) ? comprehension_to_cartesianarray(node) : CompilerTools.AstWalker.ASTWALK_RECURSE
end


@doc """
Translate all comprehension in an AST into equivalent code that uses cartesianarray call.
"""
macro comprehend(ast)
  AstWalker.AstWalk(ast, process_node, nothing)
  Core.eval(current_module(), ast)
end

end


