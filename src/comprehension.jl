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

module Comprehension

export @comprehend

import CompilerTools
import ..API

"""
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
    #headers[i] = Expr(:(=), indices[i], Expr(:call, GlobalRef(Base, :getindex), ranges[i], params[i]))
    #first(v) + (i-1)*step(v)
    first = GlobalRef(Base, :first)
    step = GlobalRef(Base, :step)
    headers[i] = :($(indices[i]) = $first($(ranges[i])) + ($(params[i]) - 1) * $step($(ranges[i])))
  end
  args = Expr(:tuple, params...)
  dims = Expr(:tuple, [ Expr(:call, GlobalRef(Base, :length), r) for r in ranges ]...)
  tmpret = gensym("tmp")
  tmpinits = [ :($idx = 1) for idx in params ]
  typetest = :(local $tmpret; if 1<0 let $(tmpinits...); $(headers...); $tmpret=$body end end)
  ast = Expr(:call, GlobalRef(API, :cartesianarray), 
                :($args -> let $(headers...); $body end), Expr(:static_typeof, tmpret), :($dims))
  Expr(:block, typetest, ast) 
end

"""
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


"""
Translate all comprehension in an AST into equivalent code that uses cartesianarray call.
"""
macro comprehend(ast)
  CompilerTools.AstWalker.AstWalk(ast, process_node, nothing)
  Core.eval(current_module(), ast)
end

end


