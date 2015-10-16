module Capture

import CompilerTools
import ..API
import ..operators

@doc """
At macro level, we translate function calls and operators that matches operator names
in our API module to direct call to those in the API module. 
"""
function process_node(node, state, top_level_number, is_top_level, read)
  if isa(node, Expr) 
    head = node.head
    if head == :comparison
      # Surprise! Expressions like x > y are not :call in AST at macro level
      opr = node.args[2]
      if isa(opr, Symbol) && in(opr, operators)
        node.args[2] = GlobalRef(API, opr)
      end
    elseif head == :call
      # f(...)
      opr = node.args[1]
      if isa(opr, Symbol) && in(opr, operators)
        node.args[1] = GlobalRef(API, opr)
      end
    elseif head == :(=) && isa(node.args[1], Expr) && node.args[1].head == :ref
      # x[...] = ...
      node.head = :call
      node.args = Any[GlobalRef(API, :setindex!), node.args[1].args[1], node.args[2], node.args[1].args[2:end]...]
    elseif node.head == :ref
      node.head = :call
      node.args = Any[GlobalRef(API, :getindex), node.args...]
    end
  end 
  CompilerTools.AstWalker.ASTWALK_RECURSE
end

end

