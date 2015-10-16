module Capture

import CompilerTools
import ..API
import ..operators

ref_assign_map = Dict{Symbol, Symbol}(
    :(+=) => :(+),
    :(-=) => :(-),
    :(*=) => :(*),
    :(/=) => :(/)
)

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
      lhs = node.args[1].args[1]
      idx = node.args[1].args[2:end]
      rhs = node.args[2]
      tmpvar = gensym()
      node.head = :block
      node.args = Any[
           Expr(:(=), tmpvar, rhs),
           Expr(:call, GlobalRef(API, :setindex!), lhs, tmpvar, idx...),
           tmpvar]
    elseif haskey(ref_assign_map, head) && isa(node.args[1], Expr) && node.args[1].head == :ref
      # x[...] += ...
      lhs = node.args[1].args[1]
      idx = node.args[1].args[2:end]
      rhs = node.args[2]
      tmpvar = gensym()
      node.head = :block
      node.args = Any[
           Expr(:(=), tmpvar, Expr(:call, ref_assign_map[head],
                                          Expr(:call, GlobalRef(API, :getindex), lhs, idx...),
                                          rhs)),
           Expr(:call, GlobalRef(API, :setindex!), lhs, tmpvar, idx...),
           tmpvar]

    elseif node.head == :ref
      node.head = :call
      node.args = Any[GlobalRef(API, :getindex), node.args...]
    end
  end 
  CompilerTools.AstWalker.ASTWALK_RECURSE
end

end

