module Capture

import CompilerTools
import ..API
import ..operators

function translate_call(node::Expr)
  opr = node.args[1]
  if isa(opr, Symbol) && in(opr, operators)
    node.args[1] = GlobalRef(API, opr)
  end
end

function process_node(node, state, top_level_number, is_top_level, read)
  if isa(node, Expr) && node.head == :call
    translate_call(node)
  end 
  CompilerTools.AstWalker.ASTWALK_RECURSE
end

end

