module Capture

#using Debug

import CompilerTools
import ..API
import ..operators
import ..binary_operators

const binary_operator_set = Set(binary_operators)

#data_source_num = 0

ref_assign_map = Dict{Symbol, Symbol}(
    :(+=) => :(+),
    :(-=) => :(-),
    :(*=) => :(*),
    :(/=) => :(/),
    :(.+=) => :(.+),
    :(.-=) => :(.-),
    :(.*=) => :(.*),
    :(./=) => :(./)
)

"""
At macro level, we translate function calls and operators that matches operator names
in our API module to direct call to those in the API module. 
"""
function process_node(node::Expr, state, top_level_number, is_top_level, read)
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
        process_operator(node, opr)
    elseif head == :(=) 
        process_assignment(node, node.args[1], node.args[2])
    elseif haskey(ref_assign_map, head) && isa(node.args[1], Expr) && node.args[1].head == :ref
        # x[...] += ...
        lhs = node.args[1].args[1]
        idx = node.args[1].args[2:end]
        rhs = node.args[2]
        tmpvar = gensym()
        node.head = :block
        node.args = Any[
        Expr(:(=), tmpvar, Expr(:call, GlobalRef(API, ref_assign_map[head]),
        Expr(:call, GlobalRef(API, :getindex), lhs, idx...),
        rhs)),
        Expr(:call, GlobalRef(API, :setindex!), lhs, tmpvar, idx...),
        tmpvar]

    elseif haskey(ref_assign_map, head) 
        # x += ...
        lhs = node.args[1]
        rhs = node.args[2]
        tmpvar = gensym()
        node.head = :(=)
        node.args = Any[ lhs, Expr(:call, GlobalRef(API, ref_assign_map[head]), lhs, rhs) ]
    elseif node.head == :ref
        node.head = :call
        node.args = Any[GlobalRef(API, :getindex), node.args...]
    end

    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_node(node::Any, state, top_level_number, is_top_level, read)
    CompilerTools.AstWalker.ASTWALK_RECURSE
end


function process_operator(node::Expr, opr::Symbol)
    api_opr = GlobalRef(API, opr)
    if in(opr, operators)
        node.args[1] = api_opr
    end
    if in(opr, binary_operator_set) && length(node.args) > 3
        # we'll turn multiple arguments into pairwise
        expr = foldl((a, b) -> Expr(:call, api_opr, a, b), node.args[2:end])
        node.args = expr.args
    end
end


function process_operator(node::Expr, opr::Any)
end

function process_assignment(node, lhs::Expr, rhs::Any)
    if lhs.head == :ref
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
    end
end

function process_assignment(node, lhs::Any, rhs::Any)
end

#function get_unique_data_source_num()
#    global data_source_num
#    data_source_num += 1
#    return data_source_num
#end

end # module Capture

