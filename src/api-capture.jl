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

macro par(args...)
    na = length(args)
    @assert (na > 0) "Expect a for loop as argument to @par"
    redVars = Array(Symbol, na-1)
    redOps = Array(Symbol, na-1)
    loop = args[end]
    if !isa(loop,Expr) || !is(loop.head,:for)
        error("malformed @par loop")
    end
    if !isa(loop.args[1], Expr) 
        error("maltformed for loop")
    end
    for i = 1:na-1
        @assert (isa(args[i], Expr) && args[i].head == :call) "Expect @par reduction in the form of var(op), but got " * string(args[i])
        v = args[i].args[1]
        op = args[i].args[2]
        println("got reduction variable ", v, " with function ", op)
        @assert (isa(v, Symbol)) "Expect reduction variable to be symbols, but got " * string(v)
        @assert (isa(op, Symbol)) "Expect reduction operator to be symbols, but got " * string(op)
        redVars[i] = v
        redOps[i] = op
    end
    if loop.args[1].head == :block
        loopheads = loop.args[1].args
    else
        loopheads = Any[loop.args[1]]
    end
    ndim = length(loopheads)
    body = loop.args[2]
    params = Array(Symbol, ndim)
    indices = Array(Symbol, ndim)
    ranges = Array(Any, ndim)
    headers = Array(Any, ndim)
    for i = 1:ndim
        r = loopheads[i]
        assert(r.head == :(=))
        indices[i] = r.args[1]
        params[i] = gensym(string(indices[i]))
        ranges[i] = r.args[2]
        headers[i] = Expr(:(=), indices[i], Expr(:call, GlobalRef(Base, :getindex), ranges[i], params[i]))
    end
    args = Expr(:tuple, params...)
    dims = Expr(:tuple, [ Expr(:call, GlobalRef(Base, :length), r) for r in ranges ]...)
    tmpret = gensym("tmp")
    tmpinits = [ :($idx = 1) for idx in params ]
    #println("indices = ", indices)
    #println("params = ", params)
    #println("ranges = ", ranges)
    #println("headers = ", headers)
    #println("tmpinits = ", tmpinits)
    #println("body = ", body)
    #if na==1
    #    thecall = :(pfor($(make_pfor_body(var, body, :therange)), length(therange)))
    #else
    #    thecall = :(preduce($(esc(reducer)),
    #                        $(make_preduce_body(reducer, var, body, :therange)), length(therange)))
    #end
    #localize_vars(quote therange = $(esc(r)); $thecall; end)
    ast =Expr(:call, GlobalRef(API, :cartesianmapreduce), 
                    :($args -> let $(headers...); $body end), :($dims))
    for i=1:na-1
        redVar = redVars[i]
        redArg = gensym(string(redVar))
        opr = redOps[i]
        if in(opr, operators)
            opr = GlobalRef(API, opr)
        end
        redBody = Expr(:(=), redVars[i], Expr(:call, opr, redVar, redArg))
        # redVals = Expr(:call, GlobalRef(Base, :copy), redVar)
        push!(ast.args, :($redArg -> begin $redBody; return $redVar; end, $redVar))
    end
    println("ast = ", ast)
    esc(ast)
    #args -> esc(loop)
end

end # module Capture

