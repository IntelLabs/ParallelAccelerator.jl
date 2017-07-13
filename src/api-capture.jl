module Capture

#using Debug

import CompilerTools
import ..API
import ..operators
import ..binary_operators
import ..rename_if_needed
import ParallelAccelerator

import CompilerTools.DebugMsg
DebugMsg.init()

const binary_operator_set = Set(binary_operators)

#data_source_num = 0

ref_assign_map = Dict{Symbol, Symbol}(
    :(+=) => :pa_api_add,
    :(-=) => :pa_api_sub,
    :(*=) => :pa_api_mul,
    :(/=) => :pa_api_div,
    :(.+=) => :pa_api_elem_add,
    :(.-=) => :pa_api_elem_sub,
    :(.*=) => :pa_api_elem_mul,
    :(./=) => :pa_api_elem_div
)

type process_node_state
    array
    dim
end

function process_node(node::Symbol, state, top_level_number, is_top_level, read)
    if node == :end
        if state == nothing
            @dprintln(3, "Found :end symbol in AST but had no array name to generate replacement length expression.")
        else
            @assert isa(state, process_node_state) "process_node state was not nothing nor a process_node_state."       
            return Expr(:call, GlobalRef(Base, :arraysize), state.array, state.dim)
        end
    end

    CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
At macro level, we translate function calls and operators that matches operator names
in our API module to direct call to those in the API module. 
"""
function process_node(node::Expr, state, top_level_number, is_top_level, read)
    head = node.head
    @dprintln(3, "api-capture process_node ", node, " head = ", head)
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
        @dprintln(3, "idx before end substitution in process_node = ", idx)
        for i = 1:length(idx)
            idx[i] = CompilerTools.AstWalker.AstWalk(idx[i], process_node, process_node_state(lhs, i))
        end
        @dprintln(3, "idx after end substitution in process_node = ", idx)
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
        node.head = ((string(head)[1] == '.') && VERSION >= v"0.5.0-dev+5381") ? :(.=) : :(=)
        node.args = Any[ lhs, Expr(:call, GlobalRef(API, ref_assign_map[head]), lhs, rhs) ]
    elseif node.head == :ref
        node.head = :call
        @dprintln(3, "ref array = ", node.args[1], " args = ", node.args[2:end])
        node.args[1] = CompilerTools.AstWalker.AstWalk(node.args[1], process_node, nothing)
        for i = 2:length(node.args)
            node.args[i] = CompilerTools.AstWalker.AstWalk(node.args[i], process_node, process_node_state(node.args[1], i-1))
        end
        @dprintln(3, "after args = ", node.args[2:end])
        node.args = Any[GlobalRef(API, :getindex), node.args...]
        return node
    end

    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_node(node::Any, state, top_level_number, is_top_level, read)
    CompilerTools.AstWalker.ASTWALK_RECURSE
end


function process_operator(node::Expr, opr::Symbol)
    rename_opr = rename_if_needed(opr)
    api_opr = GlobalRef(API, rename_opr)
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
        @dprintln(3, "idx before end substitution in process_assignment = ", idx)
        for i = 1:length(idx)
            idx[i] = CompilerTools.AstWalker.AstWalk(idx[i], process_node, process_node_state(lhs, i))
        end
        @dprintln(3, "idx after end substitution in process_assignment = ", idx)
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

function translate_par(args)
    na = length(args)
    @assert (na > 0) "Expect a for loop as argument to @par"
    redVars = Array{Symbol}(na-1)
    redOps = Array{Symbol}(na-1)
    loop = args[end]
    if !isa(loop,Expr) || !(loop.head === :for)
        error("malformed @par loop")
    end
    if !isa(loop.args[1], Expr) 
        error("maltformed for loop")
    end
    for i = 1:na-1
        @assert (isa(args[i], Expr) && args[i].head == :call) "Expect @par reduction in the form of var(op), but got " * string(args[i])
        v = args[i].args[1]
        op = args[i].args[2]
        #println("got reduction variable ", v, " with function ", op)
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
    params = Array{Symbol}(ndim)
    indices = Array{Symbol}(ndim)
    ranges = Array{Any}(ndim)
    headers = Array{Any}(ndim)
    for i = 1:ndim
        r = loopheads[i]
        assert(r.head == :(=))
        indices[i] = r.args[1]
        params[i] = gensym(string(indices[i]))
        ranges[i] = r.args[2]
        #headers[i] = Expr(:(=), indices[i], Expr(:call, GlobalRef(Base, :getindex), ranges[i], params[i]))
        #first(v) + (i-1)*step(v)
        first = GlobalRef(Base, :first)
        step = GlobalRef(Base, :step)
        headers[i] = :($(indices[i]) = $first($(ranges[i])) + ($(params[i]) - 1) * $step($(ranges[i])))
        # if range is 1:N, generate simple assignment instead of formula
        # this helps array index analysis (motivated by Kernel score in HPAT)
        if isa(ranges[i], Expr) && isStart1UnitRange(ranges[i])
            headers[i] = :($(indices[i]) = $(params[i]))
        end
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
                    :($args -> let $(headers...); $body; return end), :($dims))
    for i=1:na-1
        redVar = redVars[i]
        redArg = gensym(string(redVar))
        opr = redOps[i]
        if in(opr, operators)
            if haskey(API.rename_forward, opr)
                opr = API.rename_forward[opr]
            end
            opr = GlobalRef(API, opr)
        end
        redBody = Expr(:(=), redVars[i], Expr(:call, opr, redVar, redArg))
        # redVals = Expr(:call, GlobalRef(Base, :copy), redVar)
        push!(ast.args, :($redArg -> begin $redBody; return $redVar; end, $redVar))
    end
    #println("ast = ", ast)
    ast
    #args -> esc(loop)
end

function isStart1UnitRange(node::Expr)
    if node.head==:(:) && length(node.args)==2 && node.args[1]==1
        return true
    end
    return false
end

function process_par_macro(node::Expr, state, top_level_number, is_top_level, read)
    if node.head === :macrocall
        #println("got par macro, args = ", node.args)
        if node.args[1] == Symbol("@par")
            ast = translate_par(node.args[2:end])
            node.head = ast.head
            node.args = ast.args
            node.typ  = ast.typ
        end
    end
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_par_macro(node::ANY, state, top_level_number, is_top_level, read)
    CompilerTools.AstWalker.ASTWALK_RECURSE
end



end # module Capture

