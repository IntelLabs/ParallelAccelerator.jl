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

function process_assignment(node, lhs::Symbol, rhs::Expr)
    if rhs.head ==:call && rhs.args[1]==:DataSource
        arr_var_expr = node.args[2].args[2]
        
        @assert arr_var_expr.args[1]==:Array || arr_var_expr.args[1]==:Matrix 
                || arr_var_expr.args[1]==:Vector "Data sources need Vector or Array or Matrix as type"
        
        if arr_var_expr.args[1]==:Matrix
            dims = 2 # Matrix type is 2D array
        elseif arr_var_expr.args[1]==:Vector
            dims = 1
        elseif arr_var_expr.args[1]==:Array
            dims = arr_var_expr.args[3]
        end

        node.args[1] = :($(node.args[1])::$arr_var_expr)
        source_typ = node.args[2].args[3]
        @assert source_typ==:HDF5 || source_typ==:TXT "Only HDF5 and TXT (text) data sources supported for now."
        call_name = symbol("__hps_data_source_$source_typ")
        
        call = Expr(:call)
        
        if source_typ==:HDF5
            hdf_var_name = node.args[2].args[4]
            hdf_file_name = node.args[2].args[5]
            call = :($(call_name)($hdf_var_name,$hdf_file_name))
        else
            txt_file_name = node.args[2].args[4]
            call = :($(call_name)($txt_file_name))
        end
        
        node.args[2] = call
#=        arr_var_expr = node.args[2].args[2]
        dims = arr_var_expr.args[3]
        @assert arr_var_expr.args[1]==:Array "Data sources need arrays as type"
        
        source_typ = node.args[2].args[3]
        @assert source_typ==:HDF5 "Only HDF5 data sources supported for now."
        
        hdf_var_name = node.args[2].args[4]
        hdf_file_name = node.args[2].args[5]

        # return :($(node.args[1]) = zeros($(arr_var_expr.args[2]),$(arr_var_expr.args[3])))
        num = get_unique_data_source_num()
        hps_source_var = symbol("__hps_data_source_$num")
        hps_source_size_var = symbol("__hps_data_source_size_$num")
        hps_source_size_call = symbol("__hps_data_source_get_size_$(dims)d")
        declare_expr = :( $hps_source_var = __hps_data_source_open($hdf_var_name,$hdf_file_name))
        size_expr = :( $hps_source_size_var = $hps_source_size_call($hps_source_var))
        return [declare_expr; size_expr]
=#
   elseif rhs.head==:call && isa(rhs.args[1],Expr) && rhs.args[1].head==:. && rhs.args[1].args[1]==:HPS
        hps_call = rhs.args[1].args[2].args[1]
        new_opr = symbol("__hps_$hps_call")
        node.args[2].args[1] = new_opr
        node.args[1] = :($lhs::Matrix{Float64})
   end
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

