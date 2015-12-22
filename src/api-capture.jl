module Capture

#using Debug

import CompilerTools
import ..API
import ..operators

#data_source_num = 0

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
        if isa(opr, Symbol) && in(opr, operators)
            node.args[1] = GlobalRef(API, opr)
        end
    elseif head == :(=) && isa(node.args[1], Symbol) && isa(node.args[2], Expr) && node.args[2].head ==:call && node.args[2].args[1]==:DataSource
        arr_var_expr = node.args[2].args[2]
        dims = arr_var_expr.args[3]
        @assert arr_var_expr.args[1]==:Array "Data sources need arrays as type"
        node.args[1] = :($(node.args[1])::$arr_var_expr)
        source_typ = node.args[2].args[3]
        @assert source_typ==:HDF5 "Only HDF5 data sources supported for now."
        hdf_var_name = node.args[2].args[4]
        hdf_file_name = node.args[2].args[5]

        call_name = symbol("__hps_data_source_$source_typ")
        call = :($(call_name)($hdf_var_name,$hdf_file_name))
        node.args[2] = call
        #@bp
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

#function get_unique_data_source_num()
#    global data_source_num
#    data_source_num += 1
#    return data_source_num
#end

end # module Capture

