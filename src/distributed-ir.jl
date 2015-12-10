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

module DistributedIR

#using Debug

using CompilerTools
import CompilerTools.DebugMsg
DebugMsg.init()
using CompilerTools.AstWalker
import CompilerTools.ReadWriteSet
using CompilerTools.LambdaHandling
import ..ParallelIR
import ..ParallelIR.isArrayType
import ..ParallelIR.getParforNode
import ..ParallelIR.isAllocation
import ..ParallelIR.TypedExpr

import ..ParallelIR.ISCAPTURED
import ..ParallelIR.ISASSIGNED
import ..ParallelIR.ISASSIGNEDBYINNERFUNCTION
import ..ParallelIR.ISCONST
import ..ParallelIR.ISASSIGNEDONCE
import ..ParallelIR.ISPRIVATEPARFORLOOP
import ..ParallelIR.PIRReduction

# ENTRY to distributedIR
function from_root(function_name, ast :: Expr)
    @assert ast.head == :lambda "Input to DistributedIR should be :lambda Expr"
    dprintln(1,"Starting main DistributedIR.from_root.  function = ", function_name, " ast = ", ast)

    linfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(ast)
    state::DistIrState = initDistState(linfo)

    dprintln(3,"DistIR state before walk: ",state)
    AstWalk(ast, get_arr_dist_info, state)
    dprintln(3,"DistIR state after walk: ",state)

    # now that we have the array info, see if parfors are distributable 
    checkParforsForDistribution(state)
    dprintln(3,"DistIR state after check: ",state)
    
    # transform body
    @assert ast.args[3].head==:body "DistributedIR: invalid lambda input"
    body = TypedExpr(ast.args[3].typ, :body, from_toplevel_body(ast.args[3].args, state)...)
    new_ast = CompilerTools.LambdaHandling.lambdaInfoToLambdaExpr(state.lambdaInfo, body)
    # ast = from_expr(ast)
    return new_ast
end

type ArrDistInfo
    isSequential::Bool      # can't be distributed; e.g. it is used in sequential code
    dim_sizes::Array{Union{SymAllGen,Int},1}      # sizes of array dimensions
    
    function ArrDistInfo(num_dims::Int)
        new(false, zeros(Int64,num_dims))
    end
end

# information about AST gathered and used in DistributedIR
type DistIrState
    # information about all arrays
    arrs_dist_info::Dict{SymGen, ArrDistInfo}
    parfor_info::Dict{Int, Array{SymGen,1}}
    lambdaInfo::LambdaInfo
    seq_parfors::Array{Int,1}
    dist_arrays::Array{SymGen,1}
    uniqueId::Int
    
    function DistIrState(linfo)
        new(Dict{SymGen, Array{ArrDistInfo,1}}(), Dict{Int, Array{SymGen,1}}(), linfo, Int[], SymGen[],0)
    end
end

function initDistState(linfo)
    state = DistIrState(linfo)
    
    #params = linfo.input_params
    vars = linfo.var_defs
    gensyms = linfo.gen_sym_typs

    # Populate the symbol table
    for sym in keys(vars)
        v = vars[sym] # v is a VarDef
        if isArrayType(v.typ)
            arrInfo = ArrDistInfo(ndims(v.typ))
            state.arrs_dist_info[sym] = arrInfo
        end 
    end

    for k in 1:length(gensyms)
        typ = gensyms[k]
        if isArrayType(typ)
            arrInfo = ArrDistInfo(ndims(typ))
            state.arrs_dist_info[GenSym(k-1)] = arrInfo
        end
    end
    return state
end

# state for get_arr_dist_info AstWalk
#=type ArrInfoState
    inParfor::
    state # DIR state
end
=#

@doc """
mark sequential arrays
"""
function get_arr_dist_info(node::Expr, state, top_level_number, is_top_level, read)
    head = node.head
    # arrays written in parfors are ok for now
    
    dprintln(3,"DistIR arr info walk Expr node: ", node)
    # length==8 since 1D only is supported for now
    if head==:(=) && isAllocation(node.args[2]) && length(node.args[2].args)==8
        arr = node.args[1]
        state.arrs_dist_info[arr].dim_sizes = [node.args[2].args[7]] # 1D hack
        dprintln(3,"DistIR arr info dim_sizes update: ", state.arrs_dist_info[arr].dim_sizes)
    elseif head==:parfor
        parfor = getParforNode(node)
        rws = parfor.rws
        
        readArrs = collect(keys(rws.readSet.arrays))
        writeArrs = collect(keys(rws.writeSet.arrays))
        allArrs = [readArrs;writeArrs]
        # keep mapping from parfors to arrays
        state.parfor_info[parfor.unique_id] = allArrs
        
        # only 1D parfors supported for now
        if !parfor.simply_indexed || length(parfor.loopNests)!=1
            dprintln(2,"DistIR arr info walk parfor sequential: ", node)
            for arr in allArrs
                state.arrs_dist_info[arr].isSequential = true
            end
            return node
        end
        
        indexVariable::SymbolNode = parfor.loopNests[1].indexVariable
        for arr in keys(rws.readSet.arrays)
             index = rws.readSet.arrays[arr]
             if length(index)!=1 || length(index[1])!=1 || index[1][1].name!=indexVariable.name
                dprintln(2,"DistIR arr info walk arr read index sequential: ", index, " ", indexVariable)
                state.arrs_dist_info[arr].isSequential = true
             end
        end
        
        for arr in keys(rws.writeSet.arrays)
             index = rws.writeSet.arrays[arr]
             if length(index)!=1 || length(index[1])!=1 || index[1][1].name!=indexVariable.name
                dprintln(2,"DistIR arr info walk arr write index sequential: ", index, " ", indexVariable)
                state.arrs_dist_info[arr].isSequential = true
             end
        end
        return node
    # arrays written in sequential code are not distributed
    elseif head!=:body && head!=:block && head!=:lambda
        rws = CompilerTools.ReadWriteSet.from_exprs([node], ParallelIR.pir_live_cb, state.lambdaInfo)
        readArrs = collect(keys(rws.readSet.arrays))
        writeArrs = collect(keys(rws.writeSet.arrays))
        allArrs = [readArrs;writeArrs]
        for arr in allArrs
            dprintln(2,"DistIR arr info walk arr in sequential code: ", arr, " ", node)
            state.arrs_dist_info[arr].isSequential = true
        end
        return node
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end


function get_arr_dist_info(ast::Any, state, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end
@doc """
All arrays of a parfor should distributable for it to be distributable.
If an array is used in any sequential parfor, it is not distributable.
"""
function checkParforsForDistribution(state::DistIrState)
    changed = true
    while changed
        changed = false
        for parfor_id in keys(state.parfor_info)
            arrays = state.parfor_info[parfor_id]
            for arr in arrays
                # all parfor arrays should have same size
                if state.arrs_dist_info[arr].isSequential ||
                        !isEqualDimSize(state.arrs_dist_info[arr].dim_sizes, state.arrs_dist_info[arrays[1]].dim_sizes)
                    changed = true
                    push!(state.seq_parfors, parfor_id)
                    for a in arrays
                        state.arrs_dist_info[a].isSequential = true
                    end
                    break
                end
            end
        end
    end
    # all arrays of distributed parfors are distributable at this point 
    for parfor_id in keys(state.parfor_info)
        if !in(state.seq_parfors, parfor_id)
            dprintln(2,"DistIR distributable parfor: ", parfor_id," arrays: ", state.parfor_info[parfor_id])
            append!(state.dist_arrays, state.parfor_info[parfor_id])
        end
    end
end

function isEqualDimSize(sizes1::Array{Union{SymAllGen,Int},1} , sizes2::Array{Union{SymAllGen,Int},1})
    if length(sizes1)!=length(sizes2)
        return false
    end
    for i in 1:length(sizes1)
        if !eqSize(sizes1[i],sizes2[i])
            return false
        end
    end
    return true
end

function eqSize(a::SymbolNode, b::SymbolNode)
    return a.name == b.name
end

function eqSize(a::Any, b::Any)
    return a==b
end

# nodes are :body of AST
function from_toplevel_body(nodes::Array{Any,1}, state)
    res::Array{Any,1} = genDistributedInit(state)
    for node in nodes
        new_exprs = from_expr(node, state)
        append!(res, new_exprs)
    end
    return res
end


function from_expr(node::Expr, state)
    head = node.head
    if head==:(=)
        return from_assignment(node, state)
    elseif head==:parfor
        return from_parfor(node, state)
    #elseif head==:block
    else
        return [node]
    end
end


function from_expr(node::Any, state)
    return [node]
end

# generates initialization code for distributed execution
function genDistributedInit(state)
    initCall = Expr(:call,TopNode(:hps_dist_init))
    numPesCall = Expr(:call,TopNode(:hps_dist_num_pes))
    nodeIdCall = Expr(:call,TopNode(:hps_dist_node_id))
    
    CompilerTools.LambdaHandling.addLocalVar(symbol("__hps_num_pes"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.lambdaInfo)
    CompilerTools.LambdaHandling.addLocalVar(symbol("__hps_node_id"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.lambdaInfo)

    num_pes_assign = Expr(:(=), :__hps_num_pes, numPesCall)
    node_id_assign = Expr(:(=), :__hps_node_id, nodeIdCall)

    return Any[initCall; num_pes_assign; node_id_assign]
end

function from_assignment(node::Expr, state)
    @assert node.head==:(=) "DistributedIR invalid assignment head"

    if isAllocation(node.args[2])
        arr = node.args[1]
        if in(arr, state.dist_arrays)
            # generate array division
            old_size = node.args[2].args[7]

            div_size_var = symbol("__hps_size_"*string(getDistNewID(state)))
            new_size_var = symbol("__hps_size_"*string(getDistNewID(state)))

            CompilerTools.LambdaHandling.addLocalVar(div_size_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.lambdaInfo)
            CompilerTools.LambdaHandling.addLocalVar(new_size_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.lambdaInfo)

            div_size_expr = :($div_size_var = $old_size/__hps_num_pes)
            new_size_expr = :($new_size_var = __hps_node_id==__hps_num_pes-1 ? $old_size-__hps_node_id*$div_size_var : $div_size_var)

            node.args[2].args[7] = new_size_var

            return [div_size_expr; new_size_expr; node] 
        end
    end
    return [node]
end

function from_parfor(node::Expr, state)
    @assert node.head==:parfor "DistributedIR invalid parfor head"

    parfor = node.args[1]

    if !in(state.seq_parfors, parfor.unique_id)
        @assert length(parfor.loopNests)==1 "DistIR only 1D PIR loop supported now"

        loopnest = parfor.loopNests[1]
        @assert loopnest.lower==1 && loopnest.step==1 "DistIR only simple PIR loops supported now"

        loop_start_var = symbol("__hps_loop_start_"*string(getDistNewID(state)))
        loop_end_var = symbol("__hps_loop_end_"*string(getDistNewID(state)))
        loop_div_var = symbol("__hps_loop_div_"*string(getDistNewID(state)))

        CompilerTools.LambdaHandling.addLocalVar(loop_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.lambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(loop_end_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.lambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(loop_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.lambdaInfo)

        loop_div_expr = :($loop_div_var = $(loopnest.upper)/__hps_num_pes)
        loop_start_expr = :($loop_start_var = __hps_node_id*$loop_div_var+1)
        loop_end_expr = :($loop_end_var = __hps_node_id==__hps_num_pes-1 ?$(loopnest.upper):(__hps_node_id+1)*$loop_div_var)

        loopnest.lower = loop_start_var
        loopnest.upper = loop_end_var

        for stmt in parfor.body
            adjust_arrayrefs(stmt, loop_start_var)
        end
        res = [loop_div_expr; loop_start_expr; loop_end_expr; node]

        dist_reductions = gen_dist_reductions(parfor.reductions, state)
        append!(res, dist_reductions)

        return res
    end
    return [node]
end

function getDistNewID(state)
    state.uniqueId+=1
    return state.uniqueId
end

function adjust_arrayrefs(stmt::Expr, loop_start_var::Symbol)
    if stmt.head==:(=) && isCall(stmt.args[2]) && isTopNode(stmt.args[2].args[1])
        topCall = stmt.args[2].args[1]
        #ref_args = stmt.args[2].args[2:end]
        if topCall.name==:unsafe_arrayref || topCall.name==:unsafe_arrayset
            @assert length(stmt.args[2].args)==3 "DistIR only 1D parfor array access supported for now"
            index_arg = stmt.args[2].args[3]
            stmt.args[2].args[3] = :($(index_arg.name)-$loop_start_var+1)
        end
    end
end

function adjust_arrayrefs(stmt::Any, loop_start_var::Symbol)
end

function isCall(node::Expr)
    return node.head==:call
end

function isCall(node::Any)
    return false
end

function isTopNode(node::TopNode)
    return true
end

function isTopNode(node::Any)
    return false
end

function gen_dist_reductions(reductions::Array{PIRReduction,1}, state)
    res = Any[]
    for reduce in reductions
        reduce_var = symbol("__hps_reduce_"*string(getDistNewID(state)))
        CompilerTools.LambdaHandling.addLocalVar(reduce_var, reduce.reductionVar.typ, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.lambdaInfo)

        reduce_var_init = Expr(:(=), reduce_var, 0)
        reduceCall = Expr(:call,TopNode(:hps_dist_reduce),reduce.reductionVar,reduce.reductionFunc, reduce_var)
        rootCopy = Expr(:(=), reduce.reductionVar, reduce_var)
        append!(res,[reduce_var_init; reduceCall; rootCopy])
    end
    return res
end

end # DistributedIR
