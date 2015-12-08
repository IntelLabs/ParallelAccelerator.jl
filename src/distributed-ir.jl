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

# ENTRY to distributedIR
function from_root(function_name, ast :: Expr)
    @assert ast.head == :lambda "Input to DistributedIR should be :lambda Expr"
    dprintln(1,"Starting main DistributedIR.from_root.  function = ", function_name, " ast = ", ast)

    linfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(ast)
    state::DistIrState = initDistState(linfo)

    dprintln(3,"state before walk: ",state)
    AstWalk(ast, get_arr_dist_info, state)
    dprintln(3,"state after walk: ",state)

    return ast
end

type ArrDistInfo
    isSequential::Bool      # can't be distributed; e.g. it is used in sequential code
    dim_sizes::Array{Int,1}      # sizes of array dimensions
    
    function ArrDistInfo(num_dims::Int)
        new(false, zeros(num_dims))
    end
end

# information about AST gathered and used in DistributedIR
type DistIrState
    # information about all arrays
    arrs_dist_info::Dict{SymGen, ArrDistInfo}
    parfor_info::Dict{Int, Array{SymGen,1}}
    lambdaInfo
    
    function DistIrState(linfo)
        new(Dict{SymGen, Array{ArrDistInfo,1}}(), Dict{Int, Array{SymGen,1}}(), linfo)
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
    
    dprintln(3,"DisIR arr info walk Expr node: ", node)
    if head==:parfor
        parfor = getParforNode(node)
        rws = parfor.rws
        
        readArrs = [k for k in keys(rws.readSet.arrays)]
        writeArrs = [k for k in keys(rws.writeSet.arrays)]
        allArrs = [readArrs;writeArrs]
        # keep mapping from parfors to arrays
        state.parfor_info[parfor.unique_id] = allArrs
        
        # only 1D parfors supported for now
        if !parfor.simply_indexed || length(parfor.loopNests)!=1
            dprintln(2,"DisIR arr info walk parfor sequential: ", node)
            for arr in allArrs
                state.arrs_dist_info[arr].isSequential = true
            end
            return node
        end
        
        indexVariable::SymbolNode = parfor.loopNests[1].indexVariable
        for arr in keys(rws.readSet.arrays)
             index = rws.readSet.arrays[arr]
             if length(index)!=1 || length(index[1])!=1 || index[1][1].name!=indexVariable.name
                dprintln(2,"DisIR arr info walk arr read index sequential: ", index, " ", indexVariable)
                state.arrs_dist_info[arr].isSequential = true
             end
        end
        
        for arr in keys(rws.writeSet.arrays)
             index = rws.writeSet.arrays[arr]
             if length(index)!=1 || length(index[1])!=1 || index[1][1].name!=indexVariable.name
                dprintln(2,"DisIR arr info walk arr write index sequential: ", index, " ", indexVariable)
                state.arrs_dist_info[arr].isSequential = true
             end
        end
        return node
    # arrays written in sequential code are not distributed
    elseif head!=:body && head!=:block && head!=:lambda
        rws = CompilerTools.ReadWriteSet.from_exprs([node], ParallelIR.pir_live_cb, state.lambdaInfo)
        readArrs = [k for k in keys(rws.readSet.arrays)]
        writeArrs = [k for k in keys(rws.writeSet.arrays)]
        allArrs = [readArrs;writeArrs]
        for arr in allArrs
            dprintln(2,"DisIR arr info walk arr in sequential code: ", arr, " ", node)
            state.arrs_dist_info[arr].isSequential = true
        end
        return node
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function get_arr_dist_info(ast::Any, state, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

end # DistributedIR
