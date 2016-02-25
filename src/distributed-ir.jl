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

import Base.show
import ..ParallelAccelerator
using CompilerTools
import CompilerTools.DebugMsg
DebugMsg.init()
using CompilerTools.AstWalker
import CompilerTools.ReadWriteSet
using CompilerTools.LambdaHandling
using CompilerTools.Helper
import ..ParallelIR
import ..ParallelIR.isArrayType
import ..ParallelIR.getParforNode
import ..ParallelIR.isAllocation
import ..ParallelIR.TypedExpr
import ..ParallelIR.get_alloc_shape

import ..ParallelIR.ISCAPTURED
import ..ParallelIR.ISASSIGNED
import ..ParallelIR.ISASSIGNEDBYINNERFUNCTION
import ..ParallelIR.ISCONST
import ..ParallelIR.ISASSIGNEDONCE
import ..ParallelIR.ISPRIVATEPARFORLOOP
import ..ParallelIR.PIRReduction

dist_ir_funcs = Set([:__hps_data_source_HDF5_open,:__hps_data_source_HDF5_read,:__hps_kmeans,
                        :__hps_data_source_TXT_open,:__hps_data_source_TXT_read, :__hps_LinearRegression, :__hps_NaiveBayes, 
                        GlobalRef(Base,:arraylen), TopNode(:arraysize), GlobalRef(Base,:reshape), TopNode(:tuple), 
                        GlobalRef(Base.LinAlg,:gemm_wrapper!)])

# ENTRY to distributedIR
function from_root(function_name, ast :: Expr)
    @assert ast.head == :lambda "Input to DistributedIR should be :lambda Expr"
    @dprintln(1,"Starting main DistributedIR.from_root.  function = ", function_name, " ast = ", ast)

    linfo = CompilerTools.LambdaHandling.lambdaExprToLambdaVarInfo(ast)
    lives = CompilerTools.LivenessAnalysis.from_expr(ast, ParallelIR.pir_live_cb, linfo)
    state::DistIrState = initDistState(linfo,lives)
    
    # find if an array should be partitioned, sequential, or shared
    getArrayDistributionInfo(ast, state)
    
    # transform body
    @assert ast.args[3].head==:body "DistributedIR: invalid lambda input"
    body = TypedExpr(ast.args[3].typ, :body, from_toplevel_body(ast.args[3].args, state)...)
    new_ast = CompilerTools.LambdaHandling.LambdaVarInfoToLambdaExpr(state.LambdaVarInfo, body)
    @dprintln(1,"DistributedIR.from_root returns function = ", function_name, " ast = ", new_ast)
    # ast = from_expr(ast)
    return new_ast
end

function getArrayDistributionInfo(ast, state)
    before_dist_arrays = [arr for arr in keys(state.arrs_dist_info)]
    
    while true
        dist_arrays = []
        @dprintln(3,"DistIR state before array info walk: ",state)
        AstWalk(ast, get_arr_dist_info, state)
        @dprintln(3,"DistIR state after array info walk: ",state)
            # all arrays not marked sequential are distributable at this point 
        for arr in keys(state.arrs_dist_info)
            if state.arrs_dist_info[arr].isSequential==false
                @dprintln(2,"DistIR distributable parfor array: ", arr)
                push!(dist_arrays,arr)
            end
        end
        # break if no new sequential array discovered
        if length(dist_arrays)==length(before_dist_arrays)
            break
        end
        before_dist_arrays = dist_arrays
    end
    state.dist_arrays = before_dist_arrays
    @dprintln(3,"DistIR state dist_arrays after array info walk: ",state.dist_arrays)
end

type ArrDistInfo
    isSequential::Bool      # can't be distributed; e.g. it is used in sequential code
    dim_sizes::Array{Union{SymAllGen,Int,Expr},1}      # sizes of array dimensions
    # assuming only last dimension is partitioned
    arr_id::Int # assign ID to distributed array to access partitioning info later
    
    function ArrDistInfo(num_dims::Int)
        new(false, zeros(Int64,num_dims))
    end
end

function show(io::IO, pnode::ParallelAccelerator.DistributedIR.ArrDistInfo)
    print(io,"seq:",pnode.isSequential," sizes:", pnode.dim_sizes)
end

# information about AST gathered and used in DistributedIR
type DistIrState
    # information about all arrays
    arrs_dist_info::Dict{SymGen, ArrDistInfo}
    parfor_info::Dict{Int, Array{SymGen,1}}
    LambdaVarInfo::LambdaVarInfo
    seq_parfors::Array{Int,1}
    dist_arrays::Array{SymGen,1}
    uniqueId::Int
    lives  :: CompilerTools.LivenessAnalysis.BlockLiveness
    # keep values for constant tuples. They are often used for allocating and reshaping arrays.
    tuple_table              :: Dict{SymGen,Array{Union{SymGen,Int},1}}
    max_label :: Int # holds the max number of all LabelNodes

    function DistIrState(linfo, lives)
        new(Dict{SymGen, Array{ArrDistInfo,1}}(), Dict{Int, Array{SymGen,1}}(), linfo, Int[], SymGen[],0, lives, 
             Dict{SymGen,Array{Union{SymGen,Int},1}}(),0)
    end
end

function show(io::IO, pnode::ParallelAccelerator.DistributedIR.DistIrState)
    println(io,"DistIrState arrs_dist_info:")
    for i in pnode.arrs_dist_info
        println(io,"  ", i)
    end
    println(io,"DistIrState parfor_info:")
    for i in pnode.parfor_info
        println(io,"  ", i)
    end
    println(io,"DistIrState seq_parfors:")
    for i in pnode.seq_parfors
        print(io," ", i)
    end
    println(io,"")
    println(io,"DistIrState dist_arrays:")
    for i in pnode.dist_arrays
        print(io," ", i)
    end
    println(io,"")
end

function initDistState(linfo::LambdaVarInfo, lives)
    state = DistIrState(linfo, lives)
    
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

"""
mark sequential arrays
"""
function get_arr_dist_info(node::Expr, state::DistIrState, top_level_number, is_top_level, read)
    head = node.head
    # arrays written in parfors are ok for now
    
    @dprintln(3,"DistIR arr info walk Expr node: ", node)
    if head==:(=)
        lhs = toSymGen(node.args[1])
        rhs = node.args[2]
        if isAllocation(rhs)
            state.arrs_dist_info[lhs].dim_sizes = get_alloc_shape(rhs.args[2:end])
            @dprintln(3,"DistIR arr info dim_sizes update: ", state.arrs_dist_info[lhs].dim_sizes)
        elseif isa(rhs,SymAllGen)
            rhs = toSymGen(rhs)
            if haskey(state.arrs_dist_info, rhs)
                state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[rhs].dim_sizes
                # lhs and rhs are sequential if either is sequential
                seq = state.arrs_dist_info[lhs].isSequential || state.arrs_dist_info[rhs].isSequential
                state.arrs_dist_info[lhs].isSequential = state.arrs_dist_info[rhs].isSequential = seq
                @dprintln(3,"DistIR arr info dim_sizes update: ", state.arrs_dist_info[lhs].dim_sizes)
            end
        elseif isa(rhs,Expr) && rhs.head==:call && in(rhs.args[1], dist_ir_funcs)
            func = rhs.args[1]
            if func==GlobalRef(Base,:reshape)
                # only reshape() with constant tuples handled
                if haskey(state.tuple_table, rhs.args[3])
                    state.arrs_dist_info[lhs].dim_sizes = state.tuple_table[rhs.args[3]]
                    @dprintln(3,"DistIR arr info dim_sizes update: ", state.arrs_dist_info[lhs].dim_sizes)
                    # lhs and rhs are sequential if either is sequential
                    seq = state.arrs_dist_info[lhs].isSequential || state.arrs_dist_info[rhs.args[2]].isSequential
                    state.arrs_dist_info[lhs].isSequential = state.arrs_dist_info[rhs.args[2]].isSequential = seq
                else
                    @dprintln(3,"DistIR arr info reshape tuple not found: ", rhs.args[3])
                    state.arrs_dist_info[lhs].isSequential = state.arrs_dist_info[rhs.args[2]].isSequential = true
                end
            elseif rhs.args[1]==TopNode(:tuple)
                ok = true
                for s in rhs.args[2:end]
                    if !(isa(s,SymbolNode) || isa(s,Int))
                        ok = false
                    end 
                end 
                if ok
                    state.tuple_table[lhs]=rhs.args[2:end]
                    @dprintln(3,"DistIR arr info tuple constant: ", lhs," ",rhs.args[2:end])
                else
                    @dprintln(3,"DistIR arr info tuple not constant: ", lhs," ",rhs.args[2:end])
                end 
            elseif func==GlobalRef(Base.LinAlg,:gemm_wrapper!)
                # determine output dimensions
                state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[rhs.args[2]].dim_sizes
                arr1 = rhs.args[5]
                t1 = (rhs.args[3]=='T')
                arr2 = rhs.args[6]
                t2 = (rhs.args[4]=='T')
                
                seq = false
                
                # result is sequential if both inputs are sequential 
                if state.arrs_dist_info[arr1].isSequential && state.arrs_dist_info[arr2].isSequential
                    seq = true
                # result is sequential but with reduction if both inputs are partitioned and second one is transposed
                # e.g. labels*points'
                elseif !state.arrs_dist_info[arr1].isSequential && !state.arrs_dist_info[arr2].isSequential && t2 && !t1
                    seq = true
                # first input is sequential but output is parallel if the second input is partitioned but not transposed
                # e.g. w*points
                elseif !state.arrs_dist_info[arr2].isSequential && !t2
                    @dprintln(3,"DistIR arr info gemm first input is sequential: ", arr1)
                    state.arrs_dist_info[arr1].isSequential = true
                # otherwise, no known pattern found, every array is sequential
                else
                    @dprintln(3,"DistIR arr info gemm all sequential: ", arr1," ", arr2)
                    state.arrs_dist_info[arr1].isSequential = true
                    state.arrs_dist_info[arr2].isSequential = true
                    seq = true
                end
                
                if seq
                    @dprintln(3,"DistIR arr info gemm output is sequential: ", lhs," ",rhs.args[2])
                end
                state.arrs_dist_info[lhs].isSequential = state.arrs_dist_info[rhs.args[2]].isSequential = seq
            end
        else
            return CompilerTools.AstWalker.ASTWALK_RECURSE
        end
        return node
    elseif head==:parfor
        parfor = getParforNode(node)
        rws = parfor.rws
        
        readArrs = collect(keys(rws.readSet.arrays))
        writeArrs = collect(keys(rws.writeSet.arrays))
        allArrs = [readArrs;writeArrs]
        # keep mapping from parfors to arrays
        state.parfor_info[parfor.unique_id] = allArrs
        seq = false
        
        if length(parfor.arrays_read_past_index)!=0 || length(parfor.arrays_written_past_index)!=0 
            @dprintln(2,"DistIR arr info walk parfor sequential: ", node)
            for arr in allArrs
                seq = true
            end
        end
        
        indexVariable::SymbolNode = parfor.loopNests[1].indexVariable
        for arr in keys(rws.readSet.arrays)
             index = rws.readSet.arrays[arr]
             if length(index)!=1 || toSymGen(index[1][end])!=toSymGen(indexVariable)
                @dprintln(2,"DistIR arr info walk arr read index sequential: ", index, " ", indexVariable)
                seq = true
             end
        end
        
        for arr in keys(rws.writeSet.arrays)
             index = rws.writeSet.arrays[arr]
             if length(index)!=1 || toSymGen(index[1][end])!=toSymGen(indexVariable)
                @dprintln(2,"DistIR arr info walk arr write index sequential: ", index, " ", indexVariable)
                seq = true
             end
        end
        for arr in allArrs
            if state.arrs_dist_info[arr].isSequential ||
                        !isEqualDimSize(state.arrs_dist_info[arr].dim_sizes, state.arrs_dist_info[allArrs[1]].dim_sizes)
                    @dprintln(2,"DistIR parfor check array: ", arr," seq: ", state.arrs_dist_info[arr].isSequential)
                    seq = true
            end
        end
        # parfor and all its arrays are sequential
        if seq
            push!(state.seq_parfors, parfor.unique_id)
            for arr in allArrs
                state.arrs_dist_info[arr].isSequential = true
            end
        end
        return node
    # functions dist_ir_funcs are either handled here or do not make arrays sequential  
    elseif head==:call && in(node.args[1], dist_ir_funcs)
        func = node.args[1]
        if func==:__hps_data_source_HDF5_read || func==:__hps_data_source_TXT_read
            @dprintln(2,"DistIR arr info walk data source read ", node)
            # will be parallel IO, intentionally do nothing
        elseif func==:__hps_kmeans
            @dprintln(2,"DistIR arr info walk kmeans ", node)
            # first array is cluster output and is sequential
            # second array is input matrix and is parallel
            state.arrs_dist_info[node.args[2]].isSequential = true
        elseif func==:__hps_LinearRegression || func==:__hps_NaiveBayes
            @dprintln(2,"DistIR arr info walk LinearRegression/NaiveBayes ", node)
            # first array is cluster output and is sequential
            # second array is input matrix and is parallel
            # third array is responses and is parallel
            state.arrs_dist_info[node.args[2]].isSequential = true
        end
        return node
    # arrays written in sequential code are not distributed
    elseif head!=:body && head!=:block && head!=:lambda
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, state.lives)
        
        all_vars = union(live_info.def, live_info.use)
        
        # ReadWriteSet is not robust enough now
        #rws = CompilerTools.ReadWriteSet.from_exprs([node], ParallelIR.pir_live_cb, state.LambdaVarInfo)
        #readArrs = collect(keys(rws.readSet.arrays))
        #writeArrs = collect(keys(rws.writeSet.arrays))
        #allArrs = [readArrs;writeArrs]
        
        for var in all_vars
            if haskey(state.arrs_dist_info, var)
                @dprintln(2,"DistIR arr info walk array in sequential code: ", var, " ", node)
                
                state.arrs_dist_info[var].isSequential = true
            end
        end
        return node
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end


function get_arr_dist_info(ast::Any, state::DistIrState, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function isEqualDimSize(sizes1::Array{Union{SymAllGen,Int,Expr},1} , sizes2::Array{Union{SymAllGen,Int,Expr},1})
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

function eqSize(a::Expr, b::Expr)
    if a.head!=b.head || length(a.args)!=length(b.args)
        return false
    end
    for i in 1:length(a.args)
        if !eqSize(a.args[i],b.args[i])
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
function from_toplevel_body(nodes::Array{Any,1}, state::DistIrState)
    state.max_label = ParallelIR.getMaxLabel(state.max_label, nodes)
    res::Array{Any,1} = genDistributedInit(state)
    for node in nodes
        new_exprs = from_expr(node, state)
        append!(res, new_exprs)
    end
    return res
end


function from_expr(node::Expr, state::DistIrState)
    head = node.head
    if head==:(=)
        return from_assignment(node, state)
    elseif head==:parfor
        return from_parfor(node, state)
    #elseif head==:block
    elseif head==:call
        return from_call(node, state)
    else
        return [node]
    end
end


function from_expr(node::Any, state::DistIrState)
    return [node]
end

# generates initialization code for distributed execution
function genDistributedInit(state::DistIrState)
    initCall = Expr(:call,TopNode(:hps_dist_init))
    numPesCall = Expr(:call,TopNode(:hps_dist_num_pes))
    nodeIdCall = Expr(:call,TopNode(:hps_dist_node_id))
    
    CompilerTools.LambdaHandling.addLocalVar(symbol("__hps_num_pes"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
    CompilerTools.LambdaHandling.addLocalVar(symbol("__hps_node_id"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

    num_pes_assign = Expr(:(=), :__hps_num_pes, numPesCall)
    node_id_assign = Expr(:(=), :__hps_node_id, nodeIdCall)

    return Any[initCall; num_pes_assign; node_id_assign]
end

function from_assignment(node::Expr, state::DistIrState)
    @assert node.head==:(=) "DistributedIR invalid assignment head"
    lhs = node.args[1]
    rhs = node.args[2]
    
    if isAllocation(rhs)
        arr = toSymGen(lhs)
        if in(arr, state.dist_arrays)
            @dprintln(3,"DistIR allocation array: ", arr)
            #shape = get_alloc_shape(node.args[2].args[2:end])
            #old_size = shape[end]
            dim_sizes = state.arrs_dist_info[arr].dim_sizes
            # generate array division
            # simple 1D partitioning of last dimension, more general partitioning needed
            # match common big data matrix reperesentation
            arr_tot_size = dim_sizes[end]

            arr_id = getDistNewID(state)
            state.arrs_dist_info[arr].arr_id = arr_id
            darr_start_var = symbol("__hps_dist_arr_start_"*string(arr_id))
            darr_div_var = symbol("__hps_dist_arr_div_"*string(arr_id))
            darr_count_var = symbol("__hps_dist_arr_count_"*string(arr_id))

            CompilerTools.LambdaHandling.addLocalVar(darr_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            CompilerTools.LambdaHandling.addLocalVar(darr_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            CompilerTools.LambdaHandling.addLocalVar(darr_count_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)


            darr_div_expr = :($darr_div_var = $(arr_tot_size)/__hps_num_pes)
            # zero-based index to match C interface of HDF5
            darr_start_expr = :($darr_start_var = __hps_node_id*$darr_div_var) 
            darr_count_expr = :($darr_count_var = __hps_node_id==__hps_num_pes-1 ? $arr_tot_size-__hps_node_id*$darr_div_var : $darr_div_var)

            rhs.args[end-1] = darr_count_var

            res = [darr_div_expr; darr_start_expr; darr_count_expr; node]
            #debug_size_print = :(println("size ",$darr_count_var))
            #push!(res,debug_size_print)
            return res
        end
    elseif isa(rhs,Expr) && rhs.head==:call && rhs.args[1]==GlobalRef(Base,:reshape)
        arr = toSymGen(lhs)
        if in(arr, state.dist_arrays)
            @dprintln(3,"DistIR reshape array: ", arr)
            dim_sizes = state.arrs_dist_info[arr].dim_sizes
            # generate array division
            # simple 1D partitioning of last dimension, more general partitioning needed
            # match common big data matrix reperesentation
            arr_tot_size = dim_sizes[end]
    
            arr_id = getDistNewID(state)
            state.arrs_dist_info[arr].arr_id = arr_id
            darr_start_var = symbol("__hps_dist_arr_start_"*string(arr_id))
            darr_div_var = symbol("__hps_dist_arr_div_"*string(arr_id))
            darr_count_var = symbol("__hps_dist_arr_count_"*string(arr_id))
    
            CompilerTools.LambdaHandling.addLocalVar(darr_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            CompilerTools.LambdaHandling.addLocalVar(darr_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            CompilerTools.LambdaHandling.addLocalVar(darr_count_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
    
    
            darr_div_expr = :($darr_div_var = $(arr_tot_size)/__hps_num_pes)
            # zero-based index to match C interface of HDF5
            darr_start_expr = :($darr_start_var = __hps_node_id*$darr_div_var) 
            darr_count_expr = :($darr_count_var = __hps_node_id==__hps_num_pes-1 ? $arr_tot_size-__hps_node_id*$darr_div_var : $darr_div_var)
    
            # create a new tuple for reshape
            tup_call = Expr(:call, TopNode(:tuple), dim_sizes[1:end-1]... , darr_count_var)
            reshape_tup_var = symbol("__hps_dist_tup_var_"*string(arr_id))
            tup_typ = CompilerTools.LambdaHandling.getType(rhs.args[3], state.LambdaVarInfo)
            CompilerTools.LambdaHandling.addLocalVar(reshape_tup_var, tup_typ, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            tup_expr = Expr(:(=),reshape_tup_var,tup_call)
            rhs.args[3] = reshape_tup_var
            res = [darr_div_expr; darr_start_expr; darr_count_expr; tup_expr; node]
            #debug_size_print = :(println("size ",$darr_count_var))
            #push!(res,debug_size_print)
            return res
        end
    elseif isa(rhs,Expr) && rhs.head==:call && rhs.args[1]==GlobalRef(Base.LinAlg,:gemm_wrapper!)

                arr1 = rhs.args[5]
                t1 = (rhs.args[3]=='T')
                arr2 = rhs.args[6]
                t2 = (rhs.args[4]=='T')
                
                # result is sequential but with reduction if both inputs are partitioned and second one is transposed
                # e.g. labels*points'
                if !state.arrs_dist_info[arr1].isSequential && !state.arrs_dist_info[arr2].isSequential && t2 && !t1 &&
                            state.arrs_dist_info[lhs].isSequential
                    @dprintln(3,"DistIR translating gemm reduce: ", node)
                    # rhs.args[1] = :__hps_gemm_reduce
                    # allocate temporary array for local gemm values
                    alloc_args = Array(Any,2)
                    out_typ = CompilerTools.LambdaHandling.getType(lhs, state.LambdaVarInfo)
                    alloc_args[1] = eltype(out_typ)
                    out_dim_sizes = state.arrs_dist_info[lhs].dim_sizes
                    alloc_args[2] = out_dim_sizes
                    alloc_call = ParallelIR.from_alloc(alloc_args)
                    reduce_num = getDistNewID(state)
                    reduce_var = symbol("__hps_gemm_reduce_"*string(reduce_num))
                    CompilerTools.LambdaHandling.addLocalVar(reduce_var, out_typ, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
                    reduce_var_init = Expr(:(=), reduce_var, Expr(:call,alloc_call...))

                    # get reduction size
                    reduce_size_var = symbol("__hps_gemm_reduce_size_"*string(reduce_num))
                    CompilerTools.LambdaHandling.addLocalVar(reduce_size_var, Int, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
                    size_expr = Expr(:(=), reduce_size_var, Expr(:call,:*,out_dim_sizes[1], out_dim_sizes[2]))

                    # add allreduce call
                    allreduceCall = Expr(:call,TopNode(:hps_dist_allreduce), reduce_var, TopNode(:add_float), rhs.args[2], reduce_size_var)
                    res_copy = Expr(:(=), lhs, rhs.args[2])
                    # replace gemm output with local var
                    #node.args[1] = reduce_var
                    rhs.args[2] = reduce_var

                    return [reduce_var_init; node; size_expr; allreduceCall; res_copy]
                # first input is sequential but output is parallel if the second input is partitioned but not transposed
                # e.g. w*points
                elseif state.arrs_dist_info[arr1].isSequential && !state.arrs_dist_info[arr2].isSequential && !t2 && !state.arrs_dist_info[lhs].isSequential
                    @dprintln(3,"DistIR arr info gemm first input is sequential: ", arr1)
                    #rhs.args[1] = :__hps_gemm_broadcast
                    @dprintln(3,"DistIR translating gemm broadcast: ", node)
                # otherwise, no known pattern found
                end
    else
        node.args[2] = from_expr(rhs,state)[1]
    end
    return [node]
end

function from_parfor(node::Expr, state)
    @assert node.head==:parfor "DistributedIR invalid parfor head"

    parfor = node.args[1]

    if !in(parfor.unique_id, state.seq_parfors)
        @dprintln(3,"DistIR translating parfor: ", parfor.unique_id)
        # TODO: assuming 1st loop nest is the last dimension
        loopnest = parfor.loopNests[1]
        # TODO: build a constant table and check the loop variables at this stage
        # @assert loopnest.lower==1 && loopnest.step==1 "DistIR only simple PIR loops supported now"

        loop_start_var = symbol("__hps_loop_start_"*string(getDistNewID(state)))
        loop_end_var = symbol("__hps_loop_end_"*string(getDistNewID(state)))
        loop_div_var = symbol("__hps_loop_div_"*string(getDistNewID(state)))

        CompilerTools.LambdaHandling.addLocalVar(loop_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
        CompilerTools.LambdaHandling.addLocalVar(loop_end_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
        CompilerTools.LambdaHandling.addLocalVar(loop_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

        #first_arr = state.parfor_info[parfor.unique_id][1]; 
        #@dprintln(3,"DistIR parfor first array ", first_arr)
        #global_size = state.arrs_dist_info[first_arr].dim_sizes[1]

        # some parfors have no arrays
        global_size = loopnest.upper

        loop_div_expr = :($loop_div_var = $(global_size)/__hps_num_pes)
        loop_start_expr = :($loop_start_var = __hps_node_id*$loop_div_var+1)
        loop_end_expr = :($loop_end_var = __hps_node_id==__hps_num_pes-1 ?$(global_size):(__hps_node_id+1)*$loop_div_var)

        loopnest.lower = loop_start_var
        loopnest.upper = loop_end_var

        for stmt in parfor.body
            adjust_arrayrefs(stmt, loop_start_var)
        end
        res = [loop_div_expr; loop_start_expr; loop_end_expr; node]

        dist_reductions = gen_dist_reductions(parfor.reductions, state)
        append!(res, dist_reductions)

        #debug_start_print = :(println("parfor start", $loop_start_var))
        #debug_end_print = :(println("parfor end", $loop_end_var))
        #push!(res,debug_start_print)
        #push!(res,debug_end_print)

        #debug_div_print = :(println("parfor div ", $loop_div_var))
        #push!(res,debug_div_print)
        #debug_pes_print = :(println("parfor pes ", __hps_num_pes))
        #push!(res,debug_pes_print)
        #debug_rank_print = :(println("parfor rank ", __hps_node_id))
        #push!(res,debug_rank_print)
        return res
    else
        # broadcast results of sequential parfors if rand() is used
        has_rand = false
        for stmt in parfor.body
            if isa(stmt,Expr) && stmt.head==:(=) && isa(stmt.args[2],Expr) && stmt.args[2].head==:call && stmt.args[2].args[1]==TopNode(:rand!)
                has_rand = true
                break
            end
        end
        if has_rand
            # only rank 0 executes rand(), then broadcasts results
            writeArrs = collect(keys(parfor.rws.writeSet.arrays))
            @assert length(writeArrs)==1 "Only one parfor output supported now"
            write_arr = writeArrs[1]
            # generate new label
            label = next_label(state)
            label_node = LabelNode(label)
            goto_node = Expr(:gotoifnot, :__hps_node_id,label)
            # get broadcast size
            bcast_size_var = symbol("__hps_bcast_size_"*string(label))
            CompilerTools.LambdaHandling.addLocalVar(bcast_size_var, Int, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            size_expr = Expr(:(=), bcast_size_var, Expr(:call,:*,1,state.arrs_dist_info[write_arr].dim_sizes...))
            bcast_expr = Expr(:call,:__hps_dist_broadcast, write_arr, bcast_size_var)

            @dprintln(3,"DistIR rand() in sequential parfor ", parfor)
            return [goto_node; node; label_node; size_expr; bcast_expr]
        end
    end
    return [node]
end

function next_label(state)
    state.max_label = state.max_label + 1
    return state.max_label
end

function from_call(node::Expr, state)
    @assert node.head==:call "Invalid call node"
    @dprintln(2,"DistIR from_call ", node)

    func = node.args[1]
    if (func==:__hps_data_source_HDF5_read || func==:__hps_data_source_TXT_read) && in(toSymGen(node.args[3]), state.dist_arrays)
        arr = toSymGen(node.args[3])
        @dprintln(3,"DistIR data source for array: ", arr)
        
        arr_id = state.arrs_dist_info[arr].arr_id 
        
        dsrc_start_var = symbol("__hps_dist_arr_start_"*string(arr_id)) 
        dsrc_count_var = symbol("__hps_dist_arr_count_"*string(arr_id)) 

        push!(node.args, dsrc_start_var, dsrc_count_var)
        return [node]
    elseif func==:__hps_kmeans && in(toSymGen(node.args[3]), state.dist_arrays)
        arr = toSymGen(node.args[3])
        @dprintln(3,"DistIR kmeans call for array: ", arr)
        
        arr_id = state.arrs_dist_info[arr].arr_id 
        
        dsrc_start_var = symbol("__hps_dist_arr_start_"*string(arr_id))
        dsrc_count_var = symbol("__hps_dist_arr_count_"*string(arr_id)) 

        push!(node.args, dsrc_start_var, dsrc_count_var, 
                state.arrs_dist_info[arr].dim_sizes[1], state.arrs_dist_info[arr].dim_sizes[end])
        return [node]
    elseif (func==:__hps_LinearRegression || func==:__hps_NaiveBayes) && in(toSymGen(node.args[3]), state.dist_arrays) && in(toSymGen(node.args[4]), state.dist_arrays)
        arr1 = toSymGen(node.args[3])
        arr2 = toSymGen(node.args[4])
        @dprintln(3,"DistIR LinearRegression/NaiveBayes call for arrays: ", arr1," ", arr2)
        
        arr1_id = state.arrs_dist_info[arr1].arr_id 
        arr2_id = state.arrs_dist_info[arr2].arr_id 
        
        dsrc_start_var1 = symbol("__hps_dist_arr_start_"*string(arr1_id))
        dsrc_count_var1 = symbol("__hps_dist_arr_count_"*string(arr1_id)) 
        
        dsrc_start_var2 = symbol("__hps_dist_arr_start_"*string(arr2_id))
        dsrc_count_var2 = symbol("__hps_dist_arr_count_"*string(arr2_id)) 

        push!(node.args, dsrc_start_var1, dsrc_count_var1,
                state.arrs_dist_info[arr1].dim_sizes[1], state.arrs_dist_info[arr1].dim_sizes[end])
        push!(node.args, dsrc_start_var2, dsrc_count_var2,
                state.arrs_dist_info[arr2].dim_sizes[1], state.arrs_dist_info[arr2].dim_sizes[end])
        return [node]
    elseif isTopNode(func) && func.name==:arraysize && in(toSymGen(node.args[2]), state.dist_arrays)
        arr = toSymGen(node.args[2])
        @dprintln(3,"found arraysize on dist array: ",node," ",arr)
        # replace last dimension size queries since it is partitioned
        if node.args[3]==length(state.arrs_dist_info[arr].dim_sizes)
            return [state.arrs_dist_info[arr].dim_sizes[end]]
        end
    elseif func==GlobalRef(Base,:arraylen) && in(toSymGen(node.args[2]), state.dist_arrays)
        arr = toSymGen(node.args[2])
        #len = parse(foldl((a,b)->"$a*$b", "1",state.arrs_dist_info[arr].dim_sizes))
        len = Expr(:call,:*, 1,state.arrs_dist_info[arr].dim_sizes...)
        @dprintln(3,"found arraylen on dist array: ",node," ",arr," len: ",len)
        @dprintln(3,"found arraylen on dist array: ",node," ",arr)
        return [len]
    end
    return [node]
end

function getDistNewID(state)
    state.uniqueId+=1
    return state.uniqueId
end

function adjust_arrayrefs(stmt::Expr, loop_start_var::Symbol)
    
    if stmt.head==:(=)
        stmt = stmt.args[2]
    end
    
    if isCall(stmt) && isTopNode(stmt.args[1])
        topCall = stmt.args[1]
        #ref_args = stmt.args[2:end]
        if topCall.name==:unsafe_arrayref || topCall.name==:unsafe_arrayset
            # TODO: simply divide the last dimension, more general partitioning needed
            index_arg = stmt.args[end]
            stmt.args[end] = :($(toSymGen(index_arg))-$loop_start_var+1)
        end
    end
end

function adjust_arrayrefs(stmt::Any, loop_start_var::Symbol)
end


function gen_dist_reductions(reductions::Array{PIRReduction,1}, state)
    res = Any[]
    for reduce in reductions
        reduce_var = symbol("__hps_reduce_"*string(getDistNewID(state)))
        CompilerTools.LambdaHandling.addLocalVar(reduce_var, reduce.reductionVar.typ, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

        reduce_var_init = Expr(:(=), reduce_var, 0)
        reduceCall = Expr(:call,TopNode(:hps_dist_reduce),reduce.reductionVar,reduce.reductionFunc, reduce_var)
        rootCopy = Expr(:(=), reduce.reductionVar, reduce_var)
        append!(res,[reduce_var_init; reduceCall; rootCopy])
    end
    return res
end

end # DistributedIR
