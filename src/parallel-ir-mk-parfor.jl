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

@doc """
Make sure the index parameters to arrayref or arrayset are Int64 or SymbolNode.
"""
function augment_sn(dim :: Int64, index_vars, range :: Array{DimensionSelector,1})
    xtyp = typeof(index_vars[dim])
    dprintln(3,"augment_sn dim = ", dim, " index_vars = ", index_vars, " range = ", range, " xtyp = ", xtyp)

    if xtyp == Int64 || xtyp == SymbolNode || xtyp == Expr
        base = index_vars[dim]
    elseif xtyp == Symbol
        base = SymbolNode(index_vars[dim],Int64)
    end

    dprintln(3,"pre-base = ", base)

    if dim <= length(range) && 
       isa(range[dim], RangeData) &&
       !isStartOneRange(range[dim].exprs)
        base = DomainIR.add(base, range[dim].offset_temp_var)
    end

    dprintln(3,"post-base = ", base)

    return base
end

@doc """
Return an expression that corresponds to getting the index_var index from the array array_name.
If "inbounds" is true then use the faster :unsafe_arrayref call that doesn't do a bounds check.
"""
function mk_arrayref1(num_dim_inputs, 
                      array_name, 
                      index_vars, 
                      inbounds, 
                      state     :: expr_state, 
                      range     :: Array{DimensionSelector,1} = DimensionSelector[])
    dprintln(3,"mk_arrayref1 typeof(index_vars) = ", typeof(index_vars))
    dprintln(3,"mk_arrayref1 array_name = ", array_name, " typeof(array_name) = ", typeof(array_name))
    elem_typ = getArrayElemType(array_name, state)
    dprintln(3,"mk_arrayref1 element type = ", elem_typ)
    dprintln(3,"mk_arrayref1 range = ", range)

    if inbounds
        fname = :unsafe_arrayref
    else
        fname = :arrayref
    end

    indsyms = [ x <= num_dim_inputs ? 
                   augment_sn(x, index_vars, range) : 
                   index_vars[x] 
                for x = 1:length(index_vars) ]
    dprintln(3,"mk_arrayref1 indsyms = ", indsyms)

    TypedExpr(
        elem_typ,
        :call,
        TopNode(fname),
        :($array_name),
        indsyms...)
end

@doc """
Return a new AST node that corresponds to setting the index_var index from the array "array_name" with "value".
The paramater "inbounds" is true if this access is known to be within the bounds of the array.
"""
function mk_arrayset1(num_dim_inputs, 
                      array_name, 
                      index_vars, 
                      value, 
                      inbounds, 
                      state :: expr_state, 
                      range :: Array{DimensionSelector,1} = DimensionSelector[])
    dprintln(3,"mk_arrayset1 typeof(index_vars) = ", typeof(index_vars))
    dprintln(3,"mk_arrayset1 array_name = ", array_name, " typeof(array_name) = ", typeof(array_name))
    elem_typ = getArrayElemType(array_name, state)  # The type of the array reference will be the element type.
    dprintln(3,"mk_arrayset1 element type = ", elem_typ)
    dprintln(3,"mk_arrayset1 range = ", range)

    # If the access is known to be within the bounds of the array then use unsafe_arrayset to forego the boundscheck.
    if inbounds
        fname = :unsafe_arrayset
    else
        fname = :arrayset
    end

    # For each index expression in "index_vars", if it isn't an Integer literal then convert the symbol to
    # a SymbolNode containing the index expression type "Int".
    indsyms = [ x <= num_dim_inputs ? 
                   augment_sn(x, index_vars, range) : 
                   index_vars[x] 
                for x = 1:length(index_vars) ]
    dprintln(3,"mk_arrayset1 indsyms = ", indsyms)

    TypedExpr(
       elem_typ,
       :call,
       TopNode(fname),
       array_name,
       :($value),
       indsyms...)
end

@doc """
The main routine that converts a reduce AST node to a parfor AST node.
"""
function mk_parfor_args_from_reduce(input_args::Array{Any,1}, state)
    # Make sure we get what we expect from domain IR.
    # There should be three entries in the array, how to initialize the reduction variable, the arrays to work on and a DomainLambda.
    assert(length(input_args) == 3)

    zero_val    = input_args[1]   # The initial value of the reduction variable.
    input_array = input_args[2]   # The array expression to reduce.

    # Handle range selector
    inputInfo   = get_mmap_input_info(input_array, state)
    dprintln(3,"inputInfo = ", inputInfo)

    dl = input_args[3]            # Get the DomainLambda from the AST node's args.
    assert(isa(dl, DomainLambda))

    dprintln(3,"mk_parfor_args_from_reduce. zero_val = ", zero_val, " type = ", typeof(zero_val))
    dprintln(3,"mk_parfor_args_from_reduce. input array = ", input_array)
    dprintln(3,"mk_parfor_args_from_reduce. DomainLambda = ", dl)

    # Verify the number of input arrays matches the number of input types in dl
    assert(length(dl.inputs) == 2)

    # Get a unique number to embed in generated code for new variables to prevent name conflicts.
    unique_node_id = get_unique_num()

    # The depth of the loop nest for the parfor is equal to the dimensions of the input_array.
    num_dim_inputs = findSelectedDimensions([inputInfo], state)
    loopNests = Array(PIRLoopNest, num_dim_inputs)

    # Create variables to use for the loop indices.
    parfor_index_syms::Array{Symbol,1} = gen_parfor_loop_indices(num_dim_inputs, unique_node_id, state)

    # Make sure each input array is a SymbolNode
    # Also, create indexed versions of those symbols for the loop body
    argtyp = typeof(input_array)
    dprintln(3,"mk_parfor_args_from_reduce input_array[1] = ", input_array, " type = ", argtyp)
    assert(argtyp <: SymNodeGen)

    reduce_body = Any[]
    atm = createTempForArray(input_array, 1, state)
    push!(reduce_body, mk_assignment_expr(atm, mk_arrayref1(num_dim_inputs, input_array, parfor_index_syms, true, state), state))

    # Create an expression to access one element of this input array with index symbols parfor_index_syms
    indexed_array = atm

    # Create empty arrays to hold pre and post statements.
    pre_statements  = Any[]
    post_statements = Any[]
    save_array_lens  = AbstractString[]
    input_array_rangeconds = Array(Any, num_dim_inputs)

    # Insert a statement to assign the length of the input arrays to a var
    for i = 1:num_dim_inputs
        save_array_start = string("parallel_ir_save_array_start_", i, "_", unique_node_id)
        save_array_step  = string("parallel_ir_save_array_step_", i, "_", unique_node_id)
        save_array_len   = string("parallel_ir_save_array_len_", i, "_", unique_node_id)
        if isWholeArray(inputInfo)
            array1_start = mk_assignment_expr(SymbolNode(symbol(save_array_start), Int), 1, state)
            array1_step  = mk_assignment_expr(SymbolNode(symbol(save_array_step), Int), 1, state)
            array1_len   = mk_assignment_expr(SymbolNode(symbol(save_array_len), Int), mk_arraylen_expr(inputInfo.array,i), state)
            input_array_rangeconds[i] = nothing
        elseif isRange(inputInfo)
            this_dim = inputInfo.range[i]
            if isa(this_dim, RangeData)
                array1_start = mk_assignment_expr(SymbolNode(symbol(save_array_start), Int), inputInfo.range[i].start, state)
                array1_step  = mk_assignment_expr(SymbolNode(symbol(save_array_step), Int), inputInfo.range[i].skip, state)
                array1_len   = mk_assignment_expr(SymbolNode(symbol(save_array_len), Int), inputInfo.range[i].last, state)
                input_array_rangeconds[i] = nothing
            else
                mask_array = this_dim
                assert(DomainIR.isbitarray(CompilerTools.LambdaHandling.getType(mask_array, state.lambdaInfo)))
                if isa(mask_array, SymbolNode) # a hack to change type to Array{Bool}
                    mask_array = SymbolNode(mask_array.name, Array{Bool, mask_array.typ.parameters[1]})
                end
                # TODO: generate dimension check on mask_array
                array1_start = mk_assignment_expr(SymbolNode(symbol(save_array_start), Int), 1, state)
                array1_step  = mk_assignment_expr(SymbolNode(symbol(save_array_step), Int), 1, state)
                array1_len   = mk_assignment_expr(SymbolNode(symbol(save_array_len), Int), mk_arraylen_expr(inputInfo.array,i), state)
                input_array_rangeconds[i] = TypedExpr(Bool, :call, TopNode(:unsafe_arrayref), mask_array, SymbolNode(parfor_index_syms[i], Int))
            end
        end 
        # add that assignment to the set of statements to execute before the parfor
        push!(pre_statements,array1_start)
        push!(pre_statements,array1_step)
        push!(pre_statements,array1_len)
        CompilerTools.LambdaHandling.addLocalVar(save_array_start, Int, ISASSIGNEDONCE | ISASSIGNED, state.lambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(save_array_step,  Int, ISASSIGNEDONCE | ISASSIGNED, state.lambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(save_array_len,   Int, ISASSIGNEDONCE | ISASSIGNED, state.lambdaInfo)
        push!(save_array_lens, save_array_len)

        loopNests[num_dim_inputs - i + 1] = PIRLoopNest(SymbolNode(parfor_index_syms[i],Int),
                                                        SymbolNode(symbol(save_array_start), Int),
                                                        SymbolNode(symbol(save_array_len),Int),
                                                        SymbolNode(symbol(save_array_step), Int))
    end

    assert(length(dl.outputs) == 1)
    out_type = dl.outputs[1]
    dprintln(3,"mk_parfor_args_from_reduce dl.outputs = ", out_type)
    reduction_output_name  = string("parallel_ir_reduction_output_",unique_node_id)
    reduction_output_snode = SymbolNode(symbol(reduction_output_name), out_type)
    dprintln(3, "Creating variable to hold reduction output = ", reduction_output_snode)
    CompilerTools.LambdaHandling.addLocalVar(reduction_output_name, out_type, ISASSIGNED, state.lambdaInfo)
    push!(post_statements, reduction_output_snode)

    # Call Domain IR to generate most of the body of the function (except for saving the output)
    dl_inputs = [reduction_output_snode, atm]
    (max_label, nested_lambda, temp_body) = nested_function_exprs(state.max_label, dl, dl_inputs)
    gensym_map = mergeLambdaIntoOuterState(state, nested_lambda)
    temp_body = CompilerTools.LambdaHandling.replaceExprWithDict!(temp_body, gensym_map)
    state.max_label = max_label
    assert(isa(temp_body,Array))
    assert(length(temp_body) == 1)
    temp_body = temp_body[1]
    assert(typeof(temp_body) == Expr)
    assert(temp_body.head == :tuple)
    assert(length(temp_body.args) == 1)
    temp_body = temp_body.args[1]

    #dprintln(3,"reduce_body = ", reduce_body, " type = ", typeof(reduce_body))
    out_body = [reduce_body; mk_assignment_expr(reduction_output_snode, temp_body, state)]

    fallthroughLabel = next_label(state)
    condExprs = Any[]
    for i = 1:num_dim_inputs
        if input_array_rangeconds[i] != nothing
            push!(condExprs, Expr(:gotoifnot, input_array_rangeconds[i], fallthroughLabel))
        end
    end
    if length(condExprs) > 0
        out_body = [ condExprs; out_body; LabelNode(fallthroughLabel) ]
    end
    #out_body = TypedExpr(out_type, :call, TopNode(:parallel_ir_reduce), reduction_output_snode, indexed_array)
    dprintln(2,"typeof(out_body) = ",typeof(out_body), " out_body = ", out_body)

    # Compute which scalars and arrays are ever read or written by the body of the parfor
    rws = CompilerTools.ReadWriteSet.from_exprs(out_body, pir_rws_cb, state.lambdaInfo)

    # Make sure that for reduce that the array indices are all of the simple variety
    simply_indexed = simpleIndex(rws.readSet.arrays) && simpleIndex(rws.writeSet.arrays)
    dprintln(2,rws)

    reduce_func = nothing

    dprintln(3,"type of reduce_body = ", typeof(temp_body))
    if typeof(temp_body) == Expr
        dprintln(3,"head of reduce body = ", temp_body.head)

        dprintln(3,"length(reduce_body.args) = ", length(temp_body.args))
        for k = 1:length(temp_body.args)
            dprintln(3,"reduce_body.args[", k, "] = ", temp_body.args[k], " type = ", typeof(temp_body.args[k]))
        end
        if temp_body.head == :call
            dprintln(3,"Found a call")
            if length(temp_body.args) != 3
                throw(string("Non-binary reduction function used."))
            end
            op = temp_body.args[1]

            if op == TopNode(:add_float) || op == TopNode(:add_int) || op == TopNode(:+)
                reduce_func = :+
            elseif op == TopNode(:mul_float) || op == TopNode(:mul_int) || op == TopNode(:*)
                reduce_func = :*
            elseif op == TopNode(:max) || op == TopNode(:min)
                reduce_func = op.name
            elseif op == TopNode(:|) 
                reduce_func = :||
            elseif op == TopNode(:&) 
                reduce_func = :&&
            end
        end
    end

    if reduce_func == nothing
        throw(string("Parallel IR only supports ", DomainIR.reduceVal, " reductions right now."))
    end

    #  makeLhsPrivate(out_body, state)

    # The parfor node that will go into the AST.
    new_parfor = ParallelAccelerator.ParallelIR.PIRParForAst(
        inputInfo,
        out_body,
        pre_statements,
        loopNests,
        [PIRReduction(reduction_output_snode, zero_val, reduce_func)],
        post_statements,
        [DomainOperation(:reduce, input_args)],
        state.top_level_number,
        rws,
        unique_node_id,
        simply_indexed)

    dprintln(3,"Lowered parallel IR = ", new_parfor)

    [new_parfor]
end


# ===============================================================================================================================


@doc """
Create a variable to hold the offset of a range offset from the start of the array.
"""
function createTempForRangeOffset(num_used, ranges :: Array{RangeData,1}, unique_id :: Int64, state :: expr_state)
    range_array = SymbolNode[]

    #for i = 1:length(ranges)
    for i = 1:num_used
        range = ranges[i]

        push!(range_array, createStateVar(state, string("parallel_ir_range_", range.start, "_", range.skip, "_", range.last, "_", i, "_", unique_id), Int64, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP))
    end

    return range_array
end

function getSName(rd :: RangeData)
    return getSName(rd.offset_temp_var)
end
function getSName(rd :: MaskSelector)
    return getSName(rd.value)
end
function getSName(rd :: SingularSelector)
    return rd.value
end

@doc """
Create a temporary variable that is parfor private to hold the value of an element of an array.
"""
function createTempForRangedArray(array_sn :: SymAllGen, range :: Array{DimensionSelector,1}, unique_id :: Int64, state :: expr_state)
    key = toSymGen(array_sn) 
    temp_type = getArrayElemType(array_sn, state)
    # Is it okay to just use range[1] here instead of all the ranges?
    return createStateVar(state, string("parallel_ir_temp_", key, "_", getSName(range[1]), "_", unique_id), temp_type, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP)
end

function createTempForRangeInfo(array_sn :: SymAllGen, unique_id :: Int64, range_num::Int, info::AbstractString, state :: expr_state)
    key = toSymGen(array_sn) 
    temp_type = getArrayElemType(array_sn, state)
    return createStateVar(state, string("parallel_ir_temp_", key, "_", unique_id, "_", range_num, info), temp_type, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP)
end

@doc """
Convert a :range Expr introduced by Domain IR into a Parallel IR data structure RangeData.
"""
function rangeToRangeData(range :: Expr, arr, range_num :: Int, state)
    dprintln(3,"rangeToRangeData for Expr")
    if range.head == :range
        start = createTempForRangeInfo(arr, 1, range_num, "start", state);
        step  = createTempForRangeInfo(arr, 1, range_num, "step", state);
        last  = createTempForRangeInfo(arr, 1, range_num, "last", state);
        range_temp_var = createStateVar(state, string("parallel_ir_range_", start, "_", skip, "_", last, "_", range_num, "_1"), Int64, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP)
        return RangeData(start, step, last, range.args[1], range.args[2], range.args[3], range_temp_var)
    elseif range.head == :tomask
        return MaskSelector(range.args[1])   # Return the BitArray mask
    else 
        throw(string(":range or :tomask expression expected"))
    end
end
function rangeToRangeData(other :: Union{SymAllGen,Number}, arr, range_num :: Int, state)
    dprintln(3,"rangeToRangeData for non-Expr")
    return SingularSelector(other)
end

function rangeToRangeData(sym, arr, range_num :: Int, state)
    return sym
end

function addRangePreoffsets(rd :: RangeData, pre_offsets :: Array{Expr,1}, state)
    push!(pre_offsets, mk_assignment_expr(rd.start, rd.exprs.start_val, state))
    push!(pre_offsets, mk_assignment_expr(rd.skip,  rd.exprs.skip_val,  state))
    push!(pre_offsets, mk_assignment_expr(rd.last,  rd.exprs.last_val,  state))
end
function addRangePreoffsets(rd :: Union{MaskSelector,SingularSelector}, pre_offsets :: Array{Expr,1}, state)
    # Intentionally do nothing.
end

@doc """
Convert the range(s) part of a :select Expr introduced by Domain IR into an array of Parallel IR data structures RangeData.
"""
function selectToRangeData(select :: Expr, pre_offsets :: Array{Expr,1}, state)
    assert(select.head == :select)

    # The select expression is an array and either a mask or range(s) to choose some portion
    # of the given array.  Here we know the selection is based on ranges.
    select_array       = select.args[1] # the array to select from
    input_array_ranges = select.args[2] # range object

    dprintln(3,"selectToRangeData: select_array = ", select_array, " input_array_ranges = ", input_array_ranges)

    range_array = DimensionSelector[]

    # The expression could be :ranges which means there are multiple dimensions or
    # a single dimension with a :range expression.
    if input_array_ranges.head == :ranges
        for i = 1:length(input_array_ranges.args)
            push!(range_array, rangeToRangeData(input_array_ranges.args[i], select.args[1], i, state))
        end
    else
        push!(range_array, rangeToRangeData(input_array_ranges, select_array, 0, state))
    end

    dprintln(3,"range_array = ", range_array)

    # Should be the full dimensions of the array.
    cur = length(range_array)
    # See how many index variables we need to loop over this array.
    # We might need less than the full number because of singular ranges, e.g., col:col.
    # Scan backward from last dimension to the first.
    for rindex = cur:-1:1
        rd = range_array[rindex]
        if isa(rd, SingularSelector)
            dprintln(3, "Found singular ending dimension so eliminated this dimension.")
            cur -= 1
        else
            # Once we find one trailing dimension that isn't eliminated then dimension shrinking has to stop.
            break
        end 
    end    

    # We only need generate extra code for non-singular ranges.
    # For singular ranges, we just use the singular range start as a constant.
    for i = 1:cur
        addRangePreoffsets(range_array[i], pre_offsets, state)
    end

    return (range_array, cur)
end

function get_mmap_input_info(input_array :: Expr, state)
    thisInfo = InputInfo()

    if is(input_array.head, :select)
        thisInfo.array = input_array.args[1]
        thisInfo.dim   = getArrayNumDims(thisInfo.array, state)
        thisInfo.out_dim = thisInfo.dim
        argtyp = typeof(thisInfo.array)
        dprintln(3,"get_mmap_input_info :select thisInfo.array = ", thisInfo.array, " type = ", argtyp, " isa = ", argtyp <: SymAllGen)
        @assert (argtyp <: SymAllGen) "input array argument type should be SymAllGen"
        select_kind = input_array.args[2].head
        @assert (select_kind==:tomask || select_kind==:range || select_kind==:ranges) ":select should have :tomask or :range or :ranges in args[2]"
        dprintln(3,"select_kind = ", select_kind)
        if select_kind == :tomask
            thisInfo.range = [MaskSelector(x) for x in input_array.args[2].args]
            thisInfo.elementTemp = createTempForArray(thisInfo.array, 1, state)
        else
            (thisInfo.range, thisInfo.out_dim) = selectToRangeData(input_array, thisInfo.pre_offsets, state)
            thisInfo.elementTemp = createTempForRangedArray(thisInfo.array, thisInfo.range, 1, state)
            thisInfo.pre_offsets = [thisInfo.pre_offsets; generatePreOffsetStatements(thisInfo.out_dim, thisInfo.range)]
        end
    else
        thisInfo.array = input_array
        thisInfo.dim   = getArrayNumDims(thisInfo.array, state)
        thisInfo.out_dim = thisInfo.dim
        thisInfo.elementTemp = createTempForArray(thisInfo.array, 1, state)
    end
    return thisInfo
end

function get_mmap_input_info(input_array :: SymAllGen, state)
    thisInfo = InputInfo()
    thisInfo.array = input_array
    thisInfo.dim   = getArrayNumDims(thisInfo.array, state)
    thisInfo.out_dim = thisInfo.dim
    thisInfo.elementTemp = createTempForArray(thisInfo.array, 1, state)
    return thisInfo
end

function gen_bitarray_mask(num_dim_inputs, thisInfo::InputInfo, parfor_index_syms::Array{Symbol,1}, state)
    # We only support bitarray selection for 1D arrays
    for i = 1:length(thisInfo.range)
        if !isa(thisInfo.range[i], MaskSelector)
            continue
        end        
        if i > 1
            throw(string("bitarray_mask only support for 1D arrays"))
        end
        mask_array = thisInfo.range[i].value
        # this hack helps Cgen by converting BitArray to Array{Bool,1}, but it causes an error in liveness analysis
        #      if isa(mask_array, SymbolNode) # a hack to change type to Array{Bool}
        #        mask_array = SymbolNode(mask_array.name, Array{Bool, mask_array.typ.parameters[1]})
        #      end
        thisInfo.rangeconds = mk_arrayref1(num_dim_inputs, mask_array, parfor_index_syms, true, state)
    end
end

function gen_pir_loopnest(pre_statements, save_array_lens, num_dim_inputs, inputInfo, unique_node_id, parfor_index_syms, state)
    loopNests = Array(PIRLoopNest, num_dim_inputs)
    # Insert a statement to assign the length of the input arrays to a var
    for i = 1:num_dim_inputs
        save_array_len = string("parallel_ir_save_array_len_", i, "_", unique_node_id)
        array1_len = mk_assignment_expr(SymbolNode(symbol(save_array_len), Int), mk_arraylen_expr(inputInfo[1],i), state)
        # add that assignment to the set of statements to execute before the parfor
        push!(pre_statements,array1_len)
        CompilerTools.LambdaHandling.addLocalVar(save_array_len, Int, ISASSIGNEDONCE | ISASSIGNED, state.lambdaInfo)
        push!(save_array_lens, save_array_len)
        loopNests[num_dim_inputs - i + 1] =
        PIRLoopNest(SymbolNode(parfor_index_syms[i],Int), 1, SymbolNode(symbol(save_array_len),Int),1)
    end
    return loopNests
end

@doc """
The main routine that converts a mmap! AST node to a parfor AST node.
"""
function mk_parfor_args_from_mmap!(input_arrays :: Array, dl :: DomainLambda, with_indices, domain_oprs, state)
    # First arg is an array of input arrays to the mmap!
    len_input_arrays = length(input_arrays)
    dprintln(2,"mk_parfor_args_from_mmap!: # input arrays = ", len_input_arrays)
    dprintln(2,"input arrays: ", input_arrays)
    dprintln(2,"dl.inputs: ", dl.inputs)
    @assert len_input_arrays>0 "mmap! should have input arrays"

    # Handle range selector
    inputInfo = InputInfo[]
    for i = 1 : length(input_arrays)
        push!(inputInfo, get_mmap_input_info(input_arrays[i],state))
    end

    # Create an expression to access one element of this input array with index symbols parfor_index_syms
    indexed_arrays = map(i->inputInfo[i].elementTemp, 1:length(inputInfo))

    # Get a unique number to embed in generated code for new variables to prevent name conflicts.
    unique_node_id = get_unique_num()

    num_dim_inputs = findSelectedDimensions(inputInfo, state)
    # Verify the number of input arrays matches the number of input types in dl
    assert(length(dl.inputs) == len_input_arrays || (with_indices && length(dl.inputs) == num_dim_inputs + len_input_arrays))

    # Create variables to use for the loop indices.
    parfor_index_syms::Array{Symbol,1} = gen_parfor_loop_indices(num_dim_inputs, unique_node_id, state)

    map(i->(gen_bitarray_mask(num_dim_inputs, inputInfo[i], parfor_index_syms, state)), 1:length(inputInfo))

    out_body = Any[]
    # Create empty arrays to hold pre and post statements.
    pre_statements  = Any[]
    post_statements = Any[]

    # Make sure each input array is a SymbolNode
    # Also, create indexed versions of those symbols for the loop body
    for(i = 1:length(inputInfo))
        push!(out_body, mk_assignment_expr(inputInfo[i].elementTemp, mk_arrayref1(num_dim_inputs, inputInfo[i].array, parfor_index_syms, true, state, inputInfo[i].range), state))
    end

    # not used here?
    save_array_lens = AbstractString[]
    # generates loopnests and updates pre_statements
    loopNests = gen_pir_loopnest(pre_statements, save_array_lens, num_dim_inputs,inputInfo,unique_node_id, parfor_index_syms, state)

    for i in inputInfo
        pre_statements = [i.pre_offsets; pre_statements]
    end

    # add local vars to state
    #for (v, d) in dl.locals
    #  CompilerTools.LambdaHandling.addLocalVar(v, d.typ, d.flag, state.lambdaInfo)
    #end

    dprintln(3,"indexed_arrays = ", indexed_arrays)
    dl_inputs = with_indices ? vcat(indexed_arrays, [SymbolNode(s, Int) for s in parfor_index_syms ]) : indexed_arrays
    dprintln(3,"dl_inputs = ", dl_inputs)
    # Call Domain IR to generate most of the body of the function (except for saving the output)
    (max_label, nested_lambda, nested_body) = nested_function_exprs(state.max_label, dl, dl_inputs)
    gensym_map = mergeLambdaIntoOuterState(state, nested_lambda)
    nested_body = CompilerTools.LambdaHandling.replaceExprWithDict!(nested_body, gensym_map, AstWalk)
    state.max_label = max_label
    out_body = [out_body; nested_body...]
    dprintln(2,"typeof(out_body) = ",typeof(out_body))
    assert(isa(out_body,Array))
    oblen = length(out_body)
    # the last output of genBody is a tuple of the outputs of the mmap!
    lbexpr::Expr = out_body[oblen]
    assert(lbexpr.head == :tuple)
    assert(length(lbexpr.args) == length(dl.outputs))

    dprintln(2,"out_body is of length ",length(out_body))
    printBody(3,out_body)

    else_body = Any[]
    elseLabel = next_label(state)
    condExprs = Any[]
    for i = 1:length(inputInfo)
        if inputInfo[i].rangeconds.head != :noop
            push!(condExprs, Expr(:gotoifnot, inputInfo[i].rangeconds, elseLabel))
        end
    end
    out_body = out_body[1:oblen-1]
    for i = 1:length(dl.outputs)
        if length(inputInfo[i].range) != 0
            tfa = createTempForRangedArray(inputInfo[i].array, inputInfo[i].range, 2, state)
        else
            tfa = createTempForArray(inputInfo[i].array, 2, state)
        end
        #tfa = createTempForArray(dl.outputs[i], 2, state)
        #tfa = createTempForArray(input_arrays[i], 2, state, array_temp_map2)
        push!(out_body, mk_assignment_expr(tfa, lbexpr.args[i], state))
        push!(out_body, mk_arrayset1(num_dim_inputs, inputInfo[i].array, parfor_index_syms, tfa, true, state, inputInfo[i].range))
        if length(condExprs) > 0
            push!(else_body, mk_assignment_expr(tfa, mk_arrayref1(num_dim_inputs, inputInfo[i].array, parfor_index_syms, true, state, inputInfo[i].range), state))
            push!(else_body, mk_arrayset1(num_dim_inputs, inputInfo[i].array, parfor_index_syms, tfa, true, state, inputInfo[i].range))
        end
    end

    # add conditional expressions to body if array elements are selected by bit arrays
    fallthroughLabel = next_label(state)
    if length(condExprs) > 0
        out_body = [ condExprs; out_body; GotoNode(fallthroughLabel); LabelNode(elseLabel); else_body; LabelNode(fallthroughLabel) ]
    end

    dprintln(3, "out_body = ", out_body)
    # Compute which scalars and arrays are ever read or written by the body of the parfor
    rws = CompilerTools.ReadWriteSet.from_exprs(out_body, pir_rws_cb, state.lambdaInfo)

    # Make sure that for mmap! that the array indices are all of the simple variety
    simply_indexed = simpleIndex(rws.readSet.arrays) && simpleIndex(rws.writeSet.arrays)
    dprintln(2,rws)

    post_statements = create_mmap!_post_statements(input_arrays, dl, state)

    new_parfor = ParallelAccelerator.ParallelIR.PIRParForAst(
        inputInfo[1],
        out_body,
        pre_statements,
        loopNests,
        PIRReduction[],
        post_statements,
        domain_oprs,
        state.top_level_number,
        rws,
        unique_node_id,
        simply_indexed)

    dprintln(3,"Lowered parallel IR = ", new_parfor)

    [new_parfor]
end

function create_mmap!_post_statements(input_arrays, dl, state)
    post_statements = Any[]
    # Is there a universal output representation that is generic and doesn't depend on the kind of domain IR input?
    #if(len_input_arrays == 1)
    if length(dl.outputs) == 1
        # If there is only one output then put that output in the post_statements
        push!(post_statements, input_arrays[1])
    else
        ret_arrays = input_arrays[1:length(dl.outputs)]
        ret_types = Any[ CompilerTools.LambdaHandling.getType(x, state.lambdaInfo) for x in ret_arrays ]
        push!(post_statements, mk_tuple_expr(ret_arrays, Core.Inference.to_tuple_type(tuple(ret_types...))))
    end
    return post_statements
end

function mk_parfor_args_from_parallel_for(args :: Array{Any,1}, state)
    @assert length(args[1]) == length(args[2])
    n_loops = length(args[1])
    loopvars = args[1]
    ranges = args[2]
    dl = args[3]
    dl_inputs = [SymbolNode(s, Int) for s in loopvars]
    (max_label, nested_lambda, nested_body) = nested_function_exprs(state.max_label, dl, dl_inputs)
    gensym_map = mergeLambdaIntoOuterState(state, nested_lambda)
    nested_body = CompilerTools.LambdaHandling.replaceExprWithDict!(nested_body, gensym_map, AstWalk)
    state.max_label = max_label
    out_body = nested_body
    # Create empty arrays to hold pre and post statements.
    pre_statements  = Any[]
    post_statements = Any[]
    loopNests = Array(PIRLoopNest, n_loops)
    unique_node_id = get_unique_num()
    rearray = RangeExprs[]
    # Insert a statement to assign the length of the input arrays to a var
    for i = 1:n_loops
        loopvar = loopvars[i]
        range = ranges[i]
        range_name = symbol("parallel_ir_range_len_$(loopvar)_$(unique_node_id)_range")
        # FIXME: We should infer the range type
        range_type = UnitRange{Int64}
        range_expr = mk_assignment_expr(SymbolNode(range_name, range_type), range)
        CompilerTools.LambdaHandling.addLocalVar(string(range_name), range_type, ISASSIGNEDONCE | ISASSIGNED, state.lambdaInfo)
        push!(pre_statements, range_expr)
        save_loop_len = string("parallel_ir_save_loop_len_", loopvar, "_", unique_node_id)
        loop_len = mk_assignment_expr(SymbolNode(symbol(save_loop_len), Int), :(length($range_name)), state)
        # add that assignment to the set of statements to execute before the parfor
        push!(pre_statements,loop_len)
        CompilerTools.LambdaHandling.addLocalVar(save_loop_len, Int, ISASSIGNEDONCE | ISASSIGNED, state.lambdaInfo)
        loopNests[n_loops - i + 1] = PIRLoopNest(SymbolNode(loopvar,Int), 1, SymbolNode(symbol(save_loop_len),Int),1)
        push!(rearray, RangeExprs(1,1,:(length($range_name))))
    end
    inputInfo = InputInfo()
    inputInfo.range = [RangeData(i) for i in rearray]

    rws = CompilerTools.ReadWriteSet.from_exprs(out_body, pir_rws_cb, state.lambdaInfo)
    simply_indexed = simpleIndex(rws.readSet.arrays) && simpleIndex(rws.writeSet.arrays)
    parfor = ParallelAccelerator.ParallelIR.PIRParForAst(
        inputInfo,
        out_body,
        pre_statements,
        loopNests,
        PIRReduction[],
        post_statements,
        [],
        state.top_level_number,
        rws,
        unique_node_id,
        simply_indexed)
    [parfor]
end

# ===============================================================================================================================

function generatePreOffsetStatement(range :: RangeData, ret)
    # Special case ranges that start with 1 as they don't require the start of the range added to the loop index.
    if !isStartOneRange(range.exprs)
        dprintln(3,"range = ", range)

        range_expr = mk_assignment_expr(range.offset_temp_var, TypedExpr(Int64, :call, GlobalRef(Base, :sub_int), range.start, 1))
        push!(ret, range_expr)
    end
end

function generatePreOffsetStatement(range :: Union{MaskSelector,SingularSelector}, ret)
    # Intentionally do nothing.
end

function generatePreOffsetStatements(num_dim_inputs, ranges :: Array{DimensionSelector,1})
    assert(length(ranges) >= num_dim_inputs)

    ret = Expr[]

    #for i = 1:length(ranges)
    for i = 1:num_dim_inputs
        range = ranges[i]
        generatePreOffsetStatement(range, ret)
    end

    return ret
end


# Create variables to use for the parfor loop indices.
function gen_parfor_loop_indices(num_dim_inputs, unique_node_id, state)
    parfor_index_syms = Array(Symbol,num_dim_inputs)
    for i = 1:num_dim_inputs
        parfor_index_var = string("parfor_index_", i, "_", unique_node_id)
        parfor_index_sym = symbol(parfor_index_var)
        CompilerTools.LambdaHandling.addLocalVar(parfor_index_sym, Int, ISASSIGNED, state.lambdaInfo)
        parfor_index_syms[i] = parfor_index_sym
    end
    return parfor_index_syms
end

@doc """
Given all the InputInfo for a Domain IR operation being lowered to Parallel IR,
determine the number of output dimensions for those arrays taking into account
that singly selected trailing dimensinos are eliminated.  Make sure that all such
arrays have the same output dimensions because this will match the loop nest size.
"""
function findSelectedDimensions(inputInfo :: Array{InputInfo,1}, state)
    assert(length(inputInfo) > 0)

    # Get the first array's output dimensions.
    res = inputInfo[1].out_dim
    for i = 2:length(inputInfo)
        # Make sure that this is the same as the first array's.
        if inputInfo[i].out_dim != res
            throw(string("Array number ", i, " does not have the same number of resulting dimensions as the first array in findSelectedDimensions."))
        end
    end

    return res
end

@doc """
In the case where a domain IR operation on an array creates a lower dimensional output,
the indexing expression needs the expression that selects those constant trailing dimensions
that are being dropped.  This function returns an array of those constant expressions for
the trailing dimensions.
"""
function getConstDims(num_dim_inputs, inputInfo :: InputInfo)
    assert(num_dim_inputs <= inputInfo.dim)

    if num_dim_inputs == inputInfo.dim
        return []
    end

    # Assume that we can only get here through a :select and :range operation and therefore
    # range should be larger than the number of output dimensions.
    assert(length(inputInfo.range) > num_dim_inputs)

    ret = []

    for i = num_dim_inputs+1:length(inputInfo.range)
        this_dim = inputInfo.range[i]
        assert(isa(this_dim, SingularSelector))
        push!(ret, this_dim.value)
    end

    return ret
end

@doc """
The main routine that converts a mmap AST node to a parfor AST node.
"""
function mk_parfor_args_from_mmap(input_arrays :: Array, dl :: DomainLambda, domain_oprs, state)
    # First arg is an array of input arrays to the mmap
    len_input_arrays = length(input_arrays)
    dprintln(2,"mk_parfor_args_from_mmap: # input arrays = ", len_input_arrays)
    dprintln(2,"input arrays: ", input_arrays)
    @assert len_input_arrays>0 "mmap should have input arrays"

    # Handle range selector
    inputInfo = InputInfo[]
    for i = 1 : length(input_arrays)
        push!(inputInfo, get_mmap_input_info(input_arrays[i], state))
    end
    dprintln(3, "inputInfo = ", inputInfo)

    # Verify the number of input arrays matches the number of input types in dl
    assert(length(dl.inputs) == length(inputInfo))

    # Create an expression to access one element of this input array with index symbols parfor_index_syms
    indexed_arrays = map(i->inputInfo[i].elementTemp, 1:length(inputInfo))
    dprintln(3, "indexed_arrays = ", indexed_arrays)

    # Get a unique number to embed in generated code for new variables to prevent name conflicts.
    unique_node_id = get_unique_num()

    num_dim_inputs = findSelectedDimensions(inputInfo, state)
    dprintln(3, "num_dim_inputs = ", num_dim_inputs)
    loopNests = Array(PIRLoopNest, num_dim_inputs)

    # Create variables to use for the loop indices.
    parfor_index_syms::Array{Symbol,1} = gen_parfor_loop_indices(num_dim_inputs, unique_node_id, state)

    map(i->(gen_bitarray_mask(num_dim_inputs, inputInfo[i], parfor_index_syms, state)), 1:length(inputInfo))

    out_body = Any[]
    # Create empty arrays to hold pre and post statements.
    pre_statements  = Any[]
    post_statements = Any[]
    # To hold the names of newly created output arrays.
    new_array_symbols = Symbol[]
    save_array_lens   = AbstractString[]

    # Make sure each input array is a SymbolNode.
    # Also, create indexed versions of those symbols for the loop body.
    for(i = 1:length(inputInfo))
        push!(out_body, mk_assignment_expr(
                           inputInfo[i].elementTemp, 
                           mk_arrayref1(
                              num_dim_inputs,
                              inputInfo[i].array, 
                              [parfor_index_syms; getConstDims(num_dim_inputs, inputInfo[i])...], 
                              true,  # inbounds is true
                              state, 
                              inputInfo[i].range), 
                           state))
    end

    # TODO - make sure any ranges for any input arrays are inbounds in the pre-statements
    # TODO - extract the lower bound of the range into a variable

    # generates loopnests and updates pre_statements
    loopNests = gen_pir_loopnest(pre_statements, save_array_lens, num_dim_inputs, inputInfo, unique_node_id, parfor_index_syms, state)

    for i in inputInfo
        pre_statements = [i.pre_offsets; pre_statements]
    end

    # add local vars to state
    #for (v, d) in dl.locals
    #  CompilerTools.LambdaHandling.addLocalVar(v, d.typ, d.flag, state.lambdaInfo)
    #end
    # Call Domain IR to generate most of the body of the function (except for saving the output)
    (max_label, nested_lambda, nested_body) = nested_function_exprs(state.max_label, dl, indexed_arrays)
    gensym_map = mergeLambdaIntoOuterState(state, nested_lambda)
    nested_body = CompilerTools.LambdaHandling.replaceExprWithDict!(nested_body, gensym_map)
    state.max_label = max_label
    out_body = [out_body; nested_body...]
    dprintln(2,"typeof(out_body) = ",typeof(out_body))
    assert(isa(out_body,Array))
    oblen = length(out_body)
    # the last output of genBody is a tuple of the outputs of the mmap
    lbexpr::Expr = out_body[oblen] 
    assert(lbexpr.head == :tuple)
    assert(length(lbexpr.args) == length(dl.outputs))

    dprintln(2,"out_body is of length ",length(out_body), " ", out_body)

    # To hold the sum of the sizes of the individual output array elements
    output_element_sizes = 0

    out_body = out_body[1:oblen-1]
    else_body = Any[]
    elseLabel = next_label(state)
    condExprs = Any[]
    for i = 1:length(inputInfo)
        if inputInfo[i].rangeconds.head != :noop
            push!(condExprs, Expr(:gotoifnot, inputInfo[i].rangeconds, elseLabel))
        end
    end
    # Create each output array
    number_output_arrays = length(dl.outputs)
    for(i = 1:number_output_arrays)
        new_array_name = string("parallel_ir_new_array_name_", unique_node_id, "_", i)
        dprintln(2,"new_array_name = ", new_array_name, " element type = ", dl.outputs[i])
        # create the expression that create a new array and assigns it to a variable whose name is in new_array_name
        if num_dim_inputs == 1
            new_ass_expr = mk_assignment_expr(SymbolNode(symbol(new_array_name), Array{dl.outputs[i],num_dim_inputs}), mk_alloc_array_1d_expr(dl.outputs[i], Array{dl.outputs[i], num_dim_inputs}, symbol(save_array_lens[1])), state)
        elseif num_dim_inputs == 2
            new_ass_expr = mk_assignment_expr(SymbolNode(symbol(new_array_name), Array{dl.outputs[i],num_dim_inputs}), mk_alloc_array_2d_expr(dl.outputs[i], Array{dl.outputs[i], num_dim_inputs}, symbol(save_array_lens[1]), symbol(save_array_lens[2])), state)
        elseif num_dim_inputs == 3
            new_ass_expr = mk_assignment_expr(SymbolNode(symbol(new_array_name), Array{dl.outputs[i],num_dim_inputs}), mk_alloc_array_3d_expr(dl.outputs[i], Array{dl.outputs[i], num_dim_inputs}, symbol(save_array_lens[1]), symbol(save_array_lens[2]), symbol(save_array_lens[3])), state)
        else
            throw(string("Only arrays up to 3 dimensions supported in parallel IR."))
        end
        # remember the array variable as a new variable added to the function and that it is assigned once (the 18)
        CompilerTools.LambdaHandling.addLocalVar(new_array_name, Array{dl.outputs[i],num_dim_inputs}, ISASSIGNEDONCE | ISASSIGNED, state.lambdaInfo)
        # add the statement to create the new output array to the set of statements to execute before the parfor
        push!(pre_statements,new_ass_expr)
        nans = symbol(new_array_name)
        push!(new_array_symbols,nans)
        nans_sn = SymbolNode(nans, Array{dl.outputs[i], num_dim_inputs})

        tfa = createTempForArray(nans_sn, 1, state)
        push!(out_body, mk_assignment_expr(tfa, lbexpr.args[i], state))
        push!(out_body, mk_arrayset1(num_dim_inputs, nans_sn, parfor_index_syms, tfa, true, state))
        if length(condExprs) > 0
            push!(else_body, mk_assignment_expr(tfa, mk_arrayref1(num_dim_inputs, inputInfo[i].array, parfor_index_syms, true, state, inputInfo[i].range), state))
            push!(else_body, mk_arrayset1(num_dim_inputs, inputInfo[i].array, parfor_index_syms, tfa, true, state, inputInfo[i].range))
        end
        # keep the sum of the sizes of the individual output array elements
        output_element_sizes = output_element_sizes + sizeof(dl.outputs)
    end
    dprintln(3,"out_body = ", out_body)

    # add conditional expressions to body if array elements are selected by bit arrays
    fallthroughLabel = next_label(state)
    if length(condExprs) > 0
        out_body = [ condExprs; out_body; GotoNode(fallthroughLabel); LabelNode(elseLabel); else_body; LabelNode(fallthroughLabel) ]
    end

    # Compute which scalars and arrays are ever read or written by the body of the parfor
    rws = CompilerTools.ReadWriteSet.from_exprs(out_body, pir_rws_cb, state.lambdaInfo)

    # Make sure that for mmap that the array indices are all of the simple variety
    simply_indexed = simpleIndex(rws.readSet.arrays) && simpleIndex(rws.writeSet.arrays)
    dprintln(2,rws)

    post_statements = create_mmap_post_statements(new_array_symbols, dl, num_dim_inputs) 

    new_parfor = ParallelAccelerator.ParallelIR.PIRParForAst(
        inputInfo[1],
        out_body,
        pre_statements,
        loopNests,
        PIRReduction[],
        post_statements,
        domain_oprs,
        state.top_level_number,
        rws,
        unique_node_id,
        simply_indexed)

    dprintln(3,"Lowered parallel IR = ", new_parfor)
    [new_parfor]
end

function create_mmap_post_statements(new_array_symbols, dl, num_dim_inputs)
    post_statements = SymbolNode[]
    # Is there a universal output representation that is generic and doesn't depend on the kind of domain IR input?
    if(length(dl.outputs)==1)
        # If there is only one output then put that output in the post_statements
        push!(post_statements,SymbolNode(new_array_symbols[1],Array{dl.outputs[1],num_dim_inputs}))
    else
        all_sn = SymbolNode[]
        assert(length(dl.outputs) == length(new_array_symbols))
        for i = 1:length(dl.outputs)
            push!(all_sn, SymbolNode(new_array_symbols[1], Array{dl.outputs[1], num_dim_inputs}))
        end
        push!(post_statements, all_sn)
    end
    return post_statements
end


# ===============================================================================================================================
