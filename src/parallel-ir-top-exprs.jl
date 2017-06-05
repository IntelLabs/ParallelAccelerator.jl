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


"""
top_level_mk_body: lower domain nodes in the AST to parfor nodes.
"""
function top_level_mk_body(ast::Array{Any,1}, depth, state)
    len  = length(ast)
    body = Any[]
    fuse_number = 1

    # Process the top-level expressions of a function and do fusion and useless assignment elimination.
    for i = 1:len
        # Record the top-level statement number in the processing state.
        state.top_level_number = i
        @dprintln(2,"Processing top-level ast #",i," depth=",depth)

        # Convert the current expression.
        new_exprs = from_expr(ast[i], depth, state, true)
        assert(isa(new_exprs,Array))
        # If conversion of current statement resulted in anything.
        if length(new_exprs) != 0
            append!(body, new_exprs)
        else
            @dprintln(3,"This statement was filtered: ", ast[i])
        end
    end
    return body
end

"""
fusion_pass: perform fusion of adjacent parfor nodes in the AST.
"""
function fusion_pass(ast::Array{Any,1}, depth, state)
    len  = length(ast)
    body = Any[]
    fuse_number = 1

    # Process the top-level expressions of a function and do fusion and useless assignment elimination.
    for i = 1:len
        # Record the top-level statement number in the processing state.
        state.top_level_number = i
        @dprintln(2,"Processing top-level ast #",i," depth=",depth)

        # If this isn't the first statement...
        if length(body) != 0
            last_node = body[end]  # Get the last statement inserted.
            @dprintln(3, "Should fuse?")
            @dprintln(3, "new = ", ast[i])
            @dprintln(3, "last = ", last_node)

            # See if the previous expression is a parfor.
            is_last_parfor = isParforAssignmentNode(last_node) || isBareParfor(last_node)
            # See if the new expression is a parfor.
            is_new_parfor  = isParforAssignmentNode(ast[i])    || isBareParfor(ast[i])
            @dprintln(3,"is_new_parfor = ", is_new_parfor, " is_last_parfor = ", is_last_parfor)

            # If both are parfors then try to fuse them.
            if is_new_parfor && is_last_parfor
                @dprintln(3,"Starting fusion ", fuse_number)
                fuse_number = fuse_number + 1
                fuse_ret = fuse(body, length(body), ast[i], state)
                if fuse_ret>0
                    # 2 means combination of old and new parfors has no output and both are dead
                    if fuse_ret==2
                        # remove last parfor and don't add anything new
                        pop!(body)
                    end
                    # If fused then the new node is consumed and no new node is added to the body.
                    continue
                end
            end

            push!(body, ast[i])
        else
            push!(body, ast[i])
        end
    end
    return body
end

"""
fusion_pass: perform fusion on the AST but callable as an external pass.  Just takes the
function name and function lambda.
"""
function fusion_pass(function_name::String, LambdaVarInfo, body::Expr)
    #LambdaVarInfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
    @assert body.head==:body "invalid body expr head"
    max_label = getMaxLabel(0, body.args)
    lives = computeLiveness(body, LambdaVarInfo)
    input_arrays = getArrayParams(LambdaVarInfo)
    state = expr_state(function_name, lives, max_label, input_arrays)
    genEquivalenceClasses(LambdaVarInfo, body, state)
    @dprintln(3,"new fusion lambda = ")
    printLambda(3, LambdaVarInfo, body)
    @dprintln(3,"new fusion equivalence classes")
    print_correlations(3, state)
    body = fusion_pass(body.args, 1, state)
    body = Expr(:body, body...)
    # return CompilerTools.LambdaHandling.LambdaVarInfoToLambda(LambdaVarInfo, body, ParallelAccelerator.ParallelIR.AstWalk)
    return LambdaVarInfo, body
end

# Remove the pre-statements from parfor nodes and expand them into the top-level expression array.
function top_level_expand_pre(body, state)

    expanded_body = Any[]

    for i = 1:length(body)
        if isParforAssignmentNode(body[i])
            parfor_assignment = body[i]
            @dprintln(3,"Expanding a parfor assignment node = ", parfor_assignment)

            the_parfor = getParforNode(parfor_assignment)
            @dprintln(3,"the_parfor = ", the_parfor)
            lhs = getLhsFromAssignment(parfor_assignment)
            rhs = getRhsFromAssignment(parfor_assignment)

            # Add all the pre-parfor statements to the expanded body.
            append!(expanded_body, the_parfor.preParFor)
            append!(expanded_body, the_parfor.hoisted)
            the_parfor.preParFor = Any[]
            the_parfor.hoisted = Any[]

            # Add just the parfor to the expanded body.  The post-parfor part removed below.
            assert(typeof(rhs) == Expr)
            rhs.typ = typeof(0) # a few lines down you can see that the last post-statement of 0 is added.
            push!(expanded_body, rhs)

            # All the post parfor part to the expanded body.
            # The regular part of the post parfor is just added.
            # The part that indicates the return values creates individual assignment statements for each thing returned.
            if isFusionAssignment(parfor_assignment)
                append!(expanded_body, the_parfor.postParFor[1:end-1])
                for j = 4:length(parfor_assignment.args)
                    push!(expanded_body, mk_assignment_expr(parfor_assignment.args[j], the_parfor.postParFor[end][j-3], state))
                end
            else
                append!(expanded_body, the_parfor.postParFor[1:end-1])
                push!(expanded_body, mk_assignment_expr(lhs, the_parfor.postParFor[end], state))
            end
            the_parfor.postParFor = Any[]
            push!(the_parfor.postParFor, 0)
            #createInstructionCountEstimate(the_parfor, state)
        elseif isBareParfor(body[i])
            rhs = body[i]
            the_parfor = rhs.args[1]

            # Add all the pre-parfor statements to the expanded body.
            append!(expanded_body, the_parfor.preParFor)
            append!(expanded_body, the_parfor.hoisted)
            the_parfor.preParFor = Any[]
            the_parfor.hoisted= Any[]

            # Add just the parfor to the expanded body.  The post-parfor part removed below.
            assert(typeof(rhs) == Expr)
            rhs.typ = typeof(0) # a few lines down you can see that the last post-statement of 0 is added.
            push!(expanded_body, rhs)

            the_parfor.postParFor = Any[]
            push!(the_parfor.postParFor, 0)
            #createInstructionCountEstimate(the_parfor, state)
        else
            push!(expanded_body, body[i])
        end
    end
    return expanded_body
end

function top_level_mk_task_graph(body, state, new_lives, loop_info)
    task_start = time_ns()

    rr = ReplacedRegion[]
    # TODO: another pass of alias analysis to re-use dead but uniquely allocated arrays
    # AliasAnalysis.analyze_lambda_body(fake_body, state.param, state.meta2_typed, new_lives)

    # Create a mapping between the minimized basic block numbers and indices in body that are in those correponding basic blocks.
    map_reduced_bb_num_to_body = Dict{Int,Array{Int,1}}()
    for i = 1:length(body)
        # Get the basic block number for the first top level number associated with this entry in body.
        bb_num = CompilerTools.LivenessAnalysis.find_bb_for_statement(i, new_lives)
        if bb_num == nothing
            if typeof(body[i]) != LabelNode
                @dprintln(0,"statement that couldn't be found in liveness analysis ", body[i])
                throw(string("find_bb_for_statement should not fail for non-LabelNodes"))
            end
            continue
        end
        # If not previously in the map then initialize it with the current body index.  Otherwise, add the current body index
        # as also mapping to its containing basic block.
        if !haskey(map_reduced_bb_num_to_body, bb_num)
            map_reduced_bb_num_to_body[bb_num] = [i]
        else
            map_reduced_bb_num_to_body[bb_num] = [map_reduced_bb_num_to_body[bb_num]; i]
        end
    end

    @dprintln(3,"map_reduced_bb_num_to_body = ", map_reduced_bb_num_to_body)

    bbs_in_task_graph_loops = Set()
    bbs = new_lives.basic_blocks
    #tgsections = TaskGraphSection[]

    if put_loops_in_task_graph
        # For each loop that loop analysis identified.
        for one_loop in loop_info.loops
            # If the loop is "simple", i.e., has no just a body and a back-edge and no conditionals or nested loops.
            if length(one_loop.members) == 2
                # This is sort of sanity checking because the way Julia creates loops, these conditions should probably always hold.
                head_bb = bbs[one_loop.head]
                back_bb = bbs[one_loop.back_edge]

                if length(head_bb.preds) == 2 &&
                    length(head_bb.succs) == 1 &&
                    length(back_bb.preds) == 1 &&
                    length(back_bb.succs) == 2
                    before_head = getNonBlock(head_bb.preds, one_loop.back_edge)
                    assert(typeof(before_head) == CompilerTools.LivenessAnalysis.BasicBlock)
                    @dprintln(3,"head_bb.preds = ", head_bb.preds, " one_loop.back_edge = ", one_loop.back_edge, " before_head = ", before_head)
                    # assert(length(before_head) == 1)
                    after_back  = getNonBlock(back_bb.succs, one_loop.head)
                    assert(typeof(after_back) == CompilerTools.LivenessAnalysis.BasicBlock)
                    #assert(length(after_back) == 1)

                    head_indices = map_reduced_bb_num_to_body[one_loop.head]
                    head_first_parfor = nothing
                    for j = 1:length(head_indices)
                        if isParforAssignmentNode(body[head_indices[j]]) || isBareParfor(body[head_indices[j]])
                            head_first_parfor = j
                            break
                        end
                    end

                    back_indices = map_reduced_bb_num_to_body[one_loop.back_edge]
                    back_first_parfor = nothing
                    for j = 1:length(back_indices)
                        if isParforAssignmentNode(body[back_indices[j]]) || isBareParfor(body[back_indices[j]])
                            back_first_parfor = j
                            break
                        end
                    end

                    if head_first_parfor != nothing || back_first_parfor != nothing
                        new_bbs_for_set = Set(one_loop.head, one_loop.back_edge, before_head.label, after_back.label)
                        assert(length(intersect(bbs_in_task_graph_loops, new_bbs_for_set)) == 0)
                        bbs_in_task_graph_loops = union(bbs_in_task_graph_loops, new_bbs_for_set)

                        before_indices = map_reduced_bb_num_to_body[before_head.label]
                        before_first_parfor = nothing
                        for j = 1:length(before_indices)
                            if isParforAssignmentNode(body[before_indices[j]]) || isBareParfor(body[body_indices[j]])
                                before_first_parfor = j
                                break
                            end
                        end

                        after_indices = map_reduced_bb_num_to_body[after_back.label]
                        after_first_parfor = nothing
                        for j = 1:length(after_indices)
                            if isParforAssignmentNode(body[after_indices[j]]) || isBareParfor(body[after_indices[j]])
                                after_first_parfor = j
                                break
                            end
                        end

                        #          bb_live_info = new_lives.basic_blocks[bb_num]
                        #          push!(replaced_regions, (first_parfor, last_parfor, makeTasks(first_parfor, last_parfor, body, bb_live_info)))

                    end
                else
                    @dprintln(1,"Found a loop with 2 members but unexpected head or back_edge structure.")
                    @dprintln(1,"head = ", head_bb)
                    @dprintln(1,"back_edge = ", back_bb)
                end
            end
        end
    end

    for i in map_reduced_bb_num_to_body
        bb_num = i[1]
        body_indices = i[2]
        bb_live_info = new_lives.basic_blocks[new_lives.cfg.basic_blocks[bb_num]]

        if !in(bb_num, bbs_in_task_graph_loops)
            if task_graph_mode == SEQUENTIAL_TASKS
                # Find the first parfor in the block.
                first_parfor = nothing
                for j = 1:length(body_indices)
                    if isParforAssignmentNode(body[body_indices[j]]) || isBareParfor(body[body_indices[j]])
                        first_parfor = body_indices[j]
                        break
                    end
                end

                # If we found a parfor in the block.
                if first_parfor != nothing
                    # Find the last parfor in the block...it might be the same as the first.
                    last_parfor = nothing
                    for j = length(body_indices):-1:1
                        if isParforAssignmentNode(body[body_indices[j]]) || isBareParfor(body[body_indices[j]])
                            last_parfor = body_indices[j]
                            break
                        end
                    end
                    assert(last_parfor != nothing)

                    # Remember this section of code as something to transform into task graph format.
                    #push!(tgsections, TaskGraphSection(first_parfor, last_parfor, body[first_parfor:last_parfor]))
                    #@dprintln(3,"Adding TaskGraphSection ", tgsections[end])

                    push!(rr, ReplacedRegion(first_parfor, last_parfor, bb_num, makeTasks(first_parfor, last_parfor, body, bb_live_info, state, task_graph_mode)))
                end
            elseif task_graph_mode == ONE_AT_A_TIME
                for j = 1:length(body_indices)
                    if taskableParfor(body[body_indices[j]])
                        # Remember this section of code as something to transform into task graph format.
                        cur_start = cur_end = body_indices[j]
                        #push!(tgsections, TaskGraphSection(cur_start, cur_end, body[cur_start:cur_end]))
                        #@dprintln(3,"Adding TaskGraphSection ", tgsections[end])

                        push!(rr, ReplacedRegion(body_indices[j], body_indices[j], bb_num, makeTasks(cur_start, cur_end, body, bb_live_info, state, task_graph_mode)))
                    end
                end
            elseif task_graph_mode == MULTI_PARFOR_SEQ_NO
                cur_start = nothing
                cur_end   = nothing
                stmts_in_batch = Int64[]

                for j = 1:length(body_indices)
                    if taskableParfor(body[body_indices[j]])
                        if cur_start == nothing
                            cur_start = cur_end = body_indices[j]
                        else
                            cur_end = body_indices[j]
                        end
                        push!(stmts_in_batch, body_indices[j])
                    else
                        if cur_start != nothing
                            @dprintln(3,"Non-taskable parfor ", stmts_in_batch, " ", body[body_indices[j]])
                            in_vars, out, locals = io_of_stmts_in_batch = getIO(stmts_in_batch, bb_live_info.statements)
                            @dprintln(3,"in_vars = ", in_vars)
                            @dprintln(3,"out_vars = ", out)
                            @dprintln(3,"local_vars = ", locals)

                            cur_in_vars, cur_out, cur_locals = io_of_stmts_in_batch = getIO([body_indices[j]], bb_live_info.statements)
                            @dprintln(3,"cur_in_vars = ", cur_in_vars)
                            @dprintln(3,"cur_out_vars = ", cur_out)
                            @dprintln(3,"cur_local_vars = ", cur_locals)

                            if isempty(intersect(out, cur_in_vars))
                                @dprintln(3,"Sequential statement doesn't conflict with batch.")
                                push!(stmts_in_batch, body_indices[j])
                                cur_end = body_indices[j]
                            else
                                # Remember this section of code (excluding current statement) as something to transform into task graph format.
                                #push!(tgsections, TaskGraphSection(cur_start, cur_end, body[cur_start:cur_end]))
                                #@dprintln(3,"Adding TaskGraphSection ", tgsections[end])

                                push!(rr, ReplacedRegion(cur_start, cur_end, bb_num, makeTasks(cur_start, cur_end, body, bb_live_info, state, task_graph_mode)))

                                cur_start = cur_end = nothing
                                stmts_in_batch = Int64[]
                            end
                        end
                    end
                end

                if cur_start != nothing
                    # Remember this section of code as something to transform into task graph format.
                    #push!(tgsections, TaskGraphSection(cur_start, cur_end, body[cur_start:cur_end]))
                    #@dprintln(3,"Adding TaskGraphSection ", tgsections[end])

                    push!(rr, ReplacedRegion(cur_start, cur_end, bb_num, makeTasks(cur_start, cur_end, body, bb_live_info, state, task_graph_mode)))
                end
            else
                throw(string("Unknown Parallel IR task graph formation mode."))
            end
        end
    end

    @dprintln(3,"Regions prior to sorting.")
    @dprintln(3,rr)
    # We replace regions in reverse order of index so that we don't mess up indices that we need to replace later.
    sort!(rr, by=x -> x.end_index, rev=true)
    @dprintln(3,"Regions after sorting.")
    @dprintln(3,rr)

    printBody(3,body)

    @dprintln(2, "replaced_regions")
    for i = 1:length(rr)
        @dprintln(2, rr[i])

        if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
            # New body starts with the pre-task graph portion
            new_body = body[1:rr[i].start_index-1]

            # Then adds calls for each task
            for j = 1:length(rr[i].tasks)
                cur_task = rr[i].tasks[j]
                @dprintln(3,"cur_task = ", cur_task, " type = ", typeof(cur_task))

                process_cur_task(cur_task, new_body, state)
            end

            # Then appends the post-task graph portion
            append!(new_body, body[rr[i].end_index+1:end])

            body = new_body

            @dprintln(3,"new_body after region ", i, " replaced")
            printBody(3,body)
        elseif ParallelAccelerator.client_intel_task_graph
            # new body starts with the pre-task graph portion
            new_body = body[1:rr[i].start_index-1]

            # then adds calls for each task
            for j = 1:length(rr[i].tasks)
                cur_task = rr[i].tasks[j]
                @dprintln(3,"cur_task = ", cur_task, " type = ", typeof(cur_task))
                if typeof(cur_task) == TaskInfo
                    @dprintln(3,"Inserting call to insert_divisible_task")
                    @dprintln(3,cur_task.function_sym, " type = ", typeof(cur_task.function_sym))

                    in_len  = length(cur_task.input_symbols)
                    mod_len = length(cur_task.modified_inputs)
                    io_len  = length(cur_task.io_symbols)
                    red_len = length(cur_task.reduction_vars)
                    @dprintln(3, "inputs, modifieds, io_sym, reductions = ", cur_task.input_symbols, " ", cur_task.modified_inputs, " ", cur_task.io_symbols, " ", cur_task.reduction_vars)

                    dims = length(cur_task.loopNests)
                    if dims > 0
                        itn = InsertTaskNode()

                        if ParallelAccelerator.client_intel_task_graph_mode == 0
                            itn.task_options = TASK_STATIC_SCHEDULER | TASK_AFFINITY_XEON
                        elseif ParallelAccelerator.client_intel_task_graph_mode == 1
                            itn.task_options = TASK_STATIC_SCHEDULER | TASK_AFFINITY_PHI
                        elseif ParallelAccelerator.client_intel_task_graph_mode == 2
                            itn.task_options = 0
                        else
                            throw(string("Unknown task graph mode option ", ParallelAccelerator.client_intel_task_graph_mode))
                        end

                        if task_graph_mode == SEQUENTIAL_TASKS
                            # intentionally do nothing
                        elseif task_graph_mode == ONE_AT_A_TIME
                            itn.task_options |= TASK_FINISH
                        elseif task_graph_mode == MULTI_PARFOR_SEQ_NO
                            # intentionally do nothing
                        else
                            throw(string("Unknown task_graph_mode."))
                        end

                        # Fill in pir_range
                        # Fill in the min and max grain sizes.
                        itn.ranges.dim = dims
                        itn.host_grain_size.dim = dims
                        itn.phi_grain_size.dim = dims
                        for l = 1:dims
                            # Note that loopNest is outer-dimension first
                            push!(itn.ranges.lower_bounds, Domain_IR.sub_expr(cur_task.loopNests[dims - l + 1].lower, 1))
                            push!(itn.ranges.upper_bounds, Domain_IR.sub_expr(cur_task.loopNests[dims - l + 1].upper, 1))
                            push!(itn.host_grain_size.sizes, 2)
                            push!(itn.phi_grain_size.sizes, 2)
                        end

                        # Fill in the arg metadata.
                        for l = 1:in_len
                            if isArrayType(cur_task.input_symbols[l].typ)
                                @dprintln(3, "is array")
                                push!(itn.args, pir_arg_metadata(cur_task.input_symbols[l], ARG_OPT_IN, create_array_access_desc(cur_task.input_symbols[l])))
                            else
                                @dprintln(3, "is not array")
                                push!(itn.args, pir_arg_metadata(cur_task.input_symbols[l], ARG_OPT_IN))
                            end
                        end
                        for l = 1:mod_len
                            @dprintln(3, "mod_len loop: ", l, " ", cur_task.modified_inputs[l])
                            if isArrayType(cur_task.modified_inputs[l].typ)
                                @dprintln(3, "is array")
                                push!(itn.args, pir_arg_metadata(cur_task.modified_inputs[l], ARG_OPT_OUT, create_array_access_desc(cur_task.modified_inputs[l])))
                            else
                                @dprintln(3, "is not array")
                                push!(itn.args, pir_arg_metadata(cur_task.modified_inputs[l], ARG_OPT_OUT))
                            end
                        end
                        for l = 1:io_len
                            @dprintln(3, "io_len loop: ", l, " ", cur_task.io_symbols[l])
                            if isArrayType(cur_task.io_symbols[l].typ)
                                @dprintln(3, "is array")
                                push!(itn.args, pir_arg_metadata(cur_task.io_symbols[l], ARG_OPT_INOUT, create_array_access_desc(cur_task.io_symbols[l])))
                            else
                                @dprintln(3, "is not array")
                                push!(itn.args, pir_arg_metadata(cur_task.io_symbols[l], ARG_OPT_INOUT))
                            end
                        end
                        for l = 1:red_len
                            @dprintln(3, "red_len loop: ", l, " ", cur_task.reduction_vars[l])
                            push!(itn.args, pir_arg_metadata(cur_task.reduction_vars[l], ARG_OPT_ACCUMULATOR))
                        end

                        # Fill in the task function.
                        itn.task_func = cur_task.task_func
                        itn.join_func = cur_task.join_func

                        @dprintln(3,"InsertTaskNode = ", itn)

                        insert_task_expr = TypedExpr(Int, :insert_divisible_task, itn)
                        push!(new_body, insert_task_expr)
                    else
                        throw(string("insert sequential task not implemented yet"))
                    end
                else
                    push!(new_body, cur_task)
                end
            end

            # Insert call to wait on the scheduler to complete all tasks.
            #push!(new_body, TypedExpr(Cint, :call, TopNode(:ccall), QuoteNode(:pert_wait_all_task), Type{Cint}, ()))

            # Then appends the post-task graph portion
            append!(new_body, body[rr[i].end_index+1:end])

            body = new_body

            @dprintln(3,"new_body after region ", i, " replaced")
            printBody(3,body)
        elseif run_as_task_decrement()
            # new body starts with the pre-task graph portion
            new_body = body[1:rr[i].start_index-1]

            # then adds calls for each task
            for j = 1:length(rr[i].tasks)
                cur_task = rr[i].tasks[j]
                assert(typeof(cur_task) == TaskInfo)

                @dprintln(3,"Inserting call to function")
                @dprintln(3,cur_task, " type = ", typeof(cur_task))
                @dprintln(3,cur_task.function_sym, " type = ", typeof(cur_task.function_sym))

                in_len = length(cur_task.input_symbols)
                out_len = length(cur_task.output_symbols)

                real_out_params = Any[]

                for k = 1:out_len
                    this_param = cur_task.output_symbols[k]
                    atype = Array{getType(this_param, state.LambdaVarInfo), 1}
                    temp_param_array = createStateVar(state, string(this_param.name, "_out_array"), atype, ISASSIGNED)
                    push!(real_out_params, temp_param_array)
                    new_temp_array = mk_alloc_array_expr(this_param.typ, atype, 1)
                    push!(new_body, mk_assignment_expr(temp_param_array, new_temp_array, state))
                end

                #push!(new_body, TypedExpr(Void, :call, cur_task.function_sym, cur_task.input_symbols..., real_out_params...))
                #push!(new_body, TypedExpr(Void, :call, TopNode(cur_task.function_sym), cur_task.input_symbols..., real_out_params...))
                push!(new_body, TypedExpr(Void, :call, mk_parallelir_ref(cur_task.function_sym), TypedExpr(pir_range_actual, :call, :pir_range_actual), cur_task.input_symbols..., real_out_params...))

                for k = 1:out_len
                    push!(new_body, mk_assignment_expr(cur_task.output_symbols[k], mk_arrayref1(real_out_params[k], 1, false, state), state))
                end
            end

            # then appends the post-task graph portion
            append!(new_body, body[rr[i].end_index+1:end])

            body = new_body

            @dprintln(3,"new_body after region ", i, " replaced")
            printBody(3,body)
        end
    end

    #    throw(string("debugging task graph"))
    if print_times
    @dprintln(1,"Task formation time = ", ns_to_sec(time_ns() - task_start))
    end

    return body
end


function recreateFromLoophead(new_body, stmt :: Expr, LoopEndDict :: Dict{LHSVar,Array{Any,1}}, state, newLambdaVarInfo)
    # Only handle 1D loophead right now.

    if stmt.head == :loophead
        assert(length(stmt.args) == 3)
        loop_id    = toLHSVar(stmt.args[1])
        loop_start = stmt.args[2]
        loop_end   = stmt.args[3]
        @dprintln(3, "recreateFromLoophead ", loop_id, " ", loop_start, " ", loop_end)
    else
        assert(length(stmt.args) == 1)
        loop_id = toLHSVar(stmt.args[1])
        if haskey(LoopEndDict, loop_id)
            append!(new_body, LoopEndDict[loop_id])
        else
            throw(string("loophead information missing for loop ", loop_id))
        end
        return nothing
    end

    assert(isa(loop_id, LHSVar))
    uniq = get_unique_num()

    if VERSION >= v"0.5.0-dev+4449"
        label_head = next_label(state)
        label_end  = next_label(state)

        num_vars = 6
        parfor_nest_level = 1
        loop_nest_level = 1

        steprange_last_var  = Symbol(string("#recreate_steprange_last_", uniq, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 0))
        steprange_first_var = Symbol(string("#recreate_steprange_first_", uniq, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 1))
        steprange_step_var  = Symbol(string("#recreate_steprange_step_", uniq, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 2))
        recreate_temp_var   = Symbol(string("#recreate_temp_", uniq, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 3))
        recreate_ssa5_var   = Symbol(string("#recreate_ssa5_", uniq, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 4))
        recreate_ssa6_var   = Symbol(string("#recreate_ssa6_", uniq, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 5))

        steprange_last_rhsvar  = CompilerTools.LambdaHandling.addLocalVariable(steprange_last_var  , Int64, ISASSIGNED, newLambdaVarInfo)
        steprange_first_rhsvar = CompilerTools.LambdaHandling.addLocalVariable(steprange_first_var , Int64, ISASSIGNED, newLambdaVarInfo)
        steprange_step_rhsvar  = CompilerTools.LambdaHandling.addLocalVariable(steprange_step_var  , Int64, ISASSIGNED, newLambdaVarInfo)
        recreate_temp_rhsvar   = CompilerTools.LambdaHandling.addLocalVariable(recreate_temp_var   , Int64, ISASSIGNED, newLambdaVarInfo)
        recreate_ssa5_rhsvar   = CompilerTools.LambdaHandling.addLocalVariable(recreate_ssa5_var   , Int64, ISASSIGNED, newLambdaVarInfo)
        recreate_ssa6_rhsvar   = CompilerTools.LambdaHandling.addLocalVariable(recreate_ssa6_var   , Int64, ISASSIGNED, newLambdaVarInfo)

        steprange_last_lhsvar  = toLHSVar(steprange_last_rhsvar)
        steprange_first_lhsvar = toLHSVar(steprange_first_rhsvar)
        steprange_step_lhsvar  = toLHSVar(steprange_step_rhsvar)
        recreate_temp_lhsvar   = toLHSVar(recreate_temp_rhsvar)
        recreate_ssa5_lhsvar   = toLHSVar(recreate_ssa5_rhsvar)
        recreate_ssa6_lhsvar   = toLHSVar(recreate_ssa6_rhsvar)

        push!(new_body, mk_assignment_expr(deepcopy(steprange_last_lhsvar), Expr(:call, GlobalRef(Base,:steprange_last), deepcopy(loop_start), 1, deepcopy(loop_end)), newLambdaVarInfo))
        push!(new_body, mk_assignment_expr(deepcopy(steprange_first_lhsvar), deepcopy(loop_start), newLambdaVarInfo))
        push!(new_body, mk_assignment_expr(deepcopy(steprange_step_lhsvar), 1, newLambdaVarInfo))
        push!(new_body, mk_assignment_expr(deepcopy(recreate_temp_lhsvar), deepcopy(steprange_first_rhsvar), newLambdaVarInfo))

#        push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "steprange_last_var = ", deepcopy(steprange_last_rhsvar)))
#        push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "steprange_first_var = ", deepcopy(steprange_first_rhsvar)))
#        push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "steprange_step_var = ", deepcopy(steprange_step_rhsvar)))

        push!(new_body, LabelNode(label_head))

#        push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "after label_head"))

        push!(new_body, mk_gotoifnot_expr(
               boxOrNot(GlobalRef(Base, :Bool),
                   Expr(:call, GlobalRef(Base, :not_int),
                           Expr(:call, GlobalRef(Base, :or_int),
                                   Expr(:call, GlobalRef(Base, :and_int),
                                           Expr(:call, GlobalRef(Base, :not_int),
                                               Expr(:call, GlobalRef(Base, :(===)), deepcopy(steprange_first_rhsvar), deepcopy(steprange_last_rhsvar))
                                           ),
                                           Expr(:call, GlobalRef(Base, :not_int),
                                               Expr(:call, GlobalRef(Base, :(===)),
                                                   Expr(:call, GlobalRef(Base, :slt_int),
                                                       0,
                                                       deepcopy(steprange_step_rhsvar)
                                                   ),
                                                   Expr(:call, GlobalRef(Base, :slt_int),
                                                       deepcopy(steprange_first_rhsvar),
                                                       deepcopy(steprange_last_rhsvar)
                                                   )
                                               )
                                           )
                                       ),
                               Expr(:call, GlobalRef(Base, :(===)), deepcopy(recreate_temp_rhsvar),
                                   boxOrNot(Int64, Expr(:call, GlobalRef(Base, :add_int), deepcopy(steprange_last_rhsvar), deepcopy(steprange_step_rhsvar)))
                               )
                           )
                   )
               )
               , label_end))

        push!(new_body, mk_assignment_expr(deepcopy(recreate_ssa5_lhsvar), deepcopy(recreate_temp_rhsvar), newLambdaVarInfo))
        push!(new_body, mk_assignment_expr(deepcopy(recreate_ssa6_lhsvar), boxOrNot(Int64, Expr(:call, GlobalRef(Base, :add_int), deepcopy(recreate_temp_rhsvar), deepcopy(steprange_step_rhsvar))), newLambdaVarInfo))
        push!(new_body, mk_assignment_expr(CompilerTools.LambdaHandling.toLHSVar(deepcopy(loop_id), newLambdaVarInfo), deepcopy(recreate_ssa5_rhsvar), newLambdaVarInfo))
        push!(new_body, mk_assignment_expr(deepcopy(recreate_temp_lhsvar), deepcopy(recreate_ssa6_rhsvar), newLambdaVarInfo))

        for_loop_end = Any[]

        push!(for_loop_end, GotoNode(label_head))
        push!(for_loop_end, LabelNode(label_end))

        LoopEndDict[loop_id] = for_loop_end
    else

        label_after_first_unless   = next_label(state)
    #    label_before_second_unless = next_label(state)
        label_after_second_unless  = next_label(state)
    #    label_last                 = next_label(state)

        gensym2_var = string("#recreate_gensym2_", uniq, "_", 0)
        gensym2_sym = Symbol(gensym2_var)
        gensym0_var = string("#recreate_gensym0_", uniq, "_", 1)
        gensym0_sym = Symbol(gensym0_var)
        pound_s1_var = string("#recreate_pound_s1_", uniq, "_", 2)
        pound_s1_sym = Symbol(pound_s1_var)
        gensym3_var = string("#recreate_gensym3_", uniq, "_", 3)
        gensym3_sym = Symbol(gensym3_var)
        gensym4_var = string("#recreate_gensym4_", uniq, "_", 4)
        gensym4_sym = Symbol(gensym4_var)
        gensym2_rhsvar  = CompilerTools.LambdaHandling.addLocalVariable(gensym2_sym, Int64, ISASSIGNED, newLambdaVarInfo)
        gensym0_rhsvar  = CompilerTools.LambdaHandling.addLocalVariable(gensym0_sym, StepRange{Int64,Int64}, ISASSIGNED, newLambdaVarInfo)
        pound_s1_rhsvar = CompilerTools.LambdaHandling.addLocalVariable(pound_s1_sym, Int64, ISASSIGNED, newLambdaVarInfo)
        gensym3_rhsvar  = CompilerTools.LambdaHandling.addLocalVariable(gensym3_sym, Int64, ISASSIGNED, newLambdaVarInfo)
        gensym4_rhsvar  = CompilerTools.LambdaHandling.addLocalVariable(gensym4_sym, Int64, ISASSIGNED, newLambdaVarInfo)

        gensym2_lhsvar  = toLHSVar(gensym2_rhsvar)
        gensym0_lhsvar  = toLHSVar(gensym0_rhsvar)
        pound_s1_lhsvar = toLHSVar(pound_s1_rhsvar)
        gensym3_lhsvar  = toLHSVar(gensym3_rhsvar)
        gensym4_lhsvar  = toLHSVar(gensym4_rhsvar)

        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "ranges = ", toRHSVar(:ranges, pir_range_actual, state.LambdaVarInfo)))
        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.lower = ", this_nest.lower))
        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.step  = ", this_nest.step))
        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.upper = ", this_nest.upper))

        push!(new_body, mk_assignment_expr(deepcopy(gensym2_lhsvar), TypedExpr(Int64, :call, GlobalRef(Base,:steprange_last), loop_start, 1, loop_end), state))
        push!(new_body, mk_assignment_expr(deepcopy(gensym0_lhsvar), TypedExpr(StepRange{Int64,Int64}, :new, StepRange{Int64,Int64}, loop_start, 1, deepcopy(gensym2_rhsvar)), state))
        push!(new_body, mk_assignment_expr(deepcopy(pound_s1_lhsvar), TypedExpr(Int64, :call, GlobalRef(Base, :getfield), deepcopy(gensym0_rhsvar), QuoteNode(:start)), state))
        push!(new_body, mk_gotoifnot_expr(TypedExpr(Bool, :call, GlobalRef(ParallelAccelerator.ParallelIR, :first_unless), deepcopy(gensym0_rhsvar), deepcopy(pound_s1_rhsvar)), label_after_second_unless))
        push!(new_body, LabelNode(label_after_first_unless))

    #       push!(new_body, Expr(:call, GlobalRef(Base,:println), GlobalRef(Base,:STDOUT), " in label_after_first_unless section"))

        push!(new_body, mk_assignment_expr(deepcopy(gensym3_lhsvar), deepcopy(pound_s1_rhsvar), state))
        push!(new_body, mk_assignment_expr(deepcopy(gensym4_lhsvar), TypedExpr(Int64, :call, GlobalRef(ParallelAccelerator.ParallelIR, :assign_gs4), deepcopy(gensym0_rhsvar), deepcopy(pound_s1_rhsvar)), state))
        push!(new_body, mk_assignment_expr(deepcopy(loop_id), deepcopy(gensym3_rhsvar), state))
        push!(new_body, mk_assignment_expr(deepcopy(pound_s1_lhsvar), deepcopy(gensym4_rhsvar), state))

        for_loop_end = Any[]

    #    push!(for_loop_end, LabelNode(label_before_second_unless))
        push!(for_loop_end, mk_gotoifnot_expr(TypedExpr(Bool, :call, GlobalRef(ParallelAccelerator.ParallelIR, :second_unless), deepcopy(gensym0_rhsvar), deepcopy(pound_s1_rhsvar)), label_after_first_unless))
        push!(for_loop_end, LabelNode(label_after_second_unless))
    #    push!(for_loop_end, LabelNode(label_last))

        LoopEndDict[loop_id] = for_loop_end
    end

    return nothing
end

#function got_here_1(x)
#println("got here! ", x)
#end

run_nested_task_directly = true

function red_alloc(elem_type, length, value)
    elem_type[deepcopy(value) for i = 1:length]
end

function process_cur_task(cur_task::TaskInfo, new_body, state)
    @dprintln(3,"process_cur_task = ", cur_task.function_sym, " type = ", typeof(cur_task.function_sym), " function = ", state.function_name)
    range_var = string(Base.function_name(cur_task.task_func),"_range_var")
    range_sym = Symbol(range_var)
    range_rhsvar = CompilerTools.LambdaHandling.addLocalVariable(range_sym, pir_range_actual, CompilerTools.LambdaHandling.ISASSIGNED, state.LambdaVarInfo)
    range_lhsvar = toLHSVar(range_rhsvar)

    run_task_directly = haskey(ENV,"PROSPECT_RUN_TASK_DIRECTLY") || (state.in_nested && run_nested_task_directly)
    @dprintln(3,"run_task_directly = ", run_task_directly, " state.in_nested = ", state.in_nested, " run_nested_task_directly = ", run_nested_task_directly)

    if run_task_directly
        @dprintln(3,"Inserting call to task function directly ", range_sym, " ", range_rhsvar)
    elseif haskey(ENV,"PROSPECT_CALL_ISF")
        @dprintln(3,"Inserting call to isf ", range_sym, " ", range_rhsvar)
#    elseif haskey(ENV,"PROSPECT_CALL_DYNAMIC_CHECK")
#        @dprintln(3,"Inserting call dynamic check ", range_sym, " ", range_rhsvar)
    else
        @dprintln(3,"Inserting call dynamic check ", range_sym, " ", range_rhsvar)
#        @dprintln(3,"Inserting call to jl_threading_run in function ", state.function_name, " ", range_sym, " ", range_rhsvar)
    end
    @dprintln(3,"Cur LambdaVarInfo = ", state.LambdaVarInfo)

    in_len  = length(cur_task.input_symbols)
    mod_len = length(cur_task.modified_inputs)
    io_len  = length(cur_task.io_symbols)
    red_len = length(cur_task.reduction_vars)
    @dprintln(3, "inputs, modifieds, io_sym, reductions = ", cur_task.input_symbols, " ", cur_task.modified_inputs, " ", cur_task.io_symbols, " ", cur_task.reduction_vars)

    dims = length(cur_task.loopNests)
    if dims > 0
        @dprintln(3,"dims > 0")
        assert(dims <= 3)
        cstr_params = Any[]
        for l = 1:dims
            # Note that loopNest is outer-dimension first
            # Should this still be outer-dimension first?  FIX FIX FIX
            push!(cstr_params, cur_task.loopNests[dims - l + 1].lower)
            push!(cstr_params, cur_task.loopNests[dims - l + 1].upper)
        end
        @dprintln(3, "cstr_params = ", cstr_params)
        cstr_expr = ParallelAccelerator.ParallelIR.pir_range_actual
        #cstr_expr = mk_parallelir_ref(:pir_range_actual, Any)
        whole_range_expr = mk_assignment_expr(deepcopy(range_lhsvar), TypedExpr(pir_range_actual, :call, cstr_expr, cstr_params...), state)
        @dprintln(3,"whole_range_expr = ", whole_range_expr)
        push!(new_body, whole_range_expr)

        #    push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "whole_range = ", range_rhsvar))

        real_args_build = Any[]
        args_type = Expr(:tuple)
        red_var_rhsvars = Any[]

        # Fill in the arg metadata.
        for l = 1:in_len
            @dprintln(3,cur_task.input_symbols[l].name, " ", typeof(cur_task.input_symbols[l].name))
            push!(real_args_build, cur_task.input_symbols[l].name)
            push!(args_type.args,  cur_task.input_symbols[l].typ)
        end
        for l = 1:mod_len
            @dprintln(3,cur_task.modified_inputs[l].name, " ", typeof(cur_task.modified_inputs[l].name))
            push!(real_args_build, cur_task.modified_inputs[l].name)
            push!(args_type.args,  cur_task.modified_inputs[l].typ)
        end
        for l = 1:io_len
            @dprintln(3,cur_task.io_symbols[l].name, " ", typeof(cur_task.io_symbols[l].name))
            push!(real_args_build, cur_task.io_symbols[l].name)
            push!(args_type.args,  cur_task.io_symbols[l].typ)
        end
        for l = 1:red_len
            push!(red_var_rhsvars, cur_task.reduction_vars[l].name)
            push!(real_args_build, cur_task.reduction_vars[l].name)
            push!(args_type.args,  cur_task.reduction_vars[l].typ)
        end

        @dprintln(3,"task_func = ", cur_task.task_func)
        #@dprintln(3,"whole_iteration_range = ", whole_iteration_range)
        @dprintln(3,"real_args_build = ", real_args_build)

        if cur_task.ret_types != Void
            red_array_lhsvar = createVarForTask(cur_task, "_red_array", Array{Any,1}, state)
#            if haskey(ENV,"PROSPECT_CALL_DYNAMIC_CHECK")
                push!(new_body, mk_assignment_expr(deepcopy(red_array_lhsvar), Expr(:call, GlobalRef(ParallelAccelerator.ParallelIR, :red_alloc), Any, TypedExpr(Int, :call, GlobalRef(Base.Threads, :nthreads)), Expr(:call, GlobalRef(Core, :tuple), red_var_rhsvars...)), state))
#            else
#                push!(new_body, mk_assignment_expr(deepcopy(red_array_lhsvar), mk_alloc_array_expr(Any, Array{Any,1}, TypedExpr(Int, :call, GlobalRef(Base.Threads, :nthreads))), state))
#            end

            red_output_tuple_typ = Tuple{cur_task.ret_types...}
            red_output_tuple_lhsvar = createVarForTask(cur_task, "_red_output_tuple", red_output_tuple_typ, state)
        end

        #push!(new_body, Expr(:call, GlobalRef(ParallelAccelerator.ParallelIR, :got_here_1), TypedExpr(pir_range_actual, :call, cstr_expr, cstr_params...)))
        task_globalref = GlobalRef(ParallelAccelerator.ParallelIR, Base.function_name(cur_task.task_func))
        #push!(new_body, Expr(:call, GlobalRef(ParallelAccelerator.ParallelIR, :got_here_1), range_lhsvar))
        if run_task_directly
          if cur_task.ret_types != Void
            insert_task_expr = TypedExpr(Void,
                                         :call,
                                         task_globalref,
                                         deepcopy(range_lhsvar),
                                         real_args_build...,
                                         deepcopy(red_array_lhsvar))
          else
            insert_task_expr = TypedExpr(Void,
                                         :call,
                                         task_globalref,
                                         deepcopy(range_lhsvar),
                                         real_args_build...)
          end
        elseif haskey(ENV,"PROSPECT_CALL_ISF")
          if cur_task.ret_types != Void
            insert_task_expr = TypedExpr(Void,
                                         :call,
                                         ParallelAccelerator.ParallelIR.isf,
                                         task_globalref,
                                         deepcopy(range_lhsvar),
                                         real_args_build...,
                                         deepcopy(red_array_lhsvar))
          else
            insert_task_expr = TypedExpr(Void,
                                         :call,
                                         ParallelAccelerator.ParallelIR.isf,
                                         task_globalref,
                                         deepcopy(range_lhsvar),
                                         real_args_build...)
          end
        #elseif haskey(ENV,"PROSPECT_CALL_DYNAMIC_CHECK")
        else
          if cur_task.ret_types != Void
            insert_task_expr = TypedExpr(Void,
                                         :call,
                                         ParallelAccelerator.ParallelIR.dynamic_check,
                                         task_globalref,
                                         deepcopy(range_lhsvar),
                                         real_args_build...,
                                         deepcopy(red_array_lhsvar))
          else
            insert_task_expr = TypedExpr(Void,
                                         :call,
                                         ParallelAccelerator.ParallelIR.dynamic_check,
                                         task_globalref,
                                         deepcopy(range_lhsvar),
                                         real_args_build...)
          end
#############   This code won't prevent nested parallelism.  When nested parallelism works then this code can be re-enabled.
#############   In the meantime, in the above section, we use a function that uses a global to make sure that only function is
#############   run in threaded mode at a time.
#        else
#            tup_var = string(Base.function_name(cur_task.task_func),"_tup_var")
#            tup_sym = Symbol(tup_var)
#            tup_rhsvar = CompilerTools.LambdaHandling.addLocalVariable(tup_sym, SimpleVector, CompilerTools.LambdaHandling.ISASSIGNED, state.LambdaVarInfo)
#            tup_lhsvar = toLHSVar(tup_rhsvar)
#
#            if cur_task.ret_types != Void
#                push!(new_body, mk_assignment_expr(deepcopy(tup_lhsvar), mk_svec_expr(GlobalRef(ParallelAccelerator.ParallelIR, :isf), task_globalref, deepcopy(range_lhsvar), real_args_build..., deepcopy(red_array_lhsvar)), state))
#            else
#                push!(new_body, mk_assignment_expr(deepcopy(tup_lhsvar), mk_svec_expr(GlobalRef(ParallelAccelerator.ParallelIR, :isf), task_globalref, deepcopy(range_lhsvar), real_args_build...), state))
#            end
#
#            svec_args = mk_svec_expr(Any)
#            insert_task_expr = TypedExpr(Void,
#                                         :call,
#                                         GlobalRef(Core, :ccall),
#                                         QuoteNode(:jl_threading_run),
#                                         GlobalRef(Base.Threads,:Void),
#                                         svec_args,
#                                         tup_lhsvar, 0)
        end
        push!(new_body, insert_task_expr)

        if cur_task.ret_types != Void
            if run_task_directly || haskey(ENV,"PROSPECT_CALL_ISF")
                for l = 1:red_len
                    push!(new_body,
                      mk_assignment_expr(
                        deepcopy(cur_task.reduction_vars[l].name),
                        TypedExpr(cur_task.ret_types[l], :call, GlobalRef(Core, :typeassert),
                            Expr(:call, GlobalRef(Base, :getfield), TypedExpr(red_output_tuple_typ, :call, GlobalRef(Base, :arrayref), deepcopy(red_array_lhsvar), TypedExpr(Int, :call, GlobalRef(Base.Threads,:threadid))), l), cur_task.ret_types[l]), state))
#                    push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "index threadid of red_array = ", deepcopy(cur_task.reduction_vars[l].name)))
                end
            else
                red_loop_index_lhsvar = createVarForTask(cur_task, "_reduction_loop_index", Int, state)
                unique_node_id = get_unique_num()
                #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "whole reduction array = ", deepcopy(red_array_lhsvar)))

                # After the jl_threading_run call, we store the first element of the reduction array into their destinations.
                for l = 1:red_len
                    push!(new_body, mk_assignment_expr(
                                       deepcopy(cur_task.reduction_vars[l].name),
                                       TypedExpr(cur_task.ret_types[l], :call, GlobalRef(Core, :typeassert),
                                         Expr(:call, GlobalRef(Base, :getfield), TypedExpr(red_output_tuple_typ, :call, GlobalRef(Base, :arrayref), deepcopy(red_array_lhsvar), 1), l), cur_task.ret_types[l]), state))
#                    push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "index 1 of red_array = ", deepcopy(cur_task.reduction_vars[l].name)))
                end

                push!(new_body, Expr(:loophead, deepcopy(red_loop_index_lhsvar), 2, TypedExpr(Int, :call, GlobalRef(Base, :arraylen), deepcopy(red_array_lhsvar))))
                for l = 1:red_len
                    @dprintln(3, "reduction_vars[l] = ", cur_task.reduction_vars[l])
                    reduction_code = callDelayedFuncWith(cur_task.join_func[l].reductionFunc,
                                                    deepcopy(cur_task.reduction_vars[l].name),
                                                    TypedExpr(cur_task.ret_types[l], :call, GlobalRef(Core, :typeassert),
                                                      Expr(:call, GlobalRef(Base, :getfield), TypedExpr(red_output_tuple_typ, :call, GlobalRef(Base, :arrayref), deepcopy(red_array_lhsvar), deepcopy(red_loop_index_lhsvar)), l), cur_task.ret_types[l]))
                    @dprintln(3, "Adding reduction code for reduction variable ", l, " with ", length(reduction_code), " statements.")

                    for stmt in reduction_code
                        @dprintln(3, "reduction stmt = ", stmt)
                        if isBareParfor(stmt)
                            @dprintln(3, "reduction stmt is parfor so recreating loops")
                            recreateLoops(new_body, stmt.args[1], state, state.LambdaVarInfo)
                        else
                            @dprintln(3, "reduction stmt is normal")
                            push!(new_body, stmt)
                        end
                    end
#                    push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "next reduction from red_array = ", deepcopy(cur_task.reduction_vars[l].name)))
                end

                push!(new_body, Expr(:loopend, deepcopy(red_loop_index_lhsvar)))

                @dprintln(3,"LambdaVarInfo after reduction code = ", state.LambdaVarInfo)
            end
        end
    else
        throw(string("insert sequential task not implemented yet"))
    end
end

function createVarForTask(cur_task, suffix, typ, state)
    new_var_name = string(Base.function_name(cur_task.task_func),suffix)
    new_var_sym = Symbol(new_var_name)
    new_var_rhsvar = CompilerTools.LambdaHandling.addLocalVariable(new_var_sym, typ, CompilerTools.LambdaHandling.ISASSIGNED, state.LambdaVarInfo)
    return toLHSVar(new_var_rhsvar)
end

function run_reduction_func(red_array :: Array{Any,1}, join_func)
    @dprintln(3,"run_reduction_fun ", red_array, " ", length(red_array), " ", join_func, " type = ", typeof(join_func))
    if length(red_array) == 1
        return red_array[1]
    else
        return foldl((x,y) -> join_func(x,y), red_array)
    end
end

function process_cur_task(cur_task::Any, new_body, state)
    push!(new_body, cur_task)
end

# TOP_LEVEL
# sequence of expressions
# ast = [ expr, ... ]
function top_level_from_exprs(ast::Array{Any,1}, depth, state)
    # Perform lowering from domain nodes to parfors.
    main_proc_start = time_ns()
    body = top_level_mk_body(ast, depth, state)
    @dprintln(3,"Body after lowering but before fusion.")
    printBody(3, body)
    if print_times
    @dprintln(1,"Lowering time = ", ns_to_sec(time_ns() - main_proc_start))
    end

    # Perform fusion.
    fusion_start = time_ns()
    body = fusion_pass(body, depth, state)
    @dprintln(3,"Body after first pass before task graph creation.")
    printBody(3, body)
    if print_times
    @dprintln(1,"Fusion time = ", ns_to_sec(time_ns() - fusion_start))
    end

    # TASK GRAPH

    expand_start = time_ns()

    body = top_level_expand_pre(body, state)

    if print_times
    @dprintln(1,"Expanding parfors time = ", ns_to_sec(time_ns() - expand_start))
    end

    @dprintln(3,"expanded_body = ")
    @dprintln(3,"lvi = ", state.LambdaVarInfo)
    printBody(3, body)

    @dprintln(3,"LambdaVarInfo = ", state.LambdaVarInfo)
    new_lives = CompilerTools.LivenessAnalysis.from_lambda(state.LambdaVarInfo, body, pir_live_cb, state.LambdaVarInfo)
    @dprintln(1,"Starting loop analysis.")
    loop_info = CompilerTools.Loops.compute_dom_loops(new_lives.cfg)
    @dprintln(1,"Finished loop analysis.")

    if hoist_allocation == 1
        body = hoistAllocation(body, new_lives, loop_info, state)
        new_lives = CompilerTools.LivenessAnalysis.from_lambda(state.LambdaVarInfo, body, pir_live_cb, state.LambdaVarInfo)
        @dprintln(1,"Starting loop analysis again.")
        loop_info = CompilerTools.Loops.compute_dom_loops(new_lives.cfg)
        @dprintln(1,"Finished loop analysis.")
    end

    @dprintln(3,"new_lives = ", new_lives)
    @dprintln(3,"loop_info = ", loop_info)

    if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE || ParallelAccelerator.getTaskMode() > 0 || run_as_task()
        @dprintln(3, "Entering top_level_mk_task_graph.")
        body = top_level_mk_task_graph(body, state, new_lives, loop_info)
    end  # end of task graph formation section

    @dprintln(3,"body after mk_task_graph = ")
    printBody(3, body)

    expanded_body = Any[]

    max_label   = getMaxLabel(0, body)
    LoopEndDict = Dict{LHSVar,Array{Any,1}}()

    for i = 1:length(body)
        @dprintln(3,"Flatten index ", i, " ", body[i], " type = ", typeof(body[i]))
            # Convert loophead and loopend into Julia loops.
        if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE &&
            typeof(body[i]) == Expr &&
            (body[i].head == :loophead || body[i].head == :loopend)
            recreateFromLoophead(expanded_body, body[i], LoopEndDict, state, state.LambdaVarInfo)
        else
            push!(expanded_body, body[i])
        end
    end

    body = expanded_body
    # CompilerTools.OptFramework.cleanupBodyLabels(expanded_body)
    return body
end
