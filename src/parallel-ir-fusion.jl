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
Returns true if any variable in the collection "vars" is used in any statement whose top level number is in "top_level_numbers".
We use expr_state "state" to get the block liveness information from which we use "def" and "use" to determine if a variable usage is present.
"""
function isSymbolsUsed(vars :: Dict{LHSVar,LHSVar}, top_level_numbers :: Array{Int,1}, state)
    @dprintln(3,"isSymbolsUsed: vars = ", vars, " typeof(vars) = ", typeof(vars), " top_level_numbers = ", top_level_numbers)
    bl = state.block_lives

    for i in top_level_numbers
        tls = CompilerTools.LivenessAnalysis.find_top_number(i, bl)
        assert(tls != nothing)

        for v in vars
            if in(v[1], tls.def) || in(v[2], tls.def)
                @dprintln(3, "isSymbolsUsed: ", v, " defined in statement ", i)
                return true
            elseif in(v[1], tls.use) || in(v[2], tls.use)
                @dprintln(3, "isSymbolsUsed: ", v, " used in statement ", i)
                return true
            end
        end
    end

    @dprintln(3, "isSymbolsUsed: ", vars, " not used in statements ", top_level_numbers)
    return false
end

function compareIndex(sn1 :: RHSVar, sn2 :: RHSVar)
    return toLHSVar(sn1) == toLHSVar(sn2)
end

function compareIndex(s1::ANY, s2::ANY)
    return s1 == s2
end

function create_merged_output_from_map(output_map, unique_id, state, sym_to_type, loweredAliasMap)
    @dprintln(3,"create_merged_output_from_map, output_map = ", output_map, " sym_to_type = ", sym_to_type)
    # If there are no outputs then return nothing.
    if length(output_map) == 0
        return (nothing, [], true, nothing, [])
    end

    # If there is only one output then all we need is the symbol to return.
    if length(output_map) == 1
        for i in output_map
            new_lhs = toRHSVar(i[1], sym_to_type[i[1]], state.LambdaVarInfo)
            new_rhs = toRHSVar(getAliasMap(loweredAliasMap, i[2]), sym_to_type[i[2]], state.LambdaVarInfo)
            return (new_lhs, [new_lhs], true, [new_rhs])
        end
    end

    lhs_order = RHSVar[]
    rhs_order = RHSVar[]
    for i in output_map
        @dprintln(3,"Working on output ", i, " lhs = ", i[1], " rhs = ", i[2])
        lhs_rhsvar = toRHSVar(i[1], sym_to_type[i[1]], state.LambdaVarInfo)
        push!(lhs_order, lhs_rhsvar)
        @dprintln(3,"lhs_rhsvar = ", lhs_rhsvar)
        rhs_rhsvar = toRHSVar(getAliasMap(loweredAliasMap, i[2]), sym_to_type[i[2]], state.LambdaVarInfo)
        push!(rhs_order, rhs_rhsvar)
        @dprintln(3,"rhs_rhsvar = ", rhs_rhsvar)
    end
    num_map = length(lhs_order)

    # Multiple outputs.

    # First, form the type of the tuple for those multiple outputs.
    tt = Expr(:tuple)
    for i = 1:num_map
        push!(tt.args, CompilerTools.LambdaHandling.getType(rhs_order[i], state.LambdaVarInfo))
    end
    temp_type = eval(tt)

    ( createRetTupleType(lhs_order, unique_id, state), lhs_order, false, rhs_order )
end

"""
Test whether we can fuse the two most recent parfor statements and if so to perform that fusion.
"""
function fuse(body, body_index, cur::Expr, state)
    global fuse_limit
    prev = body[body_index]

    # Handle the debugging case where we want to limit the amount of parfor fusion to a certain number.
    if fuse_limit == 0
        return false
    end
    if fuse_limit > 0
        global fuse_limit = fuse_limit - 1
    end

    @dprintln(2, "Testing if fusion is possible.")
    prev_parfor = getParforNode(prev)
    cur_parfor  = getParforNode(cur)

    sym_to_type = Dict{LHSVar, DataType}()

    @dprintln(2, "prev = ", prev)
    @dprintln(2, "cur = ", cur)
    @dprintln(2, "prev.typ = ", prev.typ)
    @dprintln(2, "cur.typ = ", cur.typ)

    prev_assignment = isAssignmentNode(prev)

    out_correlation = getParforCorrelation(prev_parfor, state)
    if out_correlation == nothing
        @dprintln(3, "Fusion will not happen because out_correlation = nothing")
        return false
    end
    in_correlation  = getParforCorrelation(cur_parfor, state)
    if in_correlation == nothing
        @dprintln(3, "Fusion will not happen because in_correlation = nothing")
        return false
    end
    @dprintln(3,"Fusion correlations ", out_correlation, " ", in_correlation)

    is_prev_multi = isFusionAssignment(prev)
    is_cur_multi  = isFusionAssignment(cur)

    prev_num_dims = length(prev_parfor.loopNests)
    cur_num_dims  = length(cur_parfor.loopNests)

    map_prev_lhs_post, map_prev_lhs_reduction = createMapLhsToParfor(prev, prev_parfor, is_prev_multi, sym_to_type, state)
    map_prev_lhs_all = merge(map_prev_lhs_post, map_prev_lhs_reduction)
    map_cur_lhs_post,  map_cur_lhs_reduction  = createMapLhsToParfor(cur,  cur_parfor,  is_cur_multi,  sym_to_type, state)
    map_cur_lhs_all  = merge(map_cur_lhs_post, map_cur_lhs_reduction)
    prev_output_arrays = collect(values(map_prev_lhs_post))
    prev_output_reduce = collect(values(map_prev_lhs_reduction))
    cur_output_arrays  = collect(values(map_cur_lhs_post))
    cur_output_reduce  = collect(values(map_cur_lhs_reduction))

    merge!(prev_parfor.array_aliases, map_prev_lhs_post)
    assert(length(cur_parfor.array_aliases) == 0)

    loweredAliasMap = createLoweredAliasMap(prev_parfor.array_aliases)

    @dprintln(3, "map_prev_lhs_post = ", map_prev_lhs_post)
    @dprintln(3, "map_prev_lhs_reduction = ", map_prev_lhs_reduction)
    @dprintln(3, "map_cur_lhs_post = ", map_cur_lhs_post)
    @dprintln(3, "map_cur_lhs_reduction = ", map_cur_lhs_reduction)
    @dprintln(3, "sym_to_type = ", sym_to_type)
    reduction_var_used = isSymbolsUsed(map_prev_lhs_reduction, cur_parfor.top_level_number, state)
    @dprintln(3, "reduction_var_used = ", reduction_var_used)
    prev_iei = iterations_equals_inputs(prev_parfor)
    cur_iei  = iterations_equals_inputs(cur_parfor)
    @dprintln(3, "iterations_equals_inputs prev and cur = ", prev_iei, " ", cur_iei)
    @dprintln(3, "loweredAliasMap = ", loweredAliasMap)
    @dprintln(3, "cur_parfor.arrays_read_past_index = ", cur_parfor.arrays_read_past_index)
    arrays_non_simply_indexed_in_cur_that_access_prev_output = intersect(cur_parfor.arrays_read_past_index, prev_output_arrays)
    @dprintln(3, "arrays_non_simply_indexed_in_cur_that_access_prev_output = ", arrays_non_simply_indexed_in_cur_that_access_prev_output)
    @dprintln(3, "prev_parfor.arrays_written_past_index = ", prev_parfor.arrays_written_past_index)
    # Compute which scalars and arrays are ever read or written by the body of the parfor
    prev_rws = CompilerTools.ReadWriteSet.from_exprs(prev_parfor.body, pir_rws_cb, state.LambdaVarInfo, state.LambdaVarInfo)
    # Compute which scalars and arrays are ever read or written by the body of the parfor
    cur_rws = CompilerTools.ReadWriteSet.from_exprs(cur_parfor.body, pir_rws_cb, state.LambdaVarInfo, state.LambdaVarInfo)
    cur_accessed = CompilerTools.ReadWriteSet.getArraysAccessed(cur_rws)
    arrays_non_simply_indexed_in_prev_that_are_read_in_cur = intersect(prev_parfor.arrays_written_past_index, cur_accessed)
    @dprintln(3, "arrays_non_simply_indexed_in_prev_that_are_read_in_cur = ", arrays_non_simply_indexed_in_prev_that_are_read_in_cur)
    if !isempty(arrays_non_simply_indexed_in_cur_that_access_prev_output)
        @dprintln(1, "Fusion won't happen because the second parfor accesses some array created by the first parfor in a non-simple way.")
    end
    if !isempty(arrays_non_simply_indexed_in_prev_that_are_read_in_cur)
        @dprintln(1, "Fusion won't happen because the first parfor wrote to some array index in a non-simple way that the second parfor needs to access.")
    end

    @dprintln(3, "Fusion prev_rws = ", prev_rws)
    @dprintln(3, "Fusion cur_rws = ", cur_rws)
    prev_write_set_array = Set(collect(keys(prev_rws.writeSet.arrays)))
    #cur_read_set_array   = Set(collect(keys(cur_rws.readSet.arrays)))
    #cur_read_with_aliases = getAllAliases(cur_read_set_array, map_prev_lhs_post)
    @dprintln(3, "Fusion prev_write = ", prev_write_set_array)
    #write_in_prev_read_in_cur = intersect(prev_write_with_aliases, cur_read_with_aliases)
    #@dprintln(3, "Fusion intersection = ", write_in_prev_read_in_cur)
    found_array_access_problem = false
    @dprintln(3, "cur_parfor.loopNests = ", cur_parfor.loopNests)
    for cur_read_array in collect(keys(cur_rws.readSet.arrays))
        with_aliases = getAllAliases(Set{LHSVar}([cur_read_array]), map_prev_lhs_post)
        @dprintln(3, "Checking array ", cur_read_array, " with aliases = ", with_aliases)
        if isempty(intersect(prev_write_set_array, with_aliases))
            continue
        end
        @dprintln(3, "Array  ", cur_read_array, " produced by prev parfor.")
        accesses = cur_rws.readSet.arrays[cur_read_array]
        for index_expr in accesses
            @dprintln(3, "Checking usage  ", index_expr)
            for index = 1:length(index_expr)
                lnindex = length(cur_parfor.loopNests) + 1 - index
                if lnindex < 1 || lnindex > length(cur_parfor.loopNests)
                    @dprintln(3, "Found index expression not just using this dimension's loop index variable")
                    found_array_access_problem = true
                    continue
                end
                good_index = cur_parfor.loopNests[lnindex].indexVariable
                @dprintln(3, "Good index here = ", good_index, " cur_index = ", index_expr[index])
                if !compareIndex(good_index, index_expr[index])
                    @dprintln(3, "Found index expression not just using this dimension's loop index variable")
                    found_array_access_problem = true
                end
            end
        end
    end

    @dprintln(3, (prev_iei ,
        cur_iei  ,
        out_correlation == in_correlation ,
        !reduction_var_used ,
        isempty(arrays_non_simply_indexed_in_cur_that_access_prev_output) ,
        isempty(arrays_non_simply_indexed_in_prev_that_are_read_in_cur) ,
        (prev_num_dims == cur_num_dims) ,
        !found_array_access_problem        ))

    if  prev_iei &&
        cur_iei  &&
        out_correlation == in_correlation &&
        !reduction_var_used &&
        isempty(arrays_non_simply_indexed_in_cur_that_access_prev_output) &&
        isempty(arrays_non_simply_indexed_in_prev_that_are_read_in_cur) &&
        (prev_num_dims == cur_num_dims) &&
        !found_array_access_problem

        @dprintln(3, "Fusion will happen here.")

        # Get the top-level statement for the previous parfor.
        prev_stmt_live_first = CompilerTools.LivenessAnalysis.find_top_number(prev_parfor.top_level_number[1], state.block_lives)
        assert(prev_stmt_live_first != nothing)
        @dprintln(2,"Prev parfor first = ", prev_stmt_live_first)
        prev_stmt_live_last = CompilerTools.LivenessAnalysis.find_top_number(prev_parfor.top_level_number[end], state.block_lives)
        assert(prev_stmt_live_last != nothing)
        @dprintln(2,"Prev parfor last = ", prev_stmt_live_last)

        # Get the top-level statement for the current parfor.
        assert(length(cur_parfor.top_level_number) == 1)
        cur_stmt_live  = CompilerTools.LivenessAnalysis.find_top_number(cur_parfor.top_level_number[1],  state.block_lives)
        assert(cur_stmt_live != nothing)
        @dprintln(2,"Cur parfor = ", cur_stmt_live)

        # Get the variables live after the previous parfor.
        live_in_prev = prev_stmt_live_first.live_in
        # def_prev     = prev_stmt_live_first.def
        def_prev = Set{LHSVar}()
        # The "def" for a fused parfor is to a first approximation the union of the "def" of each statement in the parfor.
        # Some variables will be eliminated by fusion so we follow this first approximation up by an intersection with the
        # live_out of the last statement of the previous parfor.
        for prev_parfor_stmt in prev_parfor.top_level_number
            constituent_stmt_live = CompilerTools.LivenessAnalysis.find_top_number(prev_parfor_stmt, state.block_lives)
            def_prev = union(def_prev, constituent_stmt_live.def)
        end
        def_prev = intersect(def_prev, prev_stmt_live_last.live_out)

        @dprintln(2,"live_in_prev = ", live_in_prev, " def_prev = ", def_prev)

        # Get the variables live after the previous parfor.
        live_out_prev = prev_stmt_live_last.live_out
        @dprintln(2,"live_out_prev = ", live_out_prev)

        # Get the live variables into the current parfor.
        live_in_cur   = cur_stmt_live.live_in
        def_cur       = cur_stmt_live.def
        @dprintln(2,"live_in_cur = ", live_in_cur, " def_cur = ", def_cur)

        # Get the variables live after the current parfor.
        live_out_cur  = cur_stmt_live.live_out
        @dprintln(2,"live_out_cur = ", live_out_cur)

        surviving_def = intersect(union(def_prev, def_cur), live_out_cur)

        new_in_prev = setdiff(live_out_prev, live_in_prev)
        new_in_cur  = setdiff(live_out_cur,  live_in_cur)
        @dprintln(3,"new_in_prev = ", new_in_prev)
        @dprintln(3,"new_in_cur = ", new_in_cur)
        @dprintln(3,"surviving_def = ", surviving_def)

        # The things that come in live to cur but leave it dead.
        not_used_after_cur = setdiff(live_out_prev, live_out_cur)
        @dprintln(2,"not_used_after_cur = ", not_used_after_cur)

        live_out_prev_aliases = getAllAliases(live_out_prev, prev_parfor.array_aliases)
        live_out_cur_aliases  = getAllAliases(live_out_cur, prev_parfor.array_aliases)
        @dprintln(2, "live_out_prev_aliases = ", live_out_prev_aliases)
        @dprintln(2, "live_out_cur_aliases  = ", live_out_cur_aliases)
        not_used_after_cur_with_aliases = setdiff(live_out_prev_aliases, live_out_cur_aliases)
        @dprintln(2,"not_used_after_cur_with_aliases = ", not_used_after_cur_with_aliases)

        unique_id = get_unique_num() # prev_parfor.unique_id

        # Output of this parfor are the new things in the current parfor plus the new things in the previous parfor
        # that don't die during the current parfor.
        output_map = Dict{LHSVar, LHSVar}()
        for i in map_prev_lhs_all
            if !in(i[1], not_used_after_cur)
                output_map[i[1]] = i[2]
            end
        end
        for i in map_cur_lhs_all
            output_map[i[1]] = i[2]
        end

        new_aliases = Dict{LHSVar, LHSVar}()
        for i in map_prev_lhs_post
            if !in(i[1], not_used_after_cur)
                new_aliases[i[1]] = i[2]
            end
        end

        @dprintln(3,"output_map = ", output_map)
        @dprintln(3,"new_aliases = ", new_aliases)

        # return code 2 if there is no output in the fused parfor
        # this means the parfor is dead and should be removed
        if length(surviving_def)==0
            @dprintln(1,"No output for the fused parfor so the parfor is dead and is being removed.")
            return 2;
        end

        first_arraylen = getFirstArrayLens(prev_parfor, prev_num_dims, state)
        @dprintln(3,"first_arraylen = ", first_arraylen)

        # Merge each part of the two parfor nodes.

        # loopNests - nothing needs to be done to the loopNests
        # But we use them to establish a mapping between corresponding indices in the two parfors.
        # Then, we use that map to convert indices in the second parfor to the corresponding ones in the first parfor.
        index_map = Dict{LHSVar, LHSVar}()
        assert(length(prev_parfor.loopNests) == length(cur_parfor.loopNests))
        for i = 1:length(prev_parfor.loopNests)
            index_map[toLHSVar(cur_parfor.loopNests[i].indexVariable)] = toLHSVar(prev_parfor.loopNests[i].indexVariable)
        end

        @dprintln(3,"array_aliases before merge ", prev_parfor.array_aliases)
        for i in map_cur_lhs_post
            from = i[1]
            to   = i[2]
            prev_parfor.array_aliases[from] = to
        end
        @dprintln(3,"array_aliases after merge ", prev_parfor.array_aliases)

        # postParFor - can merge everything but the last entry in the postParFor's.
        # The last entries themselves need to be merged extending the tuple if the prev parfor's output stays live and
        # just keeping the cur parfor output if the prev parfor output dies.
        (new_lhs, all_rets, single, output_items) = create_merged_output_from_map(output_map, unique_id, state, sym_to_type, loweredAliasMap)
        @dprintln(3,"new_lhs = ", new_lhs)
        @dprintln(3,"all_rets = ", all_rets)
        @dprintln(3,"single = ", single)
        @dprintln(3,"output_items = ", output_items)
        output_items_set = live_out_cur
        output_items_with_aliases = getAllAliases(output_items_set, prev_parfor.array_aliases)

        @dprintln(3,"output_items_set = ", output_items_set)
        @dprintln(3,"output_items_with_aliases = ", output_items_with_aliases)

        escaping_sets = Set{LHSVar}()
        # Create a dictionary of arrays to the last variable containing the array's value at the current index space.
        save_body = prev_parfor.body
        arrayset_dict = Dict{LHSVar, RHSVar}()
        for i = 1:length(save_body)
            x = save_body[i]
            if isArraysetCall(x)
                # Here we have a call to arrayset.
                array_name = x.args[2]
                value      = x.args[3]
                array_var  = toLHSVar(array_name)
                assert(isa(array_name, RHSVar))
                assert(isa(value, RHSVar))
                array_var = toLHSVar(array_name)
                arrayset_dict[array_var] = value
                if getDesc(array_var, state.LambdaVarInfo) & ISCAPTURED == ISCAPTURED
                    push!(escaping_sets, array_var)
                end
            elseif typeof(x) == Expr && x.head == :(=)
                lhs = x.args[1]
                rhs = x.args[2]
                #assert(isa(lhs, RHSVar))
                if isArrayrefCall(rhs)
                    array_name = rhs.args[2]
                    assert(isa(array_name, RHSVar))
                    array_var = toLHSVar(array_name)
#                    arrayset_dict[array_var] = lhs
                    if getDesc(array_var, state.LambdaVarInfo) & ISCAPTURED == ISCAPTURED
                        push!(escaping_sets, array_var)
                    end
                end
            end
        end
        @dprintln(3,"arrayset_dict = ", arrayset_dict)

        # Extend the arrayset_dict to include the lhs of the prev parfor.
        for i in map_prev_lhs_post
            lhs_sym = i[1]
            rhs_sym = i[2]
            arrayset_dict[lhs_sym] = arrayset_dict[rhs_sym]
            if getDesc(lhs_sym, state.LambdaVarInfo) & ISCAPTURED == ISCAPTURED
                push!(escaping_sets, lhs_sym)
            end
        end
        @dprintln(3,"arrayset_dict = ", arrayset_dict)
        # make sure escaping sets include aliases and prev's output_items
        escaping_sets = union(escaping_sets, not_used_after_cur_with_aliases)
        for (k,v) in prev_parfor.array_aliases
            if in(k, escaping_sets)
                push!(escaping_sets, toLHSVar(v))
            end
        end
        @dprintln(3,"escaping_sets = ", escaping_sets)

        # body - Append cur body to prev body but replace arrayset's in prev with a temp variable
        # and replace arrayref's in cur with the same temp.
        arrays_set_in_cur_body = Set{LHSVar}()
        # Convert the cur_parfor body.
        new_cur_body = map(x -> substitute_cur_body(x, arrayset_dict, index_map, arrays_set_in_cur_body, loweredAliasMap, state), cur_parfor.body)
        arrays_set_in_cur_body_with_aliases = getAllAliases(arrays_set_in_cur_body, prev_parfor.array_aliases)
        @dprintln(3,"arrays_set_in_cur_body = ", arrays_set_in_cur_body)
        @dprintln(3,"arrays_set_in_cur_body_with_aliases = ", arrays_set_in_cur_body_with_aliases)
        combined = union(arrays_set_in_cur_body_with_aliases, not_used_after_cur_with_aliases)
        @dprintln(2,"combined = ", combined)

        prev_parfor.body = Any[]
        for i = 1:length(save_body)
            new_body_line = substitute_arrayset(save_body[i], combined, output_items_with_aliases, escaping_sets)
            @dprintln(3,"new_body_line = ", new_body_line)
            if new_body_line != nothing
                push!(prev_parfor.body, new_body_line)
            end
        end
        append!(prev_parfor.body, new_cur_body)
        @dprintln(2,"New body = ")
        printBody(2, prev_parfor.body)

        # preParFor - Append cur preParFor to prev parParFor but eliminate array creation from
        # prevParFor where the array is in allocs_to_eliminate.
        removed_allocs = Set{LHSVar}()
        @dprintln(3, "preParFor cur prestatements are array input = ", parforArrayInput(prev_parfor))
        filtered_cur_pre = filter(x -> !is_eliminated_arraylen(x), cur_parfor.preParFor)
        prev_parfor.preParFor = [ filter(x -> !is_eliminated_allocation_map(x, output_items_with_aliases, removed_allocs), prev_parfor.preParFor);
                                  map(x -> substitute_arraylen(x, first_arraylen) , filtered_cur_pre)]
#                                  parforArrayInput(prev_parfor) ? map(x -> substitute_arraylen(x, first_arraylen) , filtered_cur_pre) : cur_parfor.preParFor]
        @dprintln(3,"removed_allocs = ", removed_allocs)
        filter!( x -> !is_eliminated_arraysize(x, removed_allocs, prev_parfor.array_aliases), prev_parfor.preParFor)
        @dprintln(2,"New preParFor = ", prev_parfor.preParFor)

        append!(prev_parfor.hoisted, cur_parfor.hoisted)
        @dprintln(2,"New hoisted = ", prev_parfor.hoisted)

        # if allocation of an array is removed, arrayset should be removed as well since the array doesn't exist anymore
        @dprintln(4,"prev_parfor.body before removing dead arrayset: ", prev_parfor.body)
        filter!( x -> !is_dead_arrayset(x, removed_allocs), prev_parfor.body)
        @dprintln(4,"prev_parfor.body after_ removing dead arrayset: ", prev_parfor.body)

        # reductions - a simple append with the caveat that you can't fuse parfor where the first has a reduction that the second one uses
        # need to check this caveat above.
        append!(prev_parfor.reductions, cur_parfor.reductions)
        @dprintln(2,"New reductions = ", prev_parfor.reductions)

        prev_parfor.postParFor = [ prev_parfor.postParFor[1:end-1]; cur_parfor.postParFor[1:end-1]]
        @dprintln(2,"New initial postParFor = ", prev_parfor.postParFor)
        push!(prev_parfor.postParFor, oneIfOnly(output_items))
        @dprintln(2,"New postParFor = ", prev_parfor.postParFor, " typeof(postParFor) = ", typeof(prev_parfor.postParFor), " ", typeof(prev_parfor.postParFor[end]))

        # original_domain_nodes - simple append
        append!(prev_parfor.original_domain_nodes, cur_parfor.original_domain_nodes)
        @dprintln(2,"New domain nodes = ", prev_parfor.original_domain_nodes)

        # top_level_number - what to do here? is this right?
        push!(prev_parfor.top_level_number, cur_parfor.top_level_number[1])

        @dprintln(3,"New lhs = ", new_lhs)
        if prev_assignment
            # The prev parfor was of the form "var = parfor(...)".
            if new_lhs != nothing
                @dprintln(2,"prev was assignment and is staying an assignment")
                # The new lhs is not empty and so this is the normal case where "prev" stays an assignment expression and we update types here and if necessary FusionSentinel.
                prev.args[1] = new_lhs
                prev.typ = getType(new_lhs, state.LambdaVarInfo)
                prev.args[2].typ = prev.typ
                # Strip off a previous FusionSentinel() if it exists in the expression.
                prev.args = prev.args[1:2]
                if !single
                    push!(prev.args, FusionSentinel())
                    append!(prev.args, all_rets)
                    @dprintln(3,"New multiple ret prev args is ", prev.args)
                end
            else
                @dprintln(2,"prev was assignment and is becoming bare")
                # The new lhs is empty and so we need to transform "prev" into an assignment expression.
                body[body_index] = TypedExpr(nothing, :parfor, prev_parfor)
            end
        else
            # The prev parfor was a bare-parfor (not in the context of an assignment).
            if new_lhs != nothing
                @dprintln(2,"prev was bare and is becoming an assignment")
                # The new lhs is not empty so the fused parfor will not be bare and "prev" needs to become an assignment expression.
                body[body_index] = mk_assignment_expr(new_lhs, prev, state)
                prev = body[body_index]
                prev.args[2].typ = CompilerTools.LambdaHandling.getType(new_lhs, state.LambdaVarInfo)
                if !single
                    push!(prev.args, FusionSentinel())
                    append!(prev.args, all_rets)
                    @dprintln(3,"New multiple ret prev args is ", prev.args)
                end
            else
                @dprintln(2,"prev was bare and is staying bare")
            end
        end

        @dprintln(2,"New parfor = ", prev_parfor)

        #throw(string("not finished"))

        return 1
    else
        @dprintln(3, "Fusion could not happen here.")
    end

    return 0

    false
end

"""
Performs the mmap to mmap! phase.
If the arguments of a mmap dies aftewards, and is not aliased, then
we can safely change the mmap to mmap!.
"""
function mmapToMmap!(LambdaVarInfo, body::Expr, lives, uniqSet)
    assert(body.head == :body)
    # For each statement in the body.
    for i =1:length(body.args)
        expr = body.args[i]
        # If the statement is an assignment.
        if isa(expr, Expr) && (expr.head === :(=))
            lhs = toLHSVar(expr.args[1], LambdaVarInfo)
            rhs = expr.args[2]
            # right now assume all
            assert(isa(lhs, RHSVar))
            lhsTyp = CompilerTools.LambdaHandling.getType(lhs, LambdaVarInfo)
            # If the right-hand side is an mmap.
            if isa(rhs, Expr) && (rhs.head === :mmap)
                args = rhs.args[1]
                tls = CompilerTools.LivenessAnalysis.find_top_number(i, lives)
                assert(tls != nothing)
                assert(CompilerTools.LivenessAnalysis.isDef(lhs, tls))
                @dprintln(4, "mmap lhs=", lhs, " args=", args, " live_out = ", tls.live_out)
                reuse = nothing
                j = 0
                # Find some input array to the mmap that is dead after this statement.
                while j < length(args)
                    j = j + 1
                    if isa(args[j], RHSVar)
                        v = toLHSVar(args[j])
                        if isa(v, LHSVar) && !in(v, tls.live_out) && in(v, uniqSet) &&
                            CompilerTools.LambdaHandling.getType(v, LambdaVarInfo) == lhsTyp
                            reuse = v  # Found a dying symbol.
                            break
                        end
                    end
                end
                # If we found a dying array whose space we can reuse.
                if !(reuse === nothing)
                    rhs.head = :mmap!   # Change to mmap!
                    @dprintln(2, "mmapToMMap!: successfully reuse ", reuse, " for ", lhs)
                    if j != 1  # The array to reuse has to be first.  If it isn't already then reorder the args to make it so.
                        # swap j-th and 1st argument
                        rhs.args[1] = DomainIR.arraySwap(rhs.args[1], 1, j)
                        rhs.args[2] = DomainIR.lambdaSwapArg(rhs.args[2], 1, j)
                        @dprintln(3, "mmapToMMap!: after swap, ", lhs, " = ", rhs)
                    end
                end
            end
        end
    end
end


"""
For every basic block, try to push domain IR statements down and non-domain IR statements up so that domain nodes
are next to each other and can be fused.
"""
function maxFusion(bl :: CompilerTools.LivenessAnalysis.BlockLiveness)
    # We will try to optimize the order in each basic block.
    for bb in collect(values(bl.basic_blocks))
        if false
            # One approach to this problem is to create a dependency graph and then use that to calculate valid reorderings
            # that maximize fusion.  We may still want to switch to this at some point but it is more complicated and the
            # simpler approach works for now.

            # Start with a pseudo-statement corresponding to the variables live-in to this basic block.
            livein     = StatementWithDeps(nothing)
            # The variable "last_write" is a dictionary mapping a symbol to the last statement to have written that variable.
            last_write = Dict{Symbol, StatementWithDeps}()
            # For each symbol live-in to this basic block, add it to "last_write".
            for def in bb.live_in
                assert(typeof(def) == Symbol)
                last_write[def] = livein
            end

            # Create the dependency graph.
            for i = 1:length(bb.statements)
                # Get the i'th statement in this basic block.
                stmt    = bb.statements[i]
                # Create an object to associate this statement with its dependencies.
                new_swd = StatementWithDeps(stmt)

                # Establish dependency links from prior statements to this one based on the last statement to have written each symbol used in this statement.
                for use in stmt.use
                    assert(typeof(use) == Symbol)
                    assert(haskey(last_write, use))
                    push!(last_write[use].deps, new_swd)
                end
                # Update last_write with any symbols def in this statement.
                for def in stmt.def
                    assert(typeof(def) == Symbol)
                    last_write[def] = new_swd
                end
            end

            topo_sort = StatementWithDeps[]
            dfsVisit(livein, 1, topo_sort)
        else
            @dprintln(3, "Doing maxFusion in block ", bb)
            # A bubble-sort style of coalescing domain nodes together in the AST.
            earliest_parfor = 1
            found_change = true

            # While the lastest pass over the AST created some change, keep searching for more interchanges that can coalesce domain nodes.
            while found_change
                found_change = false

                # earliest_parfor is an optimization that we don't have to scan every statement in the block but only those statements from
                # the first parfor to the last statement in the block.
                i = earliest_parfor
                # @dprintln(3,"bb.statements = ", bb.statements)
                earliest_parfor = length(bb.statements)

                while i < length(bb.statements)
                    cur  = bb.statements[i]
                    next = bb.statements[i+1]
                    cannot_move_next = mustRemainLastStatementInBlock(next.tls.expr)
                    @dprintln(3,"maxFusion cur = ", cur.tls.expr)
                    @dprintln(3,"maxFusion next = ", next.tls.expr)
                    @dprintln(3,"maxFusion next.use = ", next.use)
                    cur_domain_node  = isDomainNode(cur.tls.expr)
                    next_domain_node = isDomainNode(next.tls.expr)
                    intersection     = intersect(cur.def, next.use)
                    @dprintln(3,"cur_domain_node = ", cur_domain_node, " next_domain_node = ", next_domain_node, " intersection = ", intersection)
                    if cur_domain_node && !cannot_move_next
                        if !next_domain_node
                          if isempty(intersection)
                            # If the current statement is a domain node and the next staterment isn't and we are allowed to move the next node
                            # in the block and the next statement doesn't use anything produced by this statement then we can switch the order of
                            # the current and next statement.
                            @dprintln(3,"bubbling domain node down")
                            (bb.statements[i], bb.statements[i+1]) = (bb.statements[i+1], bb.statements[i])
                            (bb.cfgbb.statements[i], bb.cfgbb.statements[i+1]) = (bb.cfgbb.statements[i+1], bb.cfgbb.statements[i])
                            (bb.cfgbb.statements[i].index, bb.cfgbb.statements[i+1].index) = (bb.cfgbb.statements[i+1].index, bb.cfgbb.statements[i].index)
                            found_change = true
                          else # intersection is not empty, but let's check if it is no longer live
                            if length(intersection) == 1 && isAssignmentNode(cur.tls.expr) && isAssignmentNode(next.tls.expr)
                                lhsvar = toLHSVar(cur.tls.expr.args[1])
                                tmpvar = toLHSVar(next.tls.expr.args[2])
                                if lhsvar == tmpvar && in(tmpvar, intersection) && !in(tmpvar, next.live_out)
                                    @dprintln(3, "next is assignment, and RHS is not live afterwards")
                                    (bb.statements[i], bb.statements[i+1]) = (bb.statements[i+1], bb.statements[i])
                                    (bb.cfgbb.statements[i], bb.cfgbb.statements[i+1]) = (bb.cfgbb.statements[i+1], bb.cfgbb.statements[i])
                                    (bb.cfgbb.statements[i].index, bb.cfgbb.statements[i+1].index) = (bb.cfgbb.statements[i+1].index, bb.cfgbb.statements[i].index)
                                    found_change = true
                                    bb.statements[i+1].def = bb.statements[i].def
                                    bb.statements[i+1].tls.expr.args[1] = bb.statements[i].tls.expr.args[1]
                                    bb.statements[i].tls.expr = nothing
                                    @dprintln(3, "after rewrite, cur = ", bb.statements[i+1])
                                end
                            end
                          end
                        else
                            if i < earliest_parfor
                                earliest_parfor = i
                            end
                        end
                    end
                    i += 1
                end
            end
        end
    end
end
