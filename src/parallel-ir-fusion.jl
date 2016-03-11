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

    sym_to_type = Dict{SymGen, DataType}()

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
    cur_accessed = CompilerTools.ReadWriteSet.getArraysAccessed(cur_parfor.rws)
    arrays_non_simply_indexed_in_prev_that_are_read_in_cur = intersect(prev_parfor.arrays_written_past_index, cur_accessed)
    @dprintln(3, "arrays_non_simply_indexed_in_prev_that_are_read_in_cur = ", arrays_non_simply_indexed_in_prev_that_are_read_in_cur)
    if !isempty(arrays_non_simply_indexed_in_cur_that_access_prev_output)
        @dprintln(1, "Fusion won't happen because the second parfor accesses some array created by the first parfor in a non-simple way.")
    end
    if !isempty(arrays_non_simply_indexed_in_prev_that_are_read_in_cur)
        @dprintln(1, "Fusion won't happen because the first parfor wrote to some array index in a non-simple way that the second parfor needs to access.")
    end

    if  prev_iei &&
        cur_iei  &&
        out_correlation == in_correlation &&
        !reduction_var_used &&
        isempty(arrays_non_simply_indexed_in_cur_that_access_prev_output) &&
        isempty(arrays_non_simply_indexed_in_prev_that_are_read_in_cur) &&
        (prev_num_dims == cur_num_dims)

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
        def_prev = Set{SymGen}()
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

        unique_id = prev_parfor.unique_id

        # Output of this parfor are the new things in the current parfor plus the new things in the previous parfor
        # that don't die during the current parfor.
        output_map = Dict{SymGen, SymGen}()
        for i in map_prev_lhs_all
            if !in(i[1], not_used_after_cur)
                output_map[i[1]] = i[2]
            end
        end
        for i in map_cur_lhs_all
            output_map[i[1]] = i[2]
        end

        new_aliases = Dict{SymGen, SymGen}()
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

        if parforArrayInput(prev_parfor)
            first_arraylen = getFirstArrayLens(prev_parfor.preParFor, prev_num_dims)
        end

        # Merge each part of the two parfor nodes.

        # loopNests - nothing needs to be done to the loopNests
        # But we use them to establish a mapping between corresponding indices in the two parfors.
        # Then, we use that map to convert indices in the second parfor to the corresponding ones in the first parfor.
        index_map = Dict{SymGen, SymGen}()
        assert(length(prev_parfor.loopNests) == length(cur_parfor.loopNests))
        for i = 1:length(prev_parfor.loopNests)
            index_map[cur_parfor.loopNests[i].indexVariable.name] = prev_parfor.loopNests[i].indexVariable.name
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

        # Create a dictionary of arrays to the last variable containing the array's value at the current index space.
        save_body = prev_parfor.body
        arrayset_dict = Dict{SymGen, SymNodeGen}()
        for i = 1:length(save_body)
            x = save_body[i]
            if isArraysetCall(x)
                # Here we have a call to arrayset.
                array_name = x.args[2]
                value      = x.args[3]
                assert(isa(array_name, SymNodeGen))
                assert(isa(value, SymNodeGen))
                arrayset_dict[toSymGen(array_name)] = value
            elseif typeof(x) == Expr && x.head == :(=)
                lhs = x.args[1]
                rhs = x.args[2]
                #assert(isa(lhs, SymNodeGen))
                if isArrayrefCall(rhs)
                    array_name = rhs.args[2]
                    assert(isa(array_name, SymNodeGen))
                    arrayset_dict[toSymGen(array_name)] = lhs
                end
            end
        end
        @dprintln(3,"arrayset_dict = ", arrayset_dict)

        # Extend the arrayset_dict to include the lhs of the prev parfor.
        for i in map_prev_lhs_post
            lhs_sym = i[1]
            rhs_sym = i[2]
            arrayset_dict[lhs_sym] = arrayset_dict[rhs_sym]
        end
        @dprintln(3,"arrayset_dict = ", arrayset_dict)

        # body - Append cur body to prev body but replace arrayset's in prev with a temp variable
        # and replace arrayref's in cur with the same temp.
        arrays_set_in_cur_body = Set{SymGen}()
        # Convert the cur_parfor body.
        new_cur_body = map(x -> substitute_cur_body(x, arrayset_dict, index_map, arrays_set_in_cur_body, loweredAliasMap, state), cur_parfor.body)
        arrays_set_in_cur_body_with_aliases = getAllAliases(arrays_set_in_cur_body, prev_parfor.array_aliases)
        @dprintln(3,"arrays_set_in_cur_body = ", arrays_set_in_cur_body)
        @dprintln(3,"arrays_set_in_cur_body_with_aliases = ", arrays_set_in_cur_body_with_aliases)
        combined = union(arrays_set_in_cur_body_with_aliases, not_used_after_cur_with_aliases)
        @dprintln(2,"combined = ", combined)

        prev_parfor.body = Any[]
        for i = 1:length(save_body)
            new_body_line = substitute_arrayset(save_body[i], combined, output_items_with_aliases)
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
        prev_parfor.preParFor = [ filter(x -> !is_eliminated_allocation_map(x, output_items_with_aliases), prev_parfor.preParFor);
                                  parforArrayInput(prev_parfor) ? map(x -> substitute_arraylen(x,first_arraylen) , filter(x -> !is_eliminated_arraylen(x), cur_parfor.preParFor)) : cur_parfor.preParFor]
        @dprintln(2,"New preParFor = ", prev_parfor.preParFor)

        # if allocation of an array is removed, arrayset should be removed as well since the array doesn't exist anymore
        @dprintln(4,"prev_parfor.body before removing dead arrayset: ", prev_parfor.body)
        filter!( x -> !is_dead_arrayset(x, output_items_with_aliases), prev_parfor.body)
        @dprintln(4,"prev_parfor.body after_ removing dead arrayset: ", prev_parfor.body)

        # reductions - a simple append with the caveat that you can't fuse parfor where the first has a reduction that the second one uses
        # need to check this caveat above.
        append!(prev_parfor.reductions, cur_parfor.reductions)
        @dprintln(2,"New reductions = ", prev_parfor.reductions)

        prev_parfor.postParFor = [ prev_parfor.postParFor[1:end-1]; cur_parfor.postParFor[1:end-1]]
        push!(prev_parfor.postParFor, oneIfOnly(output_items))
        @dprintln(2,"New postParFor = ", prev_parfor.postParFor, " typeof(postParFor) = ", typeof(prev_parfor.postParFor), " ", typeof(prev_parfor.postParFor[end]))

        # original_domain_nodes - simple append
        append!(prev_parfor.original_domain_nodes, cur_parfor.original_domain_nodes)
        @dprintln(2,"New domain nodes = ", prev_parfor.original_domain_nodes)

        # top_level_number - what to do here? is this right?
        push!(prev_parfor.top_level_number, cur_parfor.top_level_number[1])

        # rws
        prev_parfor.rws = CompilerTools.ReadWriteSet.from_exprs(prev_parfor.body, pir_rws_cb, state.LambdaVarInfo)

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


# mmapInline() helper function
function modify!(dict, lhs, i)
    if haskey(dict, lhs)
        push!(dict[lhs], i)
    else
        dict[lhs] = Int[i]
    end
end

# function that inlines from src mmap into dst mmap
function inline!(src, dst, lhs)
    args = dst.args[1]
    pos = 0
    for k = 1:length(args)
        s = args[k]
        if isa(s, SymbolNode) s = s.name end
        if s == lhs
            pos = k
            break
        end
    end
    @assert pos>0 "mmapInline(): position of mmap output not found"
    args = vcat(args[1:pos-1], args[pos+1:end], src.args[1])
    f = src.args[2]
    assert(length(f.outputs)==1)
    dst.args[1] = args
    g = dst.args[2]
    inputs = g.inputs
    g_inps = length(inputs) 
    inputs = vcat(inputs[1:pos-1], inputs[pos+1:end], f.inputs)
    linfo = g.linfo
    # add escaping variables from f into g, since mergeLambdaVarInfo only deals
    # with nesting lambda, but not parallel ones.
    for (v, d) in f.linfo.escaping_defs
        if !CompilerTools.LambdaHandling.isEscapingVariable(v, linfo)
            CompilerTools.LambdaHandling.addEscapingVariable(d, linfo)
        end
    end
    # gensym_map = CompilerTools.LambdaHandling.mergeLambdaVarInfo(linfo, f.linfo)
    tmp_t = f.outputs[1]
    dst.args[2] = DomainLambda(inputs, g.outputs, 
    (linfo, args) -> begin
        fb = f.genBody(linfo, args[g_inps:end])
        tmp_v = CompilerTools.LambdaHandling.addGenSym(tmp_t, linfo)
        expr = TypedExpr(tmp_t, :(=), tmp_v, fb[end].args[1])
        gb = g.genBody(linfo, vcat(args[1:pos-1], [tmp_v], args[pos:g_inps-1]))
        return [fb[1:end-1]; [expr]; gb]
    end, linfo)
    DomainIR.mmapRemoveDupArg!(dst)
end

# mmapInline() helper function
function eliminateShapeAssert(dict, lhs, body)
    if haskey(dict, lhs)
        for k in dict[lhs]
            @dprintln(3, "MI: eliminate shape assert at line ", k)
            body.args[k] = nothing
        end
    end
end

# mmapInline() helper function
function check_used(defs, usedAt, shapeAssertAt, expr::Expr,i)
    if is(expr.head, :assertEqShape)
        # handle assertEqShape separately, do not consider them
        # as valid references
        for arg in expr.args
            s = isa(arg, SymbolNode) ? arg.name : arg 
            if isa(s, Symbol) || isa(s, GenSym)
                modify!(shapeAssertAt, s, i)
            end
        end
    else
        for arg in expr.args
            check_used(defs, usedAt, shapeAssertAt, arg,i)
        end
    end
end

function check_used(defs, usedAt, shapeAssertAt, expr::Union{Symbol,GenSym},i)
    if haskey(usedAt, expr) # already used? remove from defs
        delete!(defs, expr)
    else
        usedAt[expr] = i
    end
end

function check_used(defs, usedAt, shapeAssertAt, expr::SymbolNode,i)
    if haskey(usedAt, expr.name) # already used? remove from defs
        @dprintln(3, "MI: def ", expr.name, " removed due to multi-use")
        delete!(defs, expr.name)
    else
        usedAt[expr.name] = i
    end 
end

function check_used(defs, usedAt, shapeAssertAt, expr::Union{Array, Tuple},i)
    for e in expr
        check_used(defs, usedAt, shapeAssertAt, e, i)
    end
end

function check_used(defs, usedAt, shapeAssertAt, expr::Any,i)
end

# mmapInline() helper function
function mmapInline_refs(expr::Expr, i::Int, uniqSet, defs::Dict{Union{Symbol, GenSym}, Int}, usedAt::Dict{Union{Symbol, GenSym}, Int}, 
    modifiedAt::Dict{Union{Symbol, GenSym}, Array{Int}}, shapeAssertAt::Dict{Union{Symbol, GenSym}, Array{Int}})
    head = expr.head
    # record usedAt, and reject those used more than once
    # record definition
    if is(head, :(=))
        lhs = expr.args[1]
        rhs = expr.args[2]
        check_used(defs, usedAt, shapeAssertAt, rhs,i)
        assert(isa(lhs, Symbol) || isa(lhs, GenSym))
        modify!(modifiedAt, lhs, i)
        if isa(rhs, Expr) && is(rhs.head, :mmap) && in(lhs, uniqSet)
            ok = true
            for j in rhs.args[1]
                if isa(j, SymbolNode) j = j.name end
                if !in(j, uniqSet) # being conservative by demanding arguments not being aliased
                    ok=false
                    break
                end
            end
            if (ok) defs[lhs] = i end
            @dprintln(3, "MI: def for ", lhs, " ok=", ok, " defs=", defs)
        end
    else
        check_used(defs, usedAt, shapeAssertAt, expr,i)
    end
    # check if args may be modified in place
    if is(head, :mmap!)
        for j in 1:length(expr.args[2].outputs)
            v = expr.args[1][j]
            if isa(v, SymbolNode)
                v = v.name
            end
            if isa(v, Symbol) || isa(v, GenSym)
                modify!(modifiedAt, v, i)
            end
        end
    elseif is(head, :stencil!)
        krnStat = expr.args[1]
        iterations = expr.args[2]
        bufs = expr.args[3]
        for k in krnStat.modified
            s = bufs[k]
            if isa(s, SymbolNode) s = s.name end
            modify!(modifiedAt, s, i)
        end
        if !((isa(iterations, Number) && iterations == 1) || krnStat.rotateNum == 0)
            for j in 1:min(krnStat.rotateNum, length(bufs))
                s = bufs[j]
                if isa(s, SymbolNode) s = s.name end
                modify!(modifiedAt, s, i)
            end
        end
    end
end

# mmapInline() helper function
function mmapInline_refs(expr::Any, i::Int, uniqSet, defs::Dict{Union{Symbol, GenSym}, Int}, usedAt::Dict{Union{Symbol, GenSym}, Int}, 
                                modifiedAt::Dict{Union{Symbol, GenSym}, Array{Int}}, shapeAssertAt::Dict{Union{Symbol, GenSym}, Array{Int}})
    check_used(defs, usedAt, shapeAssertAt, expr,i)
end


"""
# If a definition of a mmap is only used once and not aliased, it can be inlined into its
# use side as long as its dependencies have not been changed.
# FIXME: is the implementation still correct when branches are present?
"""
function mmapInline(ast::Expr, lives, uniqSet)
    body = ast.args[3]
    defs = Dict{Union{Symbol, GenSym}, Int}()
    usedAt = Dict{Union{Symbol, GenSym}, Int}()
    modifiedAt = Dict{Union{Symbol, GenSym}, Array{Int}}()
    shapeAssertAt = Dict{Union{Symbol, GenSym}, Array{Int}}()
    assert(isa(body, Expr) && is(body.head, :body))

    # first do a loop to see which def is only referenced once
    for i =1:length(body.args)
        mmapInline_refs(body.args[i], i, uniqSet, defs, usedAt, modifiedAt, shapeAssertAt)
    end
    @dprintln(3, "MI: defs = ", defs)
    # print those that are used once

    revdefs = Dict()
    for (lhs, i) in defs
        revdefs[i] = lhs
    end
    for i in sort!([keys(revdefs)...])
        lhs = revdefs[i]
        expr = body.args[i] # must be an assignment
        src = expr.args[2]  # must be a mmap
        # @dprintln(3, "MI: def of ", lhs)
        if haskey(usedAt, lhs)
            j = usedAt[lhs]
            ok = true
            # @dprintln(3, "MI: def of ", lhs, " at line ", i, " used by line ", j)
            for v in src.args[1]
                @assert isa(v,Symbol) || isa(v,GenSym) || isa(v,SymbolNode) "mmapInline(): Arguments of mmap should be Symbol or GenSym or SymbolNode."
                if isa(v, SymbolNode) v = v.name end
                if haskey(modifiedAt, v)
                    for k in modifiedAt[v]
                        if k >= i && k <= j
                            ok = false
                            break
                        end
                    end
                end
                if (!ok) break end
            end 
            if (!ok) continue end
            @dprintln(3, "MI: found mmap: ", lhs, " can be inlined into defintion of line ", j)
            dst = body.args[j]
            if isa(dst, Expr) && is(dst.head, :(=))
                dst = dst.args[2]
            end
            if isa(dst, Expr) && is(dst.head, :mmap) && in(lhs, dst.args[1])
                # inline mmap into mmap
                inline!(src, dst, lhs)
                body.args[i] = nothing
                eliminateShapeAssert(shapeAssertAt, lhs, body)
                @dprintln(3, "MI: result: ", body.args[j])
            elseif isa(dst, Expr) && is(dst.head, :mmap!) && in(lhs, dst.args[1])
                s = dst.args[1][1]
                if isa(s, SymbolNode) s = s.name end
                if s == lhs 
                    # when lhs is the inplace array that dst operates on
                    # change dst to mmap
                    inline!(src, dst, lhs)
                    dst.head = :mmap
                else
                    # otherwise just normal inline
                    inline!(src, dst, lhs)
                end
                body.args[i] = nothing
                eliminateShapeAssert(shapeAssertAt, lhs, body)
                # inline mmap into mmap!
                @dprintln(3, "MI: result: ", body.args[j])
            else
                # otherwise ignore, e.g., when dst is some simple assignments.
            end
        end
    end
end


"""
Performs the mmap to mmap! phase.
If the arguments of a mmap dies aftewards, and is not aliased, then
we can safely change the mmap to mmap!.
"""
function mmapToMmap!(ast, lives, uniqSet)
    LambdaVarInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaVarInfo(ast)
    body = ast.args[3]
    assert(isa(body, Expr) && is(body.head, :body))
    # For each statement in the body.
    for i =1:length(body.args)
        expr = body.args[i]
        # If the statement is an assignment.
        if isa(expr, Expr) && is(expr.head, :(=))
            lhs = expr.args[1]
            rhs = expr.args[2]
            # right now assume all
            assert(isa(lhs, SymAllGen))
            lhsTyp = CompilerTools.LambdaHandling.getType(lhs, LambdaVarInfo) 
            # If the right-hand side is an mmap.
            if isa(rhs, Expr) && is(rhs.head, :mmap)
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
                    v = args[j]
                    v = isa(v, SymbolNode) ? v.name : v
                    if (isa(v, Symbol) || isa(v, GenSym)) && !in(v, tls.live_out) && in(v, uniqSet) &&
                        CompilerTools.LambdaHandling.getType(v, LambdaVarInfo) == lhsTyp
                        reuse = v  # Found a dying symbol.
                        break
                    end
                end
                # If we found a dying array whose space we can reuse.
                if !is(reuse, nothing)
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
                    cur_domain_node  = isDomainNode(cur.tls.expr)  
                    next_domain_node = isDomainNode(next.tls.expr) 
                    intersection     = intersect(cur.def, next.use)
                    @dprintln(3,"cur_domain_node = ", cur_domain_node, " next_domain_node = ", next_domain_node, " intersection = ", intersection)
                    if cur_domain_node && !cannot_move_next
                        if !next_domain_node && isempty(intersection)
                            # If the current statement is a domain node and the next staterment isn't and we are allowed to move the next node
                            # in the block and the next statement doesn't use anything produced by this statement then we can switch the order of
                            # the current and next statement.
                            @dprintln(3,"bubbling domain node down")
                            (bb.statements[i], bb.statements[i+1]) = (bb.statements[i+1], bb.statements[i])
                            (bb.cfgbb.statements[i], bb.cfgbb.statements[i+1]) = (bb.cfgbb.statements[i+1], bb.cfgbb.statements[i])
                            (bb.cfgbb.statements[i].index, bb.cfgbb.statements[i+1].index) = (bb.cfgbb.statements[i+1].index, bb.cfgbb.statements[i].index)
                            found_change = true
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


