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

#using Debug

type findAllocationsState
    allocs :: Dict{LHSVar, Int}
    arrays_stored_in_arrays :: Set{LHSVar}
    LambdaVarInfo

    function findAllocationsState(lvi)
        return new(Dict{LHSVar, Int}(), Set{LHSVar}(), lvi)
    end
end

function findAllocations(x :: Expr, state :: findAllocationsState, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    if is_top_level
        @dprintln(3, "findAllocations is_top_level")
        if isAssignmentNode(x) && isAllocation(x.args[2])
           lhs = toLHSVar(x.args[1])
           @dprintln(3, "findAllocations found allocation ", lhs)
           if !haskey(state.allocs, lhs)
               state.allocs[lhs] = 0
           end
           state.allocs[lhs] = state.allocs[lhs] + 1
        end
    end
    if isArraysetCall(x)
        value = x.args[3]   # the value written into the array
        vtyp  = CompilerTools.LambdaHandling.getType(value, state.LambdaVarInfo)
        @dprintln(3, "findAllocations found arrayset with value ", value, " of type ", vtyp, " ", isArrayType(vtyp), " ", typeof(value))
        if isArrayType(vtyp) && isa(value, RHSVar)
            @dprintln(3, "findAllocations added ", value, " to set")
            push!(state.arrays_stored_in_arrays, toLHSVar(value))
        end
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function findAllocations(x :: ANY, state :: findAllocationsState, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Try to hoist allocations outside the loop if possible.
"""
function hoistAllocation(ast::Array{Any,1}, lives, domLoop::DomLoops, state :: expr_state)
    # Only allocations that are not aliased can be safely hoisted.
    # Note that we must rule out simple re-assignment in alias analysis to be conservative about object uniqueness
    # (instead of just variable uniqueness).
    body = CompilerTools.LambdaHandling.getBody(ast, CompilerTools.LambdaHandling.getReturnType(state.LambdaVarInfo))
    uniqSet = AliasAnalysis.from_lambda(state.LambdaVarInfo, body, lives, pir_alias_cb, nothing; noReAssign = true)
    @dprintln(3, "HA: uniqSet = ", uniqSet)
    for l in domLoop.loops
        @dprintln(3, "HA: loop from block ", l.head, " to ", l.back_edge)
        headBlk = lives.cfg.basic_blocks[ l.head ]
        tailBlk = lives.cfg.basic_blocks[ l.back_edge ]
        if length(headBlk.preds) != 2
            continue
        end
        preBlk = nothing
        for blk in headBlk.preds
            if blk.label != tailBlk.label
                preBlk = blk
                break
            end
        end

        #if ((preBlk === nothing) || length(preBlk.statements) == 0) continue end
        if (preBlk === nothing) continue end
        tls = lives.basic_blocks[ preBlk ]

        # sometimes the preBlk has no statements
        # in this case we go to preBlk's previous block to find the previous statement of the current loop (for allocations to be inserted)
        while length(preBlk.statements)==0
            if length(preBlk.preds)==1
                preBlk = next(preBlk.preds,start(preBlk.preds))[1]
            end
        end
        if length(preBlk.statements)==0 continue end
        preHead = preBlk.statements[end].index

        head = headBlk.statements[1].index
        tail = tailBlk.statements[1].index
        @dprintln(3, "HA: line before head is ", ast[preHead-1])
        # Is iterating through statement indices this way safe?
        for i = head:tail
            if isAssignmentNode(ast[i]) && isAllocation(ast[i].args[2])
                @dprintln(3, "HA: found allocation at line ", i, ": ", ast[i])
                lhs = ast[i].args[1]
                rhs = ast[i].args[2]
                lhs = toLHSVar(lhs)
                if in(lhs, uniqSet) && (haskey(state.array_length_correlation, lhs))
                    c = state.array_length_correlation[lhs]
                    for (d, v) in state.symbol_array_correlation
                        if v == c
                            ok = true

                            for j = 1:length(d)
                                if !(isa(d[j],Int) || in(d[j], tls.live_out))
                                    ok = false
                                    break
                                end
                            end
                            @dprintln(3, "HA: found correlation dimension ", d, " ", ok, " ", length(rhs.args)-6)
                            if ok && length(rhs.args) - 6 == 2 * length(d) # dimension must match
                                rhs.args = rhs.args[1:6]
                                for s in d
                                    push!(rhs.args, s)
                                    push!(rhs.args, 0)
                                end
                                @dprintln(3, "HA: hoist ", ast[i], " out of loop before line ", head)
                                ast = [ ast[1:preHead-1]; ast[i]; ast[preHead:i-1]; ast[i+1:end] ]
                                break
                            end
                        end
                    end
                end
            end
        end
    end
    return ast
end

function isDeadCall(rhs::Expr, live_out)
    if isCall(rhs)
        fun = getCallFunction(rhs)
        args = getCallArguments(rhs)
        if in(fun, CompilerTools.LivenessAnalysis.wellknown_all_unmodified)
            @dprintln(3, rhs)
            return true
        elseif in(fun, CompilerTools.LivenessAnalysis.wellknown_only_first_modified) &&
                !in(toLHSVar(args[1]), live_out)
            return true
        end
    end
    return false
end

function isDeadCall(rhs::ANY, live_out)
    return false
end

type DictInfo
    live_info
    expr
end

"""
State for the remove_no_deps and insert_no_deps_beginning phases.
"""
type RemoveNoDepsState
    lives             :: CompilerTools.LivenessAnalysis.BlockLiveness
    top_level_no_deps :: Array{Any,1}
    hoistable_scalars :: Set{LHSVar}
    dict_sym          :: Dict{LHSVar, DictInfo}
    change            :: Bool

    function RemoveNoDepsState(l, hs)
        new(l, Any[], hs, Dict{LHSVar, DictInfo}(), false)
    end
end

"""
Works with remove_no_deps below to move statements with no dependencies to the beginning of the AST.
"""
function insert_no_deps_beginning(node, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    if is_top_level && top_level_number == 1
        return [data.top_level_no_deps; node]
    end
    nothing
end

function remove_no_deps(node :: NewvarNode, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    if is_top_level
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        if live_info != nothing
            if !in(node.slot, live_info.live_out)
                return CompilerTools.AstWalker.ASTWALK_REMOVE
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
# This routine gathers up nodes that do not use
# any variable and removes them from the AST into top_level_no_deps.  This works in conjunction with
# insert_no_deps_beginning above to move these statements with no dependencies to the beginning of the AST
# where they can't prevent fusion.
"""
function remove_no_deps(node :: Expr, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    @dprintln(3,"remove_no_deps starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(3,"remove_no_deps node = ", node, " type = ", typeof(node))
    @dprintln(3,"node.head: ", node.head)
    head = node.head

    if is_top_level
        @dprintln(3,"remove_no_deps is_top_level")

        if head==:gotoifnot
            # Empty the state at the end or begining of a basic block
            data.dict_sym = Dict{LHSVar,DictInfo}()
        end

        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        # Remove line number statements.
        if head == :line || head == :meta
            return CompilerTools.AstWalker.ASTWALK_REMOVE
        end
        if live_info == nothing
            @dprintln(3,"remove_no_deps no live_info")
        else
            @dprintln(3,"remove_no_deps live_info = ", live_info)
            @dprintln(3,"remove_no_deps live_info.use = ", live_info.use)

            if isa(node, Number) || isa(node, RHSVar)
                @dprintln(3,"Eliminating dead node: ", node)
                return CompilerTools.AstWalker.ASTWALK_REMOVE
            elseif isAssignmentNode(node)
                @dprintln(3,"Is an assignment node.")
                lhs = node.args[1]
                @dprintln(4,lhs)
                rhs = node.args[2]
                @dprintln(4,rhs)

                if isa(rhs, Expr) && ((rhs.head === :parfor) || (rhs.head === :mmap!))
                    # Always keep parfor assignment in order to work with fusion
                    @dprintln(3, "keep assignment due to parfor or mmap! node")
                    return node
                end
                if isa(lhs, RHSVar)
                    lhs_sym = toLHSVar(lhs)
                    @dprintln(3,"remove_no_deps found assignment with lhs symbol ", lhs, " ", rhs, " typeof(rhs) = ", typeof(rhs))
                    # Remove a dead store
                    if !in(lhs_sym, live_info.live_out)
                        data.change = true
                        @dprintln(3,"remove_no_deps lhs is NOT live out")
                        if hasNoSideEffects(rhs) || isDeadCall(rhs, live_info.live_out)
                            @dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
                            return CompilerTools.AstWalker.ASTWALK_REMOVE
                        else
                            # Just eliminate the assignment but keep the rhs
                            @dprintln(3,"Eliminating dead variable but keeping rhs, dead = ", lhs_sym)
                            return rhs
                        end
                    else
                        @dprintln(3,"remove_no_deps lhs is live out")
                        if isa(rhs, RHSVar)
                            rhs_sym = toLHSVar(rhs)
                            @dprintln(3,"remove_no_deps rhs is symbol ", rhs_sym)
                            if !in(rhs_sym, live_info.live_out)
                                @dprintln(3,"remove_no_deps rhs is NOT live out")
                                if haskey(data.dict_sym, rhs_sym)
                                    di = data.dict_sym[rhs_sym]
                                    di_live = di.live_info
                                    prev_expr = di.expr

                                    if !in(lhs_sym, di_live.live_out)
                                        prev_expr.args[1] = lhs_sym
                                        delete!(data.dict_sym, rhs_sym)
                                        data.dict_sym[lhs_sym] = DictInfo(di_live, prev_expr)
                                        @dprintln(3,"Lhs is live but rhs is not so substituting rhs for lhs ", lhs_sym, " => ", rhs_sym)
                                        @dprintln(3,"New expr = ", prev_expr)
                                        return CompilerTools.AstWalker.ASTWALK_REMOVE
                                    else
                                        delete!(data.dict_sym, rhs_sym)
                                        @dprintln(3,"Lhs is live but rhs is not.  However, lhs is read between def of rhs and current statement so not substituting.")
                                    end
                                end
                            else
                                @dprintln(3,"Lhs and rhs are live so forgetting assignment ", lhs_sym, " ", rhs_sym)
                                delete!(data.dict_sym, rhs_sym)
                            end
                        else
                            data.dict_sym[lhs_sym] = DictInfo(live_info, node)
                            @dprintln(3,"Remembering assignment for symbol ", lhs_sym, " ", rhs)
                        end
                    end
                end
            else
                @dprintln(3,"Not an assignment node.")
            end

            for j = live_info.use
                delete!(data.dict_sym, j)
            end

            # Here we try to determine which scalar assigns can be hoisted to the beginning of the function.
            #
            # If this statement defines some variable.
            if !isempty(live_info.def)
                @dprintln(3, "Checking if the statement is hoistable.")
                @dprintln(3, "Previous hoistables = ", data.hoistable_scalars)
                # Assume that hoisting is safe until proven otherwise.
                dep_only_on_parameter = true
                # Look at all the variables on which this statement depends.
                # If any of them are not a hoistable scalar then we can't hoist the current scalar definition.
                for i in live_info.use
                    if !in(i, data.hoistable_scalars)
                        @dprintln(3, "Could not hoist because the statement depends on :", i)
                        dep_only_on_parameter = false
                        break
                    end
                end

                # See if there are any calls with side-effects that could prevent moving.
                sews = SideEffectWalkState()
                ParallelAccelerator.ParallelIR.AstWalk(node, hasNoSideEffectWalk, sews)
                if sews.hasSideEffect
                    dep_only_on_parameter = false
                end

                if dep_only_on_parameter
                    @dprintln(3,"Statement does not have any side-effects.")
                    # If this statement is defined in more than one place then it isn't hoistable.
                    for i in live_info.def
                        @dprintln(3,"Checking if ", i, " is multiply defined.")
                        @dprintln(4,"data.lives = ", data.lives)
                        if CompilerTools.LivenessAnalysis.countSymbolDefs(i, data.lives) > 1
                            @dprintln(3, "Could not hoist because the function has multiple definitions of: ", i)
                            dep_only_on_parameter = false
                            break
                        end
                    end

                    if dep_only_on_parameter
                        @dprintln(3,"remove_no_deps removing ", node, " because it only depends on hoistable scalars.")
                        push!(data.top_level_no_deps, node)
                        # If the defs in this statement are hoistable then other statements which depend on them may also be hoistable.
                        for i in live_info.def
                            push!(data.hoistable_scalars, i)
                        end
                        return CompilerTools.AstWalker.ASTWALK_REMOVE
                    end
                else
                    @dprintln(3,"Statement DOES have any side-effects.")
                end
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function remove_no_deps(node::Union{LabelNode,GotoNode}, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    if is_top_level
        # Empty the state at the end or begining of a basic block
        data.dict_sym = Dict{LHSVar,DictInfo}()
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end


function remove_no_deps(node :: Union{LineNumberNode,RHSVar,Number}, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    if is_top_level
        # remove line number nodes, bare RHSVAr and numeric constants
        return CompilerTools.AstWalker.ASTWALK_REMOVE
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function remove_no_deps(node::ANY, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end


"""
Empty statements can be added to the AST by some passes in ParallelIR.
This pass over the statements of the :body excludes such "nothing" statements from the new :body.
"""
function removeNothingStmts(args :: Array{Any,1}, state)
    newBody = Any[]
    for i = 1:length(args)
        if args[i] != nothing
            push!(newBody, args[i])
        end
    end
    return newBody
end


"""
Holds liveness information for the remove_dead AstWalk phase.
"""
type RemoveDeadState
    lives::CompilerTools.LivenessAnalysis.BlockLiveness
    linfo
end

function remove_dead(node :: NewvarNode, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    if is_top_level
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        if live_info != nothing
            if !in(node.slot, live_info.live_out)
                return CompilerTools.AstWalker.ASTWALK_REMOVE
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
An AstWalk callback that uses liveness information in "data" to remove dead stores.
"""
function remove_dead(node::Expr, data::RemoveDeadState, top_level_number, is_top_level, read)
    @dprintln(3,"remove_dead starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(3,"remove_dead node = ", node, " type = ", typeof(node))
    @dprintln(3,"node.head = ", node.head)

    if is_top_level
        @dprintln(3,"remove_dead is_top_level")
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        if live_info != nothing
            @dprintln(3,"remove_dead live_info = ", live_info)
            @dprintln(3,"remove_dead live_info.use = ", live_info.use)

            if isa(node, Number) || isa(node, RHSVar)
                @dprintln(3,"Eliminating dead node: ", node)
                return CompilerTools.AstWalker.ASTWALK_REMOVE
            elseif isAssignmentNode(node)
                @dprintln(3,"Is an assignment node.")
                lhs = node.args[1]
                @dprintln(4,lhs)
                rhs = node.args[2]
                @dprintln(4,rhs)

                if isa(lhs,RHSVar)
                    lhs_sym = toLHSVar(lhs)
                    @dprintln(3,"remove_dead found assignment with lhs symbol ", lhs, " ", rhs, " typeof(rhs) = ", typeof(rhs))
                    # Remove a dead store
                    if !in(lhs_sym, live_info.live_out)
                        @dprintln(3,"remove_dead lhs is NOT live out")
                        if hasNoSideEffects(rhs) || isDeadCall(rhs, live_info.live_out)
                            @dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
                            return CompilerTools.AstWalker.ASTWALK_REMOVE
                        else
                            # Just eliminate the assignment but keep the rhs
                            @dprintln(3,"Eliminating dead variable but keeping rhs, dead = ", lhs_sym, " rhs = ", rhs)
                            return rhs
                        end
                    end
                end
            elseif isInvoke(node)
                @dprintln(3,"isInvoke. head = ", node.head, " type = ", typeof(node.args[2]), " name = ", node.args[2])
                if hasNoSideEffects(node) || isDeadCall(node, live_info.live_out)
                    @dprintln(3,"Eliminating dead call. node = ", node)
                    return CompilerTools.AstWalker.ASTWALK_REMOVE
                end
            elseif isCall(node)
                @dprintln(3,"isCall. head = ", node.head, " type = ", typeof(node.args[1]), " name = ", node.args[1])
                if hasNoSideEffects(node) || isDeadCall(node, live_info.live_out)
                    @dprintln(3,"Eliminating dead call. node = ", node)
                    return CompilerTools.AstWalker.ASTWALK_REMOVE
                end
            end
        else
            @dprintln(3,"remove_dead no live_info!")
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function remove_dead(node::Union{RHSVar,Number}, data :: RemoveDeadState, top_level_number, is_top_level, read)
    if is_top_level
        @dprintln(3,"remove_dead is_top_level removing ", node)
        return CompilerTools.AstWalker.ASTWALK_REMOVE
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function remove_dead(node::PIRParForAst, data::RemoveDeadState, top_level_number, is_top_level, read)
    if is_top_level
        @dprintln(3,"remove_dead is_top_level parfor")
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        if live_info != nothing
            parfor_live_out = live_info.live_out
            @dprintln(3,"remove_dead parfor live_out = ", parfor_live_out)
            # node.preParFor = remove_dead_recursive_body(node.preParFor, data.linfo, parfor_live_out)
            # node.hoisted = remove_dead_recursive_body(node.hoisted, data.linfo, parfor_live_out)
            node.body = remove_dead_recursive_body(node.body, data.linfo, parfor_live_out)
            # node.postParFor = remove_dead_recursive_body(node.postParFor, data.linfo, parfor_live_out)
            return node
        end
    end
end

function remove_dead(node::ANY, data :: RemoveDeadState, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function remove_dead_recursive_body(body, linfo, parfor_live_out)
    body_lives = CompilerTools.LivenessAnalysis.from_lambda(linfo, body, pir_live_cb, linfo)
    @dprintln(3,"remove_dead parfor body lives = ", body_lives)
    # live_out variables of the parfor are added to live_out of all
    # basic blocks and statements so remove_dead doesn't remove them
    for bb in body_lives.basic_blocks
        bb[2].live_out = union(parfor_live_out, bb[2].live_out)
        stmts = bb[2].statements
        for j = 1:length(stmts)
            stmts[j].live_out = union(parfor_live_out, stmts[j].live_out)
        end
    end
    # body can now be traversed using exteneded liveness info
    new_body = AstWalk(TypedExpr(nothing, :body, body...), remove_dead, RemoveDeadState(body_lives,linfo))
    return new_body.args
end

"""
State to aide in the transpose propagation phase.
"""
type TransposePropagateState
    lives  :: CompilerTools.LivenessAnalysis.BlockLiveness
    transpose_map :: Dict{LHSVar, LHSVar} # transposed output -> matrix in

    function TransposePropagateState(l)
        new(l, Dict{LHSVar, LHSVar}())
    end
end

function transpose_propagate(node::ANY, data::TransposePropagateState, top_level_number, is_top_level, read)
    @dprintln(3,"transpose_propagate starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(3,"transpose_propagate node = ", node)

    if is_top_level
        @dprintln(3,"transpose_propagate is_top_level")
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)

        if live_info != nothing
            # Remove matrices from data.transpose_map if either original or transposed matrix is modified by this statement.
            # For each symbol modified by this statement...
            for def in live_info.def
                @dprintln(4,"Symbol ", def, " is modifed by current statement.")
                # For each transpose map we currently have recorded.
                for mat in data.transpose_map
                    @dprintln(4,"Current mat in data.transpose_map = ", mat)
                    # If original or transposed matrix is modified by the statement.
                    if def == mat[1] || def==mat[2]
                    #@bp
                        @dprintln(3,"transposed or original matrix is modified so removing ", mat," from data.transpose_map.")
                        # Then remove the lhs = rhs entry from copies.
                        delete!(data.transpose_map, mat[1])
                    end
                end
            end
        end
    end
    return transpose_propagate_helper(node, data)
end

function transpose_propagate_helper(node::Expr, data::TransposePropagateState)

    if (node.head === :gotoifnot)
        # Only transpose propagate within a basic block.  this is now a new basic block.
        empty!(data.transpose_map)
    elseif isAssignmentNode(node) && isCall(node.args[2])
        @dprintln(3,"Is an assignment call node.")
        lhs = toLHSVar(node.args[1])
        rhs = node.args[2]
        func = getCallFunction(rhs)
        if isBaseFunc(func,:transpose!)
            @dprintln(3,"transpose_propagate transpose! found.")
            args = getCallArguments(rhs)
            original_matrix = toLHSVar(args[2])
            transpose_var1 = toLHSVar(args[1])
            transpose_var2 = lhs
            data.transpose_map[transpose_var1] = original_matrix
            data.transpose_map[transpose_var2] = original_matrix
        elseif isBaseFunc(func,:transpose)
            @dprintln(3,"transpose_propagate transpose found.")
            args = getCallArguments(rhs)
            original_matrix = toLHSVar(args[1])
            transpose_var = lhs
            data.transpose_map[transpose_var] = original_matrix
        elseif isBaseFunc(func,:gemm_wrapper!)
            @dprintln(3,"transpose_propagate GEMM found.")
            args = getCallArguments(rhs)
            A = toLHSVar(args[4])
            if haskey(data.transpose_map, A)
                args[4] = data.transpose_map[A]
                args[2] = 'T'
                @dprintln(3,"transpose_propagate GEMM replace transpose arg 1.")
            end
            B = toLHSVar(args[5])
            if haskey(data.transpose_map, B)
                args[5] = data.transpose_map[B]
                args[3] = 'T'
                @dprintln(3,"transpose_propagate GEMM replace transpose arg 2.")
            end
            rhs.args = rhs.head == :invoke ? [ rhs.args[1:2]; args ] : [ rhs.args[1]; args ]
        elseif isBaseFunc(func,:gemv!)
            args = getCallArguments(rhs)
            A = toLHSVar(args[3])
            if haskey(data.transpose_map, A)
                args[3] = data.transpose_map[A]
                args[2] = 'T'
            end
            rhs.args = rhs.head == :invoke ? [ rhs.args[1:2]; args ] : [ rhs.args[1]; args ]
        # replace arraysize() calls to the transposed matrix with original
        elseif isBaseFunc(func, :arraysize)
            args = getCallArguments(rhs)
            if haskey(data.transpose_map, args[1])
                args[1] = data.transpose_map[args[1]]
                if args[2] ==1
                    args[2] = 2
                elseif args[2] ==2
                    args[2] = 1
                else
                    throw("transpose_propagate matrix dim error")
                end
            end
            rhs.args = rhs.head == :invoke ? [ rhs.args[1:2]; args ] : [ rhs.args[1]; args ]
        end
        return node
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function transpose_propagate_helper(node::Union{LabelNode,GotoNode}, data::TransposePropagateState)
    # Only transpose propagate within a basic block.  this is now a new basic block.
    empty!(data.transpose_map)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function transpose_propagate_helper(node::ANY, data::TransposePropagateState)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

# don't recurse inside DomainLambda
function transpose_propagate_helper(node::DomainLambda, data::TransposePropagateState)
    return node
end

const max_unroll_size = 12

function is_small_loop(lower::Int, upper::Int, step::Int)
    return (upper-lower)/step <= max_unroll_size
end

is_small_loop(lower::ANY, upper::ANY, step::ANY) = false

function unroll_const_parfors(node::Expr, data, top_level_number, is_top_level, read)
    if node.head==:parfor
        parfor = node.args[1]
        # TODO: extend for multi dimensional case
        if length(parfor.loopNests)>1
            return CompilerTools.AstWalker.ASTWALK_RECURSE
        end
        indexVariable = parfor.loopNests[1].indexVariable
        lower = parfor.loopNests[1].lower
        upper = parfor.loopNests[1].upper
        step = parfor.loopNests[1].step
        if !is_small_loop(lower, upper, step)
            return CompilerTools.AstWalker.ASTWALK_RECURSE
        end
        @dprintln(3,"unroll_const_parfors unrolling parfor ", node)
        out = deepcopy(parfor.preParFor)
        append!(out, deepcopy(parfor.hoisted))
        for i in lower:step:upper
            body = deepcopy(parfor.body)
            replaceExprWithDict!(body, Dict{LHSVar,Any}(toLHSVar(indexVariable)=>i), nothing, AstWalk)
            append!(out,body)
        end
        append!(out, deepcopy(parfor.postParFor))
        return Expr(:block, out...)
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end
function unroll_const_parfors(node::ANY, data, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
unroll returns :block exprs that need to be flattened
"""
function flatten_blocks(node::Expr, data, top_level_number, is_top_level, read)
    if node.head==:body
        #@dprintln(3,"flatten_blocks body found ", node)
        node.args = flatten_blocks_args(node.args)
        return node
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function flatten_blocks(node::PIRParForAst, data, top_level_number, is_top_level, read)
    node.body = flatten_blocks_args(node.body)
    return node
end

function flatten_blocks_args(args::Vector{Any})
    out = Any[]
    for arg in args
        #println("flatten_blocks body arg ", arg)
        if isa(arg,Expr) && arg.head==:block
            @dprintln(3,"flatten_blocks block found ", arg)
            for b_arg in arg.args
                bo_arg = AstWalk(b_arg, flatten_blocks, nothing)
                push!(out, bo_arg)
            end
            continue
        end
        o_arg = AstWalk(arg, flatten_blocks, nothing)
        push!(out, o_arg)
    end
    return out
end


function flatten_blocks(node::ANY, data, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
State to aide in the copy propagation phase.
"""
type CopyPropagateState
    lives  :: CompilerTools.LivenessAnalysis.BlockLiveness
    copies :: Dict{LHSVar, Union{LHSVar,Number}}
    # if ISASSIGNEDONCE flag is set for a variable, its safe to keep it across block boundaries
    safe_copies :: Dict{LHSVar, Union{LHSVar,Number}}
    # variables that shouldn't be copied (e.g. parfor reduction Variables)
    no_copy::Vector{LHSVar}
    linfo

    function CopyPropagateState(l,li)
        new(l,Dict{LHSVar, Union{LHSVar,Number}}(),Dict{LHSVar, Union{LHSVar,Number}}(),LHSVar[],li)
    end
end

"""
In each basic block, if there is a "copy" (i.e., something of the form "a = b") then put
that in copies as copies[a] = b.  Then, later in the basic block if you see the symbol
"a" then replace it with "b".  Note that this is not SSA so "a" may be written again
and if it is then it must be removed from copies.
"""
function copy_propagate(node::ANY, data::CopyPropagateState, top_level_number, is_top_level, read)
    @dprintln(3,"copy_propagate starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(3,"copy_propagate node = ", node, " type = ", typeof(node))
    @dprintln(3,"copy_propagate data = ", data.copies, " safe: ", data.safe_copies)
    @dprintln(3,"copy_propagate is_top_level ", is_top_level)

    # liveness information of top-level nodes is enough for finding defs
    if is_top_level
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        if live_info!=nothing
            # Remove elements from data.copies if the original RHS is modified by this statement.
            # For each symbol modified by this statement...
            for def in live_info.def
                @dprintln(4,"Symbol ", def, " is modifed by current statement.")
                # For each copy we currently have recorded.
                for copy in data.copies
                    @dprintln(4,"Current entry in data.copies = ", copy)
                    # If the rhs of the copy is modified by the statement.
                    if def == copy[2]
                        @dprintln(3,"RHS of data.copies is modified so removing ", copy," from data.copies.")
                        # Then remove the lhs = rhs entry from copies.
                        delete!(data.copies, copy[1])
                    elseif def == copy[1]
                        # LHS is def.  We can maintain the mapping if RHS is dead.
                        if in(copy[2], live_info.live_out)
                            @dprintln(3,"LHS of data.copies is modified and RHS is live so removing ", copy," from data.copies.")
                            # Then remove the lhs = rhs entry from copies.
                            delete!(data.copies, copy[1])
                        end
                    end
                end
            end
        else
            @dprintln(3,"copy_propagate no live_info! ")
        end
    end
    return copy_propagate_helper(node, data, top_level_number, is_top_level, read)
end

function copy_propagate_helper(node::Expr,
                               data::CopyPropagateState,
                               top_level_number,
                               is_top_level,
                               read)
    @dprintln(3,"node.head = ", node.head)

    if node.head==:gotoifnot
        # Only copy propagate within a basic block.  this is now a new basic block.
        # if ISASSIGNEDONCE flag is set for a variable, its safe to keep it across block boundaries
        data.copies = copy(data.safe_copies)
    elseif isAssignmentNode(node)
        @dprintln(3,"Is an assignment node.")
        # ignore LambdaInfo nodes generated by domain-ir that are essentially dead nodes here
        # TODO: should these nodes be traversed here recursively?
        if isa(node.args[2],LambdaInfo)
            return node
        end
        lhs = AstWalk(node.args[1], copy_propagate, data)
        @dprintln(4,"lhs = ", lhs)
        rhs = node.args[2] = AstWalk(node.args[2], copy_propagate, data)
        @dprintln(4,"rhs = ", rhs)
        # sometimes lhs can already be replaced with a constant
        if !isa(lhs, RHSVar)
            return node
        end
        node.args[1] = lhs
        if !in(lhs,data.no_copy) && (isa(rhs, RHSVar) || (isa(rhs, Number) && !isa(rhs,Complex))) # TODO: fix complex number case
            lhs = toLHSVar(lhs)
            rhs = toLHSVarOrNum(rhs)
            desc = CompilerTools.LambdaHandling.getDesc(lhs, data.linfo)
            if desc & ISASSIGNEDBYINNERFUNCTION != ISASSIGNEDBYINNERFUNCTION
                @dprintln(3,"Creating copy, lhs = ", lhs, " rhs = ", rhs)
                # Record that the left-hand side is a copy of the right-hand side.
                data.copies[lhs] = rhs
                if (desc & ISASSIGNEDONCE == ISASSIGNEDONCE) &&
                    (isa(rhs, Number) || CompilerTools.LambdaHandling.getDesc(rhs, data.linfo) & ISASSIGNEDONCE == ISASSIGNEDONCE)
                    @dprintln(3,"Creating safe copy, lhs = ", lhs, " rhs = ", rhs)
                    #@bp
                    data.safe_copies[lhs] = rhs
                end
            end
        end
        return node
    elseif isCall(node)
        return evaluate_constant_calls(node)
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function evaluate_constant_calls(node::Expr)
    @assert isCall(node) "call Expr expected"
    func = getCallFunction(node)
    args = getCallArguments(node)
    if func==GlobalRef(Core.Intrinsics,:xor_int) && isa(args[1],Int) && isa(args[2],Int)
        @dprintln(3,"copy_propagate replacing constant call Core.Intrinsics.xor_int = ", node)
        return eval(quote Core.Intrinsics.xor_int($(args[1]), $(args[2])) end )
    elseif func==GlobalRef(Core.Intrinsics,:flipsign_int) && isa(args[1],Int) && isa(args[2],Int)
        @dprintln(3,"copy_propagate replacing constant call Core.Intrinsics.flipsign_int = ", node)
        return eval(quote Core.Intrinsics.flipsign_int($(args[1]), $(args[2])) end )
    elseif func==GlobalRef(Core.Intrinsics,:box) && isa(args[2],Number) && typeof(args[2])==args[1]
        @dprintln(3,"copy_propagate replacing constant call Core.Intrinsics.box = ", node)
        return args[2]
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function copy_propagate_helper(node::PIRParForAst, data::CopyPropagateState, top_level_number, is_top_level, read)
    # remove loopnest and reduction vars from data, then recurse
    loopnest_vars = map(x->toLHSVar(x.indexVariable), node.loopNests)
    reduction_vars = map(x->toLHSVar(x.reductionVar), node.reductions)
    rem_vars = [loopnest_vars; reduction_vars]
    @dprintln(3,"copy_propagate parfor loop and reduce vars to remove: ", rem_vars)
    map(x->delete!(data.copies,x) ,rem_vars)
    map(x->delete!(data.safe_copies,x) ,rem_vars)
    append!(data.no_copy, reduction_vars)
    @dprintln(3,"copy_propagate new data = ", data.copies, " safe: ", data.safe_copies)
    # update loopnest variables
    for ln in node.loopNests
        ln.lower = AstWalk(ln.lower, copy_propagate, data)
        ln.upper = AstWalk(ln.upper, copy_propagate, data)
        ln.step = AstWalk(ln.step, copy_propagate, data)
    end
    old_lives = data.lives
    body_lives = CompilerTools.LivenessAnalysis.from_lambda(data.linfo, node.body, pir_live_cb, data.linfo)
    data.lives = body_lives
    @dprintln(3,"copy_propagate parfor body lives = ", data.lives)
    new_body = AstWalk(TypedExpr(nothing, :body, node.body...), copy_propagate, data)
    node.body = new_body.args
    data.lives = old_lives
    return node
end

function copy_propagate_helper(node::DelayedFunc, data::CopyPropagateState, top_level_number, is_top_level, read)
    # don't touch DelayedFunc for now
    # TODO: hanlde DelayedFunc (push/pop copy values around them? reduction vars?)
    return node
end

function copy_propagate_helper(node::Union{LabelNode,GotoNode}, data::CopyPropagateState, top_level_number, is_top_level, read)
    # Only copy propagate within a basic block.  this is now a new basic block.
    # if ISASSIGNEDONCE flag is set for a variable, its safe to keep it across block boundaries
    data.copies = copy(data.safe_copies)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function copy_propagate_helper(node::Union{Symbol,RHSVar},
                               data::CopyPropagateState,
                               top_level_number,
                               is_top_level,
                               read)

    lhsVar = toLHSVar(node)
    if haskey(data.copies, lhsVar)
        @dprintln(3,"Replacing ", lhsVar, " with ", data.copies[lhsVar])
        tmp_node = data.copies[lhsVar]
        return isa(tmp_node, Symbol) ? toRHSVar(tmp_node, getType(node, data.linfo), data.linfo) : tmp_node
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function copy_propagate_helper(node::DomainLambda,
                               data::CopyPropagateState,
                               top_level_number,
                               is_top_level,
                               read)

    @dprintln(3,"Found DomainLambda in copy_propagate, dl = ", node)
    inner_linfo = node.linfo
    inner_body = node.body
    # replaceExprWithDict!() expects values to be valid variables to import (no SSAValue, no name clashes with local variables etc.)
    dict = filter( (k,v)->!(isa(v,RHSVar) && (isa(v,GenSym) ||
       isLocalVariable(lookupVariableName(v,data.linfo),inner_linfo))), data.safe_copies)
    replaceExprWithDict!(node, convert(Dict{LHSVar,Any},dict), data.linfo, AstWalk)
    inner_lives = computeLiveness(inner_body, inner_linfo)
    node.body = AstWalk(inner_body, copy_propagate, CopyPropagateState(inner_lives,inner_linfo))

    return node
end

function copy_propagate_helper(node::ANY,
                               data::CopyPropagateState,
                               top_level_number,
                               is_top_level,
                               read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function create_equivalence_classes_assignment(lhs::RHSVar, rhs::RHSVar, state)
    rhs = toLHSVar(rhs)
    lhs = toLHSVar(lhs)

    rhs_corr = getOrAddArrayCorrelation(rhs, state)
    @dprintln(3,"assignment correlation lhs = ", lhs, " type = ", typeof(lhs))
    # if an array has correlation already, there might be a case of multiple assignments
    # in this case, try to make sure sizes are the same or assign a new negative value otherwise
    if haskey(state.array_length_correlation, lhs)
        prev_corr = state.array_length_correlation[lhs]
        prev_size = []
        rhs_size = []
        for (d, v) in state.symbol_array_correlation
            if v==prev_corr
                prev_size = d
            end
            if v==rhs_corr
                rhs_size = d
            end
        end
        if prev_size==[] || rhs_size==[] || prev_size!=rhs_size
            # can't make sure sizes are always equal, assign negative correlation to lhs
            state.array_length_correlation[lhs] = getNegativeCorrelation(state)
            @dprintln(3, "multiple assignment detected, negative correlation assigned for ", lhs)
        end
    else
        lhs_corr = getOrAddArrayCorrelation(toLHSVar(lhs), state)
        merge_correlations(state, lhs_corr, rhs_corr)
        @dprintln(3,"Correlations after assignment merge into lhs")
        print_correlations(3, state)
    end

    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function sizeNoTuples(x, state)
    for s in x
        stype = CompilerTools.LambdaHandling.getType(s, state.LambdaVarInfo)  # get size type
        if stype <: Tuple
            @dprintln(3,"Found Tuple in sizes for array correlation.")
            return false
        end
    end
    return true
end

function create_equivalence_classes_assignment(lhs, rhs::Expr, state)
    @dprintln(4,lhs)
    @dprintln(4,rhs)

    if rhs.head == :assertEqShape
        # assertEqShape lets us know that the array mentioned in the assertEqShape node must have the same shape.
        @dprintln(3,"Creating array length assignment from assertEqShape")
        from_assertEqShape(rhs, state)
    elseif rhs.head == :alloc
        # Here an array on the left-hand side is being created from size specification on the right-hand side.
        # Map those array sizes to the corresponding array equivalence class.
        sizes = Any[ x for x in rhs.args[2]]
        n = length(sizes)
        assert(n >= 1 && n <= 3)
        if sizeNoTuples(sizes, state)
            @dprintln(3, "Detected :alloc array allocation. dims = ", sizes)
            checkAndAddSymbolCorrelation(lhs, state, sizes)
        end
    elseif rhs.head == :foreigncall
        fun = rhs.args[1]
        args = rhs.args[2:end]
        @dprintln(3, "Detected foreigncall rhs in from_assignment.")
        @dprintln(3, "fun = ", fun, " args = ", args)
        if fun == QuoteNode(:jl_alloc_array_1d)
            dim1 = args[5]
            @dprintln(3, "Detected 1D array allocation. dim1 = ", dim1, " type = ", typeof(dim1))
            checkAndAddSymbolCorrelation(lhs, state, Any[dim1])
        elseif fun == QuoteNode(:jl_alloc_array_2d)
            dim1 = args[5]
            dim2 = args[7]
            @dprintln(3, "Detected 2D array allocation. dim1 = ", dim1, " dim2 = ", dim2)
            checkAndAddSymbolCorrelation(lhs, state, Any[dim1, dim2])
        elseif fun == QuoteNode(:jl_alloc_array_3d)
            dim1 = args[5]
            dim2 = args[7]
            dim3 = args[9]
            @dprintln(3, "Detected 2D array allocation. dim1 = ", dim1, " dim2 = ", dim2, " dim3 = ", dim3)
            checkAndAddSymbolCorrelation(lhs, state, Any[dim1, dim2, dim3])
        end
    elseif isCall(rhs)
        @dprintln(3, "Detected call rhs in from_assignment.")
        @dprintln(3, "from_assignment call, arg1 = ", rhs.args[1])
        if length(rhs.args) > 1
            @dprintln(3, " arg2 = ", rhs.args[2])
        end
        fun = getCallFunction(rhs)
        args = getCallArguments(rhs)
        if isBaseFunc(fun, :ccall)
            # Same as :alloc above.  Detect an array allocation call and map the specified array sizes to an array equivalence class.
            if args[1] == QuoteNode(:jl_alloc_array_1d)
                dim1 = args[6]
                @dprintln(3, "Detected 1D array allocation. dim1 = ", dim1, " type = ", typeof(dim1))
                checkAndAddSymbolCorrelation(lhs, state, Any[dim1])
            elseif args[1] == QuoteNode(:jl_alloc_array_2d)
                dim1 = args[6]
                dim2 = args[8]
                @dprintln(3, "Detected 2D array allocation. dim1 = ", dim1, " dim2 = ", dim2)
                checkAndAddSymbolCorrelation(lhs, state, Any[dim1, dim2])
            elseif args[1] == QuoteNode(:jl_alloc_array_3d)
                dim1 = args[6]
                dim2 = args[8]
                dim3 = args[10]
                @dprintln(3, "Detected 2D array allocation. dim1 = ", dim1, " dim2 = ", dim2, " dim3 = ", dim3)
                checkAndAddSymbolCorrelation(lhs, state, Any[dim1, dim2, dim3])
            end
        elseif isBaseFunc(fun, :hvcat)
            dimSizes = [rhs.args[2]...]
            firstDimSize = dimSizes[1]
            all_same = true
            for i = 2:length(dimSizes)
                if dimSizes[i] != firstDimSize
                    all_same = false
                end
            end
            if length(dimSizes) > 2
                @dprintln(1, "hvcat equivalence classes not supported for more than 2 dimensions yet.")
            elseif !all_same
                @dprintln(1, "hvcat equivalence classes does not support differing dimensions.")
            else
                checkAndAddSymbolCorrelation(lhs, state, Any[length(dimSizes), firstDimSize])
            end
        elseif isBaseFunc(fun, :vect)
            @dprintln(3, "found vect, args: ", args)
            len = length(args)
            checkAndAddSymbolCorrelation(lhs, state, Any[len])
        # first arg of gemm/v are assigned to output
        elseif isBaseFunc(fun, :gemm_wrapper!) || isBaseFunc(fun, :gemv!)
            return create_equivalence_classes_assignment(lhs, args[1], state)
        elseif isBaseFunc(fun, :arraylen)
            # This is the other direction.  Takes an array and extract dimensional information that maps to the array's equivalence class.
            array_param = args[1]                  # length takes one param, which is the array
            assert(isa(array_param, RHSVar))
            array_param_type = CompilerTools.LambdaHandling.getType(array_param, state.LambdaVarInfo) # get its type
            if ndims(array_param_type) == 1            # can only associate when number of dimensions is 1
                dim_symbols = Any[toLHSVar(lhs)]
                @dprintln(3,"Adding symbol correlation from arraylen, name = ", array_param, " dims = ", dim_symbols)
                checkAndAddSymbolCorrelation(toLHSVar(array_param), state, dim_symbols)
            end
        elseif isBaseFunc(fun, :arraysize)
            # This is the other direction.  Takes an array and extract dimensional information that maps to the array's equivalence class.
            if length(args) == 1
                array_param = args[1]                  # length takes one param, which is the array
                assert(isa(array_param, TypedVar))         # should be a TypedVar
                array_param_type = getType(array_param, state.LambdaVarInfo)  # get its type
                array_dims = ndims(array_param_type)
                dim_symbols = Any[]
                for dim_i = 1:array_dims
                    push!(dim_symbols, lhs[dim_i])
                end
                lhsVar = toLHSVar(args[1])
                @dprintln(3,"Adding symbol correlation from arraysize, name = ", lhsVar, " dims = ", dim_symbols)
                checkAndAddSymbolCorrelation(lhsVar, state, dim_symbols)
            elseif length(args) == 2
                @dprintln(1,"Can't establish symbol to array length correlations yet in the case where dimensions are extracted individually.")
            else
                throw(string("arraysize AST node didn't have 2 or 3 arguments."))
            end
        elseif isBaseFunc(fun, :reshape) || fun==GlobalRef(ParallelAccelerator.API,:reshape)
            # rhs.args[2] is the array to be reshaped, lhs is the result, rhs.args[3] is a tuple with new shape
            if haskey(state.tuple_table, args[2])
                @dprintln(3,"reshape tuple found in tuple_table = ", state.tuple_table[args[2]])
                checkAndAddSymbolCorrelation(lhs, state, state.tuple_table[args[2]])
            end
        elseif isBaseFunc(fun, :tuple)
            @dprintln(3,"tuple added to tuple_table = ", args)
            # no need to check since checkAndAddSymbolCorrelation already checks symbols
            state.tuple_table[lhs]=args
            #ok = true
            #for s in args
            #    if !(isa(s,TypedVar) || isa(s,Int))
            #        ok = false
            #    end
            #end
            #if ok
            #    state.tuple_table[lhs]=args[1:end]
            #end
        end
    elseif rhs.head == :mmap! || rhs.head == :mmap || rhs.head == :map! || rhs.head == :map
        # Arguments to these domain operations implicit assert that equality of sizes so add/merge equivalence classes for the arrays to this operation.
        rhs_corr = extractArrayEquivalencies(rhs, state)
        @dprintln(3,"lhs = ", lhs, " type = ", typeof(lhs))
        if rhs_corr != nothing && isa(lhs, RHSVar)
            lhs = toLHSVar(lhs)
            # if an array has correlation already, there might be a case of multiple assignments
            # in this case, try to make sure sizes are the same or assign a new negative value otherwise
            if haskey(state.array_length_correlation, lhs)
                prev_corr = state.array_length_correlation[lhs]
                prev_size = []
                rhs_size = []
                for (d, v) in state.symbol_array_correlation
                    if v==prev_corr
                        prev_size = d
                    end
                    if v==rhs_corr
                        rhs_size = d
                    end
                end
                if prev_size==[] || rhs_size==[] || prev_size!=rhs_size
                    # can't make sure sizes are always equal, assign negative correlation to lhs
                    state.array_length_correlation[lhs] = getNegativeCorrelation(state)
                    @dprintln(3, "multiple assignment detected, negative correlation assigned for ", lhs)
                end
            else
                lhs_corr = getOrAddArrayCorrelation(toLHSVar(lhs), state)
                merge_correlations(state, lhs_corr, rhs_corr)
                @dprintln(3,"Correlations after map merge into lhs")
                print_correlations(3, state)
            end
        end
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function create_equivalence_classes_assignment(lhs, rhs::ANY, state)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function getNegativeCorrelation(state)
    state.multi_correlation -= 1
    return state.multi_correlation
end

function print_correlations(level, state)
    if !isempty(state.array_length_correlation)
        dprintln(level,"array_length_correlations = ")
        for (k,v) in state.array_length_correlation
            dprintln(level, "    ", k," => ",v)
        end
    end
    if !isempty(state.symbol_array_correlation)
        dprintln(level,"symbol_array_correlations = ")
        for (k,v) in state.symbol_array_correlation
            dprintln(level, "    ", k," => ",v)
        end
    end
    if !isempty(state.range_correlation)
        dprintln(level,"range_correlations = ")
        for i in state.range_correlation
            dprint(level, "    ")
            for j in i[1]
                if isa(j, RangeData)
                    dprint(level, j.exprs, " ")
                else
                    dprint(level, j, " ")
                end
            end
            dprintln(level, " => ", i[2])
        end
    end
end

"""
AstWalk callback to determine the array equivalence classes.
"""
function create_equivalence_classes(node :: Expr, state :: expr_state, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    @dprintln(3,"create_equivalence_classes starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(3,"create_equivalence_classes node = ", node, " type = ", typeof(node))
    @dprintln(3,"node.head: ", node.head)
    print_correlations(3, state)

    if node.head == :lambda
        save_LambdaVarInfo  = state.LambdaVarInfo
        linfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(node)
        state.LambdaVarInfo = linfo
        AstWalk(body, create_equivalence_classes, state)
        state.LambdaVarInfo = save_LambdaVarInfo
        return node
    end

    # FIXME: why?
    # We can only extract array equivalences from top-level statements.
    # get equivalences from non-top-level also
    # use case: pre statements of parfors in HPAT matrix multiply optimization
    if true #is_top_level
        @dprintln(3,"create_equivalence_classes is_top_level")

        if isAssignmentNode(node)
            # Here the node is an assignment.
            @dprintln(3,"Is an assignment node.")
            # return value here since this function can replace arraysize() calls
            return create_equivalence_classes_assignment(toLHSVar(node.args[1]), node.args[2], state)
        else
            if node.head == :mmap! || node.head == :mmap || node.head == :map! || node.head == :map
                extractArrayEquivalencies(node, state)
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function create_equivalence_classes(node :: ANY, state :: expr_state, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    @dprintln(3,"create_equivalence_classes starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(3,"create_equivalence_classes node = ", node, " type = ", typeof(node))
    @dprintln(3,"Not an assignment or expr node.")
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
do not recurse into domain lambdas since there could be variable name conflicts
"""
function create_equivalence_classes(node::DomainLambda, state :: expr_state, top_level_number :: Int64, is_top_level :: Bool, read :: Bool)
    return node
end

"""
Given an array whose name is in "x", allocate a new equivalence class for this array.
"""
function addUnknownArray(x :: LHSVar, state :: expr_state)
    @dprintln(3, "addUnknownArray x = ", x, " next = ", state.next_eq_class)
    m = state.next_eq_class
    state.next_eq_class += 1
    state.array_length_correlation[x] = m + 1
end

"""
Given an array of RangeExprs describing loop nest ranges, allocate a new equivalence class for this range.
"""
function addUnknownRange(x :: Array{DimensionSelector,1}, state :: expr_state)
    m = state.next_eq_class
    state.next_eq_class += 1
    state.range_correlation[x] = m + 1
end

"""
If we somehow determine that two sets of correlations are actually the same length then merge one into the other.
"""
function merge_correlations(state, unchanging, eliminate)
    if unchanging < 0 || eliminate < 0
        @dprintln(3,"merge_correlations will not merge because ", unchanging, " and/or ", eliminate, " represents an array that is multiply defined within the function.")
        return unchanging
    end

    # For each array in the dictionary.
    for i in state.array_length_correlation
        # If it is in the "eliminate" class...
        if i[2] == eliminate
            # ...move it to the "unchanging" class.
            state.array_length_correlation[i[1]] = unchanging
        end
    end
    # The symbol_array_correlation shares the equivalence class space so
    # do the same re-numbering here.
    for i in state.symbol_array_correlation
        if i[2] == eliminate
            state.symbol_array_correlation[i[1]] = unchanging
        end
    end
    # The range_correlation shares the equivalence class space so
    # do the same re-numbering here.
    for i in state.range_correlation
        if i[2] == eliminate
            state.range_correlation[i[1]] = unchanging
        end
    end

    return unchanging
end

"""
If we somehow determine that two arrays must be the same length then
get the equivalence classes for the two arrays and merge those equivalence classes together.
"""
function add_merge_correlations(old_sym :: LHSVar, new_sym :: LHSVar, state :: expr_state)
    @dprintln(3, "add_merge_correlations ", old_sym, " ", new_sym)
    print_correlations(3, state)
    old_corr = getOrAddArrayCorrelation(old_sym, state)
    new_corr = getOrAddArrayCorrelation(new_sym, state)
    ret = merge_correlations(state, old_corr, new_corr)
    @dprintln(3, "add_merge_correlations post")
    print_correlations(3, state)

    return ret
end

"""
Return a correlation set for an array.  If the array was not previously added then add it and return it.
"""
function getOrAddArrayCorrelation(x :: LHSVar, state :: expr_state)
    if !haskey(state.array_length_correlation, x)
        @dprintln(3,"Correlation for array not found = ", x)
        addUnknownArray(x, state)
    end
    state.array_length_correlation[x]
end

"""
"node" is a domainIR node.  Take the arrays used in this node, create an array equivalence for them if they
don't already have one and make sure they all share one equivalence class.
"""
function extractArrayEquivalencies(node :: Expr, state)
    input_args = node.args

    # Make sure we get what we expect from domain IR.
    # There should be two entries in the array, another array of input array symbols and a DomainLambda type
    if(length(input_args) < 2)
        throw(string("extractArrayEquivalencies input_args length should be at least 2 but is ", length(input_args)))
    end

    # First arg is an array of input arrays to the mmap!
    input_arrays = input_args[1]
    len_input_arrays = length(input_arrays)
    @dprintln(2,"Number of input arrays: ", len_input_arrays)
    @dprintln(3,"input_arrays =  ", input_arrays)
    assert(len_input_arrays > 0)

    # Second arg is a DomainLambda
    ftype = typeof(input_args[2])
    @dprintln(2,"extractArrayEquivalencies function = ",input_args[2])
    if(ftype != DomainLambda)
        throw(string("extractArrayEquivalencies second input_args should be a DomainLambda but is of type ", typeof(input_args[2])))
    end

#    if !isa(input_arrays[1], RHSVar)
#        @dprintln(1, "extractArrayEquivalencies input_arrays[1] is not RHSVar")
#        return nothing
#    end

    inputInfo = InputInfo[]
    for i = 1 : length(input_arrays)
        push!(inputInfo, get_mmap_input_info(input_arrays[i], state))
    end
#    num_dim_inputs = findSelectedDimensions(inputInfo, state)
    @dprintln(3, "inputInfo = ", inputInfo)

    main_length_correlation = getCorrelation(inputInfo[1], state)
    # Get the correlation set of the first input array.
    #main_length_correlation = getOrAddArrayCorrelation(toLHSVar(input_arrays[1]), state)

    # Make sure each input array is a TypedVar
    # Also, create indexed versions of those symbols for the loop body
    for i = 2:length(inputInfo)
        @dprintln(3,"extractArrayEquivalencies input_array[i] = ", input_arrays[i], " type = ", typeof(input_arrays[i]))
        this_correlation = getCorrelation(inputInfo[i], state)
        # Verify that all the inputs are the same size by verifying they are in the same correlation set.
        if this_correlation != main_length_correlation
            merge_correlations(state, main_length_correlation, this_correlation)
        end
    end

    @dprintln(3,"extractArrayEq result")
    print_correlations(3, state)
    return main_length_correlation
end

"""
Make sure all the dimensions are TypedVars or constants.
Make sure each dimension variable is assigned to only once in the function.
Extract just the dimension variables names into dim_names and then register the correlation from lhs to those dimension names.
"""
function checkAndAddSymbolCorrelation(lhs :: LHSVar, state, dim_array)
    dim_names = Union{RHSVar,Int}[]

    for i = 1:length(dim_array)
        # constant sizes are either TypedVars, Symbols or Ints, TODO: expand to GenSyms that are constant
        if !(isa(dim_array[i],RHSVar) || isa(dim_array[i], Int))
            @dprintln(3, "checkAndAddSymbolCorrelation dim not Int or RHSVar ", dim_array[i])
            return false
        end
        dim_array[i] = toLHSVar(dim_array[i])
        desc = 0
        if isa(dim_array[i],Symbol) && !(CompilerTools.LambdaHandling.getType(dim_array[i], state.LambdaVarInfo)<:Int)
            @dprintln(3, "checkAndAddSymbolCorrelation dim symbol not Int ", dim_array[i])
            throw(string("Dimension not an Int"))
        end
        if !isa(dim_array[i],Int)
            desc = CompilerTools.LambdaHandling.getDesc(dim_array[i], state.LambdaVarInfo)
        end
        # FIXME: description of input parameters not changed in function is always 0?
        if !isa(dim_array[i],Int) && ((desc & ISASSIGNED == ISASSIGNED) && !(desc & ISASSIGNEDONCE == ISASSIGNEDONCE))
            @dprintln(3, "checkAndAddSymbolCorrelation dim not Int or assigned once ", dim_array[i])
            return false
        end
        push!(dim_names, dim_array[i])
    end

    @dprintln(3, "Will establish array length correlation for const size lhs = ", lhs, " dims = ", dim_names)
    getOrAddSymbolCorrelation(lhs, state, dim_names)
    return true
end

"""
Gets (or adds if absent) the range correlation for the given array of RangeExprs.
"""
function getOrAddRangeCorrelation(array, ranges :: Array{DimensionSelector,1}, state :: expr_state)
    @dprintln(3, "getOrAddRangeCorrelation for ", array, " with ranges = ", ranges)
    if print_times
        @dprintln(3, "with hash = ", hash(ranges))
    end
    print_correlations(3, state)

    num_dims = length(ranges)
    num_range = 0
    num_mask = 0
    num_single = 0
    masks = MaskSelector[]

    # We can't match on array of RangeExprs so we flatten to Array of Any
    for i = 1:length(ranges)
        if isa(ranges[i], MaskSelector)
            num_mask += 1
            push!(masks, ranges[i])
        elseif isa(ranges[i], SingularSelector)
            num_single += 1
        else
            num_range += 1
        end
    end
    all_mask = (num_mask == num_dims)
    @dprintln(3, "Selector Types: ", num_range, " ", num_mask, " ", num_single, " ", all_mask)

    if !haskey(state.range_correlation, ranges)
        @dprintln(3,"Exact match for correlation for range not found = ", ranges)
        # Look for an equivalent but non-exact range in the dictionary.
        nonExactCorrelation = nonExactRangeSearch(ranges, state.range_correlation)
        if nonExactCorrelation == nothing
            @dprintln(3, "No non-exact match so adding new range")

            if num_mask == 1 && num_range == 0 # one mask'ed dimension and all rest singular
                # If there is only one mask used and all the rest of the dimensions are singular then the
                # iteration space is equivalent to the mask dimension.  So, we get the mask array and find
                # out the correlation for that array and make this range have the same correlation.
                mask_array = masks[1].value
                mask_correlation = getCorrelation(mask_array, state)
                state.range_correlation[ranges] = mask_correlation
                @dprintln(3, "Only one mask dimension and rest singular to correlating this range with mask's correlation.")
            else
                range_corr = addUnknownRange(ranges, state)

                # If all the dimensions are selected based on masks then the iteration space
                # is that of the entire array and so we can establish a correlation between the
                # DimensionSelector and the whole array.
                if all_mask
                    masked_array_corr = getOrAddArrayCorrelation(toLHSVar(array), state)
                    @dprintln(3, "All dimension selectors are masks so establishing correlation to main array ", masked_array_corr)
                    range_corr = merge_correlations(state, masked_array_corr, range_corr)

                    if length(ranges) == 1
                        print_correlations(3, state)
                        mask_correlation = getCorrelation(ranges[1].value, state)

                        @dprintln(3, "Range length is 1 so establishing correlation between range ", range_corr, " and the mask ", ranges[1].value, " with correlation ", mask_correlation)
                        range_corr = merge_correlations(state, mask_correlation, range_corr)
                    end
                end
            end
        else
            # Found an equivalent range.
            @dprintln(3, "Adding non-exact range match to class ", nonExactCorrelation)
            state.range_correlation[ranges] = nonExactCorrelation
        end
        @dprintln(3, "getOrAddRangeCorrelation final correlations")
        print_correlations(3, state)
    end
    state.range_correlation[ranges]
end

"""
A new array is being created with an explicit size specification in dims.
"""
function getOrAddSymbolCorrelation(array :: LHSVar, state :: expr_state, dims :: Array{Union{RHSVar,Int},1})
    if !haskey(state.symbol_array_correlation, dims)
        # We haven't yet seen this combination of dims used to create an array.
        @dprintln(3,"Correlation for symbol set not found, dims = ", dims)
        if haskey(state.array_length_correlation, array)
            return state.symbol_array_correlation[dims] = state.array_length_correlation[array]
        else
            # Create a new array correlation number for this array and associate that number with the dim sizes.
            return state.symbol_array_correlation[dims] = addUnknownArray(array, state)
        end
    else
        @dprintln(3,"Correlation for symbol set found, dims = ", dims)
        # We have previously seen this combination of dim sizes used to create an array so give the new
        # array the same array length correlation number as the previous one.
        return state.array_length_correlation[array] = state.symbol_array_correlation[dims]
    end
end

type ReplaceConstArraysizesData
    lives             :: CompilerTools.LivenessAnalysis.BlockLiveness
    linfo
    array_length_correlation
    symbol_array_correlation
    saved_available_variables::Vector{LHSVar}
    # empty for now
    range_correlation        :: Dict{Array{DimensionSelector,1},Int}
    function ReplaceConstArraysizesData(lv,li,al,sa)
        new(lv,li,al,sa,Vector{LHSVar}(),Dict{Array{DimensionSelector,1},Int}())
    end
end

"""
Replace arraysize() calls for arrays with known constant sizes.
Constant size is Int constants, as well as assigned once variables which are
in symbol_array_correlation. Variables should be assigned before current statement, however.
"""
function replaceConstArraysizes(node::Expr, data::ReplaceConstArraysizesData, top_level_number::Int64, is_top_level::Bool, read::Bool)
    @dprintln(4,"replaceConstArraysizes starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(4,"replaceConstArraysizes node = ", node, " type = ", typeof(node))
    @dprintln(4,"node.head: ", node.head)
    print_correlations(4, data)

    if node.head==:parfor
        replaceConstArraysizes_parfor(node.args[1], data, top_level_number)
        return node
    end

    if !isCall(node) return CompilerTools.AstWalker.ASTWALK_RECURSE end
    func = getCallFunction(node)
    if !isBaseFunc(func,:arraysize) && !isBaseFunc(func,:arraylen)
        return CompilerTools.AstWalker.ASTWALK_RECURSE
    end

    @dprintln(3, "replaceConstArraysizes size call found ", node)
    live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
    @dprintln(3, "replaceConstArraysizes live info ", live_info)
    args = getCallArguments(node)
    arr = toLHSVar(args[1])
    # get array sizes if available, there could be multiple
    size_syms_arr = get_array_correlation_symbols(arr, data)

    if length(size_syms_arr)==0
        @dprintln(3, "replaceConstArraysizes correlation symbol not found ", node)
        print_correlations(3, data)
        return CompilerTools.AstWalker.ASTWALK_RECURSE
    end
    @dprintln(3, "replaceConstArraysizes correlation symbols: ", size_syms_arr)
    # need to make sure size variables are available in this statement to replace
    available_variables = get_available_variables(top_level_number, data)

    if isBaseFunc(getCallFunction(node), :arraysize)
        dim_ind = args[2] # dimension number
        if !isa(dim_ind,Int)
            @dprintln(3, "arraysize() index is not constant ", node)
            return CompilerTools.AstWalker.ASTWALK_RECURSE
        end
        for size_syms in size_syms_arr
            res = size_syms[dim_ind]
            # only replace when the size is constant or a valid live variable
            # the size variable is live by construction
            # since the array is allocated once, its allocation variables are live (in(res, live_info.live_in) not needed)
            # check def since a symbol correlation might be defined with current arraysize() in reverse direction
            if isa(res,Int) || (in(res,available_variables) && live_info!=nothing && !in(res,live_info.def) )
                @dprintln(3, "arraysize() replaced: ", node," res ", res)
                return res
            end
        end
    end

    if isBaseFunc(getCallFunction(node), :arraylen)
        for size_syms in size_syms_arr
            # make sure all dimension sizes are either constant or valid symbols (not reverse defined)
            if mapreduce(x-> isa(x,Int) || (in(x,available_variables) && live_info!=nothing && !in(x,live_info.def)), &, size_syms)
                res = mk_mult_int_expr(size_syms)
                @dprintln(3, "arraylen() replaced: ", node," res ", res)
                return res
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function replaceConstArraysizes_parfor(parfor::PIRParForAst, data, top_level_number)
    body_lives = CompilerTools.LivenessAnalysis.from_lambda(data.linfo, parfor.body, pir_live_cb, data.linfo)
    available_variables = get_available_variables(top_level_number, data)
    new_data = ReplaceConstArraysizesData(body_lives, data.linfo,
       data.array_length_correlation, data.symbol_array_correlation)
    new_data.saved_available_variables = union(available_variables, data.saved_available_variables)
    @dprintln(3,"replaceConstArraysizes parfor body lives = ", data.lives)
    new_body = AstWalk(TypedExpr(nothing, :body, parfor.body...), replaceConstArraysizes, new_data)
    parfor.body = new_body.args
end

function get_available_variables(top_level_number, data)
    # get dominant block information
    dom_dict = CompilerTools.CFGs.compute_dominators(data.lives.cfg)
    bb_index = CompilerTools.LivenessAnalysis.find_bb_for_statement(top_level_number, data.lives)
    @dprintln(3, "get_available_variables dom_dict ", dom_dict, " bb_index ", bb_index)
    available_variables = Set{LHSVar}()
    available_variables = union(data.saved_available_variables, available_variables)
    # input parameters are also available
    available_variables = union(getInputParametersAsLHSVar(data.linfo), available_variables)
    if bb_index==nothing
        return available_variables
    end
    dom_bbs = dom_dict[bb_index]

    # find def variables for dominant blocks except current one
    for i in dom_bbs
        if i==bb_index continue end
        bb = CompilerTools.LivenessAnalysis.getBasicBlockFromBlockNumber(i, data.lives)
        available_variables = union(bb.def, available_variables)
    end
    bb = CompilerTools.LivenessAnalysis.getBasicBlockFromBlockNumber(bb_index, data.lives)
    # find def variables for previous statements of same block
    for stmts in bb.statements
      if stmts.tls.index < top_level_number
          available_variables = union(stmts.def, available_variables)
      end
    end
    #live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, state.block_lives)
    #available_variables = union(live_info.live_in, available_variables)
    @dprintln(3, "get_available_variables returns ", available_variables)
    return available_variables
end

"""
Find correlation symbols of an array if available. Return empty array otherwise.
"""
function get_array_correlation_symbols(arr::LHSVar, state)
    out = []
    if haskey(state.array_length_correlation, arr)
        arr_class = state.array_length_correlation[arr]
        for (d, v) in state.symbol_array_correlation
            if v==arr_class
                push!(out, d)
            end
        end
    end
    return out
end

function replaceConstArraysizes(node::ANY, state::ReplaceConstArraysizesData, top_level_number::Int64, is_top_level::Bool, read::Bool)
    @dprintln(4,"replaceConstArraysizes starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(4,"replaceConstArraysizes node = ", node, " type = ", typeof(node))
    @dprintln(4,"Not an expr node.")
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Replace arraysize calls in range correlations like 1:arraysize() which are generated
from expressions like A[:,i] (kmeans_gen case).
"""
function replaceConstArraysizesRangeCorrelations(range_correlations::Dict{Array{DimensionSelector,1},Int}, state::ReplaceConstArraysizesData)
    @dprintln(3,"replaceConstArraysizesRangeCorrelations ", keys(range_correlations))
    for a in keys(range_correlations)
        for r in a
            @dprintln(3,"replaceConstArraysizesRangeCorrelations ", r)
            replaceConstArraysizesRange(r, state)
        end
    end
    nothing
end

function replaceConstArraysizesRange(r::RangeData, state::ReplaceConstArraysizesData)
    @dprintln(3,"replaceConstArraysizesRangeCorrelations ", r.exprs.last_val)
    r.exprs.last_val = replaceConstArraysizesRangeNode(r.exprs.last_val, state)
end

replaceConstArraysizesRange(r::ANY, state::ReplaceConstArraysizesData) = nothing

function replaceConstArraysizesRangeNode(node::Expr, state::ReplaceConstArraysizesData)
    @dprintln(3,"replaceConstArraysizesRangeCorrelations RangeExprs last_val ", node)
    if !isCall(node) return node end
    func = getCallFunction(node)
    if !isBaseFunc(func,:arraysize) && !isBaseFunc(func,:arraylen)
        return node
    end
    @dprintln(3,"replaceConstArraysizesRangeCorrelations size call found", node)
    args = getCallArguments(node)
    arr = toLHSVar(args[1])
    # get array sizes if available, there could be multiple
    size_syms_arr = get_array_correlation_symbols(arr, state)
    available_variables = getInputParametersAsLHSVar(state.linfo)
    if isBaseFunc(func, :arraysize)
        dim_ind::Int = args[2] # dimension number
        for size_syms in size_syms_arr
            res = size_syms[dim_ind]
            if isa(res,Int) || in(res,available_variables)
                @dprintln(3, "arraysize() in range replaced: ", node," res ", res)
                return res
            end
        end
    end

    if isBaseFunc(func, :arraylen)
        for size_syms in size_syms_arr
            if mapreduce(x-> isa(x,Int) || in(x,available_variables), &, size_syms)
                res = mk_mult_int_expr(size_syms)
                @dprintln(3, "arraylen() in range replaced: ", node," res ", res)
                return res
            end
        end
    end
    return node
end

replaceConstArraysizesRangeNode(node::ANY, state::ReplaceConstArraysizesData) = node

"""
Implements one of the main ParallelIR passes to remove assertEqShape AST nodes from the body if they are statically known to be in the same equivalence class.
"""
function removeAssertEqShape(args :: Array{Any,1}, state)
    newBody = Any[]
    for i = 1:length(args)
        # Add the current statement to the new body unless the statement is an assertEqShape Expr and the array in the assertEqShape are known to be the same size.
        if !(typeof(args[i]) == Expr && args[i].head == :assertEqShape && from_assertEqShape(args[i], state))
            push!(newBody, args[i])
        end
    end
    return newBody
end

"""
Create array equivalences from an assertEqShape AST node.
There are two arrays in the args to assertEqShape.
"""
function from_assertEqShape(node::Expr, state)
    @dprintln(3,"from_assertEqShape ", node)
    a1 = node.args[1]    # first array to compare
    a2 = node.args[2]    # second array to compare
    a1_corr = getOrAddArrayCorrelation(toLHSVar(a1), state)  # get the length set of the first array
    a2_corr = getOrAddArrayCorrelation(toLHSVar(a2), state)  # get the length set of the second array
    if a1_corr == a2_corr
        # If they are the same then return an empty array so that the statement is eliminated.
        @dprintln(3,"assertEqShape statically verified and eliminated for ", a1, " and ", a2)
        return true
    else
        @dprintln(3,"a1 = ", a1, " ", a1_corr, " a2 = ", a2, " ", a2_corr, " correlations")
        print_correlations(3, state)
        # If assertEqShape is called on e.g., inputs, then we can't statically eliminate the assignment
        # but if the assert doesn't fire then we do thereafter know that the arrays are in the same length set.
        merge_correlations(state, a1_corr, a2_corr)
        @dprintln(3,"assertEqShape NOT statically verified.  Merge correlations")
        print_correlations(3, state)
        return false
    end
end

"""
State for the remove_no_deps and insert_no_deps_beginning phases.
"""
type HoistInvariants
    lives             :: CompilerTools.LivenessAnalysis.BlockLiveness
    hoisted_stmts     :: Array{Any,1}
    hoistable_scalars :: Set{LHSVar}
    LambdaVarInfo
    fas               :: findAllocationsState

    function HoistInvariants(l, hs, lvi, fas)
        new(l, Any[], hs, lvi, fas)
    end
end

"""
# This routine gathers up nodes that do not use
# any variable and removes them from the AST into top_level_no_deps.  This works in conjunction with
# insert_no_deps_beginning above to move these statements with no dependencies to the beginning of the AST
# where they can't prevent fusion.
"""
function hoist_invariants(node :: Expr, data :: HoistInvariants, top_level_number, is_top_level, read)
    @dprintln(3,"hoist_invariants starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(3,"hoist_invariants node = ", node, " type = ", typeof(node))
    @dprintln(3,"node.head: ", node.head)
    head = node.head

    if is_top_level
        @dprintln(3,"hoist_invariants is_top_level")

        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        if live_info == nothing
            @dprintln(3,"hoist_invariants no live_info")
        else
            @dprintln(3,"hoist_invariants live_info = ", live_info)
            @dprintln(3,"hoist_invariants live_info.use = ", live_info.use)

            # Here we try to determine which scalar assigns can be hoisted to the beginning of the function.
            #
            # If this statement defines some variable.
            if !isempty(live_info.def)
                @dprintln(3, "Checking if the statement is hoistable.")
                @dprintln(3, "Previous hoistables = ", data.hoistable_scalars)
                hoistable_names = [lookupVariableName(x, data.LambdaVarInfo) for x in data.hoistable_scalars]
                @dprintln(3, "Previous hoistable names = ", hoistable_names)

                # Assume that hoisting is safe until proven otherwise.
                dep_only_on_parameter = true
                # Look at all the variables on which this statement depends.
                # If any of them are not a hoistable scalar then we can't hoist the current scalar definition.
                for i in live_info.use
                    if !in(i, data.hoistable_scalars)
                        @dprintln(3, "Could not hoist because the statement depends on :", i)
                        dep_only_on_parameter = false
                        break
                    end
                end

                # See if there are any calls with side-effects that could prevent moving.
                sews = SideEffectWalkState()
                ParallelAccelerator.ParallelIR.AstWalk(node, hasNoSideEffectWalk, sews)
                if sews.hasSideEffect
                    dep_only_on_parameter = false
                end

                if dep_only_on_parameter
                    @dprintln(3,"Statement does not have any side-effects.")
                    # If this statement is defined in more than one place then it isn't hoistable.
                    for i in live_info.def
                        @dprintln(3,"Checking if ", i, " is multiply defined.")
                        @dprintln(4,"data.lives = ", data.lives)
                        def_type = CompilerTools.LambdaHandling.getType(i, data.LambdaVarInfo)
                        # Two strategies here.  If the type of the def is a bits type then we can simply check for multiple definition.
                        # If it is a non-bits type (which can alias) then we need to do a scan of allocations and get all the aliases
                        # for those allocations and see if any of those allocations survive the parfor.  There are two possibilities for
                        # arrays allocated in a parfor: 1) either things are put into an array and then some reduction is run across the
                        # array to summarize it to a scalar or 2) the array is allocated per iteration and stored into an array of arrays.
                        if isbits(def_type)
                            if CompilerTools.LivenessAnalysis.countSymbolDefs(i, data.lives) > 1
                                @dprintln(3, "Could not hoist because the function has multiple definitions of: ", i)
                                dep_only_on_parameter = false
                                break
                            end
                        else
                            if !haskey(data.fas.allocs, i)
                                @dprintln(3, "Non-bits type def is not in alloc set in findAllocationStats. ", i)
                                dep_only_on_parameter = false
                                break
                            end
                            if data.fas.allocs[i] != 1
                                @dprintln(3, i, " was allocated ", data.fas.allocs[i], " times in the parfor body.")
                                dep_only_on_parameter = false
                                break
                            end
                            if length(data.fas.arrays_stored_in_arrays) != 0
                                # TODO Implement alias analysis and also an array could be put in
                                # an object and then that stored in the array and we should catch that
                                # as well.
                                @dprintln(3, i, " was allocated but there are arrays stored in arrays in this parfor and we haven't implemented the alias analysis yet to know if the allocated array is escaping the parfor.")
                                dep_only_on_parameter = false
                                break
                            end
                        end
                    end

                    if dep_only_on_parameter
                        @dprintln(3,"hoist_invariants removing ", node, " because it only depends on hoistable scalars.")
                        push!(data.hoisted_stmts, node)
                        # If the defs in this statement are hoistable then other statements which depend on them may also be hoistable.
                        for i in live_info.def
                            push!(data.hoistable_scalars, i)
                        end
                        return CompilerTools.AstWalker.ASTWALK_REMOVE
                    end
                else
                    @dprintln(3,"Statement DOES have side-effects.")
                end
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function hoist_invariants(node::ANY, data :: HoistInvariants, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end
