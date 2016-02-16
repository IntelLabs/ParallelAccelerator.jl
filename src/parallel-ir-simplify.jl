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

"""
Try to hoist allocations outside the loop if possible.
"""
function hoistAllocation(ast::Array{Any,1}, lives, domLoop::DomLoops, state :: expr_state)
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

        # TODO: is this correct?
        if (is(preBlk, nothing) || length(preBlk.statements) == 0) continue end
        # if is(preBlk, nothing) continue end
        tls = lives.basic_blocks[ preBlk ]

        
        # TODO: is this correct?
        preHead = preBlk.statements[end].index
        #preHead = headBlk.statements[1].index-1
        
        head = headBlk.statements[1].index
        tail = tailBlk.statements[1].index
        @dprintln(3, "HA: line before head is ", ast[preHead-1])
        # Is iterating through statement indices this way safe?
        for i = head:tail
            
            if isAssignmentNode(ast[i]) && isAllocation(ast[i].args[2])
                @dprintln(3, "HA: found allocation at line ", i, ": ", ast[i])
                lhs = ast[i].args[1]
                rhs = ast[i].args[2]
                if isa(lhs, SymbolNode) lhs = lhs.name end
                if (haskey(state.array_length_correlation, lhs))
                    c = state.array_length_correlation[lhs]
                    for (d, v) in state.symbol_array_correlation
                        if v == c
                            ok = true
                            for j = 1:length(d)
                                if !in(d[j], tls.live_out)
                                    ok = false
                                    break
                                end
                            end
                            @dprintln(3, "HA: found correlation dimension ", d, " ", ok, " ", length(rhs.args)-6)
                            if ok && length(rhs.args) - 6 == 2 * length(d) # dimension must match
                                rhs.args = rhs.args[1:6]
                                for s in d
                                    push!(rhs.args, SymbolNode(s, Int))
                                    push!(rhs.args, 0)
                                end
                                @dprintln(3, "HA: hoist ", ast[i], " out of loop before line ", head)
                                ast = [ ast[1:preHead-1], ast[i], ast[preHead:i-1], ast[i+1:end] ]
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
    hoistable_scalars :: Set{SymGen}
    dict_sym          :: Dict{SymGen, DictInfo}
    change            :: Bool

    function RemoveNoDepsState(l, hs)
        new(l, Any[], hs, Dict{SymGen, DictInfo}(), false)
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
            data.dict_sym = Dict{SymGen,DictInfo}()
        end

        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        # Remove line number statements.
        if head == :line
            return CompilerTools.AstWalker.ASTWALK_REMOVE
        end
        if live_info == nothing
            @dprintln(3,"remove_no_deps no live_info")
        else
            @dprintln(3,"remove_no_deps live_info = ", live_info)
            @dprintln(3,"remove_no_deps live_info.use = ", live_info.use)

            if isa(node, Number) || isa(node, SymAllGen)
                @dprintln(3,"Eliminating dead node: ", node)
                return CompilerTools.AstWalker.ASTWALK_REMOVE
            elseif isAssignmentNode(node)
                @dprintln(3,"Is an assignment node.")
                lhs = node.args[1]
                @dprintln(4,lhs)
                rhs = node.args[2]
                @dprintln(4,rhs)

                if isa(rhs, Expr) && (is(rhs.head, :parfor) || is(rhs.head, :mmap!))
                    # Always keep parfor assignment in order to work with fusion
                    @dprintln(3, "keep assignment due to parfor or mmap! node")
                    return node
                end
                if isa(lhs, SymAllGen)
                    lhs_sym = toSymGen(lhs)
                    @dprintln(3,"remove_no_deps found assignment with lhs symbol ", lhs, " ", rhs, " typeof(rhs) = ", typeof(rhs))
                    # Remove a dead store
                    if !in(lhs_sym, live_info.live_out)
                        data.change = true
                        @dprintln(3,"remove_no_deps lhs is NOT live out")
                        if hasNoSideEffects(rhs)
                            @dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
                            return CompilerTools.AstWalker.ASTWALK_REMOVE
                        else
                            # Just eliminate the assignment but keep the rhs
                            @dprintln(3,"Eliminating dead variable but keeping rhs, dead = ", lhs_sym)
                            return rhs
                        end
                    else
                        @dprintln(3,"remove_no_deps lhs is live out")
                        if isa(rhs, SymAllGen)
                            rhs_sym = toSymGen(rhs)
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

                if dep_only_on_parameter 
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
                end
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function remove_no_deps(node::Union{LabelNode,GotoNode}, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    if is_top_level
        # Empty the state at the end or begining of a basic block
        data.dict_sym = Dict{SymGen,DictInfo}()
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end


function remove_no_deps(node :: LineNumberNode, data :: RemoveNoDepsState, top_level_number, is_top_level, read)
    if is_top_level
        # remove line number nodes
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
    lives :: CompilerTools.LivenessAnalysis.BlockLiveness
end

"""
An AstWalk callback that uses liveness information in "data" to remove dead stores.
"""
function remove_dead(node, data :: RemoveDeadState, top_level_number, is_top_level, read)
    @dprintln(3,"remove_dead starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(3,"remove_dead node = ", node, " type = ", typeof(node))
    if typeof(node) == Expr
        @dprintln(3,"node.head = ", node.head)
    end
    ntype = typeof(node)

    if is_top_level
        @dprintln(3,"remove_dead is_top_level")
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)
        if live_info != nothing
            @dprintln(3,"remove_dead live_info = ", live_info)
            @dprintln(3,"remove_dead live_info.use = ", live_info.use)

            if isAssignmentNode(node)
                @dprintln(3,"Is an assignment node.")
                lhs = node.args[1]
                @dprintln(4,lhs)
                rhs = node.args[2]
                @dprintln(4,rhs)

                if typeof(lhs) == SymbolNode || typeof(lhs) == Symbol
                    lhs_sym = getSName(lhs)
                    @dprintln(3,"remove_dead found assignment with lhs symbol ", lhs, " ", rhs, " typeof(rhs) = ", typeof(rhs))
                    # Remove a dead store
                    if !in(lhs_sym, live_info.live_out)
                        @dprintln(3,"remove_dead lhs is NOT live out")
                        if hasNoSideEffects(rhs)
                            @dprintln(3,"Eliminating dead assignment. lhs = ", lhs, " rhs = ", rhs)
                            return CompilerTools.AstWalker.ASTWALK_REMOVE
                        else
                            # Just eliminate the assignment but keep the rhs
                            @dprintln(3,"Eliminating dead variable but keeping rhs, dead = ", lhs_sym, " rhs = ", rhs)
                            return rhs
                        end
                    end
                end
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end


"""
State to aide in the copy propagation phase.
"""
type CopyPropagateState
    lives  :: CompilerTools.LivenessAnalysis.BlockLiveness
    copies :: Dict{SymGen, Union{SymGen,Number}}

    function CopyPropagateState(l, c)
        new(l,c)
    end
end

"""
In each basic block, if there is a "copy" (i.e., something of the form "a = b") then put
that in copies as copies[a] = b.  Then, later in the basic block if you see the symbol
"a" then replace it with "b".  Note that this is not SSA so "a" may be written again
and if it is then it must be removed from copies.
"""
function copy_propagate(node :: ANY, data :: CopyPropagateState, top_level_number, is_top_level, read)
    @dprintln(3,"copy_propagate starting top_level_number = ", top_level_number, " is_top = ", is_top_level)
    @dprintln(3,"copy_propagate node = ", node, " type = ", typeof(node))
    if typeof(node) == Expr
        @dprintln(3,"node.head = ", node.head)
    end
    ntype = typeof(node)

    if is_top_level
        @dprintln(3,"copy_propagate is_top_level")
        live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, data.lives)

        if live_info != nothing
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
        end

        if isa(node, LabelNode) || isa(node, GotoNode) || (isa(node, Expr) && is(node.head, :gotoifnot))
            # Only copy propagate within a basic block.  this is now a new basic block.
            data.copies = Dict{SymGen, SymGen}() 
        elseif isAssignmentNode(node)
            @dprintln(3,"Is an assignment node.")
            lhs = node.args[1] = AstWalk(node.args[1], copy_propagate, data)
            @dprintln(4,lhs)
            rhs = node.args[2] = AstWalk(node.args[2], copy_propagate, data)
            @dprintln(4,rhs)

            if isa(rhs, SymAllGen) || (isa(rhs, Number) && !isa(rhs,Complex)) # TODO: fix complex number case
                @dprintln(3,"Creating copy, lhs = ", lhs, " rhs = ", rhs)
                # Record that the left-hand side is a copy of the right-hand side.
                data.copies[toSymGen(lhs)] = toSymGenOrNum(rhs)
            end
            return node
        end
    end

    return copy_propagate_helper(node, data, top_level_number, is_top_level, read)
end

function copy_propagate_helper(node::Union{Symbol,GenSym},
                               data::CopyPropagateState,
                               top_level_number,
                               is_top_level,
                               read)

    if haskey(data.copies, node)
        @dprintln(3,"Replacing ", node, " with ", data.copies[node])
        return data.copies[node]
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function copy_propagate_helper(node::SymbolNode,
                               data::CopyPropagateState,
                               top_level_number,
                               is_top_level,
                               read)

    if haskey(data.copies, node.name)
        @dprintln(3,"Replacing ", node.name, " with ", data.copies[node.name])
        tmp_node = data.copies[node.name]
        return isa(tmp_node, Symbol) ? SymbolNode(tmp_node, node.typ) : tmp_node
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function copy_propagate_helper(node::DomainLambda,
                               data::CopyPropagateState,
                               top_level_number,
                               is_top_level,
                               read)

    @dprintln(3,"Found DomainLambda in copy_propagate, dl = ", node)
    intersection_dict = Dict{SymGen,Any}()
    # For each statement in this basic block of the form "A = B".
    for copy in data.copies
        # If the DomainLambda has an escaping def of the left-hand side.
        if haskey(node.linfo.escaping_defs, copy[1])
            ed = node.linfo.escaping_defs[copy[1]]
            @dprintln(3, "Found escaping_def ", copy[1], " that is also a lhs in data.copies. ed = ", ed, " type = ", typeof(ed))
            if typeof(copy[2]) == Symbol
                ed.name = copy[2]
                intersection_dict[copy[1]] = SymbolNode(copy[2], ed.typ)
                delete!(node.linfo.escaping_defs, copy[1])
                node.linfo.escaping_defs[copy[2]] = ed
            else
                @dprintln(3, "copy_progation didn't replace in DomainLambda since rhs is GenSym")
            end
        end
    end
    @dprintln(3,"Intersection dict = ", intersection_dict)
    @dprintln(3,"node.linfo.escaping_defs = ", node.linfo.escaping_defs)
    if !isempty(intersection_dict)
        origBody      = node.genBody
        newBody(linfo, args) = CompilerTools.LambdaHandling.replaceExprWithDict(origBody(linfo, args), intersection_dict)
        node.genBody  = newBody
        return node
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function copy_propagate_helper(node::ANY,
                               data::CopyPropagateState,
                               top_level_number,
                               is_top_level,
                               read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
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
        sizes = rhs.args[2]
        n = length(sizes)
        assert(n >= 1 && n <= 3)
        @dprintln(3, "Detected :alloc array allocation. dims = ", sizes)
        checkAndAddSymbolCorrelation(lhs, state, sizes)            
    elseif rhs.head == :call
        @dprintln(3, "Detected call rhs in from_assignment.")
        @dprintln(3, "from_assignment call, arg1 = ", rhs.args[1])
        if length(rhs.args) > 1
            @dprintln(3, " arg2 = ", rhs.args[2])
        end
        if rhs.args[1] == TopNode(:ccall)
            # Same as :alloc above.  Detect an array allocation call and map the specified array sizes to an array equivalence class.
            if rhs.args[2] == QuoteNode(:jl_alloc_array_1d)
                dim1 = rhs.args[7]
                @dprintln(3, "Detected 1D array allocation. dim1 = ", dim1, " type = ", typeof(dim1))
                checkAndAddSymbolCorrelation(lhs, state, [dim1])            
            elseif rhs.args[2] == QuoteNode(:jl_alloc_array_2d)
                dim1 = rhs.args[7]
                dim2 = rhs.args[9]
                @dprintln(3, "Detected 2D array allocation. dim1 = ", dim1, " dim2 = ", dim2)
                checkAndAddSymbolCorrelation(lhs, state, [dim1, dim2])            
            elseif rhs.args[2] == QuoteNode(:jl_alloc_array_3d)
                dim1 = rhs.args[7]
                dim2 = rhs.args[9]
                dim3 = rhs.args[11]
                @dprintln(3, "Detected 2D array allocation. dim1 = ", dim1, " dim2 = ", dim2, " dim3 = ", dim3)
                checkAndAddSymbolCorrelation(lhs, state, [dim1, dim2, dim3])            
            end
        elseif rhs.args[1] == TopNode(:arraylen) || rhs.args[1] == GlobalRef(Core.Intrinsics, :arraylen)
            # This is the other direction.  Takes an array and extract dimensional information that maps to the array's equivalence class.
            array_param = rhs.args[2]                  # length takes one param, which is the array
            assert(isa(array_param, SymbolNode) || isa(array_param, GenSym)) # should be a SymbolNode or GenSym
            array_param_type = CompilerTools.LambdaHandling.getType(array_param, state.LambdaVarInfo) # get its type
            if ndims(array_param_type) == 1            # can only associate when number of dimensions is 1
                dim_symbols = [getSName(lhs)]
                @dprintln(3,"Adding symbol correlation from arraylen, name = ", array_param, " dims = ", dim_symbols)
                checkAndAddSymbolCorrelation(isa(array_param, SymbolNode) ? array_param.name : array_param, state, dim_symbols)
            end
        elseif rhs.args[1] == TopNode(:arraysize)
            # replace arraysize calls when size is known and constant
            arr = toSymGen(rhs.args[2])
            if isa(rhs.args[3],Int) && haskey(state.array_length_correlation, arr)
                arr_class = state.array_length_correlation[arr]
                for (d, v) in state.symbol_array_correlation
                    if v==arr_class
                        #
                        res = d[rhs.args[3]]
                        node = TypedExpr(Int64,:(=),lhs, res)
                        @dprintln(3, "arraysize() replaced: ", node)
                        return node
                    end
                end
            end
            # This is the other direction.  Takes an array and extract dimensional information that maps to the array's equivalence class.
            if length(rhs.args) == 2
                array_param = rhs.args[2]                  # length takes one param, which is the array
                assert(typeof(array_param) == SymbolNode)  # should be a SymbolNode
                array_param_type = array_param.typ         # get its type
                array_dims = ndims(array_param_type)
                dim_symbols = Symbol[]
                for dim_i = 1:array_dims
                    push!(dim_symbols, lhs[dim_i])
                end
                @dprintln(3,"Adding symbol correlation from arraysize, name = ", rhs.args[2].name, " dims = ", dim_symbols)
                checkAndAddSymbolCorrelation(rhs.args[2].name, state, dim_symbols)
            elseif length(rhs.args) == 3
                @dprintln(1,"Can't establish symbol to array length correlations yet in the case where dimensions are extracted individually.")
            else
                throw(string("arraysize AST node didn't have 2 or 3 arguments."))
            end
        elseif rhs.args[1]==GlobalRef(Base,:reshape)
            # rhs.args[2] is the array to be reshaped, lhs is the result, rhs.args[3] is a tuple with new shape
            if haskey(state.tuple_table, rhs.args[3])
                checkAndAddSymbolCorrelation(lhs, state, state.tuple_table[rhs.args[3]])
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
            end
        end
    elseif rhs.head == :mmap! || rhs.head == :mmap || rhs.head == :map! || rhs.head == :map 
        # Arguments to these domain operations implicit assert that equality of sizes so add/merge equivalence classes for the arrays to this operation.
        rhs_corr = extractArrayEquivalencies(rhs, state)
        @dprintln(3,"lhs = ", lhs, " type = ", typeof(lhs))
        if rhs_corr != nothing && isa(lhs, SymAllGen)
            lhs_corr = getOrAddArrayCorrelation(toSymGen(lhs), state) 
            merge_correlations(state, lhs_corr, rhs_corr)
            @dprintln(3,"Correlations after map merge into lhs")
            print_correlations(3, state)
        end
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function create_equivalence_classes_assignment(lhs, rhs::ANY, state)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function print_correlations(level, state)
    if !isempty(state.array_length_correlation)
        dprintln(level,"array_length_correlations = ", state.array_length_correlation)
    end
    if !isempty(state.symbol_array_correlation)
        dprintln(level,"symbol_array_correlations = ", state.symbol_array_correlation)
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
        state.LambdaVarInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaVarInfo(node)
        body = CompilerTools.LambdaHandling.getBody(node)
        AstWalk(body, create_equivalence_classes, state)
        state.LambdaVarInfo = save_LambdaVarInfo
        return node
    end

    # We can only extract array equivalences from top-level statements.
    if is_top_level
        @dprintln(3,"create_equivalence_classes is_top_level")

        if isAssignmentNode(node)
            # Here the node is an assignment.
            @dprintln(3,"Is an assignment node.")
            # return value here since this function can replace arraysize() calls
            return create_equivalence_classes_assignment(node.args[1], node.args[2], state)
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

#    if !isa(input_arrays[1], SymAllGen)
#        @dprintln(1, "extractArrayEquivalencies input_arrays[1] is not SymAllGen")
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
    #main_length_correlation = getOrAddArrayCorrelation(toSymGen(input_arrays[1]), state)

    # Make sure each input array is a SymbolNode
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
Make sure all the dimensions are SymbolNodes or constants.
Make sure each dimension variable is assigned to only once in the function.
Extract just the dimension variables names into dim_names and then register the correlation from lhs to those dimension names.
"""
function checkAndAddSymbolCorrelation(lhs :: SymGen, state, dim_array)
    dim_names = Union{SymGen,Int}[]

    for i = 1:length(dim_array)
        # constant sizes are either SymbolNodes or Ints, TODO: expand to GenSyms that are constant
        if !(typeof(dim_array[i])==SymbolNode || typeof(dim_array[i])==Int)
            return false
        end
        if isa(dim_array[i], SymbolNode) dim_array[i]=dim_array[i].name end
        if !(isa(dim_array[i],Int) || CompilerTools.LambdaHandling.getDesc(dim_array[i], state.LambdaVarInfo) & ISASSIGNEDONCE == ISASSIGNEDONCE)
            return false
        end
        push!(dim_names, dim_array[i])
    end

    @dprintln(3, "Will establish array length correlation for const size lhs = ", lhs, " dims = ", dim_names)
    getOrAddSymbolCorrelation(lhs, state, dim_names)
    return true
end


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
    a1_corr = getOrAddArrayCorrelation(toSymGen(a1), state)  # get the length set of the first array
    a2_corr = getOrAddArrayCorrelation(toSymGen(a2), state)  # get the length set of the second array
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
