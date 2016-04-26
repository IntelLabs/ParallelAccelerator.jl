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
 
function flattenParfors(function_name, ast::Expr)
    flatten_start = time_ns()

    assert(ast.head == :lambda)
    @dprintln(1,"Starting flattenParfors.  function = ", function_name, " ast = ", ast)

    #bt = backtrace() ;
    #s = sprint(io->Base.show_backtrace(io, bt))
    #@dprintln(3, "from_root backtrace ")
    #@dprintln(3, s)
 
    start_time = time_ns()

    LambdaVarInfo = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
    body = CompilerTools.LambdaHandling.getBody(ast)

    expanded_args = Any[]
    flattenParfors(expanded_args, body.args, LambdaVarInfo)
    args = expanded_args

    @dprintln(1,"Flattening parfor bodies time = ", ns_to_sec(time_ns() - flatten_start))

    @dprintln(3, "After flattening")
    for j = 1:length(args)
        @dprintln(3, args[j])
    end

    if shortcut_array_assignment != 0
        fake_body = CompilerTools.LambdaHandling.LambdaVarInfoToLambda(LambdaVarInfo, args)
        new_lives = CompilerTools.LivenessAnalysis.from_expr(fake_body, pir_live_cb, LambdaVarInfo)

        for i = 1:length(args)
            node = args[i]
            if isAssignmentNode(node)
                lhs = node.args[1]
                rhs = node.args[2]
                @dprintln(3,"shortcut_array_assignment = ", node)
                if typeof(lhs) == SymbolNode && isArrayType(lhs) && typeof(rhs) == SymbolNode
                    @dprintln(3,"shortcut_array_assignment to array detected")
                    live_info = CompilerTools.LivenessAnalysis.find_top_number(i, new_lives)
                    if !in(rhs.name, live_info.live_out)
                        @dprintln(3,"rhs is dead")
                        # The RHS of the assignment is not live out so we can do a special assignment where the j2c_array for the LHS takes over the RHS and the RHS is nulled.
                        push!(node.args, RhsDead())
                    end
                end
            end
        end
    end

    lambda = CompilerTools.LambdaHandling.LambdaVarInfoToLambda(LambdaVarInfo, args)
    return lambda
end

function flattenParfors(out_body :: Array{Any,1}, in_body :: Array{Any,1}, linfo :: LambdaVarInfo)
    for i = 1:length(in_body)
        @dprintln(3,"Flatten index ", i, " ", in_body[i], " type = ", typeof(in_body[i]))
        if isBareParfor(in_body[i])
            flattenParfor(out_body, in_body[i].args[1], linfo)
        else
            push!(out_body, in_body[i])
        end
    end
end

"""
Takes a new array of body statements in the process of construction in "new_body" and takes a parfor to add to that
body.  This parfor is in the nested (parfor code is in the parfor node itself) temporary form we use for fusion although 
pre-statements and post-statements are already elevated by this point.  We replace this nested form with a non-nested
form where we have a parfor_start and parfor_end to delineate the parfor code.
"""
function flattenParfor(new_body, the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, linfo :: LambdaVarInfo)
    @dprintln(2,"Flattening ", the_parfor)

    private_set = getPrivateSet(the_parfor.body, linfo)
    private_array = collect(private_set)

    # Output to the new body that this is the start of a parfor.
    push!(new_body, TypedExpr(Int64, :parfor_start, PIRParForStartEnd(the_parfor.loopNests, the_parfor.reductions, the_parfor.instruction_count_expr, private_array)))
    # Output the body of the parfor as top-level statements in the new function body and convert any other parfors we may find.
    flattenParfors(new_body, the_parfor.body, linfo)
    # Output to the new body that this is the end of a parfor.
    push!(new_body, TypedExpr(Int64, :parfor_end, PIRParForStartEnd(deepcopy(the_parfor.loopNests), deepcopy(the_parfor.reductions), deepcopy(the_parfor.instruction_count_expr), deepcopy(private_array))))
    nothing
end
