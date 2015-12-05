 function flattenParfors(function_name, ast::Expr)
    flatten_start = time_ns()

    println("before flattening:")
    println(ast)
    assert(ast.head == :lambda)
    dprintln(1,"Starting main ParallelIR.from_expr.  function = ", function_name, " ast = ", ast)

    start_time = time_ns()

    lambdaInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(ast)
    body = CompilerTools.LambdaHandling.getBody(ast)

    args = body.args
    expanded_args = Any[]

    for i = 1:length(args)
        dprintln(3,"Flatten index ", i, " ", args[i], " type = ", typeof(args[i]))
        if isBareParfor(args[i])
            flattenParfor(expanded_args, args[i].args[1])
        else
            push!(expanded_args, args[i])
        end
    end

    args = expanded_args

    dprintln(1,"Flattening parfor bodies time = ", ns_to_sec(time_ns() - flatten_start))

    dprintln(3, "After flattening")
    for j = 1:length(args)
        dprintln(3, args[j])
    end

    if shortcut_array_assignment != 0
        fake_body = CompilerTools.LambdaHandling.lambdaInfoToLambdaExpr(lambdaInfo, TypedExpr(CompilerTools.LambdaHandling.getReturnType(lambdaInfo), :body, args...))
        new_lives = CompilerTools.LivenessAnalysis.from_expr(fake_body, pir_live_cb, lambdaInfo)

        for i = 1:length(args)
            node = args[i]
            if isAssignmentNode(node)
                lhs = node.args[1]
                rhs = node.args[2]
                dprintln(3,"shortcut_array_assignment = ", node)
                if typeof(lhs) == SymbolNode && isArrayType(lhs) && typeof(rhs) == SymbolNode
                    dprintln(3,"shortcut_array_assignment to array detected")
                    live_info = CompilerTools.LivenessAnalysis.find_top_number(i, new_lives)
                    if !in(rhs.name, live_info.live_out)
                        dprintln(3,"rhs is dead")
                        # The RHS of the assignment is not live out so we can do a special assignment where the j2c_array for the LHS takes over the RHS and the RHS is nulled.
                        push!(node.args, RhsDead())
                    end
                end
            end
        end
    end

    body.args = args
    lambda = CompilerTools.LambdaHandling.lambdaInfoToLambdaExpr(lambdaInfo, body)
    println("after  flattening:")
    println(lambda)
    return lambda
end