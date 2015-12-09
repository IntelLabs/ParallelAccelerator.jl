
@doc """
Find the basic block before the entry to a loop.
"""
function getNonBlock(head_preds, back_edge)
    assert(length(head_preds) == 2)

    # Scan the predecessors of the head of the loop and the non-back edge must be the block prior to the loop.
    for i in head_preds
        if i.label != back_edge
            return i
        end
    end

    throw(string("All entries in head preds list were back edges."))
end

@doc """
Store information about a section of a body that will be translated into a task.
"""
type ReplacedRegion
    start_index
    end_index
    bb
    tasks
end

type EntityType
    name :: SymAllGen
    typ
end

@doc """
Structure for storing information about task formation.
"""
type TaskInfo
    task_func       :: Function                  # The Julia task function that we generated for a task.
    function_sym
    join_func                                    # The name of the C join function that we constructed and forced into the C file.
    input_symbols   :: Array{EntityType,1}       # Variables that are need as input to the task.
    modified_inputs :: Array{EntityType,1} 
    io_symbols      :: Array{EntityType,1}
    reduction_vars  :: Array{EntityType,1}
    code
    loopNests       :: Array{PIRLoopNest,1}      # holds information about the loop nests
end

@doc """
Translated to pert_range_Nd_t in the task runtime.
This represents an iteration space.
dim is the number of dimensions in the iteration space.
lower_bounds contains the lower bound of the iteration space in each dimension.
upper_bounds contains the upper bound of the iteration space in each dimension.
lower_bounds and upper_bounds can be expressions.
"""
type pir_range
    dim :: Int
    lower_bounds :: Array{Any,1}
    upper_bounds :: Array{Any,1}
    function pir_range()
        new(0, Any[], Any[])
    end
end

@doc """
Similar to pir_range but used in circumstances where the expressions must have already been evaluated.
Therefore the arrays are typed as Int64.
Up to 3 dimensional iteration space constructors are supported to make it easier to do code generation later.
"""
type pir_range_actual
    dim :: Int
    lower_bounds :: Array{Int64, 1}
    upper_bounds :: Array{Int64, 1}
    function pir_range_actual()
        new(0, Int64[], Int64[])
    end
    function pir_range_actual(l1, u1)
        new(1, Int64[l1], Int64[u1])
    end
    function pir_range_actual(l1, u1, l2, u2)
        new(2, Int64[l1; l2], Int64[u1; u2])
    end
    function pir_range_actual(l1, u1, l2, u2, l3, u3)
        new(3, Int64[l1; l2; l3], Int64[u1; u2; u3])
    end
    function pir_range_actual(larray :: Array{Int64,1}, uarray :: Array{Int64,1})
        assert(length(larray) == length(uarray))
        new(length(larray), larray, uarray)
    end
end

ARG_OPT_IN = 1
ARG_OPT_OUT = 2
ARG_OPT_INOUT = 3
#ARG_OPT_VALUE = 4
ARG_OPT_ACCUMULATOR = 5

type pir_aad_dim
    len
    a1
    a2
    l_b
    u_b
end

@doc """
Describes an array.
row_major is true if the array is stored in row major format.
dim_info describes which portion of the array is accessed for a given point in the iteration space.
"""
type pir_array_access_desc
    dim_info :: Array{pir_aad_dim, 1}
    row_major :: Bool

    function pir_array_access_desc()
        new(pir_aad_dim[],false)
    end
end

@doc """
Create an array access descriptor for "array".
Presumes that for point "i" in the iteration space that only index "i" is accessed.
"""
function create1D_array_access_desc(array :: SymbolNode)
    ret = pir_array_access_desc()
    push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 1), 1, 0, 0, 0 ))
    ret
end

@doc """
Create an array access descriptor for "array".
Presumes that for points "(i,j)" in the iteration space that only indices "(i,j)" is accessed.
"""
function create2D_array_access_desc(array :: SymbolNode)
    ret = pir_array_access_desc()
    push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 1), 1, 0, 0, 0 ))
    push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 2), 1, 0, 0, 0 ))
    ret
end

@doc """
Create an array access descriptor for "array".
"""
function create_array_access_desc(array :: SymbolNode)
    if array.typ.parameters[2] == 1
        return create1D_array_access_desc(array)
    elseif array.typ.parameters[2] == 2
        return create2D_array_access_desc(array)
    else
        throw(string("Greater than 2D arrays not supported in create_array_access_desc."))
    end
end

@doc """
A Julia representation of the argument metadata that will be passed to the runtime.
"""
type pir_arg_metadata
    value   :: SymbolNode
    options :: Int
    access  # nothing OR pir_array_access_desc

    function pir_arg_metadata()
        new(nothing, 0, 0, nothing)
    end

    function pir_arg_metadata(v, o)
        new(v, o, nothing)
    end

    function pir_arg_metadata(v, o, a)
        new(v, o, a)
    end
end

@doc """
A Julia representation of the grain size that will be passed to the runtime.
"""
type pir_grain_size
    dim   :: Int
    sizes :: Array{Int,1}

    function pir_grain_size()
        new(0, Int[])
    end
end

TASK_FINISH = 1 << 16           # 0x10000
TASK_PRIORITY  = 1 << 15        # 0x08000
TASK_SEQUENTIAL = 1 << 14       # 0x04000
TASK_AFFINITY_XEON = 1 << 13    # 0x02000
TASK_AFFINITY_PHI = 1 << 12     # 0x01000
TASK_STATIC_SCHEDULER = 1 << 11 # 0x00800

@doc """
A data type containing the information that CGen uses to generate a call to pert_insert_divisible_task.
"""
type InsertTaskNode
    ranges :: pir_range
    args   :: Array{pir_arg_metadata,1}
    task_func :: Any
    join_func :: AbstractString  # empty string for no join function
    task_options :: Int
    host_grain_size :: pir_grain_size
    phi_grain_size :: pir_grain_size

    function InsertTaskNode()
        new(pir_range(), pir_arg_metadata[], nothing, string(""), 0, pir_grain_size(), pir_grain_size())
    end
end

@doc """
If run_as_tasks is positive then convert this parfor to a task and decrement the count so that only the
original number run_as_tasks if the number of tasks created.
"""
function run_as_task_decrement()
    if run_as_tasks == 0
        return false
    end
    if run_as_tasks == -1
        return true
    end
    global run_as_tasks = run_as_tasks - 1
    return true
end

@doc """
Return true if run_as_task_decrement would return true but don't update the run_as_tasks count.
"""
function run_as_task()
    if run_as_tasks == 0
        return false
    end
    return true
end

put_loops_in_task_graph = false

limit_task = -1
function PIRLimitTask(x)
    global limit_task = x
end

# These two functions are just to maintain the interface with the old PSE system for the moment.
function PIRFlatParfor(x)
end
function PIRPreEq(x)
end

pir_stop = 0
function PIRStop(x)
    global pir_stop = x
end

polyhedral = 0
function PIRPolyhedral(x)
    global polyhedral = x
end

num_threads_mode = 0
function PIRNumThreadsMode(x)
    global num_threads_mode = x
end

stencil_tasks = 1
function PIRStencilTasks(x)
    global stencil_tasks = x
end

reduce_tasks = 0 
function PIRReduceTasks(x)
    global reduce_tasks = x
end

@doc """
Returns true if the "node" is a parfor and the task limit hasn't been exceeded.
Also controls whether stencils or reduction can become tasks.
"""
function taskableParfor(node)
    dprintln(3,"taskableParfor for: ", node)
    if limit_task == 0
        dprintln(3,"task limit exceeded so won't convert parfor to task")
        return false
    end
    if isParforAssignmentNode(node) || isBareParfor(node)
        dprintln(3,"Found parfor node, stencil: ", stencil_tasks, " reductions: ", reduce_tasks)
        if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
            return true
        end

        the_parfor = getParforNode(node)

        for i = 1:length(the_parfor.original_domain_nodes)
            if (the_parfor.original_domain_nodes[i].operation == :stencil! && stencil_tasks == 0) ||
                (the_parfor.original_domain_nodes[i].operation == :reduce && reduce_tasks == 0)
                return false
            end
        end
        if limit_task > 0
            global limit_task = limit_task - 1
        end
        return true
    end
    false
end

type eic_state
    non_calls      :: Float64    # estimated instruction count for non-calls
    fully_analyzed :: Bool
    lambdaInfo     :: Union{CompilerTools.LambdaHandling.LambdaInfo, Void}
end

ASSIGNMENT_COST = 1.0
RETURN_COST = 1.0
ARRAYSET_COST = 4.0
UNSAFE_ARRAYSET_COST = 2.0
ARRAYREF_COST = 4.0
UNSAFE_ARRAYREF_COST = 2.0
FLOAT_ARITH = 1.0

call_costs = Dict{Any,Any}()
call_costs[(TopNode(:mul_float),(Float64,Float64))] = 1.0
call_costs[(TopNode(:div_float),(Float64,Float64))] = 1.0
call_costs[(TopNode(:add_float),(Float64,Float64))] = 1.0
call_costs[(TopNode(:sub_float),(Float64,Float64))] = 1.0
call_costs[(TopNode(:neg_float),(Float64,))] = 1.0
call_costs[(TopNode(:mul_float),(Float32,Float32))] = 1.0
call_costs[(TopNode(:div_float),(Float32,Float32))] = 1.0
call_costs[(TopNode(:add_float),(Float32,Float32))] = 1.0
call_costs[(TopNode(:sub_float),(Float32,Float32))] = 1.0
call_costs[(TopNode(:neg_float),(Float32,))] = 1.0
call_costs[(TopNode(:mul_int),(Int64,Int64))] = 1.0
call_costs[(TopNode(:div_int),(Int64,Int64))] = 1.0
call_costs[(TopNode(:add_int),(Int64,Int64))] = 1.0
call_costs[(TopNode(:sub_int),(Int64,Int64))] = 1.0
call_costs[(TopNode(:sle_int),(Int64,Int64))] = 1.0
call_costs[(TopNode(:sitofp),(DataType,Int64))] = 1.0
call_costs[(:log10,(Float64,))] = 160.0
call_costs[(:erf,(Float64,))] = 75.0

@doc """
A sentinel in the instruction count estimation process.
Before recursively processing a call, we add a sentinel for that function so that if we see that
sentinel later we know we've tried to recursively process it and so can bail out by setting
fully_analyzed to false.
"""
type InProgress
end

@doc """
Generate an instruction count estimate for a call instruction.
"""
function call_instruction_count(args, state :: eic_state, debug_level)
    func  = args[1]
    fargs = args[2:end]

    dprintln(3,"call_instruction_count: func = ", func, " fargs = ", fargs)
    sig_expr = Expr(:tuple)
    sig_expr.args = map(x -> CompilerTools.LivenessAnalysis.typeOfOpr(x, state.lambdaInfo), fargs)
    signature = eval(sig_expr)
    fs = (func, signature)

    # If we've previously cached an instruction count estimate then use it.
    if haskey(call_costs, fs)
        res = call_costs[fs]
        if res == nothing
            dprintln(debug_level, "Didn't process call to function ", func, " ", signature, " ", args)
            state.fully_analyzed = false
            return nothing
        end
        # See the comment on the InProgress type above for how this prevents infinite recursive analysis.
        if typeof(res) == InProgress
            dprintln(debug_level, "Got recursive call to function ", func, " ", signature, " ", args)
            state.fully_analyzed = false
            return nothing
        end
    else
        # See the comment on the InProgress type above for how this prevents infinite recursive analysis.
        call_costs[fs] = InProgress()
        # If not then try to generate an instruction count estimated.
        res = generate_instr_count(func, signature)
        call_costs[fs] = res
        if res == nothing
            # If we couldn't do it then set fully_analyzed to false.
            dprintln(debug_level, "Didn't process call to function ", func, " ", signature, " ", args)
            state.fully_analyzed = false
            return nothing
        end
    end

    assert(typeof(res) == Float64)
    state.non_calls = state.non_calls + res
    return nothing
end

@doc """
Try to figure out the instruction count for a given call.
"""
function generate_instr_count(function_name, signature)
    # Estimate instructions for some well-known functions.
    if function_name == TopNode(:arrayset) || function_name == TopNode(:arrayref)
        call_costs[(function_name, signature)] = 4.0
        return call_costs[(function_name, signature)]
    elseif function_name == TopNode(:unsafe_arrayset) || function_name == TopNode(:unsafe_arrayref)
        call_costs[(function_name, signature)] = 2.0
        return call_costs[(function_name, signature)]
    elseif function_name == TopNode(:safe_arrayref)
        call_costs[(function_name, signature)] = 6.0
        return call_costs[(function_name, signature)]
    elseif function_name == TopNode(:box)
        call_costs[(function_name, signature)] = 20.0
        return call_costs[(function_name, signature)]
    elseif function_name == TopNode(:lt_float) || 
        function_name == TopNode(:le_float) ||
        function_name == TopNode(:not_int)
        call_costs[(function_name, signature)] = 1.0
        return call_costs[(function_name, signature)]
    end

    function_name = process_function_name(function_name)

    ftyp = typeof(function_name)

    if ftyp != Function
        dprintln(3,"generate_instr_count: instead of Function, got ", ftyp, " ", function_name)
    end

    if ftyp == IntrinsicFunction
        dprintln(3, "generate_instr_count: found IntrinsicFunction = ", function_name)
        call_costs[(function_name, signature)] = nothing
        return call_costs[(function_name, signature)]
    end

    if typeof(function_name) != Function || !isgeneric(function_name)
        dprintln(3, "generate_instr_count: function_name is not a Function = ", function_name)
        call_costs[(function_name, signature)] = nothing
        return call_costs[(function_name, signature)]
    end

    m = methods(function_name, signature)
    if length(m) < 1
        return nothing
        #    error("Method for ", function_name, " with signature ", signature, " is not found")
    end

    ct = ParallelAccelerator.Driver.code_typed(function_name, signature)      # get information about code for the given function and signature

    dprintln(2,"generate_instr_count ", function_name, " ", signature)
    state = eic_state(0, true, nothing)
    # Try to estimate the instruction count for the other function.
    AstWalk(ct[1], estimateInstrCount, state)
    dprintln(2,"instruction count estimate for parfor = ", state)
    # If so then cache the result.
    if state.fully_analyzed
        call_costs[(function_name, signature)] = state.non_calls
    else
        call_costs[(function_name, signature)] = nothing
    end
    return call_costs[(function_name, signature)]
end

function process_function_name(function_name::Expr)
    dprintln(3,"eval'ing Expr to Function")
    function_name = eval(function_name)
    return function_name
end

function process_function_name(function_name::GlobalRef)
    #dprintln(3,"Calling getfield")
    function_name = eval(function_name)
    #function_name = getfield(function_name.mod, function_name.name)
    return function_name
end

function process_function_name(function_name::Any)
    return function_name
end

@doc """
AstWalk callback for estimating the instruction count.
"""
function estimateInstrCount(ast, state :: eic_state, top_level_number, is_top_level, read)
    debug_level = 2

    asttyp = typeof(ast)
    if asttyp == Expr
        if is_top_level
            dprint(debug_level,"instruction count estimator: Expr ")
        end
        head = ast.head
        args = ast.args
        typ  = ast.typ
        if is_top_level
            dprintln(debug_level,head, " ", args)
        end
        if head == :lambda
            state.lambdaInfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(ast)
        elseif head == :body
            # skip
        elseif head == :block
            # skip
        elseif head == :(.)
            #        args = from_exprs(args, depth+1, callback, cbdata, top_level_number, read)
        elseif head == :(=)
            state.non_calls = state.non_calls + ASSIGNMENT_COST
        elseif head == :(::)
            # skip
        elseif head == :return
            state.non_calls = state.non_calls + RETURN_COST
        elseif head == :call
            call_instruction_count(args, state, debug_level)
        elseif head == :call1
            dprintln(debug_level,head, " not handled in instruction counter")
            #        args = from_call(args, depth, callback, cbdata, top_level_number, read)
            # TODO?: tuple
        elseif head == :line
            # skip
        elseif head == :copy
            dprintln(debug_level,head, " not handled in instruction counter")
            # turn array copy back to plain Julia call
            #        head = :call
            #        args = vcat(:copy, args)
        elseif head == :copyast
            #        dprintln(2,"copyast type")
            # skip
        elseif head == :gotoifnot
            #        dprintln(debug_level,head, " not handled in instruction counter")
            #        assert(length(args) == 2)
            #        args[1] = get_one(from_expr(args[1], depth, callback, cbdata, top_level_number, false, read))
            #      state.fully_analyzed = false
            state.non_calls = state.non_calls + 1
        elseif head == :new
            dprintln(debug_level,head, " not handled in instruction counter")
            #        args = from_exprs(args,depth, callback, cbdata, top_level_number, read)
        elseif head == :arraysize
            dprintln(debug_level,head, " not handled in instruction counter")
            #        assert(length(args) == 2)
            #        args[1] = get_one(from_expr(args[1], depth, callback, cbdata, top_level_number, false, read))
            #        args[2] = get_one(from_expr(args[2], depth, callback, cbdata, top_level_number, false, read))
        elseif head == :alloc
            dprintln(debug_level,head, " not handled in instruction counter")
            #        assert(length(args) == 2)
            #        args[2] = from_exprs(args[2], depth, callback, cbdata, top_level_number, read)
        elseif head == :boundscheck
            dprintln(debug_level,head, " not handled in instruction counter")
            # skip
        elseif head == :type_goto
            dprintln(debug_level,head, " not handled in instruction counter")
            #        assert(length(args) == 2)
            #        args[1] = get_one(from_expr(args[1], depth, callback, cbdata, top_level_number, false, read))
            #        args[2] = get_one(from_expr(args[2], depth, callback, cbdata, top_level_number, false, read))
            state.fully_analyzed = false
        elseif head == :enter
            dprintln(debug_level,head, " not handled in instruction counter")
            # skip
            state.fully_analyzed = false
        elseif head == :leave
            dprintln(debug_level,head, " not handled in instruction counter")
            # skip
            state.fully_analyzed = false
        elseif head == :the_exception
            # skip
            state.fully_analyzed = false
        elseif head == :&
            # skip
        else
            dprintln(1,"instruction count estimator: unknown Expr head :", head, " ", ast)
        end
    elseif asttyp == Symbol
        #skip
    elseif asttyp == SymbolNode # name, typ
        #skip
    elseif asttyp == TopNode    # name
        #skip
    elseif asttyp == GlobalRef
        #dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #    mod = ast.mod
        #    name = ast.name
        #    typ = typeof(mod)
        state.non_calls = state.non_calls + 1
    elseif asttyp == QuoteNode
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #    value = ast.value
        #TODO: fields: value
    elseif asttyp == LineNumberNode
        #skip
    elseif asttyp == LabelNode
        #dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
    elseif asttyp == GotoNode
        #dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
        #state.fully_analyzed = false
        state.non_calls = state.non_calls + 1
    elseif asttyp == DataType
        #skip
    elseif asttyp == ()
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
    elseif asttyp == ASCIIString
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
    elseif asttyp == NewvarNode
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
    elseif asttyp == Void
        #skip
        #elseif asttyp == Int64 || asttyp == Int32 || asttyp == Float64 || asttyp == Float32
    elseif isbits(asttyp)
        #skip
    elseif isa(ast,Tuple)
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #    new_tt = Expr(:tuple)
        #    for i = 1:length(ast)
        #      push!(new_tt.args, get_one(from_expr(ast[i], depth, callback, cbdata, top_level_number, false, read)))
        #    end
        #    new_tt.typ = asttyp
        #    ast = eval(new_tt)
    elseif asttyp == Module
        #skip
    elseif asttyp == NewvarNode
        dprintln(debug_level,asttyp, " not handled in instruction counter ", ast)
        #skip
    else
        dprintln(1,"instruction count estimator: unknown AST (", typeof(ast), ",", ast, ")")
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
Takes a parfor and walks the body of the parfor and estimates the number of instruction needed for one instance of that body.
"""
function createInstructionCountEstimate(the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, state :: expr_state)
    if num_threads_mode == 1 || num_threads_mode == 2 || num_threads_mode == 3
        dprintln(2,"instruction count estimate for parfor = ", the_parfor)
        new_state = eic_state(0, true, state.lambdaInfo)
        for i = 1:length(the_parfor.body)
            AstWalk(the_parfor.body[i], estimateInstrCount, new_state)
        end
        # If fully_analyzed is true then there's nothing that couldn't be analyzed so store the instruction count estimate in the parfor.
        if new_state.fully_analyzed
            the_parfor.instruction_count_expr = new_state.non_calls
        else
            the_parfor.instruction_count_expr = nothing
        end
        dprintln(2,"instruction count estimate for parfor = ", the_parfor.instruction_count_expr)
    end
end

@doc """
Marks an assignment statement where the left-hand side can take over the storage from the right-hand side.
"""
type RhsDead
end

# Task Graph Modes
SEQUENTIAL_TASKS = 1    # Take the first parfor in the block and the last parfor and form tasks for all parallel and sequential parts inbetween.
ONE_AT_A_TIME = 2       # Just forms tasks out of one parfor at a time.
MULTI_PARFOR_SEQ_NO = 3 # Forms tasks from multiple parfor in sequence but not sequential tasks.

task_graph_mode = ONE_AT_A_TIME
@doc """
Control how blocks of code are made into tasks.
"""
function PIRTaskGraphMode(x)
    global task_graph_mode = x
end



if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE

#println_mutex = Mutex()

function tprintln(args...)
  for a in args
    ccall(:puts, Cint, (Cstring,), string(a))
  end
end

type dimlength
    dim
    len
end

type isf_range
    dim
    lower_bound
    upper_bound
end

function chunk(rs :: Int, re :: Int, divisions :: Int)
    assert(divisions >= 1)
    total = (re - rs) + 1 
    if divisions == 1
        return (rs, re, re + 1)
    else
        len, rem = divrem(total, divisions)
        res_end = rs + len - 1
        return (rs, res_end, res_end + 1)
    end
end

function isfRangeToActual(build :: Array{isf_range,1}, dims)
    unsort = sort(build, by=x->x.dim)
    ParallelAccelerator.ParallelIR.pir_range_actual(map(x -> x.lower_bound, unsort), map(x -> x.upper_bound, unsort))    
end

function divide_work(assignments :: Array{ParallelAccelerator.ParallelIR.pir_range_actual,1}, 
                     build       :: Array{isf_range, 1}, 
                     start_thread, end_thread, dims, index)
    num_threads = (end_thread - start_thread) + 1

#    dprintln(3,"divide_work num_threads = ", num_threads, " build = ", build, " start = ", start_thread, " end = ", end_thread, " dims = ", dims, " index = ", index)
    assert(num_threads >= 1)
    if num_threads == 1
        assert(length(build) <= length(dims))

        if length(build) == length(dims)
            pra = isfRangeToActual(build, dims)
            assignments[start_thread] = pra
        else 
            new_build = [build[1:(index-1)]; isf_range(dims[index].dim, 1, dims[index].len)]
            divide_work(assignments, new_build, start_thread, end_thread, dims, index+1)
        end
    else
        assert(index <= length(dims))
        total_len = sum(map(x -> x.len, dims[index:end]))
        percentage = dims[index].len / total_len
        divisions_for_this_dim = Int(round(num_threads * percentage))

        chunkstart = 1
        chunkend   = dims[index].len

        threadstart = start_thread
        threadend   = end_thread

#        dprintln(3, "total = ", total_len, " percentage = ", percentage, " divisions = ", divisions_for_this_dim)
        for i = 1:divisions_for_this_dim
            (ls, le, chunkstart)  = chunk(chunkstart,  chunkend,  divisions_for_this_dim - i + 1)
            (ts, te, threadstart) = chunk(threadstart, threadend, divisions_for_this_dim - i + 1)
            divide_work(assignments, [build[1:(index-1)]; isf_range(dims[index].dim, ls, le)], ts, te, dims, index+1) 
        end
 
    end
end

function divide_fis(full_iteration_space :: ParallelAccelerator.ParallelIR.pir_range_actual, numtotal :: Int)
#    dprintln(3, "divide_fis space = ", full_iteration_space, " numtotal = ", numtotal)
    dims = sort(dimlength[dimlength(i, full_iteration_space.upper_bounds[i] - full_iteration_space.lower_bounds[i] + 1)  for i = 1:full_iteration_space.dim], by=x->x.len, rev=true)
#    dprintln(3, "dims = ", dims)
    assignments = Array(ParallelAccelerator.ParallelIR.pir_range_actual, numtotal)
    divide_work(assignments, isf_range[], 1, numtotal, dims, 1)
    return assignments
end

@doc """
An intermediate scheduling function for passing to jl_threading_run.
It takes the task function to run, the full iteration space to run and the normal argument to the task function in "rest..."
"""
function isf(t :: Function, 
             full_iteration_space :: ParallelAccelerator.ParallelIR.pir_range_actual,
             rest...)
#    ccall(:puts, Cint, (Cstring,), "Running isf.")
    tid = Base.Threads.threadid()
#    ta = [typeof(x) for x in rest]
#    Base.Threads.lock!(println_mutex)
#    tprintln("Starting isf. tid = ", tid, " space = ", full_iteration_space, " ta = ", ta)
#    Base.Threads.unlock!(println_mutex)

    if full_iteration_space.dim == 1
        # Compute how many iterations to run.
        num_iters = full_iteration_space.upper_bounds[1] - full_iteration_space.lower_bounds[1] + 1

#        Base.Threads.lock!(println_mutex)
#        tprintln("tid = ", tid, " num_iters = ", num_iters)
#        Base.Threads.unlock!(println_mutex)

        # Handle the case where iterations is less than the core count.
        if num_iters <= nthreads()
            if tid <= num_iters
                return t(ParallelAccelerator.ParallelIR.pir_range_actual(tid,tid), rest...)
            else
                return nothing
            end
        else
            # one dimensional scheduling
            len, rem = divrem(num_iters, nthreads())
            ls = (len * (tid-1)) + 1
            if tid == nthreads()
                le = full_iteration_space.upper_bounds[1]
            else
                le = (len * tid)
            end
#            Base.Threads.lock!(println_mutex)
#            tprintln("tid = ", tid, " ls = ", ls, " le = ", le, " func = ", Base.function_name(t))
#            Base.Threads.unlock!(println_mutex)
            try 
              tres = t(ParallelAccelerator.ParallelIR.pir_range_actual(ls,le), rest...)
            catch something
             # println("Call to t created exception ", something)
              ccall(:puts, Cint, (Cstring,), "caught some exception")
            end 
#            tprintln("After t call tid = ", tid)
            return tres
        end
    elseif full_iteration_space.dim >= 2
        assignments = divide_fis(full_iteration_space, nthreads())

        try 
#            msg = string("Running num_threads = ", nthreads(), " tid = ", tid, " assignment = ", assignments[tid])
#            ccall(:puts, Cint, (Cstring,), msg)
            tres = t(assignments[tid], rest...)
        catch something
            bt = catch_backtrace()
            s = sprint(io->Base.show_backtrace(io, bt))
            ccall(:puts, Cint, (Cstring,), string(s))

            msg = string("caught some exception num_threads = ", nthreads(), " tid = ", tid, " assignment = ", assignments[tid])
            ccall(:puts, Cint, (Cstring,), msg)
            msg = string(something)
            ccall(:puts, Cint, (Cstring,), msg)
        end 
        return tres
    end
end

end # end if THREADS_MODE



@doc """
For a given start and stop index in some body and liveness information, form a set of tasks.
"""
function makeTasks(start_index, stop_index, body, bb_live_info, state, task_graph_mode)
    task_list = Any[]
    seq_accum = Any[]
    dprintln(3,"makeTasks starting")

    # ONE_AT_A_TIME mode should only have parfors in the list of things to replace so we assert if we see a non-parfor in this mode.
    # SEQUENTIAL_TASKS mode bunches up all non-parfors into one sequential task.  Maybe it would be better to do one sequential task per non-parfor stmt?
    # MULTI_PARFOR_SEQ_NO mode converts parfors to tasks but leaves non-parfors as non-tasks.  This implies that the code calling this function has ensured
    #     that none of the non-parfor stmts depend on the completion of a parfor in the batch.

    if ParallelAccelerator.getPseMode() != ParallelAccelerator.THREADS_MODE
        if task_graph_mode == SEQUENTIAL_TASKS
            task_finish = false
        elseif task_graph_mode == ONE_AT_A_TIME
            task_finish = true
        elseif task_graph_mode == MULTI_PARFOR_SEQ_NO
            task_finish = false
        else
            throw(string("Unknown task_graph_mode in makeTasks"))
        end
    end

    for j = start_index:stop_index
        if taskableParfor(body[j])
            # is a parfor node
            if length(seq_accum) > 0
                assert(task_graph_mode == SEQUENTIAL_TASKS)
                st = seqTask(seq_accum, bb_live_info.statements, body, state)
                dprintln(3,"Adding sequential task to task_list. ", st)
                push!(task_list, st)
                seq_accum = Any[]
            end
            ptt = parforToTask(j, bb_live_info.statements, body, state)
            dprintln(3,"Adding parfor task to task_list. ", ptt)
            push!(task_list, ptt)
        else
            # is not a parfor node
            assert(task_graph_mode != ONE_AT_A_TIME)
            if task_graph_mode == SEQUENTIAL_TASKS
                # Collect the non-parfor stmts in a batch to be turned into one sequential task.
                push!(seq_accum, body[j])
            else
                dprintln(3,"Adding non-parfor node to task_list. ", body[j])
                # MULTI_PARFOR_SEQ_NO mode.
                # Just add the non-parfor stmts directly to the output statements.
                push!(task_list, body[j])
            end
        end
    end

    #if ParallelAccelerator.getPseMode() != ParallelAccelerator.THREADS_MODE
        # If each task doesn't wait to finish then add a call to pert_wait_all_task to wait for the batch to finish.
     #   if !task_finish
            #julia_root      = ParallelAccelerator.getJuliaRoot()
            #runtime_libpath = string(julia_root, "/intel-runtime/lib/libintel-runtime.so")
            #runtime_libpath = ParallelAccelerator.runtime_libpath

            #call_wait = Expr(:ccall, Expr(:tuple, QuoteNode(:pert_wait_all_task), runtime_libpath), :Void, Expr(:tuple))
            #push!(task_list, call_wait) 

      #      call_wait = TypedExpr(Void, 
      #      :call, 
      #      TopNode(:ccall), 
      #      Expr(:call1, TopNode(:tuple), QuoteNode(:pert_wait_all_task), runtime_libpath), 
      #      :Void, 
      #      Expr(:call1, TopNode(:tuple)))
      #      push!(task_list, call_wait)

            #    call_wait = quote ccall((:pert_wait_all_task, $runtime_libpath), Void, ()) end
            #    assert(typeof(call_wait) == Expr && call_wait.head == :block)
            #    append!(task_list, call_wait.args) 
       # end
    #end

    task_list
end

@doc """
Given a set of statement IDs and liveness information for the statements of the function, determine
which symbols are needed at input and which symbols are purely local to the functio.
"""
function getIO(stmt_ids, bb_statements)
    assert(length(stmt_ids) > 0)

    # Get the statements out of the basic block statement array such that those statement's ID's are in the stmt_ids ID array.
    stmts_for_ids = filter(x -> in(x.tls.index, stmt_ids) , bb_statements)
    # Make sure that we found a statement for every statement ID.
    if length(stmt_ids) != length(stmts_for_ids)
        dprintln(0,"length(stmt_ids) = ", length(stmt_ids))
        dprintln(0,"length(stmts_for_ids) = ", length(stmts_for_ids))
        dprintln(0,"stmt_ids = ", stmt_ids)
        dprintln(0,"stmts_for_ids = ", stmts_for_ids)
        assert(length(stmt_ids) == length(stmts_for_ids))
    end
    # The initial set of inputs is those variables "use"d by the first statement.
    # The inputs to the task are those variables used in the set of statements before they are defined in any of those statements.
    cur_inputs = stmts_for_ids[1].use
    # Keep track of variables defined in the set of statements processed thus far.
    cur_defs   = stmts_for_ids[1].def
    for i = 2:length(stmts_for_ids)
        # For each additional statement, the new set of inputs is the previous set plus uses in the current statement except for those symbols already defined in the function.
        cur_inputs = union(cur_inputs, setdiff(stmts_for_ids[i].use, cur_defs))
        # For each additional statement, the defs are just union with the def for the current statement.
        cur_defs   = union(cur_defs, stmts_for_ids[i].def)
    end
    IntrinsicSet = Set()
    # We will ignore the :Intrinsics symbol as it isn't something you need to pass as a param.
    push!(IntrinsicSet, :Intrinsics)
    # Task functions don't return anything.  They must return via an input parameter so outputs should be empty.
    outputs = setdiff(intersect(cur_defs, stmts_for_ids[end].live_out), IntrinsicSet)
    cur_defs = setdiff(cur_defs, IntrinsicSet)
    cur_inputs = setdiff(filter(x -> !(is(x, :Int64) || is(x, :Float32)), cur_inputs), IntrinsicSet)
    # The locals are those things defined that aren't inputs or outputs of the function.
    cur_inputs, outputs, setdiff(cur_defs, union(cur_inputs, outputs))
end

@doc """
Returns an expression to construct a :colon object that contains the start of a range, the end and the skip expression.
"""
function mk_colon_expr(start_expr, skip_expr, end_expr)
    TypedExpr(Any, :call, :colon, start_expr, skip_expr, end_expr)
end

@doc """
Returns an expression to get the start of an iteration range from a :colon object.
"""
function mk_start_expr(colon_sym)
    TypedExpr(Any, :call, TopNode(:start), colon_sym)
end

@doc """
Returns a :next call Expr that gets the next element of an iteration range from a :colon object.
"""
function mk_next_expr(colon_sym, start_sym)
    TypedExpr(Any, :call, TopNode(:next), colon_sym, start_sym)
end


@doc """
Returns a :gotoifnot Expr given a condition "cond" and a label "goto_label".
"""
function mk_gotoifnot_expr(cond, goto_label)
    TypedExpr(Any, :gotoifnot, cond, goto_label)
end

@doc """
Just to hold the "found" Bool that says whether a unsafe variant was replaced with a regular version.
"""
type cuw_state
    found
    function cuw_state()
        new(false)
    end
end

@doc """
The AstWalk callback to find unsafe arrayset and arrayref variants and
replace them with the regular Julia versions.  Sets the "found" flag
in the state when such a replacement is performed.
"""
function convertUnsafeWalk(x, state, top_level_number, is_top_level, read)
    use_dbg_level = 3
    dprintln(use_dbg_level,"convertUnsafeWalk ", x)

    if typeof(x) == Expr
        dprintln(use_dbg_level,"convertUnsafeWalk is Expr")
        if x.head == :call
            dprintln(use_dbg_level,"convertUnsafeWalk is :call")
            if x.args[1] == TopNode(:unsafe_arrayset)
                x.args[1] = TopNode(:arrayset)
                state.found = true
                return x
            elseif x.args[1] == TopNode(:unsafe_arrayref)
                x.args[1] = TopNode(:arrayref)
                state.found = true
                return x
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

@doc """
Remove unsafe array access Symbols from the incoming "stmt".
Returns the updated statement if something was modifed, else returns "nothing".
"""
function convertUnsafe(stmt)
    dprintln(3,"convertUnsafe: ", stmt)
    state = cuw_state() 
    # Uses AstWalk to do the pattern match and replace.
    res = AstWalk(stmt, convertUnsafeWalk, state)
    # state.found is set if the callback convertUnsafeWalk found and replaced an unsafe variant.
    if state.found
        dprintln(3, "state.found ", state, " ", res)
        dprintln(3,"Replaced unsafe: ", res)
        return res
    else
        return nothing
    end
end

@doc """
Try to remove unsafe array access Symbols from the incoming "stmt".  If successful, then return the updated
statement, else return the unmodified statement.
"""
function convertUnsafeOrElse(stmt)
    res = convertUnsafe(stmt)
    if res == nothing
        res = stmt
    end
    return res
end

function first_unless(gs0 :: StepRange{Int64,Int64}, pound :: Int64)
    res = 
     !(
       (
         (!(gs0.start == gs0.stop)) && 
         (!((0 < gs0.step) == (gs0.start < gs0.stop)))
       ) || 
       (pound == (gs0.stop + gs0.step))
      )
    dprintln(4,"first_unless res = ", res)
    return res
end

function assign_gs4(gs0 :: StepRange{Int64,Int64}, pound :: Int64)
    pound + gs0.step
end

function second_unless(gs0 :: StepRange{Int64,Int64}, pound :: Int64)
    res = 
     !(
       !(
         (
           (!(gs0.start == gs0.stop)) && 
           (!((0 < gs0.step) == (gs0.start < gs0.stop)))
         ) || 
         (pound == (gs0.stop + gs0.step))
        )
      ) 
    dprintln(4,"second_unless res = ", res)
    return res
end

precompile(first_unless, (StepRange{Int64,Int64}, Int64))
precompile(assign_gs4, (StepRange{Int64,Int64}, Int64))
precompile(second_unless, (StepRange{Int64,Int64}, Int64))

@doc """
This is a recursive routine to reconstruct a regular Julia loop nest from the loop nests described in PIRParForAst.
One call of this routine handles one level of the loop nest.
If the incoming loop nest level is more than the number of loops nests in the parfor then that is the spot to
insert the body of the parfor into the new function body in "new_body".
"""
function recreateLoopsInternal(new_body, the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, loop_nest_level, next_available_label, state, newLambdaInfo)
    dprintln(3,"recreateLoopsInternal ", loop_nest_level, " ", next_available_label)
    if loop_nest_level > length(the_parfor.loopNests)
        # A loop nest level greater than number of nests in the parfor means we can insert the body of the parfor here.
        # For each statement in the parfor body.
        for i = 1:length(the_parfor.body)
            dprintln(3, "Body index ", i)
            # Convert any unsafe_arrayref or sets in this statements to regular arrayref or arrayset.
            # But if it was labeled as "unsafe" then output :boundscheck false Expr so that Julia won't generate a boundscheck on the array access.
            cu_res = convertUnsafe(the_parfor.body[i])
            dprintln(3, "cu_res = ", cu_res)
            if cu_res != nothing
                push!(new_body, Expr(:boundscheck, false)) 
                push!(new_body, cu_res)
                push!(new_body, Expr(:boundscheck, Expr(:call, TopNode(:getfield), Base, QuoteNode(:pop))))
            else
                push!(new_body, the_parfor.body[i])
            end
        end
    else
        # See the following example from the REPL for how Julia structures loop nests into labels and gotos.
        # Our code below generates the same structure for each loop in the parfor.

        # function f1(x,y,z)
        #   for i = x:z:y
        #     println(i)
        #   end
        # end
        
        #  (Base.not_int)(
        #     (Base.or_int)(
        #        (Base.and_int)(
        #           (Base.not_int)(
        #              (top(getfield))(GenSym(0),:start)::Int64 === (top(getfield))(GenSym(0),:stop)::Int64::Bool
        #           ),
        #           (Base.not_int)(
        #              (Base.slt_int)(0,(top(getfield))(GenSym(0),:step)::Int64)::Bool === (Base.slt_int)((top(getfield))(GenSym(0),:start)::Int64,(top(getfield))(GenSym(0),:stop)::Int64)::Bool::Bool
        #           )
        #        ),
        #        #s1::Int64 === (Base.add_int)(
        #                         (top(getfield))(GenSym(0),:stop)::Int64,(top(getfield))(GenSym(0),:step)::Int64
        #                       )
        #     )
        #  )

        #  (Base.not_int)(
        #     (Base.not_int)(
        #        (Base.or_int)(
        #           (Base.and_int)(
        #              (Base.not_int)(
        #                 (top(getfield))(GenSym(0),:start)::Int64 === (top(getfield))(GenSym(0),:stop)::Int64::Bool
        #              ),
        #              (Base.not_int)(
        #                 (Base.slt_int)(0,(top(getfield))(GenSym(0),:step)::Int64)::Bool === (Base.slt_int)((top(getfield))(GenSym(0),:start)::Int64,(top(getfield))(GenSym(0),:stop)::Int64)::Bool::Bool
        #              )
        #           ),
        #           #s1::Int64 === (Base.box)(Base.Int,(Base.add_int)((top(getfield))(GenSym(0),:stop)::Int64,(top(getfield))(GenSym(0),:step)::Int64))::Int64::Bool
        #        )
        #     )
        #  )

        #   :($(Expr(:lambda, Any[:x,:y,:z], Any[Any[Any[:x,Int64,0],Any[:y,Int64,0],Any[:z,Int64,0],Any[symbol("#s1"),Int64,2],Any[:i,Int64,18],Any[symbol("##xs#7035"),Tuple{Int64},0]],Any[],Any[StepRange{Int64,Int64},Tuple{Int64,Int64},Int64,Int64,Int64],Any[]], :(begin  # none, line 2:
        #          GenSym(2) = (Base.steprange_last)(x::Int64,z::Int64,y::Int64)::Int64
        #          GenSym(0) = $(Expr(:new, StepRange{Int64,Int64}, :(x::Int64), :(z::Int64), GenSym(2)))
        #          #s1 = (top(getfield))(GenSym(0),:start)::Int64
        #          unless (Base.box)(Base.Bool,(Base.not_int)((Base.box)(Base.Bool,(Base.or_int)((Base.box)(Base.Bool,(Base.and_int)((Base.box)(Base.Bool,(Base.not_int)((top(getfield))(GenSym(0),:start)::Int64 === (top(getfield))(GenSym(0),:stop)::Int64::Bool))::Bool,(Base.box)(Base.Bool,(Base.not_int)((Base.slt_int)(0,(top(getfield))(GenSym(0),:step)::Int64)::Bool === (Base.slt_int)((top(getfield))(GenSym(0),:start)::Int64,(top(getfield))(GenSym(0),:stop)::Int64)::Bool::Bool))::Bool))::Bool,#s1::Int64 === (Base.box)(Base.Int,(Base.add_int)((top(getfield))(GenSym(0),:stop)::Int64,(top(getfield))(GenSym(0),:step)::Int64))::Int64::Bool))::Bool))::Bool goto 1
        #          2: 
        #          GenSym(3) = #s1::Int64
        #          GenSym(4) = (Base.box)(Base.Int,(Base.add_int)(#s1::Int64,(top(getfield))(GenSym(0),:step)::Int64))::Int64
        #          i = GenSym(3)
        #          #s1 = GenSym(4) # none, line 3:
        #          (Base.println)(Base.STDOUT,i::Int64)
        #          3: 
        #          unless (Base.box)(Base.Bool,(Base.not_int)((Base.box)(Base.Bool,(Base.not_int)((Base.box)(Base.Bool,(Base.or_int)((Base.box)(Base.Bool,(Base.and_int)((Base.box)(Base.Bool,(Base.not_int)((top(getfield))(GenSym(0),:start)::Int64 === (top(getfield))(GenSym(0),:stop)::Int64::Bool))::Bool,(Base.box)(Base.Bool,(Base.not_int)((Base.slt_int)(0,(top(getfield))(GenSym(0),:step)::Int64)::Bool === (Base.slt_int)((top(getfield))(GenSym(0),:start)::Int64,(top(getfield))(GenSym(0),:stop)::Int64)::Bool::Bool))::Bool))::Bool,#s1::Int64 === (Base.box)(Base.Int,(Base.add_int)((top(getfield))(GenSym(0),:stop)::Int64,(top(getfield))(GenSym(0),:step)::Int64))::Int64::Bool))::Bool))::Bool))::Bool goto 2
        #          1: 
        #          0: 
        #          return
        #      end::Void))))

        this_nest = the_parfor.loopNests[loop_nest_level]

        label_after_first_unless   = next_available_label
        label_before_second_unless = next_available_label + 1
        label_after_second_unless  = next_available_label + 2
        label_last                 = next_available_label + 3

        num_vars = 5

        gensym2_var = string("#recreate_gensym2_", (loop_nest_level-1) * num_vars + 0)
        gensym2_sym = symbol(gensym2_var)
        gensym0_var = string("#recreate_gensym0_", (loop_nest_level-1) * num_vars + 1)
        gensym0_sym = symbol(gensym0_var)
        pound_s1_var = string("#recreate_pound_s1_", (loop_nest_level-1) * num_vars + 2)
        pound_s1_sym = symbol(pound_s1_var)
        gensym3_var = string("#recreate_gensym3_", (loop_nest_level-1) * num_vars + 3)
        gensym3_sym = symbol(gensym3_var)
        gensym4_var = string("#recreate_gensym4_", (loop_nest_level-1) * num_vars + 4)
        gensym4_sym = symbol(gensym4_var)
        CompilerTools.LambdaHandling.addLocalVar(gensym2_sym, Int64, ISASSIGNED, newLambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(gensym0_sym, StepRange{Int64,Int64}, ISASSIGNED, newLambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(pound_s1_sym, Int64, ISASSIGNED, newLambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(gensym3_sym, Int64, ISASSIGNED, newLambdaInfo)
        CompilerTools.LambdaHandling.addLocalVar(gensym4_sym, Int64, ISASSIGNED, newLambdaInfo)

        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "ranges = ", SymbolNode(:ranges, pir_range_actual)))
        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.lower = ", this_nest.lower))
        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.step  = ", this_nest.step))
        #push!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.upper = ", this_nest.upper))

           push!(new_body, mk_assignment_expr(SymbolNode(gensym2_sym,Int64), Expr(:call, GlobalRef(Base,:steprange_last), convertUnsafeOrElse(this_nest.lower), convertUnsafeOrElse(this_nest.step), convertUnsafeOrElse(this_nest.upper)), state))
           push!(new_body, mk_assignment_expr(SymbolNode(gensym0_sym,StepRange{Int64,Int64}), Expr(:new, StepRange{Int64,Int64}, convertUnsafeOrElse(this_nest.lower), convertUnsafeOrElse(this_nest.step), SymbolNode(gensym2_sym,Int64)), state))
           push!(new_body, mk_assignment_expr(SymbolNode(pound_s1_sym,Int64), Expr(:call, TopNode(:getfield), SymbolNode(gensym0_sym,StepRange{Int64,Int64}), QuoteNode(:start)), state))
           push!(new_body, mk_gotoifnot_expr(TypedExpr(Bool, :call, mk_parallelir_ref(:first_unless), SymbolNode(gensym0_sym,StepRange{Int64,Int64}), SymbolNode(pound_s1_sym,Int64)), label_after_second_unless))
           push!(new_body, LabelNode(label_after_first_unless))

#           push!(new_body, Expr(:call, GlobalRef(Base,:println), GlobalRef(Base,:STDOUT), " in label_after_first_unless section"))

           push!(new_body, mk_assignment_expr(SymbolNode(gensym3_sym,Int64), SymbolNode(pound_s1_sym,Int64), state))
           push!(new_body, mk_assignment_expr(SymbolNode(gensym4_sym,Int64), Expr(:call, mk_parallelir_ref(:assign_gs4), SymbolNode(gensym0_sym,StepRange{Int64,Int64}), SymbolNode(pound_s1_sym,Int64)), state))
           push!(new_body, mk_assignment_expr(this_nest.indexVariable, SymbolNode(gensym3_sym,Int64), state))
           push!(new_body, mk_assignment_expr(SymbolNode(pound_s1_sym,Int64), SymbolNode(gensym4_sym,Int64), state))

           recreateLoopsInternal(new_body, the_parfor, loop_nest_level + 1, next_available_label + 4, state, newLambdaInfo)

           push!(new_body, LabelNode(label_before_second_unless))
           push!(new_body, mk_gotoifnot_expr(TypedExpr(Bool, :call, mk_parallelir_ref(:second_unless), SymbolNode(gensym0_sym,StepRange{Int64,Int64}), SymbolNode(pound_s1_sym,Int64)), label_after_first_unless))
           push!(new_body, LabelNode(label_after_second_unless))
           push!(new_body, LabelNode(label_last))
    end
end

@doc """
In threads mode, we can't have parfor_start and parfor_end in the code since Julia has to compile the code itself and so
we have to reconstruct a loop infrastructure based on the parfor's loop nest information.  This function takes a parfor
and outputs that parfor to the new function body as regular Julia loops.
"""
function recreateLoops(new_body, the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, state, newLambdaInfo)
    max_label = getMaxLabel(0, the_parfor.body)
    dprintln(2,"recreateLoops ", the_parfor, " max_label = ", max_label)
    # Call the internal loop re-construction code after initializing which loop nest we are working with and the next usable label ID (max_label+1).
    recreateLoopsInternal(new_body, the_parfor, 1, max_label + 1, state, newLambdaInfo)
    nothing
end

function toTaskArgName(x :: Symbol, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    x
end
function toTaskArgName(x :: SymbolNode, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    x.name
end
function toTaskArgName(x :: GenSym, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    newstr = string("parforToTask_gensym_", x.id)
    ret    = symbol(newstr)
    gsmap[x] = CompilerTools.LambdaHandling.VarDef(ret, CompilerTools.LambdaHandling.getType(x, lambdaInfo), 0)
    return ret
end

function toTaskArgVarDef(x :: Symbol, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    CompilerTools.LambdaHandling.getVarDef(x, lambdaInfo)
end
function toTaskArgVarDef(x :: SymbolNode, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    CompilerTools.LambdaHandling.getVarDef(x.name, lambdaInfo)
end
function toTaskArgVarDef(x :: GenSym, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, lambdaInfo)
    gsmap[x]
end

@doc """
Given a parfor statement index in "parfor_index" in the "body"'s statements, create a TaskInfo node for this parfor.
"""
function parforToTask(parfor_index, bb_statements, body, state)
    assert(typeof(body[parfor_index]) == Expr)
    assert(body[parfor_index].head == :parfor)  # Make sure we got a parfor node to convert.
    the_parfor = body[parfor_index].args[1]     # Get the PIRParForAst object from the :parfor Expr.
    dprintln(3,"parforToTask = ", the_parfor)

    # Create an array of the reduction vars used in this parfor.
    reduction_vars = Symbol[]
    for i in the_parfor.reductions
        push!(reduction_vars, i.reductionVar.name)
    end

    # The call to getIO determines which variables are live at input to this parfor, live at output from this parfor
    # or just used exclusively in the parfor.  These latter become local variables.
    in_vars , out, locals = getIO([parfor_index], bb_statements)
    dprintln(3,"in_vars = ", in_vars)
    dprintln(3,"out_vars = ", out)
    dprintln(3,"local_vars = ", locals)

    # Convert Set to Array
    in_array_names   = Any[]
    modified_symbols = Any[]
    io_symbols       = Any[]
    for i in in_vars
        # Determine for each input var whether the variable is just read, just written, or both.
        swritten = CompilerTools.ReadWriteSet.isWritten(i, the_parfor.rws)
        sread    = CompilerTools.ReadWriteSet.isRead(i, the_parfor.rws)
        sio      = swritten & sread
        if in(i, reduction_vars)
            # reduction_vars must be initialized before the parfor and updated during the parfor 
            # so must be io
            assert(sio)   
        elseif sio
            # If input is both read and written then remember this symbol is input and output.
            push!(io_symbols, i)       
        elseif swritten
            push!(modified_symbols, i)
        else
            push!(in_array_names, i)
        end
    end
    if length(out) != 0
        throw(string("out variable of parfor task not supported right now."))
    end

    # Start to form the lambda VarDef array for the locals to the task function.
    locals_array = CompilerTools.LambdaHandling.VarDef[]
    gensyms = Any[]
    gensyms_table = Dict{SymGen, Any}()
    for i in locals
        if isa(i, Symbol) 
            push!(locals_array, CompilerTools.LambdaHandling.getVarDef(i,state.lambdaInfo))
        elseif isa(i, GenSym) 
            push!(gensyms, CompilerTools.LambdaHandling.getType(i,state.lambdaInfo))
            gensyms_table[i] = GenSym(length(gensyms) - 1)
        else
            assert(false)
        end
    end

    # Form an array of argument access flags for each argument.
    arg_types = Cint[]
    push!(arg_types, ARG_OPT_IN) # for ranges parameter
    for i = 1:length(in_array_names)
        push!(arg_types, ARG_OPT_IN)
    end
    for i = 1:length(modified_symbols)
        push!(arg_types, ARG_OPT_OUT)
    end
    for i = 1:length(io_symbols)
        push!(arg_types, ARG_OPT_INOUT)
    end
    for i in 1:length(reduction_vars)
        push!(arg_types, ARG_OPT_ACCUMULATOR)
    end

    dprintln(3,"in_array_names = ", in_array_names)
    dprintln(3,"modified_symbols = ", modified_symbols)
    dprintln(3,"io_symbols = ", io_symbols)
    dprintln(3,"reduction_vars = ", reduction_vars)
    dprintln(3,"locals_array = ", locals_array)
    dprintln(3,"gensyms = ", gensyms)
    dprintln(3,"gensyms_table = ", gensyms_table)
    dprintln(3,"arg_types = ", arg_types)

    # Will hold GenSym's that would become parameters but can't so need to be made
    # into symbols.
    gsmap = Dict{GenSym,CompilerTools.LambdaHandling.VarDef}()

    # Form an array including symbols for all the in and output parameters plus the additional iteration control parameter "ranges".
    # If we detect a GenSym in the parameter list we replace it with a symbol derived from
    # the GenSym number and add it to gsmap.
    all_arg_names = [:ranges;
                     map(x -> toTaskArgName(x, gsmap, state.lambdaInfo), in_array_names);
                     map(x -> toTaskArgName(x, gsmap, state.lambdaInfo), modified_symbols);
                     map(x -> toTaskArgName(x, gsmap, state.lambdaInfo), io_symbols);
                     map(x -> toTaskArgName(x, gsmap, state.lambdaInfo), reduction_vars)]

    dprintln(3,"gsmap = ", gsmap)
    # Add gsmap to gensyms_table so that the GenSyms in the body can be translated to
    # the corresponding new input parameter name.
    for gsentry in gsmap
      gensyms_table[gsentry[1]] = gsentry[2].name
    end 

    # Form a tuple that contains the type of each parameter.
    all_arg_types_tuple = Expr(:tuple)
    all_arg_types_tuple.args = [
        pir_range_actual;
        map(x -> CompilerTools.LambdaHandling.getType(x, state.lambdaInfo), in_array_names);
        map(x -> CompilerTools.LambdaHandling.getType(x, state.lambdaInfo), modified_symbols);
        map(x -> CompilerTools.LambdaHandling.getType(x, state.lambdaInfo), io_symbols);
        map(x -> CompilerTools.LambdaHandling.getType(x, state.lambdaInfo), reduction_vars)]
    all_arg_type = eval(all_arg_types_tuple)
    # Forms VarDef's for the local variables to the task function.
    args_var = CompilerTools.LambdaHandling.VarDef[]
    push!(args_var, CompilerTools.LambdaHandling.VarDef(:ranges, pir_range_actual, 0))
    append!(args_var, 
      [ map(x -> toTaskArgVarDef(x, gsmap, state.lambdaInfo), in_array_names);
        map(x -> toTaskArgVarDef(x, gsmap, state.lambdaInfo), modified_symbols);
        map(x -> toTaskArgVarDef(x, gsmap, state.lambdaInfo), io_symbols);
        map(x -> toTaskArgVarDef(x, gsmap, state.lambdaInfo), reduction_vars)])
    dprintln(3,"all_arg_names = ", all_arg_names)
    dprintln(3,"all_arg_type = ", all_arg_type)
    dprintln(3,"args_var = ", args_var)

    unique_node_id = get_unique_num()

    # The name of the new task function.
    task_func_name = string("task_func_",unique_node_id)
    task_func_sym  = symbol(task_func_name)

    # Just stub out the new task function...the body and lambda will be replaced below.
    task_func = @eval function ($task_func_sym)($(all_arg_names...))
        throw(string("Some task function's body was not replaced."))
    end
    dprintln(3,"task_func = ", task_func)

    # DON'T DELETE.  Forces function into existence.
    unused_ct = ParallelAccelerator.Driver.code_typed(task_func, all_arg_type)
    dprintln(3, "unused_ct = ", unused_ct)

    newLambdaInfo = CompilerTools.LambdaHandling.LambdaInfo()
    CompilerTools.LambdaHandling.addInputParameters(deepcopy(args_var), newLambdaInfo)
    CompilerTools.LambdaHandling.addLocalVariables(deepcopy(locals_array), newLambdaInfo)
    # Change all variables in the task function to have ASSIGNED desc.
    for vd in newLambdaInfo.var_defs
        var_def = vd[2]
        var_def.desc = CompilerTools.LambdaHandling.ISASSIGNED
    end
    newLambdaInfo.gen_sym_typs = gensyms

    # Creating the new body for the task function.
    task_body = TypedExpr(Int, :body)
    saved_loopNests = deepcopy(the_parfor.loopNests)

    #  for i in all_arg_names
    #    push!(task_body.args, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), string(i), " = ", Symbol(i)))
    #  end

    if ParallelAccelerator.getTaskMode() != ParallelAccelerator.NO_TASK_MODE
        for i = 1:length(the_parfor.loopNests)
            # Put outerloop first in the loopNest
            j = length(the_parfor.loopNests) - i + 1
            the_parfor.loopNests[j].lower = TypedExpr(Int64, :call, TopNode(:add_int),
            TypedExpr(Int64, :call, TopNode(:unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, TopNode(:getfield), :ranges, QuoteNode(:lower_bounds)), i),
            1)
            the_parfor.loopNests[j].upper = TypedExpr(Int64, :call, TopNode(:add_int), 
            TypedExpr(Int64, :call, TopNode(:unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, TopNode(:getfield), :ranges, QuoteNode(:upper_bounds)), i),
            1)
        end
    elseif ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
        for i = 1:length(the_parfor.loopNests)
            # Put outerloop first in the loopNest
            j = length(the_parfor.loopNests) - i + 1
            the_parfor.loopNests[j].lower = TypedExpr(Int64, :call, TopNode(:unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, TopNode(:getfield), :ranges, QuoteNode(:lower_bounds)), i)
            the_parfor.loopNests[j].upper = TypedExpr(Int64, :call, TopNode(:unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, TopNode(:getfield), :ranges, QuoteNode(:upper_bounds)), i)
        end
    end

    dprintln(3, "Before recreation or flattening")

    # Add the parfor stmt to the task function body.
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
        #push!(task_body.args, Expr(:call, GlobalRef(Base,:println), GlobalRef(Base,:STDOUT), "in task func"))
        recreateLoops(task_body.args, the_parfor, state, newLambdaInfo)
    else
        flattenParfor(task_body.args, the_parfor)
    end

    # Add the return statement to the end of the task function.
    # If this is not a reduction parfor then return "nothing".
    # If it is a reduction in threading mode, return a tuple of the reduction variables.
    # The threading infrastructure will then call a user-specified reduction function.
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE &&
       length(the_parfor.reductions) > 0
        ret_tt = Expr(:tuple)
        ret_tt.args = map(x -> x.reductionVar.typ, the_parfor.reductions)
        ret_types = eval(ret_tt)
        dprintln(3, "ret_types = ", ret_types)

        ret_names = map(x -> x.reductionVar.name, the_parfor.reductions)
        dprintln(3, "ret_names = ", ret_names)

        push!(task_body.args, TypedExpr(ret_types, :return, mk_tuple_expr(ret_names, ret_types)))
    else
        #push!(task_body.args, TypedExpr(Void, :return, 0))
        push!(task_body.args, TypedExpr(Void, :return, nothing))
    end

    task_body = CompilerTools.LambdaHandling.replaceExprWithDict!(task_body, gensyms_table)
    # Create the new :lambda Expr for the task function.
    code = CompilerTools.LambdaHandling.lambdaInfoToLambdaExpr(newLambdaInfo, task_body)

    dprintln(3, "New task = ", code)

    m = methods(task_func, all_arg_type)
    if length(m) < 1
        error("Method for ", task_func_name, " is not found")
    else
        dprintln(3,"New ", task_func_name, " = ", m)
    end
    def = m[1].func.code
    dprintln(3, "def = ", def, " type = ", typeof(def))
    dprintln(3, "tfunc = ", def.tfunc)
    def.tfunc[2] = ccall(:jl_compress_ast, Any, (Any,Any), def, code)

    m = methods(task_func, all_arg_type)
    def = m[1].func.code
    if ParallelAccelerator.getPseMode() != ParallelAccelerator.THREADS_MODE
        def.j2cflag = convert(Int32,6)
        ccall(:set_j2c_task_arg_types, Void, (Ptr{UInt8}, Cint, Ptr{Cint}), task_func_name, length(arg_types), arg_types)
    end
    precompile(task_func, all_arg_type)
    dprintln(3, "def post = ", def, " type = ", typeof(def))

    if DEBUG_LVL >= 3
        task_func_ct = ParallelAccelerator.Driver.code_typed(task_func, all_arg_type)
        if length(task_func_ct) == 0
            println("Error getting task func code.\n")
        else
            task_func_ct = task_func_ct[1]
            println("Task func code for ", task_func)
            println(task_func_ct)    
        end
    end

    # If this task has reductions, then we create a Julia buffer that holds a C function that we build up in the section below.
    # We then force this C code into the rest of the C code generated by CGen with a special call.
    reduction_func_name = string("")
    if length(the_parfor.reductions) > 0
        if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
            if true
            reduction_func_name = string("reduction_func_",unique_node_id)
            fstring = string("function ", reduction_func_name, "(in1, in2)\n")
            fstring = string(fstring, "return (")

            for i = 1:length(the_parfor.reductions)
                fstring = string(fstring, "in1[", i, "]")
                if the_parfor.reductions[i].reductionFunc == :+
                    fstring = string(fstring, "+")
                elseif the_parfor.reductions[i].reductionFunc == :*
                    fstring = string(fstring, "*")
                else
                    throw(string("Unsupported reduction function ", the_parfor.reductions[i].reductionFunc, " during join function generation."))
                end

                fstring = string(fstring, "in2[", i, "]")

                # Add comma between tuple elements if there is at least one more tuple element.
                if i != length(the_parfor.reductions)
                    fstring = string(fstring, " , ")
                end
            end
 
            fstring = string(fstring, ")\nend\n")
            dprintln(3, "Reduction function for threads mode = ", fstring)
            fparse  = parse(fstring)
            feval   = eval(fparse)
            dprintln(3, "Reduction function for threads done.")
            end
        else
            # The name of the new reduction function.
            reduction_func_name = string("reduction_func_",unique_node_id)

            the_types = AbstractString[]
            for i = 1:length(the_parfor.reductions)
                if the_parfor.reductions[i].reductionVar.typ == Float64
                    push!(the_types, "double")
                elseif the_parfor.reductions[i].reductionVar.typ == Float32
                    push!(the_types, "float")
                elseif the_parfor.reductions[i].reductionVar.typ == Int64
                    push!(the_types, "int64_t")
                elseif the_parfor.reductions[i].reductionVar.typ == Int32
                    push!(the_types, "int32_t")
                else
                    throw(string("Unsupported reduction var type ", the_parfor.reductions[i].reductionVar.typ))
                end
            end

            # The reduction function doesn't return anything.
            c_reduction_func = string("void _")
            c_reduction_func = string(c_reduction_func, reduction_func_name)
            # All reduction functions have the same signature so that the runtime can call back into them.
            # The accumulator is effectively a pointer to Any[].  This accumulator is initialized by the runtime with the initial reduction value for the given type.
            # The new_reduction_vars are also a pointer to Any[] and contains new value to merge into the accumulating reduction var in accumulator.
            c_reduction_func = string(c_reduction_func, "(void **accumulator, void **new_reduction_vars) {\n")
            for i = 1:length(the_parfor.reductions)
                # For each reduction, put one statement in the C function.  Figure out here if it is a + or * reduction.
                if the_parfor.reductions[i].reductionFunc == :+
                    this_op = "+"
                elseif the_parfor.reductions[i].reductionFunc == :*
                    this_op = "*"
                else
                    throw(string("Unsupported reduction function ", the_parfor.reductions[i].reductionFunc, " during join function generation."))
                end
                # Add the statement to the function basically of this form *(accumulator[i]) = *(accumulator[i]) + *(new_reduction_vars[i]).
                # but instead of "+" use the operation type determined above stored in "this_op" and before you can dereference each element, you have to cast
                # the pointer to the appropriate type for this reduction element as stored in the_types[i].
                c_reduction_func = string(c_reduction_func, "*((", the_types[i], "*)accumulator[", i-1, "]) = *((", the_types[i], "*)accumulator[", i-1, "]) ", this_op, " *((", the_types[i], "*)new_reduction_vars[", i-1, "]);\n")
            end
            c_reduction_func = string(c_reduction_func, "}\n")
            dprintln(3,"Created reduction function is:")
            dprintln(3,c_reduction_func)

            # Tell CGen to put this reduction function directly into the C code.
            ccall(:set_j2c_add_c_code, Void, (Ptr{UInt8},), c_reduction_func)
        end
    end

    dprintln(3,"End of parforToTask")

    ret = TaskInfo(task_func,     # The task function that we just generated of type Function.
                    task_func_sym, # The task function's Symbol name.
                    reduction_func_name, # The name of the C reduction function created for this task.
                    map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.lambdaInfo)), in_array_names),
                    map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.lambdaInfo)), modified_symbols),
                    map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.lambdaInfo)), io_symbols),
                    map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.lambdaInfo)), reduction_vars),
                    code,          # The AST for the task function.
                    saved_loopNests)
    return ret
end

@doc """
Form a task out of a range of sequential statements.
This is not currently implemented.
"""
function seqTask(body_indices, bb_statements, body, state)
    getIO(body_indices, bb_statements)  
    throw(string("seqTask construction not implemented yet."))
    TaskInfo(:FIXFIXFIX, :FIXFIXFIX, Any[], Any[], nothing, PIRLoopNest[])
end



