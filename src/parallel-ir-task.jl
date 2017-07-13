
"""
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

"""
Store information about a section of a body that will be translated into a task.
"""
type ReplacedRegion
    start_index
    end_index
    bb
    tasks
end

type EntityType
    name :: RHSVar
    typ
end

"""
Structure for storing information about task formation.
"""
type TaskInfo
    task_func       :: Function                  # The Julia task function that we generated for a task.
    function_sym
    join_func                                    # The name of the C join function that we constructed and forced into the C file.
    ret_types                                    # Tuple containing the types of reduction variables.
    input_symbols   :: Array{EntityType,1}       # Variables that are needed as input to the task.
    modified_inputs :: Array{EntityType,1} 
    io_symbols      :: Array{EntityType,1}
    reduction_vars  :: Array{EntityType,1}
    code
    loopNests       :: Array{PIRLoopNest,1}      # Holds information about the loop nests.
end

function show(io::IO, ti :: TaskInfo)
    println("Task function = ", ti.task_func)
    println("function_sym = ", ti.function_sym)
    println("join_func = ", ti.join_func)
    println("ret_types = ", ti.ret_types)
    println("input_symbols = ", ti.input_symbols)
    println("modified_inputs = ", ti.modified_inputs)
    println("io_symbols = ", ti.io_symbols)
    println("reduction_vars = ", ti.reduction_vars)
    println("code = ", ti.code)
    println("loopNests = ", ti.loopNests)
end

"""
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

"""
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

"""
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

"""
Create an array access descriptor for "array".
Presumes that for point "i" in the iteration space that only index "i" is accessed.
"""
function create1D_array_access_desc(array :: TypedVar)
    ret = pir_array_access_desc()
    push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 1), 1, 0, 0, 0 ))
    ret
end

"""
Create an array access descriptor for "array".
Presumes that for points "(i,j)" in the iteration space that only indices "(i,j)" is accessed.
"""
function create2D_array_access_desc(array :: TypedVar)
    ret = pir_array_access_desc()
    push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 1), 1, 0, 0, 0 ))
    push!(ret.dim_info, pir_aad_dim(mk_arraylen_expr(array, 2), 1, 0, 0, 0 ))
    ret
end

"""
Create an array access descriptor for "array".
"""
function create_array_access_desc(array :: TypedVar)
    if array.typ.parameters[2] == 1
        return create1D_array_access_desc(array)
    elseif array.typ.parameters[2] == 2
        return create2D_array_access_desc(array)
    else
        throw(string("Greater than 2D arrays not supported in create_array_access_desc."))
    end
end

"""
A Julia representation of the argument metadata that will be passed to the runtime.
"""
type pir_arg_metadata
    value   :: TypedVar
    options :: Int
    access  # nothing OR pir_array_access_desc

    function pir_arg_metadata(v, o)
        new(v, o, nothing)
    end

    function pir_arg_metadata(v, o, a)
        new(v, o, a)
    end
end

"""
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

"""
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

"""
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

"""
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

#num_threads_mode = 0
#function PIRNumThreadsMode(x)
#    global num_threads_mode = x
#end

stencil_tasks = 1
function PIRStencilTasks(x)
    global stencil_tasks = x
end

reduce_tasks = 0 
function PIRReduceTasks(x)
    global reduce_tasks = x
end

"""
Returns true if the "node" is a parfor and the task limit hasn't been exceeded.
Also controls whether stencils or reduction can become tasks.
"""
function taskableParfor(node)
    @dprintln(3,"taskableParfor for: ", node)
    if limit_task == 0
        @dprintln(3,"task limit exceeded so won't convert parfor to task")
        return false
    end
    if isParforAssignmentNode(node) || isBareParfor(node)
        @dprintln(3,"Found parfor node, stencil: ", stencil_tasks, " reductions: ", reduce_tasks)
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
    LambdaVarInfo     :: Union{CompilerTools.LambdaHandling.LambdaVarInfo, Void}
end

ASSIGNMENT_COST = 1.0
RETURN_COST = 1.0
ARRAYSET_COST = 4.0
UNSAFE_ARRAYSET_COST = 2.0
ARRAYREF_COST = 4.0
UNSAFE_ARRAYREF_COST = 2.0
FLOAT_ARITH = 1.0

call_costs = Dict{Any,Any}()
call_costs[(:mul_float,(Float64,Float64))] = 1.0
call_costs[(:div_float,(Float64,Float64))] = 1.0
call_costs[(:add_float,(Float64,Float64))] = 1.0
call_costs[(:sub_float,(Float64,Float64))] = 1.0
call_costs[(:neg_float,(Float64,))] = 1.0
call_costs[(:mul_float,(Float32,Float32))] = 1.0
call_costs[(:div_float,(Float32,Float32))] = 1.0
call_costs[(:add_float,(Float32,Float32))] = 1.0
call_costs[(:sub_float,(Float32,Float32))] = 1.0
call_costs[(:neg_float,(Float32,))] = 1.0
call_costs[(:mul_int,(Int64,Int64))] = 1.0
call_costs[(:div_int,(Int64,Int64))] = 1.0
call_costs[(:add_int,(Int64,Int64))] = 1.0
call_costs[(:sub_int,(Int64,Int64))] = 1.0
call_costs[(:sle_int,(Int64,Int64))] = 1.0
call_costs[(:sitofp,(DataType,Int64))] = 1.0
call_costs[(:log10,(Float64,))] = 160.0
call_costs[(:erf,(Float64,))] = 75.0

"""
A sentinel in the instruction count estimation process.
Before recursively processing a call, we add a sentinel for that function so that if we see that
sentinel later we know we've tried to recursively process it and so can bail out by setting
fully_analyzed to false.
"""
type InProgress
end

"""
Generate an instruction count estimate for a call instruction.
"""
function call_instruction_count(args, state :: eic_state, debug_level)
    func  = args[1]
    func = isa(func, TopNode) ? func.name : func
    func = isa(func, GlobalRef) ? func.name : func
    fargs = args[2:end]

    @dprintln(3,"call_instruction_count: func = ", func, " fargs = ", fargs)
    sig_expr = Expr(:tuple)
    sig_expr.args = map(x -> CompilerTools.LivenessAnalysis.typeOfOpr(x, state.LambdaVarInfo), fargs)
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

"""
Try to figure out the instruction count for a given call.
"""
function generate_instr_count(function_name, signature)
    # Estimate instructions for some well-known functions.
    if function_name == :arrayset || function_name == :arrayref
        call_costs[(function_name, signature)] = 4.0
        return call_costs[(function_name, signature)]
    elseif function_name == :unsafe_arrayset || function_name == :unsafe_arrayref
        call_costs[(function_name, signature)] = 2.0
        return call_costs[(function_name, signature)]
    elseif function_name == :safe_arrayref
        call_costs[(function_name, signature)] = 6.0
        return call_costs[(function_name, signature)]
    elseif function_name == :box
        call_costs[(function_name, signature)] = 20.0
        return call_costs[(function_name, signature)]
    elseif function_name == :lt_float || 
        function_name == :le_float ||
        function_name == :not_int
        call_costs[(function_name, signature)] = 1.0
        return call_costs[(function_name, signature)]
    end

    function_name = process_function_name(function_name)

    ftyp = typeof(function_name)

    if ftyp != Function
        @dprintln(3,"generate_instr_count: instead of Function, got ", ftyp, " ", function_name)
    end

    if ftyp == IntrinsicFunction
        @dprintln(3, "generate_instr_count: found IntrinsicFunction = ", function_name)
        call_costs[(function_name, signature)] = nothing
        return call_costs[(function_name, signature)]
    end

    if typeof(function_name) != Function || !isgeneric(function_name)
        @dprintln(3, "generate_instr_count: function_name is not a Function = ", function_name)
        call_costs[(function_name, signature)] = nothing
        return call_costs[(function_name, signature)]
    end

    m = methods(function_name, signature)
    if length(m) < 1
        return nothing
        #    error("Method for ", function_name, " with signature ", signature, " is not found")
    end

    ct = ParallelAccelerator.Driver.code_typed(function_name, signature)      # get information about code for the given function and signature

    @dprintln(2,"generate_instr_count ", function_name, " ", signature)
    state = eic_state(0, true, nothing)
    # Try to estimate the instruction count for the other function.
    AstWalk(ct, estimateInstrCount, state)
    @dprintln(2,"instruction count estimate for parfor = ", state)
    # If so then cache the result.
    if state.fully_analyzed
        call_costs[(function_name, signature)] = state.non_calls
    else
        call_costs[(function_name, signature)] = nothing
    end
    return call_costs[(function_name, signature)]
end

function process_function_name(function_name::Union{Expr,GlobalRef})
    @dprintln(3,"eval'ing", typeof(function_name), "to Function")
    function_name = eval(function_name)
    return function_name
end

function process_function_name(function_name::Any)
    return function_name
end

"""
AstWalk callback for estimating the instruction count.
"""
function estimateInstrCount(ast::Expr, state :: eic_state, top_level_number, is_top_level, read)
    debug_level = 2

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
        linfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
        state.LambdaVarInfo = linfo
    elseif head == :body || head == :block || head == :(::) || head == :line || head == :& || head == :(.) || head == :copyast
        # skip
    elseif head == :(=)
        state.non_calls = state.non_calls + ASSIGNMENT_COST
    elseif head == :return
        state.non_calls = state.non_calls + RETURN_COST
    elseif ast == :call || ast == :invoke
        call_instruction_count(getCallArguments(ast), state, debug_level)
    elseif head == :call1
        dprintln(debug_level,head, " not handled in instruction counter")
        # TODO?: tuple
    elseif head == :gotoifnot
        state.non_calls = state.non_calls + 1
    elseif head == :copy || head == :new || head == :arraysize || head == :alloc || head == :boundscheck
        dprintln(debug_level, head, " not handled in instruction counter")
    elseif head == :type_goto || head == :enter || head == :leave
        dprintln(debug_level, head, " not handled in instruction counter")
        state.fully_analyzed = false
    elseif head == :the_exception
        state.fully_analyzed = false
    else
        @dprintln(1,"instruction count estimator: unknown Expr head :", head, " ", ast)
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function estimateInstrCount(ast::Union{Symbol,TypedVar,TopNode,LineNumberNode,LabelNode,DataType,Void,Module},
                            state :: eic_state,
                            top_level_number,
                            is_top_level,
                            read)
    # skip
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function estimateInstrCount(ast::Union{GlobalRef,GotoNode},
                            state :: eic_state,
                            top_level_number,
                            is_top_level,
                            read)
    state.non_calls = state.non_calls + 1
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function estimateInstrCount(ast::Union{QuoteNode,AbstractString,Tuple,NewvarNode},
                            state :: eic_state,
                            top_level_number,
                            is_top_level,
                            read)
    debug_level = 2
    dprintln(debug_level, typeof(ast), " not handled in instruction counter ", ast)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function estimateInstrCount(ast::Any, state :: eic_state, top_level_number, is_top_level, read)
    asttyp = typeof(ast)
    if isbits(asttyp)
        # skip
    else
        @dprintln(1,"instruction count estimator: unknown AST (", asttyp, ",", ast, ")")
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Takes a parfor and walks the body of the parfor and estimates the number of instruction needed for one instance of that body.
"""
function createInstructionCountEstimate(the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, state :: expr_state)
    if num_threads_mode == 1 || num_threads_mode == 2 || num_threads_mode == 3
        @dprintln(2,"instruction count estimate for parfor = ", the_parfor)
        new_state = eic_state(0, true, state.LambdaVarInfo)
        for i = 1:length(the_parfor.body)
            AstWalk(the_parfor.body[i], estimateInstrCount, new_state)
        end
        # If fully_analyzed is true then there's nothing that couldn't be analyzed so store the instruction count estimate in the parfor.
        if new_state.fully_analyzed
            the_parfor.instruction_count_expr = new_state.non_calls
        else
            the_parfor.instruction_count_expr = nothing
        end
        @dprintln(2,"instruction count estimate for parfor = ", the_parfor.instruction_count_expr)
    end
end

"""
Marks an assignment statement where the left-hand side can take over the storage from the right-hand side.
"""
type RhsDead
end

# Task Graph Modes
SEQUENTIAL_TASKS = 1    # Take the first parfor in the block and the last parfor and form tasks for all parallel and sequential parts inbetween.
ONE_AT_A_TIME = 2       # Just forms tasks out of one parfor at a time.
MULTI_PARFOR_SEQ_NO = 3 # Forms tasks from multiple parfor in sequence but not sequential tasks.

task_graph_mode = ONE_AT_A_TIME
"""
Control how blocks of code are made into tasks.
"""
function PIRTaskGraphMode(x)
    global task_graph_mode = x
end



if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE

println_mutex = Mutex()
one_at_a_time_mutex = Mutex()

function lock_one()
    Base.Threads.lock(one_at_a_time_mutex)
end
function unlock_one()
    Base.Threads.unlock(one_at_a_time_mutex)
end

function ThreadSafePrintln(args...)
    Base.Threads.lock(println_mutex)
    println(args...)
    Base.Threads.unlock(println_mutex)
end

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

precompile(chunk, (Int, Int, Int))

function isfRangeToActual(build :: Array{isf_range,1})
    bcopy = deepcopy(build)
    unsort = sort(bcopy, by=x->x.dim)
    ParallelAccelerator.ParallelIR.pir_range_actual(map(x -> x.lower_bound, unsort), map(x -> x.upper_bound, unsort))    
end

precompile(isfRangeToActual,(Array{isf_range,1},))

function divide_work(full_iteration_space :: ParallelAccelerator.ParallelIR.pir_range_actual,
                     assignments :: Array{ParallelAccelerator.ParallelIR.pir_range_actual,1}, 
                     build       :: Array{isf_range, 1}, 
                     start_thread, end_thread, dims, index)
    num_threads = (end_thread - start_thread) + 1

#    @dprintln(3,"divide_work num_threads = ", num_threads, " build = ", build, " start = ", start_thread, " end = ", end_thread, " dims = ", dims, " index = ", index)
    assert(num_threads >= 1)
    if num_threads == 1
        assert(length(build) <= length(dims))

        if length(build) == length(dims)
            pra = isfRangeToActual(build)
            assignments[start_thread] = pra
        else 
            new_build = [build[1:(index-1)]; isf_range(dims[index].dim, full_iteration_space.lower_bounds[dims[index].dim], full_iteration_space.upper_bounds[dims[index].dim])]
            divide_work(full_iteration_space, assignments, new_build, start_thread, end_thread, dims, index+1)
        end
    else
        assert(index <= length(dims))
        total_len = sum(map(x -> x.len, dims[index:end]))
        if total_len == 0
            divisions_for_this_dim = num_threads
        else
            percentage = dims[index].len / total_len
            divisions_for_this_dim = Int(round(num_threads * percentage))
#            @dprintln(3, "total = ", total_len, " percentage = ", percentage, " divisions = ", divisions_for_this_dim)
        end

        chunkstart = full_iteration_space.lower_bounds[dims[index].dim]
        chunkend   = full_iteration_space.upper_bounds[dims[index].dim]

        threadstart = start_thread
        threadend   = end_thread

        for i = 1:divisions_for_this_dim
            (ls, le, chunkstart)  = chunk(chunkstart,  chunkend,  divisions_for_this_dim - i + 1)
            (ts, te, threadstart) = chunk(threadstart, threadend, divisions_for_this_dim - i + 1)
            divide_work(full_iteration_space, assignments, [build[1:(index-1)]; isf_range(dims[index].dim, ls, le)], ts, te, dims, index+1) 
        end
 
    end
end

precompile(divide_work, (ParallelAccelerator.ParallelIR.pir_range_actual, Array{ParallelAccelerator.ParallelIR.pir_range_actual,1}, Array{isf_range, 1}, Int64, Int64, Array{dimlength,1}, Int64))

function divide_fis(full_iteration_space :: ParallelAccelerator.ParallelIR.pir_range_actual, numtotal :: Int)
#    @dprintln(3, "divide_fis space = ", full_iteration_space, " numtotal = ", numtotal)
    dims = sort(dimlength[dimlength(i, full_iteration_space.upper_bounds[i] - full_iteration_space.lower_bounds[i] + 1)  for i = 1:full_iteration_space.dim], by=x->x.len, rev=true)
#    @dprintln(3, "dims = ", dims)
    assignments = Array{ParallelAccelerator.ParallelIR.pir_range_actual}(numtotal)
    divide_work(full_iteration_space, assignments, isf_range[], 1, numtotal, dims, 1)
    return assignments
end

precompile(divide_fis, (ParallelAccelerator.ParallelIR.pir_range_actual, Int64))

if false

# Only for testing.
function isf(t :: Function, 
             full_iteration_space :: ParallelAccelerator.ParallelIR.pir_range_actual,
             rest...)
   #println("isf ", t, " ", full_iteration_space, " ", rest...)
   println("isf ", t, " ", full_iteration_space)
   return t(full_iteration_space, rest...)
end

else

in_thread_region = false

"""
This function checks global in_thread_region to see if we are already in a threaded region and if not then it enters one.
Return true if the code was executed with all threads and false if executed by the current thread.
"""
function dynamic_check(
             t :: Function, 
             full_iteration_space :: ParallelAccelerator.ParallelIR.pir_range_actual,
             rest...)
    if in_thread_region
        t(full_iteration_space, rest...)
        return false
    else
#        println("Entering threaded region.")
        global in_thread_region = true
        ccall("jl_threading_run", Void, (Any,), (Core.svec(isf, t, full_iteration_space, rest...)))
        global in_thread_region = false
#        println("Exiting threaded region.")
        return true
    end
end

"""
An intermediate scheduling function for passing to jl_threading_run.
It takes the task function to run, the full iteration space to run and the normal argument to the task function in "rest..."
"""
function isf(t :: Function, 
             full_iteration_space :: ParallelAccelerator.ParallelIR.pir_range_actual,
             rest...)
#    ccall(:puts, Cint, (Cstring,), "Running isf.")
#println(STDERR, "Running isf. ", t, "args typ=", Tuple{map(typeof, rest)...})
    tid = Base.Threads.threadid()
#    ta = [typeof(x) for x in rest]
#    Base.Threads.lock!(println_mutex)
#    tprintln("Starting isf. tid = ", tid, " space = ", full_iteration_space, " ta = ", ta)
#    Base.Threads.unlock!(println_mutex)
#   println("isf ", t, " ", full_iteration_space, " ", rest..., " tid = ", tid, " nthreads = ", nthreads())

    if full_iteration_space.dim == 1
        # Compute how many iterations to run.
        num_iters = full_iteration_space.upper_bounds[1] - full_iteration_space.lower_bounds[1] + 1

#        Base.Threads.lock!(println_mutex)
#        tprintln("tid = ", tid, " num_iters = ", num_iters)
#        Base.Threads.unlock!(println_mutex)

#println("num_iters = ", num_iters)
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
#println("ls, le = ", ls, ", ", le)
              return t(ParallelAccelerator.ParallelIR.pir_range_actual(ls,le), rest...)
#println("after ls, le = ", ls, ", ", le)
            catch something
#println("caught ", something)
             # println("Call to t created exception ", something)
              bt = catch_backtrace()
              s = sprint(io->Base.show_backtrace(io, bt))
              ccall(:puts, Cint, (Cstring,), string(s))
  
              msg = string("caught some exception task = ", t, " num_threads = ", nthreads(), " tid = ", tid, " fis = ", full_iteration_space, " ", ls, " ", le)
              ccall(:puts, Cint, (Cstring,), msg)
              msg = string(something)
              ccall(:puts, Cint, (Cstring,), msg)
              throw(msg)
            end 
#            tprintln("After t call tid = ", tid)
#println("After t call.")
        end
    elseif full_iteration_space.dim >= 2
        assignments = ParallelAccelerator.ParallelIR.pir_range_actual[]
        try
            assignments = divide_fis(full_iteration_space, nthreads())
        catch something
#            Base.Threads.lock!(println_mutex)
            msg = string("Error dividing up work for threads, range = ", full_iteration_space, " num_threads = ", nthreads(), " tid = ", tid)
#            Base.Threads.unlock!(println_mutex)
            ccall(:puts, Cint, (Cstring,), msg)

            bt = catch_backtrace()
            s = sprint(io->Base.show_backtrace(io, bt))
            ccall(:puts, Cint, (Cstring,), string(s))

            msg = string(something)
            ccall(:puts, Cint, (Cstring,), msg)
            throw(msg)
        end

        try 
            #msg = string("Running num_threads = ", nthreads(), " tid = ", tid, " assignment = ", assignments[tid])
            #ccall(:puts, Cint, (Cstring,), msg)
            return t(assignments[tid], rest...)
        catch something
            msg = string("caught some exception num_threads = ", nthreads(), " tid = ", tid, " assignments = ", assignments)
            ccall(:puts, Cint, (Cstring,), msg)

            bt = catch_backtrace()
            s = sprint(io->Base.show_backtrace(io, bt))
            ccall(:puts, Cint, (Cstring,), string(s))

            msg = string(something)
            ccall(:puts, Cint, (Cstring,), msg)
            throw(msg)
        end 
    end
end

end

end # end if THREADS_MODE



"""
For a given start and stop index in some body and liveness information, form a set of tasks.
"""
function makeTasks(start_index, stop_index, body, bb_live_info, state, task_graph_mode)
    task_list = Any[]
    seq_accum = Any[]
    @dprintln(3,"makeTasks starting")

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
                @dprintln(3,"Adding sequential task to task_list. ", st)
                push!(task_list, st)
                seq_accum = Any[]
            end
            ptt = parforToTask(j, bb_live_info.statements, body, state)
            @dprintln(3,"Adding parfor task to task_list. ", ptt)
            push!(task_list, ptt)
        else
            # is not a parfor node
            assert(task_graph_mode != ONE_AT_A_TIME)
            if task_graph_mode == SEQUENTIAL_TASKS
                # Collect the non-parfor stmts in a batch to be turned into one sequential task.
                push!(seq_accum, body[j])
            else
                @dprintln(3,"Adding non-parfor node to task_list. ", body[j])
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

"""
Given a set of statement IDs and liveness information for the statements of the function, determine
which symbols are needed at input and which symbols are purely local to the function.
"""
function getIO(stmt_ids, bb_statements)
    assert(length(stmt_ids) > 0)

    # Get the statements out of the basic block statement array such that those statement's ID's are in the stmt_ids ID array.
    stmts_for_ids = filter(x -> in(x.tls.index, stmt_ids) , bb_statements)
    @dprintln(3, "stmts_for_ids = ", stmts_for_ids)

    # Make sure that we found a statement for every statement ID.
    if length(stmt_ids) != length(stmts_for_ids)
        @dprintln(0,"length(stmt_ids) = ", length(stmt_ids))
        @dprintln(0,"length(stmts_for_ids) = ", length(stmts_for_ids))
        @dprintln(0,"stmt_ids = ", stmt_ids)
        @dprintln(0,"stmts_for_ids = ", stmts_for_ids)
        assert(length(stmt_ids) == length(stmts_for_ids))
    end
    # The initial set of inputs is those variables "use"d by the first statement.
    # The inputs to the task are those variables used in the set of statements before they are defined in any of those statements.
    cur_inputs = stmts_for_ids[1].use
    @dprintln(3,"getIO cur_inputs = ", cur_inputs)
    # Keep track of variables defined in the set of statements processed thus far.
    cur_defs   = stmts_for_ids[1].def
    @dprintln(3,"getIO cur_defs = ", cur_defs)
    for i = 2:length(stmts_for_ids)
        # For each additional statement, the new set of inputs is the previous set plus uses in the current statement except for those symbols already defined in the function.
        cur_inputs = union(cur_inputs, setdiff(stmts_for_ids[i].use, cur_defs))
        @dprintln(3,"getIO i = ", i, " cur_inputs = ", cur_inputs)
        # For each additional statement, the defs are just union with the def for the current statement.
        cur_defs   = union(cur_defs, stmts_for_ids[i].def)
        @dprintln(3,"getIO i = ", i, " cur_defs = ", cur_defs)
    end
    IntrinsicSet = Set()
    # We will ignore the :Intrinsics symbol as it isn't something you need to pass as a param.
    push!(IntrinsicSet, :Intrinsics)
    # Task functions don't return anything.  They must return via an input parameter so outputs should be empty.
    @dprintln(3, "end liveout = ", stmts_for_ids[end].live_out)
    outputs = setdiff(intersect(cur_defs, stmts_for_ids[end].live_out), IntrinsicSet)
    @dprintln(3, "outputs = ", outputs)
    cur_defs = setdiff(cur_defs, IntrinsicSet)
    cur_inputs = setdiff(filter(x -> !((x === :Int64) || (x === :Float32)), cur_inputs), IntrinsicSet)
    # The locals are those things defined that aren't inputs or outputs of the function.
    cur_inputs, outputs, setdiff(cur_defs, union(cur_inputs, outputs))
end

"""
Returns an expression to construct a :colon object that contains the start of a range, the end and the skip expression.
"""
function mk_colon_expr(start_expr, skip_expr, end_expr)
    TypedExpr(Any, :call, :colon, start_expr, skip_expr, end_expr)
end

"""
Returns an expression to get the start of an iteration range from a :colon object.
"""
function mk_start_expr(colon_sym)
    TypedExpr(Any, :call, GlobalRef(Base, :start), colon_sym)
end

"""
Returns a :next call Expr that gets the next element of an iteration range from a :colon object.
"""
function mk_next_expr(colon_sym, start_sym)
    TypedExpr(Any, :call, GlobalRef(Base, :next), colon_sym, start_sym)
end

"""
Returns a :gotoifnot Expr given a condition "cond" and a label "goto_label".
"""
function mk_gotoifnot_expr(cond, goto_label)
    TypedExpr(Any, :gotoifnot, cond, goto_label)
end

"""
Just to hold the "found" Bool that says whether a unsafe variant was replaced with a regular version.
"""
type cuw_state
    found
    function cuw_state()
        new(false)
    end
end

function safe_arrayref(arr, default, index1)
    if index1 > 0 && index1 <= size(arr, 1)
        return arr[index1]
    else
        return default
    end
end

function safe_arrayref(arr, default, index1, index2)
    if index1 > 0 && index1 <= size(arr,1) && 
       index2 > 0 && index2 <= size(arr,2)
        return arr[index1, index2]
    else
        return default
    end
end

function safe_arrayref(arr, default, index1, index2, index3)
    if index1 > 0 && index1 <= size(arr,1) && 
       index2 > 0 && index2 <= size(arr,2) &&
       index3 > 0 && index3 <= size(arr,3)
        return arr[index1, index2, index3]
    else
        return default
    end
end

"""
The AstWalk callback to find unsafe arrayset and arrayref variants and
replace them with the regular Julia versions.  Sets the "found" flag
in the state when such a replacement is performed.
"""
function convertUnsafeWalk(x::Expr, state, top_level_number, is_top_level, read)
    use_dbg_level = 4
    dprintln(use_dbg_level,"convertUnsafeWalk ", x)

    dprintln(use_dbg_level,"convertUnsafeWalk is Expr")
    if x.head == :call
        dprintln(use_dbg_level,"convertUnsafeWalk is :call")
        if isBaseFunc(x.args[1], :unsafe_arrayset)
            x.args[1] = GlobalRef(Base, :arrayset)
            state.found = true
            return x
        elseif isBaseFunc(x.args[1], :unsafe_arrayref)
            x.args[1] = GlobalRef(Base, :arrayref)
            state.found = true
            return x
        elseif isBaseFunc(x.args[1], :safe_arrayref)
            @dprintln(3,"Replace Base.safe_arrayref with ParallelAccelerator.ParallelIR.safe_arrayref.")
            x.args[1] = GlobalRef(ParallelAccelerator.ParallelIR, :safe_arrayref)
            state.found = true
            return x
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function convertUnsafeWalk(x::ANY, state, top_level_number, is_top_level, read)
    use_dbg_level = 4
    dprintln(use_dbg_level,"convertUnsafeWalk ", x)

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Remove unsafe array access Symbols from the incoming "stmt".
Returns the updated statement if something was modifed, else returns "nothing".
"""
function convertUnsafe(stmt)
    use_dbg_level = 4

    @dprintln(use_dbg_level, "convertUnsafe: ", stmt)
    state = cuw_state() 
    # Uses AstWalk to do the pattern match and replace.
    res = AstWalk(stmt, convertUnsafeWalk, state)
    # state.found is set if the callback convertUnsafeWalk found and replaced an unsafe variant.
    if state.found
        @dprintln(use_dbg_level, "state.found ", state, " ", res)
        @dprintln(use_dbg_level, "Replaced unsafe: ", res)
        return res
    else
        return nothing
    end
end

"""
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

function boxArraysetValueWalk(x::Expr, state, top_level_number, is_top_level, read)
    if x.head == :call
        if isBaseFunc(x.args[1], :unsafe_arrayset) || isBaseFunc(x.args[1], :arrayset)
            @dprintln(3, "boxArraysetValueWalk x = ", x)
            vtyp = CompilerTools.LambdaHandling.getType(x.args[3], state)
            @dprintln(3, "x.args[3] = ", x.args[3], " type = ", vtyp)

            if vtyp <: Complex
                #(Base.arrayset)(x,$(Expr(:new, Complex{Float64}, :((Core.getfield)(z,:re)::Float64), :((Core.getfield)(z,:im)::Float64))),y)::Array{Complex{Float64},1}
                @dprintln(3, "vtyp is Complex")
                x.args[3] = Expr(:new, vtyp, Expr(:call, GlobalRef(Core,:getfield), deepcopy(x.args[3]), QuoteNode(:re)), Expr(:call, GlobalRef(Core,:getfield), deepcopy(x.args[3]), QuoteNode(:im)))
            else
                x.args[3] = Expr(:call, GlobalRef(Base, :box), vtyp, x.args[3])
                return x
            end
        end
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function boxArraysetValueWalk(x::ANY, state, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function boxArraysetValue(stmt, linfo :: LambdaVarInfo)
    return AstWalk(stmt, boxArraysetValueWalk, linfo)
end

function removeInputArgNewvarnode(x::NewvarNode, linfo :: CompilerTools.LambdaHandling.LambdaVarInfo, top_level_number, is_top_level, read)
    @dprintln(3,"removeInputArgNewvarnode = ", x)
    @dprintln(3,x.slot, " ", typeof(x.slot), " ", linfo.input_params)
    if isa(x.slot, Symbol) && in(x.slot, linfo.input_params)
        return CompilerTools.AstWalker.ASTWALK_REMOVE
    elseif isa(x.slot, LHSVar) && in(x.slot, CompilerTools.LambdaHandling.getInputParametersAsLHSVar(linfo))
        return CompilerTools.AstWalker.ASTWALK_REMOVE
    end

    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function removeInputArgNewvarnode(x::ANY, linfo :: CompilerTools.LambdaHandling.LambdaVarInfo, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
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
    @dprintln(4,"first_unless res = ", res)
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
    @dprintln(4,"second_unless res = ", res)
    return res
end

precompile(first_unless, (StepRange{Int64,Int64}, Int64))
precompile(assign_gs4, (StepRange{Int64,Int64}, Int64))
precompile(second_unless, (StepRange{Int64,Int64}, Int64))

DEBUG_TASK_FUNCTIONS = false

function addToBody!(new_body, x, line_num)
    push!(new_body, x) 
    push!(new_body, LineNumberNode(line_num))
    return line_num + 1
end

"""
This is a recursive routine to reconstruct a regular Julia loop nest from the loop nests described in PIRParForAst.
One call of this routine handles one level of the loop nest.
If the incoming loop nest level is more than the number of loops nests in the parfor then that is the spot to
insert the body of the parfor into the new function body in "new_body".
"""
function recreateLoopsInternal(new_body, the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, parfor_nest_level, loop_nest_level, state, newLambdaVarInfo, line_num)
    @dprintln(3,"recreateLoopsInternal loop_nest_level=", loop_nest_level, " parfor_nest_level=", parfor_nest_level)
    if loop_nest_level > length(the_parfor.loopNests) 
        @dprintln(3, "Body size ", length(the_parfor.body))
        # A loop nest level greater than number of nests in the parfor means we can insert the body of the parfor here.
        # For each statement in the parfor body.
        tmp_body = relabel(the_parfor.body, state)
        for i = 1:length(the_parfor.body)
            @dprintln(3, "Body index ", i, " ", the_parfor.body[i])
            # Convert any unsafe_arrayref or sets in this statements to regular arrayref or arrayset.
            # But if it was labeled as "unsafe" then output :boundscheck false Expr so that Julia won't generate a boundscheck on the array access.
            if isBareParfor(the_parfor.body[i])
                @dprintln(3,"Detected nested parfor.  Converting it to a loop.")
                line_num = recreateLoopsInternal(new_body, the_parfor.body[i].args[1], parfor_nest_level + 1, 1, state, newLambdaVarInfo, line_num)
            else
                cu_res = convertUnsafe(the_parfor.body[i])
                @dprintln(3, "cu_res = ", cu_res)
                if cu_res == nothing
                    @dprintln(3, "unmodified stmt = ", the_parfor.body[i])
                end
                if cu_res != nothing
                    if !DEBUG_TASK_FUNCTIONS
                        if VERSION >= v"0.5.0-dev+4449"
                            line_num = addToBody!(new_body, Expr(:inbounds, true), line_num) 
                        else
                            line_num = addToBody!(new_body, Expr(:boundscheck, false), line_num) 
                        end
                    end
                    line_num = addToBody!(new_body, deepcopy(cu_res), line_num)
                    if DEBUG_TASK_FUNCTIONS
                        if isAssignmentNode(cu_res)
                            line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), string(CompilerTools.LambdaHandling.lookupVariableName(cu_res.args[1], newLambdaVarInfo)), " = ", deepcopy(the_parfor.body[i].args[1])), line_num)
                            #line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "last assignment = ", deepcopy(cu_res.args[1])), line_num)
                        else
                           # line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "ranges = ", deepcopy(toLHSVar(:ranges, newLambdaVarInfo))), line_num)
                           line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "after stmt"), line_num)
                        end
                    end
                    if !DEBUG_TASK_FUNCTIONS
                        if VERSION >= v"0.5.0-dev+4449"
                            line_num = addToBody!(new_body, Expr(:inbounds, :pop), line_num) 
                        else
                            line_num = addToBody!(new_body, Expr(:boundscheck, Expr(:call, GlobalRef(Base, :getfield), Base, QuoteNode(:pop))), line_num)
                        end
                    end
                else
                    line_num = addToBody!(new_body, deepcopy(the_parfor.body[i]), line_num)
                    if DEBUG_TASK_FUNCTIONS
                        if isAssignmentNode(the_parfor.body[i])
                            line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), string(CompilerTools.LambdaHandling.lookupVariableName(the_parfor.body[i].args[1], newLambdaVarInfo)), " = ", deepcopy(the_parfor.body[i].args[1])), line_num)
                        else
                            #line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "ranges = ", deepcopy(toLHSVar(:ranges, newLambdaVarInfo))), line_num)
                            line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "after stmt"), line_num)
                        end
                    end
                end
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

        uid = the_parfor.unique_id
        this_nest = the_parfor.loopNests[loop_nest_level]

        if VERSION >= v"0.5.0-dev+4449"
# 1       SSAValue(2) = (Base.steprange_last)(x,z,y)::Int64
# 2       SSAValue(3) = x
# 3       SSAValue(4) = z
# 4       #temp# = SSAValue(3)
# 5       15: 
# 6       unless (Base.box)(Base.Bool,
#                    (Base.not_int)(
#                        (Base.box)(Base.Bool,
#                            (Base.or_int)(
#                                (Base.box)(Base.Bool,
#                                    (Base.and_int)(
#                                        (Base.box)(Base.Bool,
#                                            (Base.not_int)((SSAValue(3) === SSAValue(2))::Bool)),
#                                        (Base.box)(Base.Bool,
#                                            (Base.not_int)(
#                                                ((Base.slt_int)(0,SSAValue(4))::Bool === (Base.slt_int)(SSAValue(3),SSAValue(2))::Bool)::Bool
#                                            )
#                                        )
#                                    )
#                                ),
#                                (#temp# === (Base.box)(Int64,(Base.add_int)(SSAValue(2),SSAValue(4))))::Bool
#                            )
#                        )
#                    )
#                ) 
#                 goto 25
# 7       SSAValue(5) = #temp#
# 8       SSAValue(6) = (Base.box)(Int64,(Base.add_int)(#temp#,SSAValue(4)))
# 9       i = SSAValue(5)
# 10      #temp# = SSAValue(6) # line 3:
# 11      (Base.println)(Base.STDOUT,i)
# 12      goto 15
# 13      25: 
# 14      return

        label_head = next_label(state)
        label_end  = next_label(state)

        num_vars = 6

        steprange_last_var  = Symbol(string("#recreate_steprange_last_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 0))
        steprange_first_var = Symbol(string("#recreate_steprange_first_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 1))
        steprange_step_var  = Symbol(string("#recreate_steprange_step_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 2))
        recreate_temp_var   = Symbol(string("#recreate_temp_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 3))
        recreate_ssa5_var   = Symbol(string("#recreate_ssa5_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 4))
        recreate_ssa6_var   = Symbol(string("#recreate_ssa6_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 5))

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
          
        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(steprange_last_lhsvar), Expr(:call, GlobalRef(Base,:steprange_last), convertUnsafeOrElse(deepcopy(this_nest.lower)), convertUnsafeOrElse(deepcopy(this_nest.step)), convertUnsafeOrElse(deepcopy(this_nest.upper))), newLambdaVarInfo), line_num) # 1
        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(steprange_first_lhsvar), convertUnsafeOrElse(deepcopy(this_nest.lower)), newLambdaVarInfo), line_num) # 2
        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(steprange_step_lhsvar), convertUnsafeOrElse(deepcopy(this_nest.step)), newLambdaVarInfo), line_num) # 3
        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(recreate_temp_lhsvar), deepcopy(steprange_first_rhsvar), newLambdaVarInfo), line_num) # 4

        if DEBUG_TASK_FUNCTIONS
        line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "steprange_last_var = ", deepcopy(steprange_last_rhsvar)), line_num)
        line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "steprange_first_var = ", deepcopy(steprange_first_rhsvar)), line_num)
        line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "steprange_step_var = ", deepcopy(steprange_step_rhsvar)), line_num)
        end

        line_num = addToBody!(new_body, LabelNode(label_head), line_num) # 5

        if DEBUG_TASK_FUNCTIONS
        line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "after label_head"), line_num)
        end

        line_num = addToBody!(new_body, mk_gotoifnot_expr(
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
               , label_end), line_num) # 6

        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(recreate_ssa5_lhsvar), deepcopy(recreate_temp_rhsvar), newLambdaVarInfo), line_num) # 7
        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(recreate_ssa6_lhsvar), boxOrNot(Int64, Expr(:call, GlobalRef(Base, :add_int), deepcopy(recreate_temp_rhsvar), deepcopy(steprange_step_rhsvar))), newLambdaVarInfo), line_num) # 8
        @dprintln(3, "this_nest.indexVariable = ", this_nest.indexVariable, " type = ", typeof(this_nest.indexVariable))
        line_num = addToBody!(new_body, mk_assignment_expr(CompilerTools.LambdaHandling.toLHSVar(deepcopy(this_nest.indexVariable), newLambdaVarInfo), deepcopy(recreate_ssa5_rhsvar), newLambdaVarInfo), line_num) # 9
        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(recreate_temp_lhsvar), deepcopy(recreate_ssa6_rhsvar), newLambdaVarInfo), line_num) # 10

        line_num = recreateLoopsInternal(new_body, the_parfor, parfor_nest_level, loop_nest_level + 1, state, newLambdaVarInfo, line_num) # 11
        line_num = addToBody!(new_body, GotoNode(label_head), line_num) # 12
        line_num = addToBody!(new_body, LabelNode(label_end), line_num) # 13

        if DEBUG_TASK_FUNCTIONS
        line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "after label_end"), line_num)
        end
else
        label_after_first_unless   = next_label(state)
#        label_before_second_unless = next_label(state)
        label_after_second_unless  = next_label(state)
#        label_last                 = next_label(state)

        num_vars = 5

        gensym2_var = string("#recreate_gensym2_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 0)
        gensym2_sym = Symbol(gensym2_var)
        gensym0_var = string("#recreate_gensym0_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 1)
        gensym0_sym = Symbol(gensym0_var)
        pound_s1_var = string("#recreate_pound_s1_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 2)
        pound_s1_sym = Symbol(pound_s1_var)
        gensym3_var = string("#recreate_gensym3_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 3)
        gensym3_sym = Symbol(gensym3_var)
        gensym4_var = string("#recreate_gensym4_", uid, "_", parfor_nest_level, "_", (loop_nest_level-1) * num_vars + 4)
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
        @dprintln(3, "gensym2_lhsvar  = ", gensym2_lhsvar)
        @dprintln(3, "gensym0_lhsvar  = ", gensym0_lhsvar)
        @dprintln(3, "pound_s1_lhsvar = ", pound_s1_lhsvar)
        @dprintln(3, "gensym3_lhsvar  = ", gensym3_lhsvar)
        @dprintln(3, "gensym4_lhsvar  = ", gensym4_lhsvar)

        if DEBUG_TASK_FUNCTIONS
#            line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "ranges = ", deepcopy(toLHSVar(:ranges, newLambdaVarInfo))), line_num)
#            line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.lower = ", convertUnsafeOrElse(deepcopy(this_nest.lower))), line_num)
#            line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.step  = ", convertUnsafeOrElse(deepcopy(this_nest.step))), line_num)
#            line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "this_nest.upper = ", convertUnsafeOrElse(deepcopy(this_nest.upper))), line_num)
        end

        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(gensym2_lhsvar), Expr(:call, GlobalRef(Base,:steprange_last), convertUnsafeOrElse(deepcopy(this_nest.lower)), convertUnsafeOrElse(deepcopy(this_nest.step)), convertUnsafeOrElse(deepcopy(this_nest.upper))), newLambdaVarInfo), line_num)
        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(gensym0_lhsvar), Expr(:new, StepRange{Int64,Int64}, convertUnsafeOrElse(deepcopy(this_nest.lower)), convertUnsafeOrElse(deepcopy(this_nest.step)), deepcopy(gensym2_rhsvar)), newLambdaVarInfo), line_num)
        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(pound_s1_lhsvar), Expr(:call, GlobalRef(Base, :getfield), deepcopy(gensym0_rhsvar), QuoteNode(:start)), newLambdaVarInfo), line_num)
        line_num = addToBody!(new_body, mk_gotoifnot_expr(TypedExpr(Bool, :call, ParallelAccelerator.ParallelIR.first_unless, deepcopy(gensym0_rhsvar), deepcopy(pound_s1_rhsvar)), label_after_second_unless), line_num)
        line_num = addToBody!(new_body, LabelNode(label_after_first_unless), line_num)

        if DEBUG_TASK_FUNCTIONS
#           line_num = addToBody!(new_body, Expr(:call, GlobalRef(Base,:println), GlobalRef(Base,:STDOUT), " in label_after_first_unless section"), line_num)
        end

        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(gensym3_lhsvar), deepcopy(pound_s1_rhsvar), newLambdaVarInfo), line_num)
        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(gensym4_lhsvar), Expr(:call, ParallelAccelerator.ParallelIR.assign_gs4, deepcopy(gensym0_rhsvar), deepcopy(pound_s1_rhsvar)), newLambdaVarInfo), line_num)
        @dprintln(3, "this_nest.indexVariable = ", this_nest.indexVariable, " type = ", typeof(this_nest.indexVariable))
        line_num = addToBody!(new_body, mk_assignment_expr(CompilerTools.LambdaHandling.toLHSVar(deepcopy(this_nest.indexVariable), newLambdaVarInfo), deepcopy(gensym3_rhsvar), newLambdaVarInfo), line_num)

        if DEBUG_TASK_FUNCTIONS
           line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "index_variable ", string(CompilerTools.LambdaHandling.lookupVariableName(this_nest.indexVariable, newLambdaVarInfo)), " = ", CompilerTools.LambdaHandling.toLHSVar(deepcopy(this_nest.indexVariable), newLambdaVarInfo)), line_num)
        end

        line_num = addToBody!(new_body, mk_assignment_expr(deepcopy(pound_s1_lhsvar), deepcopy(gensym4_rhsvar), newLambdaVarInfo), line_num)

        line_num = recreateLoopsInternal(new_body, the_parfor, parfor_nest_level, loop_nest_level + 1, state, newLambdaVarInfo, line_num)

#        line_num = addToBody!(new_body, LabelNode(label_before_second_unless), line_num)
        line_num = addToBody!(new_body, mk_gotoifnot_expr(TypedExpr(Bool, :call, ParallelAccelerator.ParallelIR.second_unless, deepcopy(gensym0_rhsvar), deepcopy(pound_s1_rhsvar)), label_after_first_unless), line_num)
        line_num = addToBody!(new_body, LabelNode(label_after_second_unless), line_num)
#        line_num = addToBody!(new_body, LabelNode(label_last), line_num)
end
    end
    if DEBUG_TASK_FUNCTIONS
       line_num = addToBody!(new_body, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "finished loop nest = ", loop_nest_level), line_num)
    end
    return line_num
end

"""
In threads mode, we can't have parfor_start and parfor_end in the code since Julia has to compile the code itself and so
we have to reconstruct a loop infrastructure based on the parfor's loop nest information.  This function takes a parfor
and outputs that parfor to the new function body as regular Julia loops.
"""
function recreateLoops(new_body, the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, state, newLambdaVarInfo, parfor_nest_level = 1)
    @dprintln(2,"recreateLoops ", the_parfor, " unique_id = ", the_parfor.unique_id)
    # Call the internal loop re-construction code after initializing which loop nest we are working with and the next usable label ID (max_label+1).
    push!(new_body, Expr(:line, 1, Symbol(string("from_parfor_", the_parfor.unique_id))))
    recreateLoopsInternal(new_body, the_parfor, parfor_nest_level, 1, state, newLambdaVarInfo, 2)
    nothing
end

#"""
#Takes a new array of body statements in the process of construction in "new_body" and takes a parfor to add to that
#body.  This parfor is in the nested (parfor code is in the parfor node itself) temporary form we use for fusion although 
#pre-statements and post-statements are already elevated by this point.  We replace this nested form with a non-nested
#form where we have a parfor_start and parfor_end to delineate the parfor code.
#"""
#function flattenParfor(new_body, the_parfor :: ParallelAccelerator.ParallelIR.PIRParForAst, linfo :: LambdaVarInfo)
#    @dprintln(2,"Flattening ", the_parfor)
#
#    private_set = getPrivateSet(the_parfor.body, linfo)
#    private_array = collect(private_set)
#
#    # Output to the new body that this is the start of a parfor.
#    push!(new_body, TypedExpr(Int64, :parfor_start, PIRParForStartEnd(the_parfor.loopNests, the_parfor.reductions, the_parfor.instruction_count_expr, private_array)))
#    # Output the body of the parfor as top-level statements in the new function body.
#    append!(new_body, the_parfor.body)
#    # Output to the new body that this is the end of a parfor.
#    push!(new_body, TypedExpr(Int64, :parfor_end, PIRParForStartEnd(deepcopy(the_parfor.loopNests), deepcopy(the_parfor.reductions), deepcopy(the_parfor.instruction_count_expr), deepcopy(private_array))))
#    nothing
#end

function getVarDef(x :: Union{RHSVar, Symbol}, LambdaVarInfo)
    CompilerTools.LambdaHandling.VarDef(
        CompilerTools.LambdaHandling.lookupVariableName(x, LambdaVarInfo),
        CompilerTools.LambdaHandling.getType(x, LambdaVarInfo),
        CompilerTools.LambdaHandling.getDesc(x, LambdaVarInfo))
end

function toTaskArgVarDef(x :: Union{RHSVar, Symbol}, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, LambdaVarInfo)
    getVarDef(x, LambdaVarInfo)
end
function toTaskArgVarDef(x :: GenSym, gsmap :: Dict{GenSym,CompilerTools.LambdaHandling.VarDef}, LambdaVarInfo)
    gsmap[x]
end

type TaskFuncVariableInfo
    name  :: Symbol
    typ   :: DataType
    value :: LHSVar # The SlotNumber that this name maps to in the task function.
end

function addToTaskFunc!(vars :: Array{LHSVar,1}, 
                        info_dict :: Dict{LHSVar,TaskFuncVariableInfo}, 
                        old_func_linfo :: CompilerTools.LambdaHandling.LambdaVarInfo, 
                        task_func_linfo :: CompilerTools.LambdaHandling.LambdaVarInfo)
    for i = 1:length(vars)
        this_var = vars[i]

        if isa(this_var, LHSRealVar) 
            name = CompilerTools.LambdaHandling.lookupVariableName(this_var, old_func_linfo)
            typ  = CompilerTools.LambdaHandling.getType(this_var, old_func_linfo) 
        elseif isa(this_var, GenSym) 
            newstr = string("parforToTask_gensym_", this_var.id)
            name   = Symbol(newstr)
            typ    = CompilerTools.LambdaHandling.getType(this_var, old_func_linfo)
        else
            assert(false)
        end
        
        newLHSVar = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(name, typ, CompilerTools.LambdaHandling.ISASSIGNED, task_func_linfo))
        info_dict[this_var] = TaskFuncVariableInfo(name, typ, newLHSVar)
    end
end

function ParforBoxify!(parfor :: PIRParForAst, linfo :: LambdaVarInfo)
    for i = 1:length(parfor.body)
       parfor.body[i] = boxArraysetValue(parfor.body[i], linfo)
    end
end

function set_reduction_array(reduction_array :: Array{Any,1}, values...)
    @dprintln(3, "set_reduction_array ", values..., " ", Base.Threads.threadid())
    reduction_array[Base.Threads.threadid()] = (values...)
end

"""
Given a parfor statement index in "parfor_index" in the "body"'s statements, create a TaskInfo node for this parfor.
"""
function parforToTask(parfor_index, bb_statements, body, state)
    assert(typeof(body[parfor_index]) == Expr)
    assert(body[parfor_index].head == :parfor)  # Make sure we got a parfor node to convert.
    the_parfor = body[parfor_index].args[1]     # Get the PIRParForAst object from the :parfor Expr.
    @dprintln(3,"(parforToTask = ", the_parfor)

    # Create an array of the reduction vars used in this parfor.
    reduction_vars = LHSVar[]
    for i in the_parfor.reductions
        push!(reduction_vars, toLHSVar(i.reductionVar))
    end
    @dprintln(3,"reduction_vars = ", reduction_vars, " type = ", typeof(reduction_vars))

    # The call to getIO determines which variables are live at input to this parfor, live at output from this parfor
    # or just used exclusively in the parfor.  These latter become local variables.
    in_vars , out, locals = getIO([parfor_index], bb_statements)
    @dprintln(3,"in_vars = ", in_vars, " type = ", typeof(in_vars))
    @dprintln(3,"out_vars = ", out, " type = ", typeof(out))
    locals = collect(locals)
    @dprintln(3,"local_vars = ", locals, " type = ", typeof(locals))
    # The following 8 lines if just for debugging purposes.
    in_vars_sym = [lookupVariableName(x, state.LambdaVarInfo) for x in in_vars]
    out_vars_sym = [lookupVariableName(x, state.LambdaVarInfo) for x in out]
    locals_vars_sym = [lookupVariableName(x, state.LambdaVarInfo) for x in locals]
    reduction_vars_sym = [lookupVariableName(x, state.LambdaVarInfo) for x in reduction_vars]
    @dprintln(3,"in_vars names = ", in_vars_sym)
    @dprintln(3,"out_vars names= ", out_vars_sym)
    @dprintln(3,"local_vars names= ", locals_vars_sym)
    @dprintln(3,"reduction_vars names= ", reduction_vars_sym)

    parfor_rws = CompilerTools.ReadWriteSet.from_exprs(the_parfor.body, pir_rws_cb, state.LambdaVarInfo, state.LambdaVarInfo)

    # Convert Set to Array
    in_array_names   = LHSVar[]
    modified_symbols = LHSVar[]
    io_symbols       = LHSVar[]
    for i in in_vars
        assert(isa(i, LHSVar))
        # Determine for each input var whether the variable is just read, just written, or both.
        swritten = CompilerTools.ReadWriteSet.isWritten(i, parfor_rws)
        sread    = CompilerTools.ReadWriteSet.isRead(i, parfor_rws)
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
        out = setdiff(out, in_vars)
        if length(out) != 0
            throw(string("out variable of parfor task not supported right now."))
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

    # This is the LambdaVarInfo for the new task function.
    newLambdaVarInfo = CompilerTools.LambdaHandling.LambdaVarInfo()

    # For each LHSVar used in the incoming code, which is relative to the original function,
    # keep a mapping to its name (as a symbol), its type, and its slot number in the new function.
    oldToTaskMap = Dict{LHSVar,TaskFuncVariableInfo}()
    # Same as above but only maps the LHSVar in the original function to the slot number in the new task function.
    oldToNewMap  = Dict{LHSVar,Any}()

    # Add all the variables needs for the task function, both parameters and locals.
    # These functions update the oldToTaskMap.
    addToTaskFunc!(in_array_names, oldToTaskMap, state.LambdaVarInfo, newLambdaVarInfo)
    addToTaskFunc!(modified_symbols, oldToTaskMap, state.LambdaVarInfo, newLambdaVarInfo)
    addToTaskFunc!(io_symbols, oldToTaskMap, state.LambdaVarInfo, newLambdaVarInfo)
    addToTaskFunc!(reduction_vars, oldToTaskMap, state.LambdaVarInfo, newLambdaVarInfo)
    addToTaskFunc!(locals, oldToTaskMap, state.LambdaVarInfo, newLambdaVarInfo)

if false
    rn = 0
    for ian in in_array_names
        map_entry = oldToTaskMap[ian]
        tm = string(map_entry.name)
        @dprintln(3,"map_entry = ", map_entry, " type = ", typeof(map_entry), " tm = ", tm, " type = ", typeof(tm))
        if contains(tm, "#")
            rn += 1
            map_entry.name = Symbol("replacement_arg_" * string(rn))
            @dprintln(3,"replaced ", tm, " with ", map_entry.name)
        end
    end
end

    # Create the oldToNewMap as described above.
    for old in oldToTaskMap
        oldToNewMap[old[1]] = old[2].value
    end

    # If this parfor has any reductions.
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE && length(the_parfor.reductions) > 0
        # Create a tuple containing the types of the reduction variables.  This will go in the TaskInfo as ret_types.
        ret_tt = Expr(:tuple)
        ret_tt.args = map(x -> CompilerTools.LambdaHandling.getType(x.reductionVar, state.LambdaVarInfo), the_parfor.reductions)
        ret_types = eval(ret_tt)
        # Here is how reductions overall work.
        # Before calling a task function, the original function allocates a reduction array whose length is the number of threads.
        # This array is passed to the task function.
        # At the end of the loop, the task function writes its computed reduction value for just this thread into the reduction array
        # index equal to this thread's threadid.
        # After the task function completes, 

        # Allocate the variable for the reduction_array parameter.
        red_array_LHSVar = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(:reduction_array, Array{Any,1}, CompilerTools.LambdaHandling.ISASSIGNED, newLambdaVarInfo))
    else
        ret_types = Void
    end

    @dprintln(3,"in_array_names = ", in_array_names)
    @dprintln(3,"modified_symbols = ", modified_symbols)
    @dprintln(3,"io_symbols = ", io_symbols)
    @dprintln(3,"reduction_vars = ", reduction_vars)
    @dprintln(3,"oldToTaskMap = ", oldToTaskMap)
    @dprintln(3,"oldToNewMap = ", oldToNewMap)
    @dprintln(3,"newLambdaVarInfo = ", newLambdaVarInfo)
    @dprintln(3,"arg_types = ", arg_types)
    @dprintln(3,"ret_types = ", ret_types)

    range_rhsvar = CompilerTools.LambdaHandling.addLocalVariable(:ranges, ParallelAccelerator.ParallelIR.pir_range_actual, CompilerTools.LambdaHandling.ISASSIGNED, newLambdaVarInfo)
    range_lhsvar = toLHSVar(range_rhsvar)

    # Form an array including symbols for all the in and output parameters plus the additional iteration control parameter "ranges".
    # If we detect a GenSym in the parameter list we replace it with a symbol derived from the GenSym number and add it to gsmap.
    all_arg_names = Symbol[:ranges;
                     map(x -> oldToTaskMap[x].name, in_array_names);
                     map(x -> oldToTaskMap[x].name, modified_symbols);
                     map(x -> oldToTaskMap[x].name, io_symbols);
                     map(x -> oldToTaskMap[x].name, reduction_vars)]
    if ret_types != Void
        push!(all_arg_names, :reduction_array)
    end
    CompilerTools.LambdaHandling.setInputParameters(all_arg_names, newLambdaVarInfo)

    # Form a tuple that contains the type of each parameter.
    all_arg_types_tuple = Expr(:tuple)
    all_arg_types_tuple.args = [
        pir_range_actual;
        map(x -> oldToTaskMap[x].typ, in_array_names);
        map(x -> oldToTaskMap[x].typ, modified_symbols);
        map(x -> oldToTaskMap[x].typ, io_symbols);
        map(x -> oldToTaskMap[x].typ, reduction_vars)]
    if ret_types != Void
        push!(all_arg_types_tuple.args, Array{Any,1})
    end
    all_arg_type = eval(all_arg_types_tuple)

    @dprintln(3,"all_arg_names = ", all_arg_names)
    @dprintln(3,"all_arg_type  = ", all_arg_type)

    unique_node_id = get_unique_num()

    # The name of the new task function.
    task_func_name = string("task_func_",unique_node_id)
    task_func_sym  = Symbol(task_func_name)

    # Just stub out the new task function...the body and lambda will be replaced below.
    task_func = @eval function ($task_func_sym)($(all_arg_names...))
        throw(string("Some task function's body was not replaced."))
    end
    @dprintln(3,"task_func = ", task_func)

    # DON'T DELETE.  Forces function into existence.
    unused_ct = ParallelAccelerator.Driver.code_typed(task_func, all_arg_type)
    @dprintln(3, "unused_ct = ", unused_ct, " type = ", typeof(unused_ct))
    newLambdaVarInfo.orig_info = unused_ct

    self_sym = Symbol("#self#")
    CompilerTools.LambdaHandling.addLocalVariable(self_sym, typeof(task_func), CompilerTools.LambdaHandling.getDefaultDesc(), newLambdaVarInfo)

    # Creating the new body for the task function.
    task_body = TypedExpr(Int, :body)
    saved_loopNests = deepcopy(the_parfor.loopNests)

    # If the initial value of the reduction variable is not a bits type then it will be passed by reference and when the task function updates
    # it then this creates a race in multi-threaded mode.  So, here we deepcopy any incoming initial reduction variable back to itself
    # so that each thread gets its own copy.
    for dc_red_var in reduction_vars
        tf_red_var = oldToTaskMap[dc_red_var].value
        @dprintln(3, "Adding deepcopy of reduction var to task function prolog. reduction_var = ", dc_red_var, " task_func_reduction_var = ", tf_red_var)
        push!(task_body.args, mk_assignment_expr(deepcopy(tf_red_var), Expr(:call, :deepcopy, deepcopy(tf_red_var)), newLambdaVarInfo))
    end

    if DEBUG_TASK_FUNCTIONS
#      push!(task_body.args, Expr(:call, ParallelAccelerator.ParallelIR.lock_one))
      for i in all_arg_names
        push!(task_body.args, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), string(i), " = ", toLHSVar(i, newLambdaVarInfo)))
      end
    end

    # If this task has reductions, then we create a Julia buffer that holds a C function that we build up in the section below.
    # We then force this C code into the rest of the C code generated by CGen with a special call.
    reduction_func_name = string("")
    if length(the_parfor.reductions) > 0
        if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
            reduction_func_name = deepcopy(the_parfor.reductions)
        end
    end

    @dprintln(3, "Before LHSVar replacement in parfor")
    @dprintln(3, the_parfor)

    CompilerTools.LambdaHandling.replaceExprWithDict!(the_parfor, oldToNewMap, state.LambdaVarInfo, ParallelAccelerator.ParallelIR.AstWalk)

    @dprintln(3, "Before ParforBoxify")
    @dprintln(3, the_parfor)

#    ParforBoxify!(the_parfor, newLambdaVarInfo)

    @dprintln(3, "Before loopNest adjustment to range")
    @dprintln(3, the_parfor)

    if ParallelAccelerator.getTaskMode() != ParallelAccelerator.NO_TASK_MODE
        for i = 1:length(the_parfor.loopNests)
            # Put outerloop first in the loopNest
            j = length(the_parfor.loopNests) - i + 1
            the_parfor.loopNests[j].lower = DomainIR.add_expr(
            TypedExpr(Int64, :call, GlobalRef(Base, :unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, GlobalRef(Base, :getfield), deepcopy(range_rhsvar), QuoteNode(:lower_bounds)), i),
            1)
            the_parfor.loopNests[j].upper = DomainIR.add_expr(
            TypedExpr(Int64, :call, GlobalRef(Base, :unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, GlobalRef(Base, :getfield), deepcopy(range_rhsvar), QuoteNode(:upper_bounds)), i),
            1)
        end
    elseif ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
        for i = 1:length(the_parfor.loopNests)
            # Put outerloop first in the loopNest
            j = length(the_parfor.loopNests) - i + 1
            the_parfor.loopNests[j].lower = TypedExpr(Int64, :call, GlobalRef(Base, :unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, GlobalRef(Base, :getfield), deepcopy(range_rhsvar), QuoteNode(:lower_bounds)), i)
            the_parfor.loopNests[j].upper = TypedExpr(Int64, :call, GlobalRef(Base, :unsafe_arrayref), TypedExpr(Array{Int64,1}, :call, GlobalRef(Base, :getfield), deepcopy(range_rhsvar), QuoteNode(:upper_bounds)), i)
        end
    end

    @dprintln(3, "After loopNest adjustment")
    @dprintln(3, the_parfor)

    if DEBUG_TASK_FUNCTIONS
        push!(task_body.args, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "starting task function ", task_func_name))
    end

    # Add the parfor stmt to the task function body.
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
        #push!(task_body.args, Expr(:call, GlobalRef(Base,:println), GlobalRef(Base,:STDOUT), "in task func"))
        recreateLoops(task_body.args, the_parfor, state, newLambdaVarInfo)
    else
        flattenParfor(task_body.args, the_parfor, newLambdaVarInfo)
    end

    @dprintln(3, "After recreateLoops")
    @dprintln(3, task_body)
    @dprintln(3, newLambdaVarInfo)

    # Add the return statement to the end of the task function.
    # If this is not a reduction parfor then return "nothing".
    # If it is a reduction in threading mode, return a tuple of the reduction variables.
    # The threading infrastructure will then call a user-specified reduction function.
    if ret_types != Void
        ret_names = map(x -> toLHSVar(x.reductionVar), the_parfor.reductions)
        @dprintln(3, "ret_names = ", ret_names)

        if DEBUG_TASK_FUNCTIONS
            push!(task_body.args, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "before call to set_reduction_array"))
        end
        # Write the reduction tuple into reduction_array[threadid]
        push!(task_body.args, TypedExpr(Array{Any,1}, # type of arrayset expr
                                        :call,
                                        GlobalRef(Base, :arrayset),
                                        deepcopy(red_array_LHSVar),
                                        TypedExpr(Tuple{ret_types...}, :call, GlobalRef(Core, :tuple), ret_names...),
                                        TypedExpr(Int, :call, GlobalRef(Base.Threads,:threadid))))
        if DEBUG_TASK_FUNCTIONS
            push!(task_body.args, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "end of task func red_array = ", deepcopy(red_array_LHSVar)))
        end
    end

    if DEBUG_TASK_FUNCTIONS
        push!(task_body.args, TypedExpr(Any, :call, :println, GlobalRef(Base,:STDOUT), "ending task function ", task_func_name))
#        push!(task_body.args, Expr(:call, ParallelAccelerator.ParallelIR.unlock_one))
    end

    push!(task_body.args, TypedExpr(Void, :return, nothing))
    newLambdaVarInfo.return_type = Void

    # Remove Newvarnode in the task function for input arguments to the task.
    task_body = AstWalk(task_body, removeInputArgNewvarnode, newLambdaVarInfo)
    @dprintln(3,"Body after Newvarnode removed for input args = ", task_body)

    # Create the new :lambda Expr for the task function.
    newLambdaVarInfo, task_body = CompilerTools.OptFramework.cleanupFunction(newLambdaVarInfo, task_body)
    code = CompilerTools.LambdaHandling.LambdaVarInfoToLambda(newLambdaVarInfo, task_body)

    @dprintln(3, "New task = ", code)
#    CompilerTools.Helper.print_by_field(code)

    CompilerTools.OptFramework.setCode(task_func, all_arg_type, code)
    precompile(task_func, all_arg_type)

    if DEBUG_LVL >= 3
        task_func_ct = ParallelAccelerator.Driver.code_typed(task_func, all_arg_type)
        println("Task func code for ", task_func)
        println(task_func_ct, " type = ", typeof(task_func_ct))    
        println("code = ", code)
        debug_lvi = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(task_func_ct)
        #println(newLambdaVarInfo)   
        println("debug_lvi = ", code)
        println(debug_lvi)   
        println(CompilerTools.LambdaHandling.getBody(task_func_ct))   

        ParallelAccelerator.Driver.code_llvm(task_func, all_arg_type)
    end
#throw(string("stop here"))

    @dprintln(3,"End of parforToTask )")

    ret = TaskInfo(task_func,           # The task function that we just generated of type Function.
                   task_func_sym,       # The task function's Symbol name.
                   reduction_func_name, # The name of the C reduction function created for this task.
                   ret_types,
                   map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.LambdaVarInfo)), in_array_names),
                   map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.LambdaVarInfo)), modified_symbols),
                   map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.LambdaVarInfo)), io_symbols),
                   map(x -> EntityType(x, CompilerTools.LambdaHandling.getType(x, state.LambdaVarInfo)), reduction_vars),
                   code,          # The AST for the task function.
                   saved_loopNests)
    return ret
end

"""
Form a task out of a range of sequential statements.
This is not currently implemented.
"""
function seqTask(body_indices, bb_statements, body, state)
    getIO(body_indices, bb_statements)  
    throw(string("seqTask construction not implemented yet."))
    TaskInfo(:FIXFIXFIX, :FIXFIXFIX, Any[], Any[], nothing, PIRLoopNest[])
end
