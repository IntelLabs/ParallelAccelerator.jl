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

#
# A prototype Julia to C++ generator
# Originally created by Jaswanth Sreeram.
#

module CGen

import CompilerTools.DebugMsg
using CompilerTools.LambdaHandling
using CompilerTools.Helper
DebugMsg.init()

using ..ParallelAccelerator
using Core: IntrinsicFunction
import ..ParallelIR
import ..ParallelIR.DelayedFunc
import CompilerTools
export setvectorizationlevel, from_root, writec, compile, link, set_include_blas, set_include_lapack
import ParallelAccelerator, ..getPackageRoot
import ParallelAccelerator.H5SizeArr_t
import ParallelAccelerator.SizeArr_t

using Compat

# uncomment this line for using Debug.jl
#using Debug


type LambdaGlobalData
    #adp::ASTDispatcher
    ompprivatelist::Array{Any, 1}
    globalConstants :: Dict{Any, Any}   # non-scalar global constants
    globalUDTs::Dict{Any, Any}
    globalUDTsOrder::Array{Any, 1}
    symboltable::Dict{Any, Any}         # mapping from some kind of variable to a type
    tupleTable::Dict{Any, Array{Any,1}} # a table holding tuple values to be used for hvcat allocation
    compiledfunctions::Array{Any, 1}
    worklist::Array{Any, 1}
    jtypes::Dict{Type, AbstractString}
    ompdepth::Int64
    head_loop_set::Set{Int}
    back_loop_set::Set{Int}
    exit_loop_set::Set{Int}
    all_loop_exits::Set{Int}
    follow_set::Dict{Int,Int}
    cond_jump_targets::Set{Int}

    function LambdaGlobalData()
        _j = Dict(
            Int8    =>  "int8_t",
            UInt8   =>  "uint8_t",
            Int16   =>  "int16_t",
            UInt16  =>  "uint16_t",
            Int32   =>  "int32_t",
            UInt32  =>  "uint32_t",
            Int64   =>  "int64_t",
            UInt64  =>  "uint64_t",
            Float16 =>  "float",
            Float32 =>  "float",
            Float64 =>  "double",
            Bool    =>  "bool",
            Char    =>  "char",
            Void    =>  "void",
            H5SizeArr_t => "hsize_t*",
            SizeArr_t => "uint64_t*"
    )

        #new(ASTDispatcher(), [], Dict(), Dict(), [], [])
        new([], Dict(), Dict(), [], Dict(), Dict(), [], [], _j, 0, Set{Int}(), Set{Int}(), Set{Int}(), Set{Int}(), Dict{Int,Int}(), Set{Int}())
    end
end

# Auto-vec level. These determine what vectorization
# flags are emitted in the code or passed to icc. These
# levels are one of:

# 0: default autovec - icc
# 1: disable autovec - icc -no-vec
# 2: force autovec   - icc with #pragma simd at OpenMP loops

const VECDEFAULT = 0
const VECDISABLE = 1
const VECFORCE = 2
USE_OMP = 1
CGEN_RAW_ARRAY_MODE = false

function set_raw_array_mode(mode=true)
    global CGEN_RAW_ARRAY_MODE = mode
end

function enableOMP()
    global USE_OMP = 1
end

function disableOMP()
    global USE_OMP = 0
end

function isDistributedMode()
    mode = "0"
    if haskey(ENV,"CGEN_MPI_COMPILE")
        mode = ENV["CGEN_MPI_COMPILE"]
    end
    return mode=="1"
end

# Globals
inEntryPoint = false
lstate = nothing
USE_HDF5 = 0
NERSC = 0
USE_DAAL = 0

if haskey(ENV, "CGEN_NO_OMP") && ENV["CGEN_NO_OMP"]=="1"
    global USE_OMP = 0
else # use setting from config file
    global USE_OMP = ParallelAccelerator.openmp_supported
end

if isDistributedMode() #&& NERSC==0
    using MPI
    MPI.Init()
end

type ExternalPatternMatchCall
    func::Function
end

external_pattern_match_call = ExternalPatternMatchCall((ast::Array{Any,1},linfo)->"")
external_pattern_match_assignment = ExternalPatternMatchCall((lhs,rhs,linfo)->"")

type CgenUserOptions
    includeStatements
    compileFlags
    linkFlags
end

userOptions = CgenUserOptions[]

function addCgenUserOptions(value)
    push!(userOptions, value)
end

"""
Other packages can set external functions for pattern matching calls in CGen
"""
function setExternalPatternMatchCall(ext_pm::Function)
    external_pattern_match_call.func = ext_pm
end

function setExternalPatternMatchAssignment(ext_pm::Function)
    external_pattern_match_assignment.func = ext_pm
end


# Set what flags pertaining to vectorization to pass to the
# C++ compiler.
vectorizationlevel = VECDEFAULT
function setvectorizationlevel(x)
    global vectorizationlevel = x
end

include_blas = false
function set_include_blas(val::Bool=true)
    global include_blas = val
end

include_lapack = false
function set_include_lapack(val::Bool=true)
    global include_lapack = val
end

# include random number generator?
include_rand = false

insertAliasCheck = true
function set_alias_check(val)
    @dprintln(3, "set_alias_check =", val)
    global insertAliasCheck = val
end

# Reset and reuse the LambdaGlobalData object across function
# frames
function resetLambdaState(l::LambdaGlobalData)
    @dprintln(3, "resetLambdaState")
    empty!(l.ompprivatelist)
    empty!(l.globalConstants)
    empty!(l.globalUDTs)
    empty!(l.globalUDTsOrder)
    empty!(l.symboltable)
    empty!(l.worklist)
    inEntryPoint = false
    l.ompdepth = 0
    empty!(l.head_loop_set)
    empty!(l.back_loop_set)
    empty!(l.exit_loop_set)
    empty!(l.all_loop_exits)
    empty!(l.follow_set)
    empty!(l.cond_jump_targets)
end


# These are primitive operators on scalars and arrays
_operators = ["*", "/", "+", "-", "<", ">"]
# these are builtins that for primitive type only
_primitive_builtins = ["min", "max"]
# These are builtin "methods" for scalars and arrays
_builtins = ["getindex", "getindex!", "setindex", "setindex!", "arrayref", "top", "box",
            "unbox", "tuple", "arraysize", "arraylen", "ccall",
            "arrayset", "getfield", "unsafe_arrayref", "unsafe_arrayset",
            "safe_arrayref", "safe_arrayset", "tupleref",
            "call1", ":jl_alloc_array_1d", ":jl_alloc_array_2d", ":jl_alloc_array_3d", "nfields",
            "_unsafe_setindex!", ":jl_new_array", "unsafe_getindex", "steprange_last",
            ":jl_array_ptr", "sizeof", "pointer", "pointerref",
            # We also consider type casting here
            "Float32", "Float64",
            "Int8", "Int16", "Int32", "Int64",
            "UInt8", "UInt16", "UInt32", "UInt64",
            "convert", "unsafe_convert", "setfield!", "string"
]

# Intrinsics
_Intrinsics = [
        "===",
        "box", "unbox",
        "bitcast",
        #arithmetic
        "neg_int", "add_int", "sub_int", "mul_int", "sle_int", "ule_int",
        "xor_int", "and_int", "or_int", "ne_int", "eq_int",
        "sdiv_int", "udiv_int", "srem_int", "urem_int", "smod_int", "ctlz_int",
        "neg_float", "add_float", "sub_float", "mul_float", "div_float",
        "neg_float_fast", "add_float_fast", "sub_float_fast", "mul_float_fast", "div_float_fast",
        "rem_float", "sqrt_llvm", "sqrt_llvm_fast", "fma_float", "muladd_float",
        "le_float", "le_float_fast", "ne_float", "ne_float_fast", "eq_float", "eq_float_fast", "copysign_float",
        "fptoui", "fptosi", "uitofp", "sitofp", "not_int",
        "nan_dom_err", "lt_float", "lt_float_fast", "slt_int", "ult_int", "abs_float", "select_value",
        "fptrunc", "fpext", "trunc_llvm", "floor_llvm", "rint_llvm",
        "trunc", "ceil_llvm", "ceil", "pow", "powf", "lshr_int",
        "checked_ssub", "checked_ssub_int", "checked_sadd", "checked_sadd_int", "checked_srem_int",
        "checked_smul", "checked_sdiv_int", "checked_udiv_int", "checked_urem_int", "flipsign_int", "check_top_bit", "shl_int", "ctpop_int",
        "checked_trunc_uint", "checked_trunc_sint", "checked_fptosi", "powi_llvm", "llvm.powi.f64", "llvm.powf.f64",
        "ashr_int", "lshr_int", "shl_int",
        "cttz_int",
        "zext_int", "sext_int"
]

tokenXlate = Dict(
    '*' => "star",
    '/' => "slash",
    '-' => "minus",
    '!' => "bang",
    '.' => "dot",
    '^' => "hat",
    '|' => "bar",
    '&' => "amp",
    '=' => "eq",
    '\\' => "backslash"
)

replacedTokens = Set("#")
scrubbedTokens = Set(",.({}):")

#= if isDistributedMode() #&& NERSC==0
    package_root = getPackageRoot()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    generated_file_dir = "$package_root/deps/generated$rank"
    if !isdir(generated_file_dir)
        mkdir(generated_file_dir)
    end
else
=#
if NERSC==1
    generated_file_dir =  ENV["SCRATCH"]*"/generated_"*ENV["SLURM_JOBID"]
    if !isdir(generated_file_dir)
        if !isDistributedMode() || MPI.Comm_rank(MPI.COMM_WORLD)==0
            #println(generated_file_dir)
            mkdir(generated_file_dir)
        end
    end
elseif CompilerTools.DebugMsg.PROSPECT_DEV_MODE || isDistributedMode()
    package_root = getPackageRoot()
    generated_file_dir = "$package_root/deps/generated"
else
    generated_file_dir = mktempdir()
end

file_counter = -1

#### End of globals ####

function generate_new_file_name()
    global file_counter
    file_counter += 1
    return "cgen_output$file_counter"
end

function CGen_finalize()
    if !CompilerTools.DebugMsg.PROSPECT_DEV_MODE && !isDistributedMode()
        rm(generated_file_dir; recursive=true)
    end
    if isDistributedMode() #&& NERSC==0
        #MPI.Finalize()
    end
end

atexit(CGen_finalize)

function __init__()
    packageroot = joinpath(dirname(@__FILE__), "..")
end

function getenv(var::AbstractString)
  ENV[var]
end

include("cgen-pattern-match.jl")

# Emit declarations and "include" directives
function from_header(isEntryPoint::Bool, linfo)
    s = from_UDTs(linfo)
    isEntryPoint ? from_includes() * from_globals(linfo) * s : s
end

function from_includes()
    packageroot = getPackageRoot()
    blas_include = ""
    if include_blas == true
        libblas = Base.libblas_name
        if mkl_lib!=""
            blas_include = "#include <mkl.h>\n"
        elseif openblas_lib!="" || sys_blas==1
            blas_include = "#include <cblas.h>\n"
        else
            blas_include = "#include \"$packageroot/deps/include/cgen_linalg.h\"\n"
        end
    end
    if include_lapack == true
        if mkl_lib!=""
            blas_include = "#include <mkl.h>\n"
        end
    end
    s = ""
    if include_rand==true
        s *= "#include <random>\n"
    end
    if isDistributedMode()
        s *= "#include <mpi.h>\n"
    end
    if USE_HDF5==1
        s *= "#include \"hdf5.h\"\n"
    end
    if USE_OMP==1 || USE_DAAL==1
        s *= "#include <omp.h>\n"
    end
    if USE_DAAL==1
        s *= """
        #include "daal.h"

        using namespace std;
        using namespace daal;
        using namespace daal::algorithms;
        using namespace daal::data_management;
        """
    end
    s *= reduce(*, "", (
    blas_include,
    "#include <stdint.h>\n",
    "#include <float.h>\n",
    "#include <limits.h>\n",
    "#include <complex>\n",
    "#include <math.h>\n",
    "#include <stdio.h>\n",
    "#include <iostream>\n",
    "#include \"$packageroot/deps/include/j2c-array.h\"\n",
    "#include \"$packageroot/deps/include/pse-types.h\"\n",
    "#include \"$packageroot/deps/include/cgen_intrinsics.h\"\n",
    "#include <sstream>\n",
    "#include <vector>\n",
    "#include <string>\n")
    )
    for userOption in userOptions
        s *= userOption.includeStatements
    end

    s *= "unsigned main_count = 0;\n"

    if VERSION >= v"0.6.0-pre"
        s *= "
              template <typename R, typename T, typename U>
              R checked_sadd_int(T x, U y) {
                  R ret;
                  ret.f0 = x + y;
                  if ((ret.f0 > x) != (y > 0)) ret.f1 = true;
                  else ret.f1 = false;
                  return ret;
              }\n 
              template <typename R, typename T, typename U>
              R checked_ssub_int(T x, U y) {
                  R ret;
                  ret.f0 = x - y;
                  if ((ret.f0 < x) != (y > 0)) ret.f1 = true;
                  else ret.f1 = false;
                  return ret;
              }\n
              template <typename R, typename T, typename U>
              R checked_smul_int(T x, U y) {
                  R ret;
                  ret.f0 = x * y;
                  if ((x != 0) && (ret.f0 / x != b)) ret.f1 = true;
                  else ret.f1 = false;
                  return ret;
              }\n
              template <typename R, typename T, typename U>
              R checked_uadd_int(T x, U y) {
                  R ret;
                  ret.f0 = x + y;
                  if ((ret.f0 < x) || (ret.f0 < y)) ret.f1 = true;
                  else ret.f1 = false;
                  return ret;
              }\n
              template <typename R, typename T, typename U>
              R checked_usub_int(T x, U y) {
                  R ret;
                  ret.f0 = x - y;
                  if (y > x) ret.f1 = true;
                  else ret.f1 = false;
                  return ret;
              }\n
              template <typename R, typename T, typename U>
              R checked_umul_int(T x, U y) {
                  R ret;
                  ret.f0 = x * y;
                  if ((x != 0) && (ret.f0 / x != b)) ret.f1 = true;
                  else ret.f1 = false;
                  return ret;
              }\n
              "
    end
             
    return s
end

function from_globals(linfo)
    global lstate
    s = ""
    for (name, value) in lstate.globalConstants
        if isa(value, Array)
            eltyp = toCtype(eltype(value))
            len   = length(value)
            dims  = ndims(value)
            shape = mapfoldl(x -> string(x), (a,b) -> a * "," * b, size(value))
            @dprintln(3,"Array alloc shape = ", shape)
            data_name = "_" * name * "_"
            value_str = "{" * mapfoldl(x -> from_expr(x, linfo), (a,b) -> a * "," * b, value) * "}"
            s *= "static $(eltyp) $(data_name)[$(len)] = " * value_str * ";\n"
            s *= "static j2c_array<$eltyp> $(name) = j2c_array<$eltyp>::new_j2c_array_$(dims)d($(data_name), $shape);\n"
        end
    end
    return s
end

# Iterate over all the user defined types (UDTs) in a function
# and emit a C++ type declaration for each
function from_UDTs(linfo)
    global lstate
    @dprintln(3,"from_UDTs globalUDTs = ", lstate.globalUDTs)
    @dprintln(3,"from_UDTs globalUDTsOrder = ", lstate.globalUDTsOrder)
    s = ""
    for udt in lstate.globalUDTsOrder
        @dprintln(3, "udt = ", udt)
        if lstate.globalUDTs[udt] == 1
            @dprintln(3, "udt not processed yet so doing it now.")
            s *= from_decl(udt, linfo)
        end
    end
#    isempty(lstate.globalUDTs) ? "" : mapfoldl((a) -> (lstate.globalUDTs[a] == 1 ? from_decl(a, linfo) : ""), *, keys(lstate.globalUDTs))
    return s
end

# Tuples are represented as structs
function from_decl(k::Tuple, linfo)
    s = "typedef struct {\n"
    for i in 1:length(k)
        s *= toCtype(k[i]) * " " * "f" * string(i-1) * ";\n"
    end
    s *= "} Tuple" *
        (!isempty(k) ? mapfoldl(canonicalize, (a, b) -> "$(a)$(b)", k) : "") * ";\n"
    if haskey(lstate.globalUDTs, k)
        lstate.globalUDTs[k] = 0
    end
    s
end

# Generic declaration emitter for non-primitive Julia DataTypes
function from_decl(k::DataType, linfo)
    if (k === UnitRange{Int64})
        if haskey(lstate.globalUDTs, k)
            lstate.globalUDTs[k] = 0
        end
        btyp, ptyp = parseParametricType(k)
        s = "typedef struct {\n\t"
        s *= toCtype(ptyp[1]) * " start;\n"
        s *= toCtype(ptyp[1]) * " stop;\n"
        s *= "} " * canonicalize(k) * ";\n"
        return s
    elseif issubtype(k, StepRange)
        if haskey(lstate.globalUDTs, k)
            lstate.globalUDTs[k] = 0
        end
        btyp, ptyp = parseParametricType(k)
        s = "typedef struct {\n\t"
        s *= toCtype(ptyp[1]) * " start;\n"
        s *= toCtype(ptyp[1]) * " step;\n"
        s *= toCtype(ptyp[2]) * " stop;\n"
        s *= "} " * canonicalize(k) * ";\n"
        return s
    elseif k.name == Tuple.name
        if haskey(lstate.globalUDTs, k)
            lstate.globalUDTs[k] = 0
        end
        btyp, ptyp = parseParametricType(k)
        s = "typedef struct {\n"
        for i in 1:length(ptyp)
            s *= toCtype(ptyp[i]) * " " * "f" * string(i-1) * ";\n"
        end
        s *= "} Tuple" * (!isempty(ptyp) ? mapfoldl(canonicalize, (a, b) -> "$(a)$(b)", ptyp) : "") * ";\n"
        return s
    elseif issubtype(k, AbstractString)
        # Strings are handled by a speciall class in j2c-array.h.
        return ""
    elseif issubtype(k, Base.LibuvStream)
        # Stream type support is limited to STDOUT now
        return "typedef FILE *" * canonicalize(k.name) * ";\n"
    else
        if haskey(lstate.globalUDTs, k)
            lstate.globalUDTs[k] = 0
        end
        ftyps = k.types
        fnames = fieldnames(k)
        dprintln(3, "k = ", k, " ftyps = ", ftyps, " fnames = ", fnames)
        s = "typedef struct {\n"
        for i in 1:length(ftyps)
            s *= toCtype(ftyps[i]) * " " * canonicalize(fnames[i]) * ";\n"
        end
        s *= "} " * canonicalize(k) * ";\n"
        return s
    end
    throw(string("Could not translate Julia Type: " * string(k)))
    return ""
end

# no type declaration for Union{}
function from_decl(k::Type{Union{}}, linfo)
    return ""
end

function from_decl(k, linfo)
    return toCtype(lookupSymbolType(k, linfo)) * " " * canonicalize(k) * ";\n"
end

function isCompositeType(t::Type)
    # TODO: Expand this to real UDTs
    b = (t<:Tuple) || (t === UnitRange{Int64}) || (t === StepRange{Int64, Int64})
    b
end

function from_lambda(ast)
    linfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
    from_lambda(linfo, body)
end

type Conditional
    head :: Int
    follow :: Int
    hasElse :: Bool
end

function hasElse(head, follow, bbs)
    succs = [x.label for x in bbs[head].succs]
    !in(follow, succs)
end

function addConditional(conditionals, head, follow, bbs, follow_set, cond_jump_targets)
    with_else = hasElse(head, follow, bbs)
    push!(conditionals, Conditional(head, follow, with_else))
    if !haskey(follow_set, follow)
        follow_set[follow] = 0
    end
    follow_set[follow] = follow_set[follow] + 1
    union!(cond_jump_targets, Set{Int}([x.label for x in bbs[head].succs]))
end

function getLoopInfo(body)
    empty!(lstate.head_loop_set)
    empty!(lstate.back_loop_set)
    empty!(lstate.exit_loop_set)
    empty!(lstate.all_loop_exits)
    empty!(lstate.follow_set)
    empty!(lstate.cond_jump_targets)

    if !recreateLoops
        return
    end

    @dprintln(3, "getLoopInfo body = ", body)
    cfg = CompilerTools.CFGs.from_lambda(body)

#    intervals = CompilerTools.CFGs.computeIntervals(cfg)
#    @dprintln(3, "intervals = ", intervals)

    blockCategories = [Set{Int}() for x = 1:3]
    for bbentry in cfg.basic_blocks
        push!(blockCategories[CompilerTools.CFGs.classifyBlock(bbentry[2])], bbentry[1])
    end
    @dprintln(3, "blockCategories = ", blockCategories)
    unaccounted = deepcopy(blockCategories[CompilerTools.CFGs.BLOCK_GOTOIFNOT])

    inv_dom = CompilerTools.CFGs.compute_inverse_dominators(cfg)
    @dprintln(3, "inverse dominators = ", inv_dom)

    loop_info = CompilerTools.Loops.compute_dom_loops(cfg)
    im_doms = CompilerTools.CFGs.compute_immediate_dominators(loop_info.dom_dict)
    @dprintln(3, "immediate_dominators = ", im_doms)
    @dprintln(3, "dominators = ", loop_info.dom_dict)

    for li in loop_info.loops
        @dprintln(3, "Loop Info = ", li)
        head_bb = cfg.basic_blocks[li.head]
        back_bb = cfg.basic_blocks[li.back_edge]
        union!(lstate.all_loop_exits, li.exits)
        unaccounted = setdiff(unaccounted, li.blocks_that_exit)

        if length(li.exits) > 1
            @dprintln(3, "Couldn't recreate loop since loop has more than one exit block.")
            continue
        end

        exit_bb = cfg.basic_blocks[first(li.exits)]

        # If the exit node has more than one predecessor then we can't eliminate the label.
        if length(exit_bb.preds) > 1
            @dprintln(3, "Couldn't recreate loop since exit block has more than one predecessor.")
            continue
        end
        # If the head node has a non-back edge predecessor that doesn't fallthrough to head then we can't eliminate the label.
        okay = true
        for hp in head_bb.preds
            if hp == back_bb
                continue
            end
            if hp.fallthrough_succ == nothing
                okay = false
                break
            end
        end
        if !okay
            @dprintln(3, "Couldn't recreate loop since head non-back edge predecessor doesn't fallthrough to head.")
            continue
        end

        @dprintln(3, "Adding loop to set to recreate.")
        push!(lstate.head_loop_set, li.head)
        push!(lstate.back_loop_set, li.back_edge)
        push!(lstate.exit_loop_set, exit_bb.label)
    end
    @dprintln(3, "Unaccounted gotoifnot blocks = ", unaccounted)

    unresolved = Set{Int}()
    conditionals = Conditional[]
    follow_set = Dict{Int,Int}()
    cond_jump_targets = Set{Int}()

    # For blocks in reverse order.
    for i = length(cfg.depth_first_numbering):-1:1
        to_check = cfg.depth_first_numbering[i]
        @dprintln(3, "2-way conditional processing ", to_check, " unaccounted = ", unaccounted)
        # If this block is a gotoifnot node that isn't part of a loop then it is a conditional.
        if in(to_check, unaccounted)
            @dprintln(3, "2-way conditional in unaccounted")
            found = false
            best_follow = CompilerTools.CFGs.CFG_EXIT_BLOCK
            # Scan the blocks in reverse order.
            # If we find a block that inverse dominates the unaccounted (to_check) block then we
            # have a potential follow block.  There may be other lesser blocks that also inverse
            # dominate so we keep going until the block we are working with is the same as to_check.
            for j = length(cfg.depth_first_numbering):-1:1
                potential_follow = cfg.depth_first_numbering[j]
                @dprintln(3, "2-way conditional checking potential follow ", potential_follow)
                if potential_follow == to_check
                    @dprintln(5,"Found follow == to_check.")
                    break
                end
                if in(potential_follow, inv_dom[to_check])
                    @dprintln(3,"Found new best_follow.")
                    best_follow = potential_follow
                end
            end
            # If we found the closest block that inverse dominates to_check then add this combination as a conditional.
            # to_check is the conditional head and best_follow is the block that follows the conditional.
            if best_follow != CompilerTools.CFGs.CFG_EXIT_BLOCK
                @dprintln(3, "2-way conditional found follow.  head = ", to_check, " follow = ", best_follow)
                addConditional(conditionals, to_check, best_follow, cfg.basic_blocks, follow_set, cond_jump_targets)
                found = true
if false
                for one_unresolved in unresolved
                    if one_unresolved != best_follow 
                        if in(to_check, loop_info.dom_dict[one_unresolved])
                            @dprintln(3, "2-way conditional processing unresolved.  head = ", one_unresolved, " follow = ", best_follow)
                            addConditional(conditionals, one_unresolved, best_follow, cfg.basic_blocks, follow_set, cond_jump_targets)
                        else
                            @dprintln(3, "2-way conditional skipping unresolved since to_check doesn't dominate unresolved.")
                        end
                    else
                        @dprintln(3, "2-way conditional skipping unresolved since head and follow are equal.")
                    end
                end
end
                unresolved = Set{Int}()
            end
            if !found
                @dprintln(3, "2-way conditional couldn't find follow....adding unresolved ", to_check)
                push!(unresolved, to_check)
            end
        end
    end

    @dprintln(3, "conditionals = ", conditionals)
    @dprintln(3, "follow_set = ", follow_set)
    @dprintln(3, "cond_jump_targets = ", cond_jump_targets)

    lstate.follow_set = follow_set
    lstate.cond_jump_targets = cond_jump_targets

    @dprintln(3, "head_loop_set = ", lstate.head_loop_set)
    @dprintln(3, "back_loop_set = ", lstate.back_loop_set)
    @dprintln(3, "exit_loop_set = ", lstate.exit_loop_set)
    @dprintln(3, "all_loop_exits = ", lstate.all_loop_exits)
    @dprintln(3, "follow_set = ", lstate.follow_set)
    @dprintln(3, "cond_jump_targets = ", lstate.cond_jump_targets)
end

type CGen_boolean_and
    lhs
    rhs
end

function mergeGotoIfNot(body :: Expr)
    assert(body.head == :body)
    new_body = []

    @dprintln(3, "Before mergeGotoIfNot. ", body)
    i = 1
    while i <= length(body.args) 
        if isa(body.args[i], Expr) && body.args[i].head == :gotoifnot
            labelId = body.args[i].args[2]
            while ((i+1) <= length(body.args)) && 
                  isa(body.args[i+1], Expr) && 
                  body.args[i+1].head == :gotoifnot && 
                  body.args[i+1].args[2] == labelId
                body.args[i+1].args[1] = CGen_boolean_and(body.args[i].args[1], body.args[i+1].args[1])
                i += 1
            end
        end
        push!(new_body, body.args[i])
        i += 1
    end

    body.args = new_body
    @dprintln(3, "After mergeGotoIfNot. ", body)

    return body
end

function from_lambda(linfo :: LambdaVarInfo, body)
    params = Symbol[ CompilerTools.LambdaHandling.lookupVariableName(x, linfo)
                     for x in CompilerTools.LambdaHandling.getInputParameters(linfo)]
    vars = CompilerTools.LambdaHandling.getLocalVariablesNoParam(linfo)

    decls = ""
    global lstate
    # Populate the symbol table
    for k in vcat(params, vars)
        t = CompilerTools.LambdaHandling.getType(k, linfo) # v is a VarDef
        setSymbolType(k, t, linfo)
        @assert t!=Any "CGen: variable " * string(k) * " cannot have Any (unresolved) type"
        if !in(k, params) && (CompilerTools.LambdaHandling.getDesc(k, linfo) & 32 != 0)
            push!(lstate.ompprivatelist, k)
        end
        # If we have user defined types, record them
        #if isCompositeType(lstate.symboltable[k]) || isUDT(lstate.symboltable[k])
        if !isPrimitiveJuliaType(t) && !isArrayOfPrimitiveJuliaType(t)
            @dprintln(3, "from_lambda adding to globalUDTs ", t)
            lstate.globalUDTs[t] = 1
            push!(lstate.globalUDTsOrder, t)
        end
    end

    if recreateConds
        body = mergeGotoIfNot(body)
    end

    getLoopInfo(body)

    bod = from_expr(body, linfo)
    @dprintln(3,"lambda params = ", params)
    @dprintln(3,"lambda vars = ", vars)
    dumpSymbolTable(lstate.symboltable)

    for k in vars
        s = lookupVariableName(k, linfo)
        @dprintln(3, "from_lambda creating decl for variable ", k, " with name ", s)
        decls *= toCtype(lookupSymbolType(k, linfo)) * " " * canonicalize(s) * ";\n"
    end
    decls * bod
end


function from_exprs(args::Array, linfo)
    s = ""
    for a in args
        @dprintln(3, "from_exprs working on = ", a)
        se = from_expr(a, linfo)
        if se != "nothing" # skip nothing statement
          s *= se * (!isempty(se) ? ";\n" : "")
        end
    end
    s
end


function dumpSymbolTable(a::Dict{Any, Any})
    @dprintln(3,"SymbolTable: ")
    for x in keys(a)
        @dprintln(3,x, " ==> ", a[x])
    end
end

function has(a, b)
    return findfirst(a, b) != 0
end

function hasfield(a, f)
    return has(fieldnames(a), f)
end

function typeAvailable(a)
    return hasfield(a, :typ)
end

function checkGlobalRefName(arg::GlobalRef, name::Symbol)
    return arg.name==name
end

function checkGlobalRefName(arg::ANY, name::Symbol)
    return false
end

lookupType(a::RHSVar, linfo) = getType(a, linfo)
lookupType(a::GlobalRef, linfo) = typeof(getfield(a.mod, a.name))
lookupType(a, linfo) = typeAvailable(a) ? a.typ : typeof(a)

function from_assignment_fix_tuple(lhs, rhs::Expr, linfo)
  # if assignment is var = (...)::tuple, add var to tupleTable to be used for hvcat allocation
  if rhs.head==:call && isBaseFunc(rhs.args[1],:tuple)
    @dprintln(3,"Found tuple assignment: ", lhs," ", rhs)
    lstate.tupleTable[lhs] = rhs.args[2:end]
  end
end

function from_assignment_fix_tuple(lhs, rhs::ANY, linfo)
end

function from_assignment(args::Array{Any,1}, linfo)
    global lstate
    lhs = args[1]
    rhs = args[2]

    from_assignment_fix_tuple(lhs, rhs, linfo)

    @dprintln(3,"from_assignment: ", lhs, " ", rhs)
    @dprintln(3,external_pattern_match_call)
    @dprintln(3,external_pattern_match_assignment)
    external_match = external_pattern_match_assignment.func(lhs, rhs, linfo)
    if external_match!=""
        @dprintln(3,"external pattern match returned something")
        return external_match
    end

    @dprintln(3,"external pattern match returned nothing")

    # 0.4 legacy code not needed anymore
    # hack to convert triangular matrix output of cholesky
    #chol_match = pattern_match_assignment_chol(lhs, rhs, linfo)
    #if chol_match!=""
    #    return chol_match
    #end

    match_hvcat = from_assignment_match_hvcat(lhs, rhs, linfo)
    if match_hvcat!=""
        return match_hvcat
    end

    match_cat_t = from_assignment_match_cat_t(lhs, rhs, linfo)
    if match_cat_t!=""
        return match_cat_t
    end

    match_hcat = from_assignment_match_hcat(lhs, rhs, linfo)
    if match_hcat!=""
        return match_hcat
    end

    match_vcat = from_assignment_match_vcat(lhs, rhs, linfo)
    if match_vcat!=""
        return match_vcat
    end

    match_iostream = from_assignment_match_iostream(lhs, rhs, linfo)
    if match_iostream!=""
        return match_iostream
    end

    match_transpose = pattern_match_assignment_transpose(lhs, rhs, linfo)
    if match_transpose!=""
        return match_transpose
    end

    lhsO = from_expr(lhs, linfo)
    rhsO = from_expr(rhs, linfo)
    if lhsO == rhsO # skip x = x due to issue with j2c_array
        return ""
    end

    if !typeAvailable(lhs) && !inSymbolTable(lhs, linfo)
        if typeAvailable(rhs)
            setSymbolType(lhs, rhs.typ, linfo)
        elseif inSymbolTable(rhs, linfo)
            setSymbolType(lhs, lookupSymbolType(rhs, linfo), linfo)
        elseif isPrimitiveJuliaType(typeof(rhs))
            setSymbolType(lhs, typeof(rhs), linfo)
        elseif isPrimitiveJuliaType(typeof(rhsO))
            setSymbolType(lhs, typeof(rhs0), linfo)
        else
            @dprintln(3,"Unknown type in assignment: ", args)
            throw("FATAL error....exiting")
        end
        # Try to refine type for certain stmts with a call in the rhs
        # that doesn't have a type
        if typeAvailable(rhs) && (rhs.typ === Any) &&
        hasfield(rhs, :head) && ((rhs.head === :call) || (rhs.head === :call1))
            m, f, t = resolveCallTarget(rhs.args,linfo, rhs.typ)
            f = string(f)
            if f == "fpext"
                @dprintln(3,"Args: ", rhs.args, " type = ", typeof(rhs.args[2]))
                setSymbolType(lhs, eval(rhs.args[2]), linfo)
                @dprintln(3,"Emitting :", rhs.args[2])
                @dprintln(3,"Set type to : ", lookupSymbolType(lhs, linfo))
            end
        end
    end
    lhsO * " = " * rhsO
end

function parseArrayType(arrayType::Type)
    return eltype(arrayType), ndims(arrayType)
end

function isPrimitiveJuliaType(t::Type)
    haskey(lstate.jtypes, t)
end

function isPrimitiveJuliaType(t::ANY)
    return false
end

function isArrayOfPrimitiveJuliaType(t::Type)
  if isArrayType(t) && isPrimitiveJuliaType(eltype(t))
    return true
  end
  return false
end

function isArrayOfPrimitiveJuliaType(t::ANY)
    return false
end


function toCtype(typ::Type{Union{}})
    return "void*"
end

function toCtype(typ::Tuple)
    return "Tuple" * mapfoldl(canonicalize, (a, b) -> "$(a)$(b)", typ)
end

function toCtype(typ::GlobalRef)
    if typ.mod == Base
        return toCtype(eval(typ))
    else
        error("Not implemented")
    end
end

# Generate a C++ type name for a Julia type
function toCtype(typ::DataType)
    if haskey(lstate.jtypes, typ)
        return lstate.jtypes[typ]
    elseif isArrayType(typ)
        atyp, dims = parseArrayType(typ)
        atyp = toCtype(atyp)
        assert(dims >= 0)
        if CGEN_RAW_ARRAY_MODE
            return "$(atyp) * "
        else
            return " j2c_array< $(atyp) > "
        end
    elseif isPtrType(typ)
        return "$(toCtype(eltype(typ))) *"
    elseif typ == Complex64
        return "std::complex<float>"
    elseif typ == Complex128
        return "std::complex<double>"
    elseif length(typ.parameters) != 0
        # For parameteric types, for now assume we have equivalent C++
        # implementations
        btyp, ptyps = parseParametricType(typ)
        return canonicalize(btyp) * mapfoldl(canonicalize, (a, b) -> a * b, ptyps)
    elseif typ <: AbstractString
        return "ASCIIString"
    elseif typ == Any
        return "void*"
    else
        return canonicalize(typ)
    end
end

function toCtype(utyp::Union)
    typ = Void
    for t in utyp.types
        if t != Void
            if typ == Void
                typ = t
            else
                error("CGen does not handle Union type: ", utyp)
            end
        end
    end
    dprintln(3, "Type ", utyp, " is being treated as ", typ)
    toCtype(typ)
end

function canonicalize_re(tok)
    s = string(tok)
    name = ""
    for c in 1:length(s)
        if isalpha(s[c]) || isdigit(s[c]) || c == '_'
            name *= string(s[c])
        elseif haskey(tokenXlate, s[c])
            name *= "_" * tokenXlate[s[c]]
        else
            name *= string("a") * string(Int(c))
        end
    end
    return name
end

function canonicalize(tok)
    global replacedTokens
    global scrubbedTokens
    s = string(tok)
    s = replace(s, scrubbedTokens, "")
    s = replace(s, replacedTokens, "p")
    s = replace(s, "∇", "del")
    for (k,v) in tokenXlate
       s = replace(s, k, v)
    end
    s = replace(s, r"[^a-zA-Z0-9]", "_")
    s
end

function parseParametricType(typ::DataType)
    return typ.name, typ.parameters
end

function parseParametricType_s(typ)
    assert(isa(typ, DataType))
    m = split(string(typ), "{"; keep=false)
    assert(length(m) >= 1)
    baseTyp = m[1]
    if length(m) == 1
        return baseTyp, ""
    end
    pTyps = split(m[2], ","; keep=false)
    if endswith(last(pTyps), "}")
        pTyps[length(pTyps)] = chop(last(pTyps))
    end
    return baseTyp, pTyps
end

function from_tupleref(args, linfo)
    # We could generate std::tuples instead of structs
    from_expr(args[1], linfo) * ".f" * string(parse(Int, (from_expr(args[2], linfo)))-1)
end

function from_safegetindex(args,linfo)
    s = ""
    src = from_expr(args[1], linfo)
    s *= src * ".SAFEARRAYELEM("
    idxs = map(x->from_expr(x,linfo), args[2:end])
    for i in 1:length(idxs)
        s *= idxs[i] * (i < length(idxs) ? "," : "")
    end
    s *= ")"
    s
end

function from_getslice(args, linfo)
    s = ""
    src = from_expr(args[1], linfo)
    s *= src * ".slice("
    idxs = Any[]
    i = 0
    for a in args[2:end]
        i = i + 1
        if (isa(a, GlobalRef) && a.name == :(:)) || isa(a, Colon)
        else
            if isCall(a) && isBaseFunc(getCallFunction(a), :UnitRange)
              args = getCallArguments(a)
              @assert (args[1] == args[2]) "Expect UnitRange to have identical start and end, but got " * string(a)
              a = args[1]
            end
            push!(idxs, string(i))
            push!(idxs, from_expr(a, linfo))
        end
    end
    for i in 1:length(idxs)
        s *= idxs[i] * (i < length(idxs) ? "," : "")
    end
    s *= ")"
    s
end

function from_getindex(args, linfo)
    # if args has any range indexing, it is slicing
    if any([(isa(a, GlobalRef) && a.name == :(:)) || isa(a, Colon) for a in args])
       return from_getslice(args, linfo)
    end
    s = ""
    src = from_expr(args[1], linfo)
    if CGEN_RAW_ARRAY_MODE
        s *= src * "["
    else
        s *= src * ".ARRAYELEM("
    end
    idxs = map(x->from_expr(x,linfo), args[2:end])
    for i in 1:length(idxs)
        s *= idxs[i] * (i < length(idxs) ? "," : "")
    end
    if CGEN_RAW_ARRAY_MODE
        s *= " - 1]"
    else
        s *= ")"
    end
    s
end

function from_setindex(args, linfo)
    s = ""
    src = from_expr(args[1], linfo)
    if CGEN_RAW_ARRAY_MODE
        s *= src * "["
    else
        s *= src * ".ARRAYELEM("
    end
    idxs = map(x->from_expr(x,linfo), args[3:end])
    for i in 1:length(idxs)
        s *= idxs[i] * (i < length(idxs) ? "," : "")
    end
    if CGEN_RAW_ARRAY_MODE
        s *= " - 1] = "
    else
        s *= ") = "
    end
    s *= from_expr(args[2], linfo)
    s
end



# unsafe_setindex! has several variations. Here we handle only one.
# For this (and other related) array indexing methods we could just do
# codegen but we would have to extend the j2c_array runtime to support
# all the other operations allowed on Julia arrays

function from_unsafe_setindex!(args, linfo)
    @assert (length(args) == 4) "Array operation unsafe_setindex! has unexpected number of arguments"
    @assert !isArrayType(args[2]) "Array operation unsafe_setindex! has unexpected format"
    src = from_expr(args[2], linfo)
    v = from_expr(args[3], linfo)
    mask = from_expr(args[4], linfo)
    "for(uint64_t i = 1; i < $(src).ARRAYLEN(); i++) {\n\t if($(mask)[i]) \n\t $(src)[i] = $(v);}\n"
end

function from_tuple(args, linfo)
    "{" * mapfoldl(x->from_expr(x, linfo), (a, b) -> "$a, $b", args) * "}"
end

function from_arraysize(args, linfo)
    s = from_expr(args[1], linfo)
    if length(args) == 1
        s *= ".ARRAYLEN()"
    else
        s *= ".ARRAYSIZE(" * from_expr(args[2],linfo) * ")"
    end
    s
end

function from_arraysize(arr, dim::Int, linfo)
    s = from_expr(arr, linfo)
    s *= ".ARRAYSIZE(" * from_expr(dim, linfo) * ")"
    s
end


function from_foreigncall(args, linfo, call_ret_typ)
    @dprintln(3,"foreigncall args:")
    @dprintln(3,"target tuple: ", args[1], " - ", typeof(args[1]))
    @dprintln(3,"return type: ", args[2])
    @dprintln(3,"input types tuple: ", args[3])
    long_inputs = args[4:end]
    @dprintln(3,"inputs tuple: ", long_inputs)

#    short_inputs = []
#    for i = 1:length(long_inputs)
#        if i%2 == 1
#            push!(short_inputs, long_inputs[i])
#        end
#    end
#    @dprintln(3,"short_inputs: ", short_inputs)

    for i in 1:length(args)
        @dprintln(3,"arg ", i, " = ", args[i])
    end
    @dprintln(3,"End of foreigncall args")
    fun = args[1]
    if isInlineable(fun, args, linfo)
        @dprintln(3,"isInlineable")
        return from_inlineable(fun, args, linfo, call_ret_typ)
    end

    if isa(fun, QuoteNode)
        s = from_symbol(fun, linfo)
#    elseif isa(fun, Expr) && ((fun.head === :call1) || (fun.head === :call))
#        s = canonicalize(string(fun.args[2]))
#        @dprintln(3,"ccall target: ", s)
    elseif isa(fun, Tuple) && length(fun) == 2
        s = canonicalize(string(fun[1]))
        @dprintln(3,"ccall target: ", s)
    else
        throw("Invalid ccall format...")
    end
    s *= "("
#    numInputs = length(args[3].args)-1
    argsStart = 4
    argsEnd = length(args)
    if contains(s, "cblas") && contains(s, "gemm")
        if mkl_lib!=""
            s *= "(CBLAS_LAYOUT) $(from_expr(args[4], linfo)), "
        else
            s *= "(CBLAS_ORDER) $(from_expr(args[4], linfo)), "
        end
        s *= "(CBLAS_TRANSPOSE) $(from_expr(args[6], linfo)), "
        s *= "(CBLAS_TRANSPOSE) $(from_expr(args[8], linfo)), "
        argsStart = 10
    end
    to_fold = args[argsStart:2:end]
    if length(to_fold) > 0
        s *= mapfoldl(x->from_expr(x,linfo), (a, b)-> "$a, $b", to_fold)
    end
    s *= ")"
    @dprintln(3,"from_foreignccall: ", s)
    s
end

function from_ccall(args, linfo, call_ret_typ)
    @dprintln(3,"ccall args:")
    @dprintln(3,"target tuple: ", args[1], " - ", typeof(args[1]))
    @dprintln(3,"return type: ", args[2])
    @dprintln(3,"input types tuple: ", args[3])
    @dprintln(3,"inputs tuple: ", args[4:end])
    for i in 1:length(args)
        @dprintln(3,args[i])
    end
    @dprintln(3,"End of ccall args")
    fun = args[1]
    if isInlineable(fun, args, linfo)
        return from_inlineable(fun, args, linfo, call_ret_typ)
    end

    if isa(fun, QuoteNode)
        s = from_symbol(fun, linfo)
    elseif isa(fun, Expr) && ((fun.head === :call1) || (fun.head === :call))
        s = canonicalize(string(fun.args[2]))
        @dprintln(3,"ccall target: ", s)
    elseif isa(fun, Tuple) && length(fun) == 2
        s = canonicalize(string(fun[1]))
        @dprintln(3,"ccall target: ", s)
    else
        throw("Invalid ccall format...")
    end
    s *= "("
#    numInputs = length(args[3].args)-1
    argsStart = 4
    argsEnd = length(args)
    if contains(s, "cblas") && contains(s, "gemm")
        if mkl_lib!=""
            s *= "(CBLAS_LAYOUT) $(from_expr(args[4], linfo)), "
        else
            s *= "(CBLAS_ORDER) $(from_expr(args[4], linfo)), "
        end
        s *= "(CBLAS_TRANSPOSE) $(from_expr(args[6], linfo)), "
        s *= "(CBLAS_TRANSPOSE) $(from_expr(args[8], linfo)), "
        argsStart = 10
    end
    to_fold = args[argsStart:2:end]
    if length(to_fold) > 0
        s *= mapfoldl(x->from_expr(x,linfo), (a, b)-> "$a, $b", to_fold)
    end
    s *= ")"
    @dprintln(3,"from_ccall: ", s)
    s
end

function from_arrayset(args, linfo)
    idxs = mapfoldl(x->from_expr(x,linfo), (a, b) -> "$a, $b", args[3:end])
    src = from_expr(args[1], linfo)
    val = from_expr(args[2], linfo)
    if CGEN_RAW_ARRAY_MODE
        return "$src[$idxs - 1] = $val"
    else
        return "$src.ARRAYELEM($idxs) = $val"
    end
end

function istupletyp(typ)
    isa(typ, DataType) && (typ.name === Tuple.name)
end

function from_setfield!(args, linfo)
    @dprintln(3,"from_setfield! args are: ", args)
    @assert (length(args)==3) "Expect 3 arguments to setfield!, but got " * string(args)
    tgt = from_expr(args[1], linfo)
    @assert (isa(args[2], QuoteNode)) "CGen only handles setfield! with a fixed field name, but not " * string(args[2])
    fld = from_symbol(args[2].value, linfo)
    tgt * "." * fld * " = " * from_expr(args[3], linfo)
end

function from_getfield(args, linfo)
    @dprintln(3,"from_getfield args are: ", args)
    tgt = from_expr(args[1], linfo)
    @dprintln(4,"from_getfield tgt: ", tgt)
    if isa(args[1], TypedVar)
      args1typ = args[1].typ
      @dprintln(4,"from_getfield args[1] is ", args[1], " args1typ: ", args1typ)
    elseif inSymbolTable(args[1], linfo)
      args1typ = lookupSymbolType(args[1], linfo)
      @dprintln(4,"from_getfield args[1] is GenSym or Symbol, args1typ: ", args1typ)
    elseif isa(args[1], LHSVar)
      args1typ = getType(args[1], linfo)
    else
      throw("Unhandled argument 1 type to getfield")
    end
    #if istupletyp(args1typ) && isPrimitiveJuliaType(eltype(args1typ))
    if istupletyp(args1typ)
        fieldtype = typeof(args[2])
        @dprintln(4,"from_getfield found tupletyp, eltype: ", eltype(args1typ), " field type: ", fieldtype)
        if isa(args[2], Int)
            @dprintln(4,"from_getfield args[2] is Int so using simple field reference via from_tupleref")
            return from_tupleref(args, linfo)
        else
            # TO-DO!
            # What we really need to do here is when you create a Tuple type that you create an
            # array containing the offsets from the start of the struct to a given field.  Then
            # use you can args[2]-1 to index that array, cast the Tuple var to a char *, add the
            # offset, cast as a pointer to the type of the field, and then dereference.
            eltyp = toCtype(eltype(args1typ))
            @dprintln(4,"from_getfield eltyp: ", eltyp)
            return "(($eltyp *)&$(tgt))[" * from_expr(args[2], linfo) * " - 1]"
        end
    elseif isa(args1typ, DataType) && isa(args[2], QuoteNode)
        @dprintln(3, "from_getfield access field ", args[2], " of a record type ", args1typ)
        return string("(", from_expr(args[1],linfo), ").", args[2].value)
    end
    throw(string("Unhandled call to getfield ",args1typ, " ", eltype(args1typ)))
    #=
    mod, tgt = resolveCallTarget(args[1], args[2:end],linfo)
    if mod == "Intrinsics"
        return from_expr(tgt, linfo)
    elseif isInlineable(tgt, args[2:end], linfo)
        return from_inlineable(tgt, args[2:end], linfo)
    end
    from_expr(mod, linfo) * "." * from_expr(tgt, linfo)
    =#
end

function from_nfields(arg::LHSVar, linfo)
    @dprintln(3,"Arg is: ", arg)
    @dprintln(3,"Arg type = ", typeof(arg))
    #string(nfields(args[1].typ))
    string(nfields(lookupSymbolType(arg, linfo)))
end

function from_nfields(arg::TypedVar, linfo)
    @dprintln(3,"Arg is: ", arg)
    @dprintln(3,"Arg type = ", typeof(arg))
    string(nfields(arg.typ))
end

function from_steprange_last(args, linfo)
  start = "(" * from_expr(args[1], linfo) * ")"
  step  = "(" * from_expr(args[2], linfo) * ")"
  stop  = "(" * from_expr(args[3], linfo) * ")"
  return "("*stop*"-("*stop*"-"*start*")%"*step*")"
end


function get_shape_from_tuple(arg::Expr, linfo)
    res = ""
    if arg.head==:call && isBaseFunc(arg.args[1], :tuple)
        shp = AbstractString[]
        for i in 2:length(arg.args)
            push!(shp, from_expr(arg.args[i], linfo))
        end
        res = foldl((a, b) -> "$a, $b", shp)
    end
    return res
end

function get_shape_from_tuple(arg::ANY, linfo)
    return ""
end

function get_alloc_shape(args, dims, linfo)
    res = ""
    # in cases like rand(s1,s2), array allocation has only a tuple
    if length(args)==7
        res = get_shape_from_tuple(args[6], linfo)
    end
    if res!=""
        return res
    end
    shp = AbstractString[]
    arg = args[6]
    if (isa(arg, Expr) && isa(arg.typ, Tuple)) ||
       (isa(arg, RHSVar) && istupletyp(getSymType(arg, linfo))) # in case where the argument is a tuple
        arg_str = from_expr(arg, linfo)
        for i in 0:dims-1
            push!(shp, arg_str * ".f" * string(i))
        end
    else
        for i in 1:dims
            push!(shp, from_expr(args[6+(i-1)*2], linfo))
        end
    end
    res = foldl((a, b) -> "$a, $b", shp)
    return res
end

function from_arrayalloc(args, linfo)
    @dprintln(3,"Array alloc args:")
    map((i)->@dprintln(3,args[i]), 1:length(args))
    @dprintln(3,"----")
    @dprintln(3,"Parsing array type: ", args[4])
    typ, dims = parseArrayType(args[4])
    @dprintln(3,"Array alloc typ = ", typ)
    @dprintln(3,"Array alloc dims = ", dims)
    typ = toCtype(typ)
    @dprintln(3,"Array alloc after ctype conversion typ = ", typ)
    shape = get_alloc_shape(args, dims, linfo)
    @dprintln(3,"Array alloc shape = ", shape)
    return "j2c_array<$typ>::new_j2c_array_$(dims)d(NULL, $shape)"
end

function from_builtins_comp(f, args, linfo)
    tgt = string(f)
    return eval(parse("from_$cmd()"))
end

function from_array_ptr(args, linfo)
    return "$(from_expr(args[4], linfo)).data"
end

function from_sizeof(args, linfo)
    if isa(args[1], RHSVar)
        t = lookupType(args[1], linfo)
    else
        t = args[1]
    end
    s = toCtype(t)
    return "sizeof($s)"
end

function from_pointer(args, linfo)
    if CGEN_RAW_ARRAY_MODE
        s = "$(from_expr(args[1], linfo))"
    else
        s =  "$(from_expr(args[1], linfo)).data"
    end
    if length(args) > 1
        s *= " + $(from_expr(args[2], linfo))"
    end
    s
end

function from_pointerref(args, linfo)
    s = ""
    if length(args) > 1
        s *= " *($(from_expr(args[1], linfo)) + $(from_expr(args[2], linfo)) - 1)"
    end
    s
end

function from_raw_pointer(args, linfo)
    if length(args) == 1
        return "$(from_expr(args[1], linfo))"
    else
        return "$(from_expr(args[1], linfo)) + $(from_expr(args[2], linfo))"
    end
end

function from_string(args, linfo)
    @dprintln(3,"from_string")
    "BaseString(" * mapfoldl(x -> from_expr(x, linfo), (a,b) -> a * "," * b, args) * ")"
end

function from_builtins(f, args, linfo, call_ret_type)
    tgt = string(f)
    @dprintln(3,"from_builtins tgt = ", tgt)
    if tgt == "getindex" || tgt == "getindex!"
        return from_getindex(args, linfo)
    elseif tgt == "setindex" || tgt == "setindex!"
        return from_setindex(args, linfo)
    elseif tgt == "top"
        return ""
    elseif tgt == "box"
        return from_box(args, linfo)
    elseif tgt == "arrayref"
        return from_getindex(args, linfo)
    elseif tgt == "tupleref"
        return from_tupleref(args, linfo)
    elseif tgt == "tuple"
        typs = Any[lookupType(x, linfo) for x in args]
        return toCtype(Tuple{typs...}) * from_tuple(args, linfo)
    elseif tgt == "arraylen"
        return from_arraysize(args, linfo)
    elseif tgt == "arraysize"
        return from_arraysize(args, linfo)
    elseif tgt == "ccall"
        return from_ccall(args, linfo, call_ret_type)
    elseif tgt == "arrayset"
        return from_arrayset(args, linfo)
    elseif tgt == ":jl_new_array" || tgt == ":jl_alloc_array_1d" || tgt == ":jl_alloc_array_2d" || tgt == ":jl_alloc_array_3d"
        return from_arrayalloc(args, linfo)
    elseif tgt == ":jl_array_ptr"
        return from_array_ptr(args, linfo)
    elseif tgt == "pointer"
        return from_pointer(args, linfo)
    elseif tgt == "pointerref"
        return from_pointerref(args, linfo)
    elseif tgt == "getfield"
        return from_getfield(args, linfo)
    elseif tgt == "setfield!"
        return from_setfield!(args, linfo)
    elseif tgt == "unsafe_arrayref"
        return from_getindex(args, linfo)
    elseif tgt == "safe_arrayref"
        return from_safegetindex(args, linfo)
    elseif tgt == "unsafe_arrayset" || tgt == "safe_arrayset"
        return from_setindex(args, linfo)
    elseif tgt == "_unsafe_setindex!"
        return from_unsafe_setindex!(args, linfo)
    elseif tgt == "nfields"
        return from_nfields(args[1], linfo)
    elseif tgt == "sizeof"
        return from_sizeof(args, linfo)
    elseif tgt =="steprange_last"
        return from_steprange_last(args, linfo)
    elseif tgt == "convert" || tgt == "unsafe_convert"
        return from_typecast(args[1], [args[2]], linfo)
    elseif tgt == "min" || tgt == "max"
        arg_x = from_expr(args[1], linfo)
        arg_y = from_expr(args[2], linfo)
        if tgt == "max"
            cmp_op = ">"
        else
            cmp_op = "<"
        end
        return "(($arg_x $cmp_op $arg_y) ? ($arg_x) : ($arg_y))"
    elseif tgt == "string"
        return from_string(args, linfo)
    elseif isdefined(Base, f)
        fval = getfield(Base, f)
        if isa(fval, DataType)
            # handle type casting
            @dprintln(3, "found a typecast: ", fval, "(", args, ")")
            return from_typecast(fval, args, linfo)
        end
    end

    @dprintln(3,"Compiling ", string(f))
    throw("Unimplemented builtin")
end

function from_typecast(typ, args, linfo)
    @assert (length(args) == 1) "Expect only one argument in " * typ * "(" * args * ")"
    return "(" * toCtype(typ) * ")" * "(" * from_expr(args[1], linfo) * ")"
end

function from_box(args, linfo)
    s = ""
    typ = args[1]
    val = args[2]
    s *= from_expr(val, linfo)
    s
end

function from_intrinsic(f :: ANY, args, linfo, call_ret_typ)
    intr = string(f)
    @dprintln(3,"Intrinsic ", intr, ", args are ", args)

    if intr == "mul_int"
        return "($(from_expr(args[1], linfo))) * ($(from_expr(args[2], linfo)))"
    elseif intr == "neg_int"
        return "-" * "(" * from_expr(args[1], linfo) * ")"
    elseif intr == "mul_float" || intr == "mul_float_fast"
        return "($(from_expr(args[1], linfo))) * ($(from_expr(args[2], linfo)))"
    elseif intr == "urem_int" || intr == "checked_urem_int"
        return "($(from_expr(args[1], linfo))) % ($(from_expr(args[2], linfo)))"
    elseif intr == "add_int"
        return "($(from_expr(args[1], linfo))) + ($(from_expr(args[2], linfo)))"
    elseif intr == "or_int"
        return "($(from_expr(args[1], linfo))) | ($(from_expr(args[2], linfo)))"
    elseif intr == "xor_int"
        return "($(from_expr(args[1], linfo))) ^ ($(from_expr(args[2], linfo)))"
    elseif intr == "and_int"
        return "($(from_expr(args[1], linfo))) & ($(from_expr(args[2], linfo)))"
    elseif intr == "sub_int"
        return "($(from_expr(args[1], linfo))) - ($(from_expr(args[2], linfo)))"
    elseif intr == "slt_int" || intr == "ult_int"
        return "($(from_expr(args[1], linfo))) < ($(from_expr(args[2], linfo)))"
    elseif intr == "sle_int" || intr == "ule_int"
        return "($(from_expr(args[1], linfo))) <= ($(from_expr(args[2], linfo)))"
    elseif intr == "lshr_int"
        return "($(from_expr(args[1], linfo))) >> ($(from_expr(args[2], linfo)))"
    elseif intr == "shl_int"
        return "($(from_expr(args[1], linfo))) << ($(from_expr(args[2], linfo)))"
    elseif intr == "bitcast"
        return "(($(toCtype(args[1])))($(from_expr(args[2], linfo))))"
    elseif intr == "checked_ssub" || intr == "checked_ssub_int"
        if VERSION >= v"0.6.0-pre"
            return "checked_ssub_int<$(toCtype(call_ret_typ))>($(from_expr(args[1], linfo)), $(from_expr(args[2], linfo)))"
        else
            return "($(from_expr(args[1], linfo))) - ($(from_expr(args[2], linfo)))"
        end
    elseif intr == "checked_sadd" || intr == "checked_sadd_int"
        if VERSION >= v"0.6.0-pre"
            return "checked_sadd_int<$(toCtype(call_ret_typ))>($(from_expr(args[1], linfo)), $(from_expr(args[2], linfo)))"
        else
            return "($(from_expr(args[1], linfo))) + ($(from_expr(args[2], linfo)))"
        end
    elseif intr == "checked_smul"
        if VERSION >= v"0.6.0-pre"
            return "checked_smul_int<$(toCtype(call_ret_typ))>($(from_expr(args[1], linfo)), $(from_expr(args[2], linfo)))"
        else
            return "($(from_expr(args[1], linfo))) * ($(from_expr(args[2], linfo)))"
        end
    elseif intr == "checked_umul"
        if VERSION >= v"0.6.0-pre"
            return "checked_umul_int<$(toCtype(call_ret_typ))>($(from_expr(args[1], linfo)), $(from_expr(args[2], linfo)))"
        else
            return "($(from_expr(args[1], linfo))) * ($(from_expr(args[2], linfo)))"
        end
    elseif intr == "zext_int"
        return "($(toCtype(args[1]))) ($(from_expr(args[2], linfo)))"
    elseif intr == "sext_int"
        return "($(toCtype(args[1]))) ($(from_expr(args[2], linfo)))"
    elseif intr == "ctlz_int"
        return " (uint64_t)(__builtin_clzll($(from_expr(args[1], linfo))))"
    elseif intr == "smod_int"
        m = from_expr(args[1], linfo)
        n = from_expr(args[2], linfo)
        return "((($m) % ($n) + ($n)) % $n)"
    elseif intr == "srem_int" || intr == "checked_srem_int"
        return "($(from_expr(args[1], linfo))) % ($(from_expr(args[2], linfo)))"
    #TODO: Check if flip semantics are the same as Julia codegen.
    # For now, we emit unary negation
    elseif intr == "copysign_float"
        return "copysign(" * from_expr(args[1], linfo) * ", " * from_expr(args[2], linfo) * ")"
    elseif intr == "flipsign_int"
        return "cgen_flipsign_int(" * from_expr(args[1], linfo) * ", " * from_expr(args[2], linfo) * ")"
    elseif intr == "check_top_bit"
        typ = typeof(args[1])
        if !isPrimitiveJuliaType(typ)
            if hasfield(args[1], :typ)
                typ = args[1].typ
            end
        end
        nshfts = 8*sizeof(typ) - 1
        oprnd = from_expr(args[1], linfo)
        return oprnd * " >> " * string(nshfts) * " == 0 ? " * oprnd * " : " * oprnd
    elseif intr == "select_value"
        return "(" * from_expr(args[1], linfo) * ")" * " ? " *
        "(" * from_expr(args[2], linfo) * ") : " * "(" * from_expr(args[3], linfo) * ")"
    elseif intr == "not_int"
        return "!" * "(" * from_expr(args[1], linfo) * ")"
    elseif intr == "ctpop_int"
        return "__builtin_popcount" * "(" * from_expr(args[1], linfo) * ")"
    elseif intr == "cttz_int"
        return "cgen_cttz_int" * "(" * from_expr(args[1], linfo) * ")"
    elseif intr == "ashr_int" || intr == "lshr_int"
        return "($(from_expr(args[1], linfo))) >> ($(from_expr(args[2], linfo)))"
    elseif intr == "shl_int"
        return "($(from_expr(args[1], linfo))) << ($(from_expr(args[2], linfo)))"
    elseif intr == "add_float" || intr == "add_float_fast"
        return "($(from_expr(args[1], linfo))) + ($(from_expr(args[2], linfo)))"
    elseif intr == "lt_float" || intr == "lt_float_fast"
        return "($(from_expr(args[1], linfo))) < ($(from_expr(args[2], linfo)))"
    elseif intr == "eq_float" || intr == "eq_int" || intr == "eq_float_fast"
        return "($(from_expr(args[1], linfo))) == ($(from_expr(args[2], linfo)))"
    elseif intr == "ne_float" || intr == "ne_int" || intr == "ne_float_fast"
        return "($(from_expr(args[1], linfo))) != ($(from_expr(args[2], linfo)))"
    elseif intr == "le_float" || intr == "le_float_fast"
        return "($(from_expr(args[1], linfo))) <= ($(from_expr(args[2], linfo)))"
    elseif intr == "neg_float" || intr == "neg_float_fast"
        return "-($(from_expr(args[1], linfo)))"
    elseif intr == "abs_float"
        return "fabs(" * from_expr(args[1], linfo) * ")"
    elseif intr == "sqrt_llvm" || intr == "sqrt_llvm_fast"
        return "sqrt(" * from_expr(args[1], linfo) * ")"
    elseif intr == "sub_float" || intr == "sub_float_fast"
        return "($(from_expr(args[1], linfo))) - ($(from_expr(args[2], linfo)))"
    elseif intr == "div_float" || intr == "div_float_fast" ||
           intr == "sdiv_int" || intr == "udiv_int" || intr == "checked_sdiv_int" || intr == "checked_udiv_int"
        return "($(from_expr(args[1], linfo))) / ($(from_expr(args[2], linfo)))"
    elseif intr == "sitofp" || intr == "fptosi" || intr == "checked_fptosi" || intr == "fptrunc" || intr == "fpext" || intr == "uitofp"
        return "(" * toCtype(args[1]) * ")" * from_expr(args[2], linfo)
    elseif intr == "trunc_llvm" || intr == "trunc"
        return "(" * toCtype(args[1]) * ") trunc(" * from_expr(args[2], linfo) * ")"
    elseif intr == "floor_llvm" || intr == "floor"
        return "floor(" * from_expr(args[1], linfo) * ")"
    elseif intr == "ceil_llvm" || intr == "ceil"
        return "ceil(" * from_expr(args[1], linfo) * ")"
    elseif intr == "rint_llvm" || intr == "rint"
        return "round(" * from_expr(args[1], linfo) * ")"
    elseif f == :(===)
        return "(" * from_expr(args[1], linfo) * " == " * from_expr(args[2], linfo) * ")"
    elseif intr == "pow" || intr == "powi_llvm"
        return "pow(" * from_expr(args[1], linfo) * ", " * from_expr(args[2], linfo) * ")"
    elseif intr == "llvm.powi.f64"
        return "pow(" * from_expr(args[4], linfo) * ", " * from_expr(args[6], linfo) * ")"
    elseif intr == "powf" || intr == "powf_llvm"
        return "powf(" * from_expr(args[1], linfo) * ", " * from_expr(args[2], linfo) * ")"
    elseif intr == "llvm.powf.f64"
        return "powf(" * from_expr(args[4], linfo) * ", " * from_expr(args[6], linfo) * ")"
    elseif intr == "nan_dom_err"
        @dprintln(3,"nan_dom_err is: ")
        for i in 1:length(args)
            @dprintln(3,args[i])
        end
        #return "assert(" * "isNan(" * from_expr(args[1], linfo) * ") && !isNan(" * from_expr(args[2], linfo) * "))"
        return from_expr(args[1], linfo)
    elseif intr in ["checked_trunc_uint", "checked_trunc_sint"]
        return "(" * toCtype(args[1]) * ")(" * from_expr(args[2], linfo) * ")"
    else
        @dprintln(3,"Intrinsic ", intr, " is known but no translation available")
        throw("Unhandled Intrinsic...")
    end
end

function from_inlineable(f, args, linfo, call_ret_typ)
    @dprintln(3,"Checking if ", f, " can be inlined")
    @dprintln(3,"Args are: ", args)
#=
    if has(_operators, string(f))
        if length(args) == 1
          return "(" * string(f) * from_expr(args[1], linfo) * ")"
        else
          s = "(" * mapfoldl(\x->from_expr(x,linfo), (a,b)->"$a"*string(f)*"$b", args) * ")"
          return s
        end
    elseif has(_builtins, string(f))
=#
    s = string(f)
    if has(_primitive_builtins, s) || has(_builtins, s)
        return from_builtins(f, args, linfo, call_ret_typ)
    elseif isBaseFunc(f, :length) && length(args) > 0 && (isArrayType(lookupType(args[1], linfo)) || isStringType(lookupType(args[1], linfo)))
        return "(" * from_expr(args[1], linfo) * ".ARRAYLEN())"
    elseif has(_Intrinsics, s)
        return from_intrinsic(f, args, linfo, call_ret_typ)
    else
        throw("Unknown Operator or Method encountered: " * s)
    end
end

function isInlineable(f, args, linfo)
    #@dprintln(3, "IsInlineable f = ", f, " type = ", typeof(f), " isBase = ", isBaseFunc(f, :length), " args = ", args)
    #if isBaseFunc(f, :length) && length(args) > 0
    #    t = lookupType(args[1], linfo)
    #    @dprintln(3,"t = ", t, " type = ", typeof(t), " isArray = ", isArrayType(t))
    #end

    #if has(_operators, string(f)) || has(_builtins, string(f)) || has(_Intrinsics, string(f))
    s = string(f)
    if has(_primitive_builtins, s) && length(args) > 0
        t = lookupType(args[1], linfo)
        isPrimitiveJuliaType(t)
    elseif isBaseFunc(f, :length) && length(args) > 0 && (isArrayType(lookupType(args[1], linfo)) || isStringType(lookupType(args[1], linfo)))
        true
    else
        has(_builtins, s) || has(_Intrinsics, s)
    end
end

function arrayToTuple(a)
    ntuple((i)->a[i], length(a))
end

function from_symbol(ast, linfo)
    if ast in [:Inf, :Inf32]
        return "INFINITY"
    end
    hasfield(ast, :name) ? canonicalize(string(ast.name)) : canonicalize(ast)
end

function from_linenumbernode(ast, linfo)
    ""
end

function from_labelnode(ast, linfo)
    if recreateLoops
        if in(ast.label, lstate.exit_loop_set)
            return ""
        end
        if in(ast.label, lstate.head_loop_set)
            return "while (1) {"
        end
    end
    if recreateConds
        if haskey(lstate.follow_set, ast.label)
            s = ""
            for i = 1:lstate.follow_set[ast.label]
                s *= "}\n"
            end
            return s
        end
        if in(ast.label, lstate.cond_jump_targets)
            return ""
        end
    end

    "label" * string(ast.label) * " : "
end

function from_ref(args, linfo)
    "&$(from_expr(args[1], linfo))"
end

function from_call1(ast::Array{Any, 1}, linfo)
    @dprintln(3,"Call1 args")
    s = ""
    for i in 2:length(ast)
        s *= from_expr(ast[i], linfo)
        @dprintln(3,ast[i])
    end
    @dprintln(3,"Done with call1 args")
    s
end

function isPendingCompilation(list, tgt, typs0)
    for i in 1:length(list)
        ast, name, typs = lstate.worklist[i]
        if name == tgt && typs == typs0
            return true
        end
    end
    return false
end

function resolveCallTarget(ast::Array{Any, 1},linfo, call_ret_typ)
    # julia doesn't have GetfieldNode anymore
    #if isdefined(:GetfieldNode) && isa(args[1],GetfieldNode) && isa(args[1].value,Module)
    #   M = args[1].value; s = args[1].name; t = ""

    @dprintln(3,"Trying to resolve target from ast::Array{Any,1} with args: ", ast)
    return resolveCallTarget(ast[1], ast[2:end],linfo, call_ret_typ)
end

#case 0:
function resolveCallTarget(f::Symbol, args::Array{Any, 1},linfo, call_ret_typ)
    M = ""
    t = ""
    s = ""
    if isInlineable(f, args, linfo)
        return M, string(f), from_inlineable(f, args, linfo, call_ret_typ)
    elseif (f === :call)
        #This means, we have a Base.call - if f is not a Function, this is translated to f(args)
        arglist = mapfoldl(x->from_expr(x,linfo), (a,b)->"$a, $b", args[2:end])
        if isa(args[1], DataType)
            t = toCtype(args[1]) * "{" * arglist * "}"
        else
            t = from_expr(args[1],linfo) * "(" * arglist * ")"
        end
    end
    return M, s, t
end

function resolveCallTarget(f::Expr, args::Array{Any, 1},linfo, call_ret_typ)
    M = ""
    t = ""
    s = ""
    if (f.head === :call) || (f.head === :call1) # :call1 is gone in v0.4
        if length(f.args) == 3 && isBaseFunc(f.args[1], :getfield) && isa(f.args[3],QuoteNode)
            s = f.args[3].value
            if isa(f.args[2],Module)
                M = f.args[2]
            elseif isa(f.args[2],GlobalRef)
                M = eval(f.args[2])
            end
        end
        @dprintln(3,"Case 0: Returning M = ", M, " s = ", s, " t = ", t)
    end
    return M, s, t
end

function resolveCallTarget(f, args::Array{Any, 1},linfo, call_ret_typ)
    @dprintln(3,"Trying to resolve target from ", f, "::", typeof(f), " with args: ", args)
    M = ""
    t = ""
    s = ""

    if isa(f, QuoteNode)
        @dprintln(3,"Removing QuoteNode.")
        M = Base
        f = f.value
        s = f
    end

    #case 1:
    if isBaseFunc(f, :getfield) && isa(args[2], QuoteNode)
        @dprintln(3,"Case 1: args[2] is ", args[2])
        fname = args[2].value
        if isa(args[1], Module)
            M = args[1]
        elseif (fname == :im || fname == :re) &&
               (isa(args[1], RHSVar) &&
                (getSymType(args[1], linfo) == Complex64 || getSymType(args[1], linfo) == Complex128))
            func = fname == :re ? "real" : "imag";
            t = func * "(" * from_expr(args[1],linfo) * ")"
        else
            #case 2:
            t = from_expr(args[1],linfo) * "." * string(fname)
            #M, _s = resolveCallTarget([args[1]])
        end
        @dprintln(3,"Case 1: Returning M = ", M, " s = ", s, " t = ", t)
    # the following appears to be dead code
    # elseif isBaseFunc(f, :getfield) && hasfield(f, :head) && (f.head === :call)
    #    @dprintln(3,"Case 2: calling")
    #    return resolveCallTarget(f,linfo)
    # case 3:
    elseif (isa(f, TopNode) || isa(f, GlobalRef)) && isInlineable(f.name, args, linfo)
        t = from_inlineable(f.name, args,linfo, call_ret_typ)
        @dprintln(3,"Case 3: Returning M = ", M, " s = ", s, " t = ", t)
    end
    @dprintln(3,"In resolveCallTarget: Returning M = ", M, " s = ", s, " t = ", t)
    return M, s, t
end

function inSymbolTable(x::RHSVar, linfo)
    @dprintln(3, "inSymbolTable RHSVar x = ", x)
    x = CompilerTools.LambdaHandling.lookupVariableName(x, linfo)
    @dprintln(3, "inSymbolTable RHSVar x = ", x)
    if !haskey(lstate.symboltable, x)
        @dprintln(3, "symboltable = ", lstate.symboltable)
    end
    haskey(lstate.symboltable, x)
end


#inSymbolTable(x, linfo) = haskey(lstate.symboltable, x)
function inSymbolTable(x, linfo) 
    @dprintln(3, "inSymbolTable x = ", x, " ", typeof(x))
    haskey(lstate.symboltable, x)
end

function lookupSymbolType(x, linfo)
    x = CompilerTools.LambdaHandling.lookupVariableName(x, linfo)
    lstate.symboltable[x]
end

function setSymbolType(x, typ, linfo)
    @dprintln(3, "setSymbolType ", x, " typ = ", typ)
    x = CompilerTools.LambdaHandling.lookupVariableName(x, linfo)
    lstate.symboltable[x] = typ
end

function typeToStr(typ)
  if isa(typ, Array) || isa(typ, Tuple)
    if isempty(typ)
        "()"
    else
        "(" * foldl(*, String[typeToStr(t) for t in typ]) * ")"
    end
  elseif isa(typ, DataType)
      if typ <: Array
          "Array{" * typeToStr(eltype(typ)) * "}"
      elseif typ <: Tuple
          "Tuple{" * foldl(*, String[typeToStr(t) for t in typ.parameters]) * "}"
      else
          string(typ)
      end
  else
    string(typ)
  end
end

function isFunctionCompiled(funStr, argTyps)
    typs = typeToStr(argTyps)
    has(lstate.compiledfunctions, (funStr, typs))
end

function setFunctionCompiled(funStr, argTyps)
    typs = typeToStr(argTyps)
    push!(lstate.compiledfunctions, (funStr, typs))
end

function from_call(ast::Array{Any, 1},linfo, call_ret_typ)

    pat_out = external_pattern_match_call.func(ast,linfo)
    if pat_out != ""
        @dprintln(3, "external pattern matched: ",ast)
        return pat_out
    end

    pat_out = pattern_match_call(ast,linfo)
    if pat_out != ""
        @dprintln(3, "pattern matched: ",ast)
        return pat_out
    end

    @dprintln(3,"Compiling call: ast = ", ast, " args are: ")
    for i in 1:length(ast)
        @dprintln(3,"Arg ", i, " = ", ast[i], " type = ", typeof(ast[i]))
    end
    # Try and find the target of the call
    mod, fun, t = resolveCallTarget(ast,linfo, call_ret_typ)

    # resolveCallTarget will eagerly try to translate the call
    # if it can. If it does, then we are done.
    if !isempty(t)
        return t;
    end

    if fun == ""
        fun = ast[1]
    end

    args = ast[2:end]
    @dprintln(3,"mod is: ", mod)
    @dprintln(3,"fun is: ", fun)
    @dprintln(3,"call Args are: ", args)

    if isInlineable(fun, args, linfo)
        @dprintln(3,"Doing with inlining ", fun, "(", args, ")")
        fs = from_inlineable(fun, args, linfo, call_ret_typ)
        return fs
    end
    @dprintln(3,"Not inlinable")
    if isa(mod, Module)
        funStr = "_" * from_expr(GlobalRef(mod, fun), linfo)
    else
        funStr = "_" * from_expr(fun, linfo)
    end

    if isBaseFunc(fun, :println) || isBaseFunc(fun, :print)
        s =  "std::cout << "
        for a in 2:length(args)
            s *= from_expr(args[a],linfo) * (a < length(args) ? "<<" : "")
        end
        if fun==:println
            s *= "<< std::endl;"
        else
            s *= ";"
        end
        return s
    end

    s = ""
    map((i)->@dprintln(3,i[2]), lstate.worklist)
    map((i)->@dprintln(3,i), lstate.compiledfunctions)
    argTyps = []
    for a in 1:length(args)
        @dprintln(3, "a = ", a, " args[a] = ", args[a], " type = ", typeof(args[a]))
        s *= from_expr(args[a],linfo) * (a < length(args) ? "," : "")
        #if !skipCompilation
            # Attempt to find type
            if typeAvailable(args[a])
                push!(argTyps, args[a].typ)
            elseif isPrimitiveJuliaType(typeof(args[a]))
                push!(argTyps, typeof(args[a]))
            elseif isa(args[a], AbstractString)
                push!(argTyps, typeof(args[a]))
            elseif inSymbolTable(args[a], linfo)
                push!(argTyps, lookupSymbolType(args[a], linfo))
            else
                @dprintln(3, "linfo = ", linfo)
                throw(string("Could not determine type for arg ", a, " to call ", mod, ".", fun, " with name ", args[a]))
            end
        #end
    end
    s = funStr * "(" * s * ")"

    # If we have previously compiled this function
    # we fallthru and simply emit the call.
    # Else we lookup the function Symbol and enqueue it
    # TODO: This needs to specialize on types
    skipCompilation = isFunctionCompiled(funStr, argTyps) ||
        isPendingCompilation(lstate.worklist, funStr, argTyps)

    @dprintln(3,"Finished translating call : ", s)
    @dprintln(3,ast[1], " : ", typeof(ast[1]), " : ", hasfield(ast[1], :head) ? ast[1].head : "")
    if !skipCompilation && (isa(fun, Symbol) || isa(fun, Function) || isa(fun, TopNode) || isa(fun, GlobalRef))
        #=
        @dprintln(3,"Worklist is: ")
        for i in 1:length(lstate.worklist)
            ast, name, typs = lstate.worklist[i]
            @dprintln(3,name);
        end
        @dprintln(3,"Compiled Functions are: ")
        for i in 1:length(lstate.compiledfunctions)
            name, typs = lstate.compiledfunctions[i]
            @dprintln(3,name, " ", typs);
        end
        =#
        @dprintln(3,"Inserting: ", fun, " : ", funStr, " : ", arrayToTuple(argTyps))
        insert(fun, mod, funStr, arrayToTuple(argTyps))
    end
    s
end

# Generate return statements. The way we handle multiple return
# values is different depending on whether this return statement is in
# the entry point or not. If it is, then the multiple return values are
# pushed into spreallocated slots in the argument list (ret0, ret1...)
# If not, just return a tuple.

function from_return(args,linfo)
    global inEntryPoint
    @dprintln(3,"Return args are: ", args)
    retExp = ""
    if length(args) == 0 || (length(args) == 1 && args[1] == nothing || isa(args[1], GlobalRef) && Base.resolve(args[1], force=true) == GlobalRef(Core, :nothing))
        return "return"
    elseif inEntryPoint
        arg1 = args[1]
        arg1_typ = Any
        if typeAvailable(arg1)
            arg1_typ = arg1.typ
        elseif isa(arg1, GenSym)
            arg1_typ = lookupSymbolType(arg1, linfo)
        end
        if istupletyp(arg1_typ)
            retTyps = arg1_typ.parameters
            for i in 1:length(retTyps)
                retExp *= "*ret" * string(i-1) * " = " * from_expr(arg1,linfo) * ".f" * string(i-1) * ";\n"
            end
        else
            # Skip the "*ret0 = ..." stmt for the special case of Void/nothing return value.
            if arg1 != nothing
                retExp = "*ret0 = " * from_expr(arg1,linfo) * ";\n"
            end
        end
        return retExp * "return"
    else
        return "return " * from_expr(args[1],linfo)
    end
end


function from_gotonode(ast, linfo)
    labelId = ast.label
    s = ""
    @dprintln(3,"Compiling goto: ", exp, " ", typeof(exp))
    if isa(exp, Expr) || isa(exp, RHSVar)
        s *= "if (!(" * from_expr(exp,linfo) * ")) "
    end
    if recreateLoops && in(labelId, lstate.head_loop_set)
        s *= "}\n"
    elseif recreateConds && haskey(lstate.follow_set, labelId)
        s *= "}\nelse {"
    else
        s *= "goto " * "label" * string(labelId)
    end
    s
end

function from_gotoifnot(args,linfo)
    exp = args[1]
    labelId = args[2]
    s = ""
    @dprintln(3,"Compiling gotoifnot: ", exp, " ", typeof(exp))
    if isa(exp, Expr) || isa(exp, RHSVar) || isa(exp, CGen_boolean_and)
        if recreateLoops && in(labelId, lstate.exit_loop_set)
            s *= "if (!(" * from_expr(exp,linfo) * ")) break"
        elseif recreateConds && in(labelId, lstate.cond_jump_targets)
            s *= "if (" * from_expr(exp,linfo) * ") {"
        else
            s *= "if (!(" * from_expr(exp,linfo) * ")) "
            s *= "goto " * "label" * string(labelId)
        end
    elseif exp == true
    elseif exp == false
        s *= "goto " * "label" * string(labelId)
    end
    s
end
#=
function from_goto(exp, labelId,linfo)
    s = ""
    @dprintln(3,"Compiling goto: ", exp, " ", typeof(exp))
    if isa(exp, Expr) || isa(exp, RHSVar)
        s *= "if (!(" * from_expr(exp,linfo) * ")) "
    end
    s *= "goto " * "label" * string(labelId)
    s
end
=#

function from_if(args,linfo)
    exp = args[1]
    true_exp = args[2]
    false_exp = args[3]
    s = ""
    @dprintln(3,"Compiling if: ", exp, " ", true_exp," ", false_exp)
    s *= from_expr(exp,linfo) * "?" * from_expr(true_exp,linfo) * ":" * from_expr(false_exp,linfo)
    s
end

function from_comparison(args,linfo)
    @assert length(args)==3 "CGen: invalid comparison"
    left = args[1]
    comp = args[2]
    right = args[3]
    s = ""
    @dprintln(3,"Compiling comparison: ", left, " ", comp," ", right)
    s *= from_expr(left,linfo) * "$comp" * from_expr(right,linfo)
    s
end

function from_globalref(ast,linfo)
    ast = Base.resolve(ast, force=true)
    mod = ast.mod
    name = ast.name
    s = canonicalize(mod) * "_" * from_symbol(name,linfo)
    @dprintln(3,"Name is: ", name, " and its type is:", typeof(name))
    # handle global constant
    if isdefined(mod, name) && ccall(:jl_is_const, Int32, (Any, Any), mod, name) == 1
        def = getfield(mod, name)
        if !isa(def, IntrinsicFunction) && !isa(def, Function)
            if isbits(def)
                return from_expr(def,linfo)
            elseif isa(def, Array)
            # global constant array
                @dprintln(3, "record global array: ", name)
                lstate.globalConstants[s] = def
            else
                error("Global definition for ", name, " :: ", typeof(def), " is not supported by CGen.")
            end
        end
    elseif mod == Base && name == :STDOUT
        s = "stdout"
    end
    s
end

function from_topnode(ast,linfo)
    canonicalize(ast)
end

function from_quotenode(ast,linfo)
    from_symbol(ast,linfo)
end

function from_line(args,linfo)
    ""
end

function from_parforend(args,linfo)
    global lstate
    s = ""
    parfor = args[1]
    lpNests = parfor.loopNests
    for i in 1:length(lpNests)
        s *= "}\n"
    end
    rdsinit = rdsepilog = rdscopy = ""
    rds = parfor.reductions
    parallel_reduction = USE_OMP==1 && lstate.ompdepth <= 1 #&& any(Bool[(isa(a->reductionFunc, Function) || isa(a->reductionVarInit, Function)) for a in rds])
    if parallel_reduction && length(rds) > 0
        @dprintln(3,"from_parforend: parallel_reduction")
        nthreadsvar = "_num_threads"
        rdsepilog = "for (unsigned i = 0; i < $nthreadsvar; i++) {\n"
        rdscleanup = ""
        for rd in rds
            rdv = rd.reductionVar
            rdvt = getSymType(rdv, linfo)
            rdvtyp = toCtype(rdvt)
            @dprintln(3,"from_parforend: rdv = ", rdv, " rdvt = ", rdvt, " rdvtyp = ", rdvtyp, " func = ", rd.reductionFunc)
            rdvar = from_expr(rdv,linfo)
            @dprintln(3,"from_parforend: rdvar = ", rdvar)
            # this is now handled either in pre_statements, or by user (in the case of explicit parfor loop).
            #rdsinit *= from_reductionVarInit(rd.reductionVarInit, rdv,linfo)
            rdvar_i = addLocalVariable(gensym(string(rdvar, "_i")), rdvt, 0, linfo)
            setSymbolType(rdvar_i, rdvt, linfo)
            @dprintln(3,"from_parforend: rdvar_i = ", rdvar_i)
            rdsepilog *= "$rdvtyp &" * from_expr(rdvar_i, linfo) * " = $(rdvar)_vec[i];\n"
            @dprintln(3,"from_parforend: rdsepilog = ", rdsepilog);
            rdsepilog *= from_reductionFunc(rd.reductionFunc, rdv, rdvar_i,linfo) * ";\n"
            @dprintln(3,"from_parforend: after reductionFunc rdsepilog = ", rdsepilog);
            if isPrimitiveJuliaType(rdvt)
                rdscleanup *= "free($(rdvar)_vec);\n";
            end
            rdscopy *= "shared_$(rdvar) = $(rdvar);\n"
        end
        rdsepilog *= "}\n" * rdscleanup

    end
    s *= USE_OMP==1 && lstate.ompdepth <=1 ? "$rdscopy }\n$rdsinit $rdsepilog }/*parforend*/\n" : "" # end block introduced by private list
    @dprintln(3,"Parforend = ", s)
    lstate.ompdepth -= 1
    s
end

function loopNestCount(loop,linfo)
    "(((" * from_expr(loop.upper,linfo) * ") + 1 - (" * from_expr(loop.lower,linfo) * ")) / (" * from_expr(loop.step,linfo) * "))"
end

#TODO: Implement task mode support here
function from_insert_divisible_task(args,linfo)
    inserttasknode = args[1]
    @dprintln(3,"Ranges: ", inserttasknode.ranges)
    @dprintln(3,"Args: ", inserttasknode.args)
    @dprintln(3,"Task Func: ", inserttasknode.task_func)
    @dprintln(3,"Join Func: ", inserttasknode.join_func)
    @dprintln(3,"Task Options: ", inserttasknode.task_options)
    @dprintln(3,"Host Grain Size: ", inserttasknode.host_grain_size)
    @dprintln(3,"Phi Grain Size: ", inserttasknode.phi_grain_size)
    throw("Task mode is not supported yet")
end

function from_loopnest(ivs, starts, stops, steps, linfo)
    vecclause = (vectorizationlevel == VECFORCE) ? "#pragma simd\n" : ""
    mapfoldl(
        (i) ->
            (i == length(ivs) ? vecclause : "") *
            "for ( $(ivs[i]) = $(starts[i]); $(ivs[i]) <= (int64_t)$(stops[i]); $(ivs[i]) += $(steps[i])) {\n",
            (a, b) -> "$a $b",
            1:length(ivs)
    )
end

function from_reductionVarInit(reductionVarInit :: ParallelIR.DelayedFunc, a, linfo)
    from_exprs(ParallelIR.callDelayedFuncWith(reductionVarInit,a), linfo)
end

function from_reductionVarInit(reductionVarInit :: Any, a, linfo)
    from_expr(a, linfo) * " = " * from_expr(reductionVarInit, linfo) * ";\n"
end

function from_reductionFunc(reductionFunc :: Symbol, a, b, linfo)
    from_expr(a, linfo) * " " * string(reductionFunc) * " " * from_expr(b, linfo)
end

function from_reductionFunc(reductionFunc :: ParallelIR.DelayedFunc, a, b, linfo)
    @dprintln(3, "from_reductionFunc for DelayedFunc")
    from_exprs(ParallelIR.callDelayedFuncWith(reductionFunc, a, b), linfo)
end

function from_reductionFunc(reductionFunc :: Any, a, b, linfo)
    throw(string("CGen Error: Unsupported redunction function: ", reductionFunc, " :: ", typeof(reductionFunc)))
end

# If the parfor body is too complicated then DomainIR or ParallelIR will set
# instruction_count_expr = nothing

# Meaning of num_threads_mode
# mode = 1 uses static insn count if it is there, but doesn't do dynamic estimation and fair core allocation between levels in a loop nest.
# mode = 2 does all of the above
# mode = 3 in addition to 2, also uses host minimum (0) and Phi minimum (10)

function pattern_match_reduce_sum(reductionFunc::DelayedFunc,linfo)
    reduce_box = reductionFunc.args[1][1].args[2]
    if reduce_box.args[1]==GlobalRef(Core.Intrinsics,:box)
        if reduce_box.args[3].args[1].name==:add_float || reduce_box.args[3].args[1].name==:add_int
            return true
        end
    elseif reduce_box.args[1]==GlobalRef(Base,:+)
        return true
    end
    return false
end

function pattern_match_reduce_sum(reductionFunc::GlobalRef,linfo)
    if reductionFunc.name==:add_float || reductionFunc.name==:add_int
        return true
    end
    return false
end

function from_parforstart(args, linfo)
    global lstate
    num_threads_mode = ParallelIR.num_threads_mode

    @dprintln(3,"from_parforstart args: ",args);

    parfor  = args[1]
    lpNests = parfor.loopNests
    private_vars = parfor.private_vars

    # Remove duplicate gensyms
    vgs = Dict()
    dups = []
    for i in 1:length(private_vars)
        p = private_vars[i]
        if isa(p, GenSym)
            if get(vgs, p.id, false)
                push!(dups, i)
            else
                vgs[p.id] = true
            end
        end
    end
    deleteat!(private_vars, dups)

    # Translate metadata for the loop nests
    ivs = map((a)->from_expr(a.indexVariable, linfo), lpNests)
    starts = map((a)->from_expr(a.lower, linfo), lpNests)
    stops = map((a)->from_expr(a.upper, linfo), lpNests)
    steps = map((a)->from_expr(a.step, linfo), lpNests)

    @dprintln(3,"ivs ",ivs);
    @dprintln(3,"starts ", starts);
    @dprintln(3,"stops ", stops);
    @dprintln(3,"steps ", steps);

    # Generate the actual loop nest
    loopheaders = from_loopnest(ivs, starts, stops, steps, linfo)

    s = ""

    # thread count related stuff
    lcountexpr = ""
    for i in 1:length(lpNests)
        lcountexpr *= "(((" * starts[i] * ") + 1 - (" * stops[i] * ")) / (" * steps[i] * "))" * (i == length(lpNests) ? "" : " * ")
    end
    nthreadsvar = "_num_threads"
    preclause = "unsigned $nthreadsvar;\n"
    nthreadsclause = ""
    instruction_count_expr = parfor.instruction_count_expr
    if num_threads_mode == 1 && instruction_count_expr != nothing
        insncount = from_expr(instruction_count_expr, linfo)
        preclause *= "$nthreadsvar = computeNumThreads(((unsigned)" * insncount * ") * (" * lcountexpr * "));\n"
        nthreadsclause = "num_threads($nthreadsvar) "
    elseif num_threads_mode == 2
        if instruction_count_expr != nothing
            insncount = from_expr(instruction_count_expr, linfo)
            preclause *= "J2cParRegionThreadCount j2c_block_region_thread_count(std::min(unsigned($lcountexpr ),computeNumThreads(((unsigned) $insncount ) * ( $lcountexpr ))),__LINE__,__FILE__);\n"
        else
            preclause *= "J2cParRegionThreadCount j2c_block_region_thread_count(" * lcountexpr * ",__LINE__,__FILE__);\n"
        end
        preclause *= "$nthreadsvar = j2c_block_region_thread_count.getUsed();\n"
        nthreadsclause = "num_threads($nthreadsvar)"
    elseif num_threads_mode == 3
        if instruction_count_expr != nothing
            insncount = from_expr(instruction_count_expr, linfo)
            preclause *= "J2cParRegionThreadCount j2c_block_region_thread_count(std::min(unsigned($lcountexpr),computeNumThreads(((unsigned) $insncount) * ($lcountexpr))),__LINE__,__FILE__, 0, 10);\n"
        else
            preclause *= "J2cParRegionThreadCount j2c_block_region_thread_count($lcountexpr,__LINE__,__FILE__, 0, 10);\n"
        end
        preclause *= "$nthreadsvar = j2c_block_region_thread_count.getUsed();\n"
        nthreadsclause = "if(j2c_block_region_thread_count.runInPar()) num_threads($nthreadsvar) "
    else
        preclause *= "$nthreadsvar = omp_get_max_threads();\n"
        nthreadsclause = "num_threads($nthreadsvar) "
    end
    @dprintln(3, "preclause = ", preclause)
    @dprintln(3, "nthreadsvar = ", nthreadsvar)
    @dprintln(3, "nthreadsclause = ", nthreadsclause)
    # Generate initializers and OpenMP clauses for reductions
    rds = parfor.reductions
    # non-symbol reductionFunc is not supported by OpenMP
    rdsextra = rdsprolog = rdsclause = ""
    @dprintln(3,"reductions = ", rds);
    lstate.ompdepth += 1
    # custom reduction only kicks in when omp parallel is produced, i.e., when ompdepth == 1
    parallel_reduction = USE_OMP==1 && lstate.ompdepth == 1 #&& any(Bool[(isa(a->reductionFunc, Function) || isa(a->reductionVarInit, Function)) for a in rds])
    for rd in rds
        rdv = rd.reductionVar
        rdvt = getSymType(rdv, linfo)
        rdvtyp = toCtype(rdvt)
        rdvar = from_expr(rdv, linfo)
        rdv_tmp = gensym(rdvar)
        advout = addLocalVariable(rdv_tmp, rdvt, 0, linfo)
        setSymbolType(advout, rdvt, linfo)
        rdvar_tmp = from_symbol(rdv_tmp, linfo)
        if parallel_reduction
            if isPrimitiveJuliaType(rdvt)
                rdsprolog *= "$rdvtyp *$(rdvar)_vec = ($rdvtyp *)malloc(sizeof($rdvtyp)*$nthreadsvar);\n"
            else
                rdsprolog *= "std::vector<$rdvtyp> $(rdvar)_vec($nthreadsvar);\n"
            end
            rdsprolog *= "for (int rds_init_loop_var = 0; rds_init_loop_var  < $nthreadsvar; rds_init_loop_var++) {\n"
            rdsprolog *= "$rdvtyp &$rdvar_tmp = $(rdvar)_vec[rds_init_loop_var];\n"
            rdsprolog *= from_reductionVarInit(rd.reductionVarInit, rdv_tmp, linfo) * "}\n"
            #push!(private_vars, rdv)
            rdsextra *= "$rdvtyp &shared_$(rdvar) = $(rdvar)_vec[omp_get_thread_num()];\n"
            rdsextra *= "$rdvtyp $(rdvar) = shared_$(rdvar);\n"
        else
            if isDistributedMode() && lstate.ompdepth == 1
                if pattern_match_reduce_sum(rd.reductionFunc, linfo) && !isArrayType(rdvt)
                    rdsclause *= "reduction(+: $(rdvar)) "
                end
            end
            # The init is now handled in pre-statements
            #rdsprolog *= from_reductionVarInit(rd.reductionVarInit, rdv, linfo)
            # parallel IR no longer produces reductionFunc as a symbol
            #if isa(rd.reductionFunc, Symbol)
            #   rdop = string(rd.reductionFunc)
            #   rdsclause *= "reduction($(rdop) : $(rdvar)) "
            #end
        end
    end
    @dprintln(3, "rdsprolog = ", rdsprolog)
    @dprintln(3, "rdsclause = ", rdsclause)

    if isDistributedMode() && lstate.ompdepth == 1 && parfor.force_simd
        s *= "$rdsprolog #pragma simd $rdsclause\n"
        s *= loopheaders
        return s
    end

    # Don't put openmp pragmas on nested parfors.
    if USE_OMP==0 || lstate.ompdepth > 1
        # Still need to prepend reduction variable initialization for non-openmp loops.
        return rdsprolog * loopheaders
    end
    private_vars = [ lookupVariableName(x, linfo) for x in private_vars ]
    # Check if there are private vars and emit the |private| clause
    @dprintln(3,"Private Vars: ")
    @dprintln(3,"-----")
    @dprintln(3,private_vars)
    @dprintln(3,"-----")
    privatevars = isempty(private_vars) ? "" : "private(" * mapfoldl(canonicalize, (a,b) -> "$a, $b", private_vars) * ")"

    s *= "{\n$preclause $rdsprolog #pragma omp parallel $nthreadsclause $privatevars\n{\n$rdsextra"
    s *= "#pragma omp for private(" * mapfoldl((a)->a, (a, b)->"$a, $b", ivs) * ") $rdsclause\n"
    s *= loopheaders
    s
end

# TODO: Should simple objects be heap allocated ?
# For now, we stick with stack allocation
function from_new(args, linfo)
    s = ""
    typ = args[1] #type of the object
    @dprintln(3,"from_new args = ", args)
    if isa(typ, GlobalRef)
        typ = getfield(typ.mod, typ.name)
    end
    if isa(typ, DataType)
        if typ <: AbstractString || typ == Complex64 || typ == Complex128
            # assert(length(args) == 3)
            @dprintln(3, "new complex number")
            s = toCtype(typ) * "(" * mapfoldl(x->from_expr(x,linfo), (a, b) -> "$a, $b", args[2:end]) * ")"
        else
            objtyp, ptyps = parseParametricType(typ)
            if isempty(ptyps)
                s = canonicalize(objtyp) * "{}"
            else
                ctyp = canonicalize(objtyp) * mapfoldl(canonicalize, (a, b) -> a * b, ptyps)
                s = ctyp * "{"
                s *= mapfoldl(x->from_expr(x,linfo), (a, b) -> "$a, $b", args[2:end]) * "}"
            end
        end
    elseif isa(typ, Expr)
        if isBaseFunc(typ.args[1], :getfield)
            typ = getfield(typ.args[2], typ.args[3].value)
            objtyp, ptyps = parseParametricType(typ)
            ctyp = canonicalize(objtyp) * (isempty(ptyps) ? "" : mapfoldl(canonicalize, (a, b) -> a * b, ptyps))
            s = ctyp * "{"
            s *= (isempty(args[2:end]) ? "" : mapfoldl(x->from_expr(x,linfo), (a, b) -> "$a, $b", args[2:end])) * "}"
        else
            throw(string("CGen Error: unhandled args in from_new ", args))
        end
    end
    s
end

function body(ast)
    ast.args[3]
end

function from_loophead(args,linfo)
    iv = from_expr(args[1],linfo)
    decl = "uint64_t"
    if inSymbolTable(args[1], linfo)
        decl = toCtype(lookupSymbolType(args[1], linfo))
    end
    start = from_expr(args[2],linfo)
    stop = from_expr(args[3],linfo)
    "for($decl $iv = $start; $iv <= $stop; $iv += 1) {\n"
end

function from_loopend(args,linfo)
    "}\n"
end

"""
Helper function to construct a :parallel_loophead node,
see from_parallel_loophead for a description of the arguments
"""
function mk_parallel_loophead(loopvars::Vector{Symbol},
                              starts::Vector{Union{Symbol,Int}},
                              stops::Vector{Union{Symbol,Int}};
                              num_threads::Int=0, schedule::AbstractString="")
    Expr(:parallel_loophead, loopvars, starts, stops,
         Set{Union{GenSym, Symbol}}(), num_threads, schedule)
end

export mk_parallel_loophead

"""
Insert an openmp parallel for loop nest
    args[1]::Vector{Symbol}            : The loop variable for loop in the nest
    args[2]::Vector{Union{Symbol,Int}} : The start value for each loop in the nest
    args[3]::Vector{Union{Symbol,Int}} : The stop value for each loop in the nest
    args[4]::Set{GenSym,Symbol}        : A list of private variables for the parallel region
    args[5]::Int (optional)            : Argument to num_threads clause
    args[6]::String (optional)         : Exact string for schedule clause (i.e. schedule(dynamic))
"""
function from_parallel_loophead(args,linfo)
    private = ""
    if length(args[4]) > 0
        private = "private("
        for var in args[4]
            private *= "$(from_expr(var, linfo)),"
        end
        private = chop(private)
        private *= ")"
    end
    num_threads = ""
    if args[5] > 0
        num_threads = "num_threads($(args[5]))"
    end
    schedule = args[6]
    inner_private = "private("
    for iv in args[1]
        inner_private *= "$(from_expr(iv, linfo)),"
    end
    inner_private = chop(inner_private)
    inner_private *= ")"

    s = "#pragma omp parallel $private $num_threads \n{\n"
    s *= "#pragma omp for $schedule collapse($(length(args[1]))) $inner_private\n"
    for (iv, start, stop) in zip(args[1], args[2], args[3])
        start = from_expr(start,linfo)
        stop = from_expr(stop,linfo)
        iv = from_expr(iv,linfo)
        s *="for(int64_t $iv = $start; $iv <= $stop; $iv += 1) {\n"
    end
    s
end

"""
Close a loopnest create by a :parallel_loophead
    args[1]::Int : The depth of the loopnest (>= 1)
"""
function from_parallel_loopend(args,linfo)
    s = "}\n"           # Close parallel region
    for i in 1:args[1]  # Close loops
        s *= "}\n"
    end
    s
end

function from_expr(ast::CGen_boolean_and, linfo)
    "((" * from_expr(ast.lhs, linfo) * ") && (" * from_expr(ast.rhs, linfo) * "))"
end

function from_expr(ast::Expr, linfo)
    s = ""
    head = ast.head
    args = ast.args
    typ = ast.typ

    @dprintln(4, "from_expr = ", ast)
    if head == :block
        @dprintln(3,"Compiling block")
        s *= from_exprs(args, linfo)

    elseif head == :body
        @dprintln(3,"Compiling body")
        if include_rand && (contains(string(ast),"rand") || contains(string(ast),"randn"))
            s *= "std::random_device cgen_rand_device;\n"
            s *= "std::uniform_real_distribution<double> cgen_distribution(0.0,1.0);\n"
            s *= "std::normal_distribution<double> cgen_n_distribution(0.0,1.0);\n"
            if USE_OMP==1
                s *= "std::vector<std::default_random_engine> cgen_rand_generator;\n"
                s *= "for(int i=0; i<omp_get_max_threads(); i++) { cgen_rand_generator.push_back(std::default_random_engine(cgen_rand_device()));}\n"
            else
                s *= "std::default_random_engine cgen_rand_generator(cgen_rand_device());\n"
            end
        end
        s *= from_exprs(args, linfo)

    elseif head == :new
        @dprintln(3,"Compiling new")
        s *= from_new(args, linfo)

    elseif head == :lambda
        @dprintln(3,"Compiling lambda")
        s *= from_lambda(ast)

    elseif head == :(=)
        @dprintln(3,"Compiling assignment ", ast)
        s *= from_assignment(args, linfo)

    elseif head == :(&)
        @dprintln(3, "Compiling ref")
        s *= from_ref(args, linfo)

    elseif head == :invoke || head == :call
        @dprintln(3,"Compiling call")
        fun  = getCallFunction(ast)
        args = getCallArguments(ast)
        s *= from_call([fun; args], linfo, typ)

    elseif head == :foreigncall
        @dprintln(3,"Compiling foreigncall")
        fun  = getCallFunction(ast)
        args = getCallArguments(ast)
        s *= from_foreigncall([fun; args], linfo, typ)

    elseif head == :call1
        @dprintln(3,"Compiling call1")
        s *= from_call1(args, linfo, typ)

    elseif head == :return
        @dprintln(3,"Compiling return")
        s *= from_return(args, linfo)

    elseif head == :line
        s *= from_line(args, linfo)

    elseif head == :gotoifnot
        @dprintln(3,"Compiling gotoifnot : ", args)
        s *= from_gotoifnot(args, linfo)
    # :if and :comparison should only come from ?: expressions we generate
    # normal ifs should be inlined to gotoifnot
    elseif head == :if
        @dprintln(3,"Compiling if : ", args)
        s *= from_if(args, linfo)
    elseif head == :comparison
        @dprintln(3,"Compiling comparison : ", args)
        s *= from_comparison(args, linfo)

    elseif head == :parfor_start
        s *= from_parforstart(args, linfo)

    elseif head == :parfor_end
        s *= from_parforend(args, linfo)

    elseif head == :insert_divisible_task
        s *= from_insert_divisible_task(args, linfo)

    elseif head == :boundscheck || head == :inbounds
        # Nothing
    elseif head == :assert
        # Nothing
    elseif head == :simdloop
        # Nothing

    # For now, we ignore meta nodes.
    elseif head == :meta
        # Nothing

    elseif head == :loophead
        s *= from_loophead(args, linfo)

    elseif head == :loopend
        s *= from_loopend(args, linfo)

    elseif head == :parallel_loophead
        s *= from_parallel_loophead(args, linfo)

    elseif head == :parallel_loopend
        s *= from_parallel_loopend(args, linfo)

    # type_goto is "a virtual control flow edge used to convey
    # type data to static_typeof, also to be removed."  We can
    # safely ignore it.
    elseif head == :type_goto
        #Nothing

    else
        @dprintln(3,"Unknown head in expression: ", head)
        throw("Unknown head")
    end
    s
end


function fromRHSVar(ast::Symbol, linfo)
    s = from_symbol(ast, linfo)
end

function fromRHSVar(ast::GenSym, linfo)
    s = string(typeof(ast)) * string(ast.id)
end

function from_expr(ast::Union{Symbol,RHSVar}, linfo)
    sym = lookupVariableName(ast, linfo)
    fromRHSVar(sym, linfo)
end

function from_expr(ast::LineNumberNode, linfo)
    s = from_linenumbernode(ast, linfo)
end

function from_expr(ast::LabelNode, linfo)
    s = from_labelnode(ast, linfo)
end

function from_expr(ast::GotoNode, linfo)
    s = from_gotonode(ast, linfo)
end

function from_expr(ast::TopNode, linfo)
    s = from_topnode(ast, linfo)
end

function from_expr(ast::QuoteNode, linfo)
    # All QuoteNode should have been explicitly handled, otherwise we translate the value inside
    return from_expr(ast.value, linfo)
end

function from_expr(ast::NewvarNode, linfo)
    s = from_newvarnode(ast, linfo)
end

function from_expr(ast::GlobalRef, linfo)
    s = from_globalref(ast, linfo)
end

function from_expr(ast::Type, linfo)
    s = "{}" # because Type are translated to empty struct
#    if isPrimitiveJuliaType(ast)
#        s = "(" * toCtype(ast) * ")"
#    else
#        throw("Unknown julia type")
#    end
    s
end

function from_expr(ast::Char, linfo)
    #s = "'$(string(ast))'"
    # We do not handle widechar in C, force a conversion here
    s = ast > Char(255) ? "(char)L" : ""
    buf=IOBuffer()
    show(buf, ast)
    s *= takebuf_string(buf)
    s
end

function from_expr(ast::Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Float16, Float32,Float64,Bool,Char,Void}, linfo)
    if (ast === Inf)
      "DBL_MAX"
    elseif (ast === Inf32)
      "FLT_MAX"
    elseif (ast === -Inf)
      "DBL_MIN"
    elseif (ast === -Inf32)
      "FLT_MIN"
    elseif isa(ast, Int64) && (ast >= (1<<31) || ast < -(1<<31))
      string("0x", hex(ast), "LL")
    elseif isa(ast, UInt64) && (ast >= (1<<32))
      string("0x", hex(ast), "ULL")
    else
      string(ast)
    end
end

function from_expr(ast::Complex64, linfo)
    "std::complex<float>(" * from_expr(ast.re, linfo) * " + " * from_expr(ast.im, linfo) * ")"
end

function from_expr(ast::Complex128, linfo)
    "std::complex<double>(" * from_expr(ast.re, linfo) * ", " * from_expr(ast.im, linfo) * ")"
end

function from_expr(ast::Complex, linfo)
    toCtype(typeof(ast)) * "{" * from_expr(ast.re, linfo) * ", " * from_expr(ast.im, linfo) * "}"
end

function from_expr(ast::AbstractString, linfo)
    return "\"$ast\""
end

function from_expr(ast::ANY, linfo)
    #s *= dispatch(lstate.adp, ast, ast)
    asttyp = typeof(ast)
    @dprintln(3,"Unknown node type encountered: ", ast, " with type: ", asttyp)
    throw(string("CGen Error: Could not translate node: ", ast, " with type: ", asttyp))
end

function resolveFunction(func::Symbol, mod::Module, typs)
    return Base.getfield(mod, func)
end

function resolveFunction(func::Symbol, typs)
    return Base.getfield(Main, func)
end

function resolveFunction_O(func::Symbol, typs)
    curModule = Base.function_module(func, typs)
    while true
        if Base.isdefined(curModule, func)
            return Base.getfield(curModule, func)
        elseif curModule == Main
            break
        else
            curModule = Base.module_parent(curModule)
        end
    end
    if Base.isdefined(Base, func)
        return Base.getfield(Base, func)
    elseif Base.isdefined(Main, func)
        return Base.getfield(Main, func)
    end
    throw(string("Unable to resolve function ", string(func)))
end

# When a function has varargs, we specialize and destructure them
# into normal singular args (ie., we get rid of the var args expression) to
# match the signature at the callsite.
# However, the body still expects a packed representation and may perform
# operations such as |getfield|. So we pack them here again.

function from_varargpack(vargs, linfo)
    @dprintln(3, "vargs = ", vargs)
    args = vargs[1]
    vsym = from_expr(args[1], linfo)
    vtyps = args[2]
    @dprintln(3, "vsym = ", vsym, " vtyps = ", vtyps)
    assert(vtyps <: Tuple)
    # skip producing the vararg declaration when its type list is empty
    isempty(vtyps.types) ? "" : toCtype(vtyps) * " " * vsym * " = " *
        "{" * mapfoldl((i) -> vsym * string(i), (a, b) -> "$a, $b", 1:length(vtyps.types)) * "};"
end

function from_formalargs(params, vararglist, unaliased, linfo)
    s = ""
    ql = unaliased ? "__restrict" : ""
    @dprintln(3,"Compiling formal args: ", params)
    dumpSymbolTable(lstate.symboltable)
    argtypes = []
    for p in 1:length(params)
        @dprintln(3,"Doing param $p: ", params[p])
        @dprintln(3,"Type is: ", typeof(params[p]))
        if isa(params[p], Expr)
            # We may have a varags expression
            assert(isa(params[p].args[1], Symbol))
            @dprintln(3,"varargs type: ", params[p], lookupSymbolType(params[p].args[1], linfo))
            varargtyp = lookupSymbolType(params[p].args[1], linfo)
            for i in 1:length(varargtyp.types)
                vtyp = varargtyp.types[i]
                cvtyp = toCtype(vtyp)
                push!(argtypes, cvtyp)
                s = isempty(s) ? s : s * ", "
                s *= cvtyp * ((isArrayType(vtyp) ? "&" : "")
                * (isArrayType(vtyp) ? " $ql " : " ")
                * canonicalize(params[p].args[1]) * string(i))
            end
            if !isPrimitiveJuliaType(varargtyp) && !isArrayOfPrimitiveJuliaType(varargtyp)
                if !haskey(lstate.globalUDTs, varargtyp)
                    @dprintln(3, "from_formalargs adding to globalUDTs ", varargtyp)
                    lstate.globalUDTs[varargtyp] = 1
                    push!(lstate.globalUDTsOrder, varargtyp)
                end
            end
        elseif inSymbolTable(params[p], linfo)
            s = isempty(s) ? s : s * ", "
            typ = lookupSymbolType(params[p], linfo)
            ptyp = toCtype(typ)
            push!(argtypes, ptyp)
            is_array = isArrayType(typ) || isStringType(typ)
            s *= ptyp * ((is_array && !CGEN_RAW_ARRAY_MODE ? "&" : "")
                * (is_array ? " $ql " : " ")
                * canonicalize(params[p]))
        else
            throw("Could not translate formal argument: " * string(params[p]))
        end
    end
    @dprintln(3,"Formal args are: ", s)
    s, argtypes
end

function from_newvarnode(args, linfo)
    ""
end

function from_callee(ast::Expr, functionName::AbstractString, linfo)
    @dprintln(3,"Ast = ", ast)
    @dprintln(3,"Starting processing for $ast")
    linfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
    params = CompilerTools.LambdaHandling.getInputParametersAsExpr(linfo)
    typ = toCtype(CompilerTools.LambdaHandling.getReturnType(linfo))
    f = Dict(ast => functionName)
    bod = from_expr(body, linfo)
    args, argtypes = from_formalargs(params, [], false, linfo)
    dumpSymbolTable(lstate.symboltable)
    s = "$typ $functionName($args) { $bod } "
    s
end



function isScalarType(typ::Type)
    !(isArrayType(typ) || isCompositeType(typ) || isStringType(typ))
end

createMain = false
function setCreateMain(val)
    global createMain = val
end

recreateLoops = false
function setRecreateLoops(val)
    global recreateLoops = val
end

recreateConds = false
function setRecreateConds(val)
    global recreateConds = val
end

# Creates an entrypoint that dispatches onto host or MIC.
# For now, emit host path only
function createEntryPointWrapper(functionName, params, args, jtyp, argtypes, alias_check = nothing)
    @dprintln(3,"createEntryPointWrapper params = ", params, ", args = (", args, ") jtyp = ", jtyp, " argtypes = ", argtypes)
    assert(length(params) == length(argtypes))
    if length(params) > 0
        paramstr = mapfoldl(canonicalize, (a,b) -> "$a, $b", params)
    else
        paramstr = ""
    end
    # length(jtyp) == 0 means the special case of Void/nothing return so add nothing extra to actualParams in that case.
    retParams = length(jtyp) == 0 ? "" : foldl((a, b) -> "$a, $b",
        [(isScalarType(jtyp[i]) ? "" : "*") * "ret" * string(i-1) for i in 1:length(jtyp)])
    @dprintln(3, " params = (", paramstr, ") retParams = (", retParams, ")")
    actualParams = paramstr * ((length(paramstr) > 0 && length(retParams) > 0) ? ", " : "") * retParams
    @dprintln(3, " actualParams = (", actualParams, ")")
    wrapperParams = "int run_where"
    if length(args) > 0
        wrapperParams *= ", $args"
    end
    allocResult = ""
    retSlot = ""
    if length(jtyp) > 0
        retSlot = ", "
        for i in 1:length(jtyp)
            delim = i < length(jtyp) ? ", " : ""
            retSlot *= toCtype(jtyp[i]) *
                (isScalarType(jtyp[i]) ? "" : "*") * "* __restrict ret" * string(i-1) * delim
            if isArrayType(jtyp[i]) || isStringType(jtyp[i])
                typ = toCtype(jtyp[i])
                allocResult *= "*ret" * string(i-1) * " = new $typ();\n"
            end
        end
    end
    #printf(\"Starting execution of CGen generated code\\n\");
    #printf(\"End of execution of CGen generated code\\n\");

    genMain = ""
    genMainParam = ""
    if createMain
       genMainParam = ", bool genMain = true"
       genMain *= "if (genMain) {\n"
       genMain *= "++main_count;\n"
       genMain *= "std::stringstream newMain;\n"
       genMain *= "std::stringstream newMainData;\n"
       genMain *= "std::stringstream newMainSh;\n"
       genMain *= "std::stringstream newMainExe;\n"
       genMain *= "newMain << \"main\" << main_count << \".cc\";\n"
       genMain *= "newMainData << \"main\" << main_count << \".data\";\n"
       genMain *= "newMainSh << \"main\" << main_count << \".sh\";\n"
       genMain *= "newMainExe << \"main\" << main_count;\n"
       genMain *= "std::cout << \"Main will be generated in file \" << newMain.str() << std::endl;\n"
       genMain *= "std::cout << \"Data for main will be in file \" << newMainData.str() << std::endl;\n"
       genMain *= "std::cout << \"Script to compile is in \" << newMainSh.str() << std::endl;\n"
       # ---------------------------------------------------------------------------------------------
       genMain *= "std::ofstream mainFileData(newMainData.str(), std::ios::out | std::ios::binary);\n"
       genMain *= "mainFileData << run_where << std::endl;\n"
       for i = 1:length(params)
           genMain *= "mainFileData << " * canonicalize(params[i]) * " << std::endl;\n"
       end
       genMain *= "mainFileData.close();\n"
       # ---------------------------------------------------------------------------------------------
       genMain *= "std::ofstream mainFile(newMain.str());\n"
       genMain *= "mainFile << \"#include \\\"\" << __FILE__ << \"\\\"\" << std::endl;\n"
       genMain *= "mainFile << \"int main(int argc, char *argv[]) {\" << std::endl;\n"
       if isDistributedMode()
           genMain *= "mainFile << \"    MPI_Init(&argc, &argv);\" << std::endl;\n"
       end
       genMain *= "mainFile << \"    std::ifstream mainFileData(\\\"\" << newMainData.str() << \"\\\", std::ios::in | std::ios::binary);\" << std::endl;\n"
       genMain *= "mainFile << \"    int runwhere;\" << std::endl;\n"
       genMain *= "mainFile << \"    mainFileData >> runwhere;\" << std::endl;\n"
       for i in 1:length(jtyp)
           genMain *= "mainFile << \"    " * toCtype(jtyp[i]) * (isScalarType(jtyp[i]) ? "" : "*") * " ret" * string(i-1) * ";\" << std::endl;\n"
       end
       for i = 1:length(argtypes)
           genMain *= "mainFile << \"    " * argtypes[i] * " " * canonicalize(params[i]) * ";\" << std::endl;\n"
           genMain *= "mainFile << \"    mainFileData >> " * canonicalize(params[i]) * ";\" << std::endl;\n"
       end
       genMain *= "mainFile << \"    _$(functionName)_(runwhere"
       for i = 1:length(params)
           genMain *= ", " * canonicalize(params[i])
       end
       for i in 1:length(jtyp)
           genMain *= ", &ret" * string(i-1)
       end
       genMain *= ", false);\" << std::endl;\n"
       if isDistributedMode()
           genMain *= "mainFile << \"    MPI_Finalize();\" << std::endl;\n"
       end
       genMain *= "mainFile << \"    return 0;\" << std::endl;\n"
       genMain *= "mainFile << \"}\" << std::endl;\n"
       genMain *= "mainFile.close();\n"
       # ---------------------------------------------------------------------------------------------
       genMain *= "std::ofstream mainFileSh(newMainSh.str());\n"
       genMain *= "mainFileSh << \"#!/bin/sh\" << std::endl;\n"
       before, after = getShellBase()
       genMain *= "mainFileSh << \"$before -o \" << newMainExe.str() << \" \" << newMain.str() << \" $after \" << std::endl;\n"
       genMain *= "mainFileSh.close();\n"
       # ---------------------------------------------------------------------------------------------
       genMain *= "}\n"
    end

    unaliased_func = functionName * "_unaliased"
    unaliased_func_call = "$unaliased_func($actualParams);"

    # OMP offload only works for unaliased calls
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.OFFLOAD1_MODE ||
       ParallelAccelerator.getPseMode() == ParallelAccelerator.OFFLOAD2_MODE
        paramoffstr = ""
        declstr = ""
        initstr = ""
        retstr = ""
        memstr = ""
        outstr = ""
        inoutstr = ""
        # parameter related processing, we treat j2c-arrays differently
        nparams = length(params)
        for i = 1:nparams
            if inSymbolTable(params[i], linfo)
                sep = i < nparams ? ", " : ""
                typ = lookupSymbolType(params[i], linfo)
                pname = canonicalize(params[i])
                tname = toCtype(typ)
                if isArrayType(typ)
                    vname = "tmparr" * string(i)
                    declstr *= "uintptr_t $vname = $pname.to_mic(run_where);\n"
                    paramoffstr *= "*($tname*)$vname" * sep
                    memstr *= "delete ($tname*)$vname;\n"
                else
                    paramoffstr *= pname * sep
                end
            else
              error("Root function cannot have non-Symbol parameter: ", params[i])
            end
        end
        if length(jtyp) > 0 && nparams > 0
            paramoffstr *= ", "
        end
        # return type related processing
        for i = 1:length(jtyp)
            sep = i < length(jtyp) ? ", " : ""
            typ = jtyp[i]
            j = string(i-1)
            rname = "ret" * j
            if isArrayType(typ)
                vname = "retval" * j
                tname = toCtype(typ)
                pname = "tmpret" * j
                declstr *= "uintptr_t $vname;\n"
                initstr *= "$tname *$pname = new $tname();\n"
                initstr *= "$vname = (uintptr_t)$pname;\n"
                outstr *= vname * sep
                paramoffstr *= pname * sep
                retstr *= "**$rname = $tname::from_mic(run_where, $vname);\n"
                memstr *= "$tname *$pname = ($tname*)$vname;\n"
                memstr *= "delete $pname;\n"
            else
                paramoffstr *= rname * sep
            end
        end
        if length(outstr) != ""
            outstr = "out(" * outstr * ")"
        end
         unaliased_func_call =
        "if (run_where >= 0) {
           $declstr
           #pragma offload target(mic:run_where) $outstr
           {
             $initstr
             $unaliased_func($paramoffstr);
           }
           $retstr
           #pragma offload target(mic:run_where)
           {
             $memstr
           }
         }
         else {
           $unaliased_func($actualParams);
         }"
    end

    # If we are forcing vectorization then we will not emit the alias check
    emitaliascheck = (vectorizationlevel == VECDEFAULT ? true : false)
    s = ""
    if emitaliascheck && alias_check != nothing
        assert(isa(alias_check, AbstractString))

        s *=
        "extern \"C\" void _$(functionName)_($wrapperParams $retSlot $genMainParam) {\n
            $genMain
            $allocResult
            if ($alias_check) {
                $functionName($actualParams);
            } else {
                $unaliased_func_call
            }
        }\n"
    else
        s *=
    "extern \"C\" void _$(functionName)_($wrapperParams $retSlot $genMainParam) {\n
        $genMain
        $allocResult
        $functionName($actualParams);
    }\n"
    end
    s
end

function set_includes(ast)
    s = string(ast)
    if contains(s,"gemm_wrapper!") || contains(s,"gemv!") || contains(s,"transpose!") || contains(s,"vecnorm") || contains(s,"transpose")
        set_include_blas(true)
    end
    if contains(s,"LinAlg.chol") ||  contains(s,"LAPACK")
        set_include_lapack(true)
    end
    if contains(s,"rand") || contains(s,"randn")
        global include_rand = true
    end
    if contains(s,"HDF5")
        global USE_HDF5 = 1
    end
    if contains(s,"__hpat_Kmeans") || contains(s,"__hpat_LinearRegression") || contains(s,"__hpat_NaiveBayes")
        global USE_DAAL = 1
    end
end

function check_params(emitunaliasedroots, params, linfo)
    # Find varargs expression if present
    global lstate
    vararglist = []
    num_array_params = 0
    canAliasCheck = insertAliasCheck
    array_list = ""
    for k in params
        if isa(k, Expr) # If k is a vararg expression
            canAliasCheck = false
            @dprintln(3,"vararg expr: ", k, k.args[1], k.head)
            if isa(k.args[1], Symbol) && inSymbolTable(k.args[1], linfo)
                push!(vararglist, (k.args[1], lookupSymbolType(k.args[1], linfo)))
                @dprintln(3,lookupSymbolType(k.args[1], linfo))
                @dprintln(3,vararglist)
            end
        else
            assert(typeof(k) == Symbol)
            ptyp = lookupSymbolType(k, linfo)
            if isArrayType(ptyp)
#                if !isArrayOfPrimitiveJuliaType(ptyp)
#                    canAliasCheck = false
#                end
                if num_array_params > 0
                    array_list *= ","
                end
                array_list *= "&" * canonicalize(k)
                num_array_params += 1
            end
        end
    end
    @dprintln(3,"canAliasCheck = ", canAliasCheck, " array_list = ", array_list)
    if canAliasCheck && num_array_params > 0 && !CGEN_RAW_ARRAY_MODE
        alias_check = "j2c_alias_test<" * string(num_array_params) * ">({{" * array_list * "}})"
        @dprintln(3,"alias_check = ", alias_check)
    else
        alias_check = nothing
    end

    # Translate arguments
    args, argtypes = from_formalargs(params, vararglist, false, linfo)

    # If emitting unaliased versions, get "restrict"ed decls for arguments
    if emitunaliasedroots
       argsunal, argunaltypes = from_formalargs(params, vararglist, true, linfo)
    else
       argsunal = ""
       argunaltypes = []
    end

    vararg_bod = isempty(vararglist) ? "" : from_varargpack(vararglist, linfo)

    return vararg_bod, args, argsunal, alias_check, argtypes
end


# This is the entry point to CGen from the PSE driver
function from_root_entry(ast, functionName::AbstractString, argtyps, array_types_in_sig :: Dict{DataType,Int64} = Dict{DataType,Int64}())
    #assert(isfunctionhead(ast))
    global inEntryPoint
    inEntryPoint = true
    global lstate
    lstate = LambdaGlobalData()
    # If we are forcing vectorization then we will not emit the unaliased versions
    # of the roots
    emitunaliasedroots = (vectorizationlevel == VECDEFAULT ? true : false)

    @dprintln(3,"============ Compiling AST for ", functionName, " ============")
    @dprintln(3,"vectorizationlevel = ", vectorizationlevel)
    @dprintln(3,"emitunaliasedroots = ", emitunaliasedroots)
    @dprintln(1,"Ast = ", ast)

    set_includes(ast)
    if isa(ast, Tuple)
        (linfo, body) = ast
    else
        linfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
    end
    @dprintln(3, "LambdaVarInfo = ", linfo)
    @dprintln(3, "body = ", body)

    params = CompilerTools.LambdaHandling.getInputParametersAsExpr(linfo)
    returnType = CompilerTools.LambdaHandling.getReturnType(linfo)
    # Translate the body
    #bod = from_expr(ast, linfo)
    bod = from_lambda(linfo, body)

    if DEBUG_LVL>=3
        dumpSymbolTable(lstate.symboltable)
    end

    vararg_bod, args, argsunal, alias_check, argtypes = check_params(emitunaliasedroots, params, linfo)
    # don't generate unaliased versions if there are no array inputs or alias check is not possible
    if alias_check==nothing emitunaliasedroots=false end
    bod = vararg_bod * bod

    @dprintln(3,"returnType = ", returnType)
    if istupletyp(returnType)
        returnType = tuple(returnType.parameters...)
    elseif returnType == Void
        # We special case a single Void/nothing return value since it is common.
        # We ignore such Void returns on the CGen side and the proxy in driver.jl will force a "nothing" return.
        returnType = ()
    else
        returnType = (returnType,)
    end

    # Create an entry point that will be called by the Julia code.
    wrapper = (emitunaliasedroots ? createEntryPointWrapper(functionName * "_unaliased", params, argsunal, returnType, argtypes) : "") * createEntryPointWrapper(functionName, params, args, returnType, argtypes, alias_check)
    rtyp = "void"
    if length(returnType) > 0
        retargs = foldl((a, b) -> "$a, $b",
           [toCtype(returnType[i]) * " * __restrict ret" * string(i-1) for i in 1:length(returnType)])
    else
        # Must be the special case for Void/nothing so don't do anything here.
        retargs = ""
    end

    comma = (length(args) > 0 && length(retargs) > 0) ? ", " : ""
    args *= comma * retargs
    argsunal *= comma * retargs

    @dprintln(3, "args = (", args, ")")
    s = "$rtyp $functionName($args)\n{\n$bod\n}\n"
    s *= emitunaliasedroots ? "$rtyp $(functionName)_unaliased($argsunal)\n{\n$bod\n}\n" : ""
    setFunctionCompiled(functionName, argtyps)
    forwards, funcs = from_worklist()
    hdr = from_header(true, linfo)
    c = hdr * forwards * funcs * s * wrapper
    resetLambdaState(lstate)

    gen_j2c_array_new = "extern \"C\"\nvoid *j2c_array_new(int key, void*data, unsigned ndim, int64_t *dims) {\nvoid *a = NULL;\nswitch(key) {\n"
    for (key, value) in array_types_in_sig
        atyp = toCtype(key)
        elemtyp = toCtype(eltype(key))
        gen_j2c_array_new *= "case " * string(value) * ":\na = new " * atyp * "((" * elemtyp * "*)data, ndim, dims);\nbreak;\n"
    end
    gen_j2c_array_new *= "default:\nfprintf(stderr, \"j2c_array_new called with invalid key %d\", key);\nassert(false);\nbreak;\n}\nreturn a;\n}\n"
    c *= gen_j2c_array_new
    flush(STDOUT)
    c
end

# This is the entry point to CGen from the PSE driver
function from_root_nonentry(ast, functionName::AbstractString, argtyps, array_types_in_sig :: Dict{DataType,Int64} = Dict{DataType,Int64}())
    global inEntryPoint
    inEntryPoint = false
    global lstate
    @dprintln(1,"Ast = ", ast)
    @dprintln(3,"functionName = ", functionName)

    set_includes(ast)
    if isa(ast, Tuple)
        linfo, body = ast
    else
        linfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
    end
    @dprintln(3,"linfo = ", linfo)
    @dprintln(3,"body = ", body)

    params = CompilerTools.LambdaHandling.getInputParametersAsExpr(linfo)
    returnType = CompilerTools.LambdaHandling.getReturnType(linfo)
    # Translate the body
    #bod = from_expr(ast, linfo)
    bod = from_lambda(linfo, body)

    vararg_bod, args, argsunal, alias_check, argtypes = check_params(false, params, linfo)
    bod = vararg_bod * bod

    #hdr = from_header(false)
    # Create an entry point that will be called by the Julia code.
    rtyp = toCtype(returnType)

    @dprintln(3, "args = (", args, ")")
    s = "$rtyp $functionName($args)\n{\n$bod\n}\n"
    forwarddecl = "$rtyp $functionName($args);\n"
    setFunctionCompiled(functionName, argtyps)
    if length(array_types_in_sig) > 0
        @dprintln(3, "Non-empty array_types_in_sig for non-entry point.")
    end
    forwarddecl, s
end

function insert(func::Any, mod::Any, name, typs)
    if mod == ""
        insert(func, name, typs)
        return
    end
    target = resolveFunction(func, mod, typs)
    @dprintln(3,"Resolved function ", func, " : ", name, " : ", typs)
    insert(target, name, typs)
end

function insert(func::TopNode, name, typs)
    target = eval(func)
    @dprintln(3,"Resolved function ", func, " : ", name, " : ", typs, " target ", target)
    insert(target, name, typs)
end

function insert(func::GlobalRef, name, typs)
    target = resolveFunction(func.name, func.mod, typs)
    @dprintln(3,"Resolved function ", func, " : ", name, " : ", typs, " target ", target)
    insert(target, name, typs)
end

function insert(func::Symbol, name, typs)
    target = resolveFunction(func, typs)
    @dprintln(3,"Resolved function ", func, " : ", name, " : ", typs, " target ", target)
    insert(target, name, typs)
end

funcToLambdaInfo(func, typs) = ParallelAccelerator.Driver.code_typed(func, typs)

function insert(func::Function, name, typs)
    global lstate
    ast = funcToLambdaInfo(func, typs)
    if !isFunctionCompiled(name,typs)
        @dprintln(3, "Adding function ", name, " to worklist. ", ast, " ", typeof(ast))
        push!(lstate.worklist, (ast, name, typs))
    end
end

function insert(func::IntrinsicFunction, name, typs)
    global lstate
    ast = funcToLambdaInfo(func, typs)
    if !isFunctionCompiled(name,typs)
        @dprintln(3, "Adding intrinsic function ", name, " to worklist.")
        push!(lstate.worklist, (ast, name, typs))
    end
end

# Translate function nodes in breadth-first order
function from_worklist()
    f = ""
    s = ""
    global lstate
    while !isempty(lstate.worklist)
        a, fname, typs = splice!(lstate.worklist, 1)
        @dprintln(3,"Checking if we compiled ", fname, " before ", typeof(a), " ", typs)
        @dprintln(3,lstate.compiledfunctions)
        if isFunctionCompiled(fname,typs)
            @dprintln(3,"Yes, skipping compilation...")
            continue
        end
        @dprintln(3,"No, compiling it now")
        empty!(lstate.symboltable)
        empty!(lstate.ompprivatelist)
        if isa(a, Symbol)
            @dprintln(3,"a is a Symbol, calling code_typed")
            a = funcToLambdaInfo(a, typs)
        end

		@dprintln(3,"============ Compiling AST for ", fname, " ============")
		fi, si = from_root_nonentry(a, fname, typs, Dict{DataType,Int64}())
		@dprintln(3,"============== C++ after compiling ", fname, " ===========")
		@dprintln(3,si)
		@dprintln(3,"============== End of C++ for ", fname, " ===========")
		@dprintln(3,"Adding ", (fname,typs), " to compiledFunctions")
		@dprintln(3,lstate.compiledfunctions)
		@dprintln(3,"Added ", (fname,typs), " to compiledFunctions")
		@dprintln(3,lstate.compiledfunctions)
		f *= fi
		s *= si
    end
    f, s
end

#
# Utility methods to write, compile and link generated code
#
import Base.write
function writec(s, outfile_name=nothing; with_headers=false)
    if outfile_name == nothing
        outfile_name = generate_new_file_name()
    end
    if !isdir(generated_file_dir)
        global generated_file_dir = mktempdir()
    end
    if with_headers
        s = from_header(true) * "extern \"C\" {\n" * s * "\n}"
    end
    cgenOutput = "$generated_file_dir/$outfile_name.cpp"
    cf = open(cgenOutput, "w")
    write(cf, s)
    @dprintln(3,"Done committing CGen code")
    close(cf)
    return outfile_name
end

function getGccName()
    if Compat.is_windows()
        return "x86_64-w64-mingw32-g++"
    else
        return "g++"
    end
end

function getShellBase(flags=[])
  # return an error if this is not overwritten with a valid compiler
  compileCommand = `echo "invalid backend compiler"`

  packageroot = getPackageRoot()
  # otherArgs = ["-DJ2C_REFCOUNT_DEBUG", "-DDEBUGJ2C"]
  otherArgs = flags

  Opts = ["-O3 "]
  if USE_DAAL==1
    DAALROOT=ENV["DAALROOT"]
    push!(Opts,"-I$DAALROOT/include ")
  end

  push!(Opts, "-std=c++11 ")

   link_Opts = flags
    linkLibs = []
    if include_blas==true
        if mkl_lib!=""
            push!(linkLibs,"-mkl ")
        elseif openblas_lib!=""
            push!(linkLibs,"-lopenblas ")
        elseif sys_blas==1
            push!(linkLibs,"-lblas ")
        end
    end
    if include_lapack==true
        if mkl_lib!=""
            push!(linkLibs,"-mkl ")
        end
    end
    if USE_HDF5==1
        if haskey(ENV,"HDF5_DIR")
            HDF5_DIR=ENV["HDF5_DIR"]
            push!(linkLibs,"-L$HDF5_DIR/lib ")
        end
#        if NERSC==1
#          HDF5_DIR=ENV["HDF5_DIR"]
#          push!(linkLibs,"-L$HDF5_DIR/lib ")
#      end
      #push!(linkLibs,"-L/usr/local/hdf5/lib -lhdf5 ")
      push!(linkLibs,"-lhdf5 ")
  end
  if USE_DAAL==1
      DAALROOT=ENV["DAALROOT"]
    push!(linkLibs,"$DAALROOT/lib/intel64_lin/libdaal_core.a ")
    if USE_OMP==1
        push!(linkLibs,"$DAALROOT/lib/intel64_lin/libdaal_thread.a ")
    else
        push!(linkLibs,"$DAALROOT/lib/intel64_lin/libdaal_sequential.a ")
    end
    push!(linkLibs,"-L$DAALROOT/../tbb/lib/intel64_lin/gcc4.4/ ")
    push!(linkLibs,"-ltbb ")
    push!(linkLibs,"-liomp5 ")
    push!(linkLibs,"-ldl ")
  end

  if backend_compiler == USE_ICC
    comp = "icpc"
    if isDistributedMode()
        if haskey(ENV,"HDF5_DIR")
            HDF5_DIR=ENV["HDF5_DIR"]
            push!(Opts, "-I$HDF5_DIR/include ")
        end
        if NERSC==1
            comp = "CC"
        else
            comp = "mpiicpc"
        end
    end
    vecOpts = (vectorizationlevel == VECDISABLE ? "-no-vec " : " ")
    if USE_OMP == 1 || USE_DAAL==1
        push!(Opts, "-qopenmp")
    end
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.OFFLOAD1_MODE ||
       ParallelAccelerator.getPseMode() == ParallelAccelerator.OFFLOAD2_MODE
        push!(Opts,"-DJ2C_ARRAY_OFFLOAD ")
        push!(Opts,"-qoffload-attribute-target=mic ")
    end
    # Generate dyn_lib
    return "$comp $(Opts...) -g $vecOpts -fpic $(otherArgs...)", "$(linkLibs...) -lm"
  elseif backend_compiler == USE_GCC
    comp = getGccName()
    if isDistributedMode()
        comp = "mpic++"
    end
    if USE_OMP == 1
        push!(Opts, "-fopenmp")
    end
    return "$comp $(Opts...) -g -fpic $(otherArgs...)", "$(linkLibs...) -lm"
  elseif backend_compiler == USE_MINGW
    throw(string("getShellBase implementation for mingw not completed."))
  end
end

function getCompileCommand(full_outfile_name, cgenOutput, flags=[])
  # return an error if this is not overwritten with a valid compiler
  compileCommand = `echo "invalid backend compiler"`

  packageroot = getPackageRoot()
  # otherArgs = ["-DJ2C_REFCOUNT_DEBUG", "-DDEBUGJ2C"]
  otherArgs = flags

  Opts = ["-O3"]
  for user_option in userOptions
      if user_option.compileFlags!=""
          push!(Opts, user_option.compileFlags)
      end
  end
  if USE_DAAL==1
    DAALROOT=ENV["DAALROOT"]
    push!(Opts,"-I$DAALROOT/include")
  end
  if backend_compiler == USE_ICC
    comp = "icpc"
    if isDistributedMode()
        if haskey(ENV,"HDF5_DIR")
            HDF5_DIR=ENV["HDF5_DIR"]
            push!(Opts, "-I$HDF5_DIR/include")
        end
        if NERSC==1
            comp = "CC"
        else
            comp = "mpiicpc"
        end
        # wd2593 turns simd errors to warnings
        # needed since #simd loops may have gotos
        push!(Opts, "-qopenmp-simd") # No parallelization, just vectorization.
        push!(Opts, "-wd2593")
    end
    vecOpts = (vectorizationlevel == VECDISABLE ? "-no-vec" : [])
    if USE_OMP == 1 || USE_DAAL==1
        push!(Opts, "-qopenmp")
    end
    if mkl_lib!=""
        push!(Opts,"-mkl")
    end
    if ParallelAccelerator.getPseMode() == ParallelAccelerator.OFFLOAD1_MODE ||
       ParallelAccelerator.getPseMode() == ParallelAccelerator.OFFLOAD2_MODE
        push!(Opts,"-DJ2C_ARRAY_OFFLOAD")
        push!(Opts,"-qoffload-attribute-target=mic")
    end
    # Generate dyn_lib
    compileCommand = `$comp $Opts -std=c++11 -g $vecOpts -fpic -c -o $full_outfile_name $otherArgs $cgenOutput`
  elseif backend_compiler == USE_GCC
    comp = getGccName()
    if isDistributedMode()
        comp = "mpic++"
    end
    if USE_OMP == 1
        push!(Opts, "-fopenmp")
    end
    push!(Opts, "-std=c++11")
    compileCommand = `$comp $Opts -g -fpic -c -o $full_outfile_name $otherArgs $cgenOutput`
  elseif backend_compiler == USE_MINGW
    gpp = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","bin","g++")
    RPMbindir = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","bin")
    incdir = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","include")

    push!(Base.Libdl.DL_LOAD_PATH,RPMbindir)
    ENV["PATH"]=ENV["PATH"]*";"*RPMbindir

    if USE_OMP == 1
        push!(Opts, "-fopenmp")
    end
    push!(Opts, "-std=c++11")
    compileCommand = `$gpp $Opts -g -fpic -I $incdir -c -o $full_outfile_name $otherArgs $cgenOutput`
  end

  return compileCommand
end

function compile(outfile_name; flags=[])
    global use_bcpp
    packageroot = getPackageRoot()

    cgenOutput = "$generated_file_dir/$outfile_name.cpp"

    if isDistributedMode() && MPI.Comm_rank(MPI.COMM_WORLD)==0
        println("Distributed-memory MPI mode.")
    end
    if !isDistributedMode() || MPI.Comm_rank(MPI.COMM_WORLD)==0

        if USE_OMP==0 && (!isDistributedMode() || MPI.Comm_rank(MPI.COMM_WORLD)==0)
            println("OpenMP is not used.")
        end

        if use_bcpp == 1
            cgenOutputTmp = "$generated_file_dir/$(outfile_name)_tmp.cpp"
            run(`cp $cgenOutput $cgenOutputTmp`)
            # make cpp code readable
            beautifyCommand = `bcpp $cgenOutputTmp $cgenOutput`
            if DEBUG_LVL < 1
                beautifyCommand = pipeline(beautifyCommand, stdout=DevNull, stderr=DevNull)
            end
            run(beautifyCommand)
        end

        full_outfile_name = `$generated_file_dir/$outfile_name.o`
        compileCommand = getCompileCommand(full_outfile_name, cgenOutput, flags)
        @dprintln(1,"Compilation command = ", compileCommand)
        run(compileCommand)
    end
end

function getLinkCommand(outfile_name, lib, flags=[])
    # return an error if this is not overwritten with a valid compiler
    linkCommand = `echo "invalid backend linker"`

    Opts = flags
    linkLibs = []
    for user_option in userOptions
        if user_option.linkFlags!=""
            push!(linkLibs, user_option.linkFlags)
        end
    end
    if include_blas==true
        if mkl_lib!=""
            push!(linkLibs,"-mkl")
        elseif openblas_lib!=""
            push!(linkLibs,"-lopenblas")
        elseif sys_blas==1
            push!(linkLibs,"-lblas")
        end
    end
    if include_lapack==true
        if mkl_lib!=""
            push!(linkLibs,"-mkl")
        end
    end
    if USE_HDF5==1
        if haskey(ENV,"HDF5_DIR")
            HDF5_DIR=ENV["HDF5_DIR"]
            push!(linkLibs,"-L$HDF5_DIR/lib")
        end
#        if NERSC==1
#          HDF5_DIR=ENV["HDF5_DIR"]
#          push!(linkLibs,"-L$HDF5_DIR/lib")
#      end
      #push!(linkLibs,"-L/usr/local/hdf5/lib -lhdf5")
      push!(linkLibs,"-lhdf5")
  end
  if USE_DAAL==1
      DAALROOT=ENV["DAALROOT"]
    push!(linkLibs,"$DAALROOT/lib/intel64_lin/libdaal_core.a")
    if USE_OMP==1
        push!(linkLibs,"$DAALROOT/lib/intel64_lin/libdaal_thread.a")
    else
        push!(linkLibs,"$DAALROOT/lib/intel64_lin/libdaal_sequential.a")
    end
    push!(linkLibs,"-L$DAALROOT/../tbb/lib/intel64_lin/gcc4.4/")
    push!(linkLibs,"-ltbb")
    push!(linkLibs,"-liomp5")
    push!(linkLibs,"-ldl")
  end
  if backend_compiler == USE_ICC
    comp = "icpc"
    if isDistributedMode()
        if NERSC==1
            comp = "CC"
        else
            comp = "mpiicpc"
        end
    end
    if USE_OMP==1 || USE_DAAL==1
        push!(Opts,"-qopenmp")
    end
    linkCommand = `$comp -g -shared $Opts -o $lib $generated_file_dir/$outfile_name.o $linkLibs -lm`
  elseif backend_compiler == USE_GCC
    comp = getGccName()
    if isDistributedMode()
        comp = "mpic++"
    end
    if USE_OMP==1
        push!(Opts,"-fopenmp")
    end
    push!(Opts, "-std=c++11")
    linkCommand = `$comp -g -shared $Opts -o $lib $generated_file_dir/$outfile_name.o $linkLibs -lm`
  elseif backend_compiler == USE_MINGW
    gpp = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","bin","g++")
    RPMbindir = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","bin")

    push!(Base.Libdl.DL_LOAD_PATH,RPMbindir)
    ENV["PATH"]=ENV["PATH"]*";"*RPMbindir

    if USE_OMP == 1
        push!(Opts, "-fopenmp")
    end
    push!(Opts, "-std=c++11")
    linkCommand = `$gpp -static-libgcc -static-libstdc++ -g -shared $Opts -o $lib $generated_file_dir/$outfile_name.o $linkLibs -lm -Wl,-static -lgomp -Wl,-Bdynamic -lpthread`
  end

  return linkCommand
end


function link(outfile_name; flags=[])
    if Compat.is_windows()
        lib = "$generated_file_dir/lib$outfile_name.dll"
    else
        lib = "$generated_file_dir/lib$outfile_name.so.1.0"
    end

    if !isDistributedMode() || MPI.Comm_rank(MPI.COMM_WORLD)==0
        linkCommand = getLinkCommand(outfile_name, lib, flags)
        @dprintln(1,"Link command = ", linkCommand)
        run(linkCommand)
        @dprintln(3,"Done CGen linking")
    end
    if isDistributedMode()
        MPI.Barrier(MPI.COMM_WORLD)
    end
    return lib
end

end # CGen module
