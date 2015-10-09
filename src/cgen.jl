#
# A prototype Julia to C++ generator
# Jaswanth Sreeram (jaswanth.sreeram@intel.com)
#

module cgen
using ..ParallelIR
using CompilerTools
export setvectorizationlevel, generate, from_root, writec, compile, link, set_debug_level, set_include_blas
import ParallelAccelerator, ..getPackageRoot

# uncomment this line for using Debug.jl
#using Debug

# This controls the debug print level.
DEBUG_LVL=0
const ENABLE_DEBUG = true

function set_debug_level(x)
    global DEBUG_LVL = x
end

if ENABLE_DEBUG==true
  # A debug print routine.
  function dprint(level,msgs...)
    if(DEBUG_LVL >= level)
      print(msgs...)
    end
  end

  # A debug print routine.
  function dprintln(level,msgs...)
    if(DEBUG_LVL >= level)
      println(msgs...)
    end
  end
else

  function dprintln(args...)
  end

  function dprint(args...)
  end

end



#=
type ASTDispatcher
  nodes::Array{Any, 1}
  dt::Dict{Any, Any}
  m::Module
	function ASTDispatcher()
		d = Dict{Any, Any}()
		n = [Expr, Symbol, SymbolNode, LineNumberNode, LabelNode,
			GotoNode, TopNode, QuoteNode, NewvarNode,
			isdefined(:GetfieldNode) ? GetfieldNode : 
				(isdefined(:GlobalRef) ? GlobalRef :
				throw(string("Neither GetfieldNode or GlobalRef defined."))), 
			:block, :body, :new, :lambda, :(=), :call, :call1,
			:return, :line, :gotoifnot, :parfor_start, :parfor_end,
			:boundscheck, :loophead, :loopend]
		for x in n
			assert(isa(x, DataType) || isa(x, Symbol))
			d[x] = symbol("from_" * lowercase(string(x)))
		end
		new(n, d, current_module())
	end	
end


function dispatch(a::ASTDispatcher, node::Any, args)
	tgt = typeof(node)
	if !haskey(a.dt, tgt)
		#dumpSymbolTable(a.dt)
		dprintln(3,"ERROR: Unexpected node: ", node, " with type = ", tgt)
		throw("Could not dispatch node")
	end
	dprintln(3,"Dispatching to call: ", a.dt[tgt], " with args: ", args)
	getfield(a.m, a.dt[tgt])(args)
end
=#

type LambdaGlobalData
	#adp::ASTDispatcher
	ompprivatelist::Array{Any, 1}
	globalUDTs::Dict{Any, Any}
	symboltable::Dict{Any, Any}
  tupleTable::Dict{Any, Array{Any,1}} # a table holding tuple values to be used for hvcat allocation
	compiledfunctions::Array{Any, 1}
	worklist::Array{Any, 1}
	jtypes::Dict{Any, Any}
	ompdepth::Int64
	function LambdaGlobalData()
		_j = Dict(
			Int8	=>	"int8_t",
			UInt8	=>	"uint8_t",
			Int16	=>	"int16_t",
			UInt16	=>	"uint16_t",
			Int32	=>	"int32_t",
			UInt32	=>	"uint32_t",
			Int64	=>	"int64_t",
			UInt64	=>	"uint64_t",
			Float16	=>	"float",
			Float32	=>	"float",
			Float64	=>	"double",
			Bool	=>	"bool",
			Char	=>	"char",
            Void    =>  "void"
	)

		#new(ASTDispatcher(), [], Dict(), Dict(), [], [])
		new([], Dict(), Dict(), Dict(), [], [], _j, 0)
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
const USE_ICC = 0
const USE_GCC = 1

# Globals
inEntryPoint = false
lstate = nothing
backend_compiler = USE_ICC
use_bcpp = 0
package_root = getPackageRoot()
#config file overrides backend_compiler variable
if isfile("$package_root/deps/generated/config.jl")
  include("$package_root/deps/generated/config.jl")
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

# Reset and reuse the LambdaGlobalData object across function
# frames
function resetLambdaState(l::LambdaGlobalData)
	empty!(l.ompprivatelist)
	empty!(l.globalUDTs)
	empty!(l.symboltable)
	empty!(l.worklist)
	inEntryPoint = false
	l.ompdepth = 0
end


# These are primitive operators on scalars and arrays
_operators = ["*", "/", "+", "-", "<", ">"]
# These are primitive "methods" for scalars and arrays
_builtins = ["getindex", "getindex!", "setindex", "setindex!", "arrayref", "top", "box", 
			"unbox", "tuple", "arraysize", "arraylen", "ccall",
			"arrayset", "getfield", "unsafe_arrayref", "unsafe_arrayset",
			"safe_arrayref", "safe_arrayset", "tupleref",
			"call1", ":jl_alloc_array_1d", ":jl_alloc_array_2d", "nfields",
			"_unsafe_setindex!", ":jl_new_array", "unsafe_getindex", "steprange_last",
            ":jl_array_ptr", "sizeof"
]

# Intrinsics
_Intrinsics = [
		"===",
		"box", "unbox",
        #arithmetic
        "neg_int", "add_int", "sub_int", "mul_int", "sle_int",
		"xor_int", "and_int", "or_int",
        "sdiv_int", "udiv_int", "srem_int", "urem_int", "smod_int",
        "neg_float", "add_float", "sub_float", "mul_float", "div_float",
		"rem_float", "sqrt_llvm", "fma_float", "muladd_float",
		"le_float", "ne_float",
        "fptoui", "fptosi", "uitofp", "sitofp", "not_int",
		"nan_dom_err", "lt_float", "slt_int", "abs_float", "select_value",
		"fptrunc", "fpext", "trunc_llvm", "floor_llvm", "rint_llvm", 
		"trunc", "ceil_llvm", "ceil", "pow", "powf", "lshr_int",
		"checked_ssub", "checked_sadd", "flipsign_int", "check_top_bit", "shl_int", "ctpop_int",
        "checked_trunc_uint", "checked_trunc_sint"
]

tokenXlate = Dict(
	'*' => "star",
	'/' => "slash",
	'-' => "minus",
	'!' => "bang",
	'.' => "dot"
)

replacedTokens = Set("#")
scrubbedTokens = Set(",.({}):")

file_counter = -1

#### End of globals ####

function generate_new_file_name()
    global file_counter
    file_counter += 1
    return "cgen_output$file_counter"
end

function cleanup_generated_files()
    package_root = getPackageRoot()
    # TODO: Check debug level and save generated files
    if false
        for file in readdir("$package_root/deps/generated")
            if file == ".gitignore"
                continue
            end
            rm("$package_root/deps/generated/$file")
        end
    end
end

atexit(cleanup_generated_files)

function __init__()
	packageroot = joinpath(dirname(@__FILE__), "..")
end

function getenv(var::AbstractString)
  ENV[var]
end

# Emit declarations and "include" directives
function from_header(isEntryPoint::Bool)
	s = from_UDTs()
	isEntryPoint ? from_includes() * s : s
end

function from_includes()
	packageroot = getPackageRoot()
    blas_include = ""
    if include_blas != false
        libblas = Base.libblas_name
        if contains(libblas, "mkl")
            blas_include = "#include <mkl.h>\n"
        else
            blas_include = "#include <cblas.h>\n"
        end
    end
	reduce(*, "", (
        blas_include,
		"#include <omp.h>\n",
		"#include <stdint.h>\n",
		"#include <math.h>\n",
		"#include <stdio.h>\n",
		"#include <iostream>\n",
		"#include \"$packageroot/deps/include/pse-runtime.h\"\n",
		"#include \"$packageroot/deps/include/j2c-array.h\"\n",
		"#include \"$packageroot/deps/include/j2c-array-pert.h\"\n",
		"#include \"$packageroot/deps/include/pse-types.h\"\n")
	)
end

# Iterate over all the user defined types (UDTs) in a function
# and emit a C++ type declaration for each
function from_UDTs()
	global lstate
	isempty(lstate.globalUDTs) ? "" : mapfoldl((a) -> (lstate.globalUDTs[a] == 1 ? from_decl(a) : ""), (a, b) -> "$a; $b", keys(lstate.globalUDTs))
end

# Tuples are represented as structs
function from_decl(k::Tuple)
	s = "typedef struct {\n"
	for i in 1:length(k)
		s *= toCtype(k[i]) * " " * "f" * string(i-1) * ";\n"
	end
	s *= "} Tuple" * 
		(!isempty(k) ? mapfoldl((a) -> canonicalize(a), (a, b) -> "$(a)$(b)", k) : "") * ";\n"
	if haskey(lstate.globalUDTs, k)
		lstate.globalUDTs[k] = 0
	end
	s
end

# Generic declaration emitter for non-primitive Julia DataTypes
function from_decl(k::DataType)
	if is(k, UnitRange{Int64})
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
		s *= "} Tuple" * (!isempty(ptyp) ? mapfoldl((a) -> canonicalize(a), (a, b) -> "$(a)$(b)", ptyp) : "") * ";\n"
		return s
	else
		if haskey(lstate.globalUDTs, k)
			lstate.globalUDTs[k] = 0
		end
		ftyps = k.types
		fnames = fieldnames(k)
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

function from_decl(k)
	return toCtype(lstate.symboltable[k]) * " " * canonicalize(k) * ";\n"
end

function isCompositeType(t)
	# TODO: Expand this to real UDTs
	b = isa(t, Tuple) || is(t, UnitRange{Int64}) || is(t, StepRange{Int64, Int64})
	b |= isa(t, DataType) && t.name == Tuple.name
	b
end

function lambdaparams(ast)
	CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(ast).input_params
end

function from_lambda(ast, args)
	s = ""
	linfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(ast)
	params = linfo.input_params
	vars = linfo.var_defs
	gensyms = linfo.gen_sym_typs

	decls = ""
	global lstate
	# Populate the symbol table	
	for k in keys(vars)
		v = vars[k] # v is a VarDef
		lstate.symboltable[k] = v.typ
		if !in(k, params) && (v.desc & 32 != 0)
			push!(lstate.ompprivatelist, k)
		end 
	end
	
	for k in 1:length(gensyms)
		lstate.symboltable[GenSym(k-1)] = gensyms[k]
	end
	bod = from_expr(args[3])
	dprintln(3,"lambda params = ", params)
	dprintln(3,"lambda vars = ", vars)
	dumpSymbolTable(lstate.symboltable)
	
	for k in keys(lstate.symboltable)
		if !in(k, params) #|| (!in(k, locals) && !in(k, params))
			# If we have user defined types, record them
			#if isCompositeType(lstate.symboltable[k]) || isUDT(lstate.symboltable[k])
			if !isPrimitiveJuliaType(lstate.symboltable[k]) && !isArrayOfPrimitiveJuliaType(lstate.symboltable[k])
				if !haskey(lstate.globalUDTs, lstate.symboltable[k])
					lstate.globalUDTs[lstate.symboltable[k]] = 1
				end
			end
			decls *= toCtype(lstate.symboltable[k]) * " " * canonicalize(k) * ";\n"
		end
	end
	decls * bod
end


function from_exprs(args::Array)
	s = ""
	for a in args
		se = from_expr(a)
		s *= se * (!isempty(se) ? ";\n" : "")
	end
	s
end


function dumpSymbolTable(a::Dict{Any, Any})
	dprintln(3,"SymbolTable: ")
	for x in keys(a)
		dprintln(3,x, " ==> ", a[x])
	end
end

function dumpDecls(a::Array{Dict{Any, ASCIIString}})
	for x in a
		for k in keys(x)
			dprintln(3,x[k], " ", k)
		end
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

function from_assignment(args::Array)
	global lstate
	lhs = args[1]
	rhs = args[2]

  # if assignment is var = (...)::tuple, add var to tupleTable to be used for hvcat allocation
  if isa(rhs,Expr) && rhs.head==:call && isa(rhs.args[1],TopNode) && rhs.args[1].name==:tuple
    dprintln(3,"Found tuple assignment: ", lhs," ", rhs)
    lstate.tupleTable[lhs] = rhs.args[2:end]
  end

  # if this is a hvcat call, the array should be allocated and initialized
  if isa(rhs,Expr) && rhs.head==:call && isa(rhs.args[1],TopNode) && rhs.args[1].name==:typed_hvcat
    dprintln(3,"Found hvcat assignment: ", lhs," ", rhs)
    s = ""
    
    @assert isa(rhs.args[2], GlobalRef) && rhs.args[2].mod==Main "Cgen expects hvcat with simple types in GlobalRef form, e.g. Main.Float64"
    typ = toCtype(eval(rhs.args[2].name))

    arr_info = rhs.args[3]
    rows = Int64[]
    if isa(arr_info, Expr) && arr_info.head==:call
      rows = arrinfo[2:end]
    else
      rows = lstate.tupleTable[arr_info]
    end
    nr = length(rows)
    nc = rows[1] # all rows should have the same size
    s *= from_expr(lhs) * " = j2c_array<$typ>::new_j2c_array_2d(NULL, $nr, $nc);\n"
    values = rhs.args[4:end]
    s *= mapfoldl((i) -> from_setindex([lhs,values[i],convert(Int64,ceil(i/nr)),(i-1)%nr+1])*";", (a, b) -> "$a $b", 1:length(values))
    return s
  end

  if isa(rhs,Expr) && rhs.head==:call && isa(rhs.args[1],GlobalRef) && rhs.args[1].name==:cat_t
    dims = rhs.args[2]
    @assert dims==2 "cgen: only 2d cat_t() is supported now"
    size = length(rhs.args[4:end])
    typ = toCtype(eval(rhs.args[3].name))
    s = ""
    s *= from_expr(lhs) * " = j2c_array<$typ>::new_j2c_array_$(dims)d(NULL, 1,$size);\n"
    values = rhs.args[4:end]
    s *= mapfoldl((i) -> from_setindex([lhs,values[i],i])*";", (a, b) -> "$a $b", 1:length(values))
    return s
  end

	lhsO = from_expr(lhs)
	rhsO = from_expr(rhs)

  if !typeAvailable(lhs)
		if typeAvailable(rhs)
			lstate.symboltable[lhs] = rhs.typ
		elseif haskey(lstate.symboltable, rhs)
			lstate.symboltable[lhs] = lstate.symboltable[rhs]
		elseif isPrimitiveJuliaType(typeof(rhs))
			lstate.symboltable[lhs] = typeof(rhs)
		elseif isPrimitiveJuliaType(typeof(rhsO))
			lstate.symboltable[lhs] = typeof(rhs0)
		else
			dprintln(3,"Unknown type in assignment: ", args)
			throw("FATAL error....exiting")
		end
		# Try to refine type for certain stmts with a call in the rhs
		# that doesn't have a type
		if typeAvailable(rhs) && is(rhs.typ, Any) &&
		hasfield(rhs, :head) && (is(rhs.head, :call) || is(rhs.head, :call1))
			m, f, t = resolveCallTarget(rhs.args)
			f = string(f)
			if f == "fpext"
				dprintln(3,"Args: ", rhs.args, " type = ", typeof(rhs.args[2]))
				lstate.symboltable[lhs] = eval(rhs.args[2])
				dprintln(3,"Emitting :", rhs.args[2])
				dprintln(3,"Set type to : ", lstate.symboltable[lhs])
			end
		end
	end
	lhsO * " = " * rhsO
end

function parseArrayType(arrayType)
	return eltype(arrayType), ndims(arrayType)
end

function isPrimitiveJuliaType(t)
	haskey(lstate.jtypes, t)
end

function isArrayOfPrimitiveJuliaType(t)
  if isArrayType(t) && isPrimitiveJuliaType(eltype(t))
    return true
  end
  return false
end


function isArrayType(typ)
	isa(typ, DataType) && ((typ.name == Array.name) || (typ.name == BitArray.name))
end

function isPtrType(typ)
    isa(typ, DataType) && typ.name == Ptr.name
end

function toCtype(typ::Tuple)
	return "Tuple" * mapfoldl((a) -> canonicalize(a), (a, b) -> "$(a)$(b)", typ)
end

# Generate a C++ type name for a Julia type
function toCtype(typ)
	if haskey(lstate.jtypes, typ)
		return lstate.jtypes[typ]
	elseif isArrayType(typ)	
		atyp, dims = parseArrayType(typ)
		atyp = toCtype(atyp)
		assert(dims >= 0)
		return " j2c_array< $(atyp) > "
    elseif isPtrType(typ)
        return "$(toCtype(eltype(typ))) *"
	elseif in(:parameters, fieldnames(typ)) && length(typ.parameters) != 0
		# For parameteric types, for now assume we have equivalent C++
		# implementations
		btyp, ptyps = parseParametricType(typ)
		return canonicalize(btyp) * mapfoldl((a) -> canonicalize(a), (a, b) -> a * b, ptyps)
	else
		return canonicalize(typ)
	end
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
	s = replace(s, r"^[^a-zA-Z]", "_")
	s = replace(s, replacedTokens, "p")
    s = replace(s, "!", "bang")
    s = replace(s, "∇", "del")
	s
end

function parseParametricType(typ)
	assert(isa(typ, DataType))
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

function from_tupleref(args)
	# We could generate std::tuples instead of structs
	from_expr(args[1]) * ".f" * string(int(from_expr(args[2]))-1)
end

function from_safegetindex(args)
	s = ""
	src = from_expr(args[1])
	s *= src * ".SAFEARRAYELEM("
	idxs = map((i)->from_expr(i), args[2:end])
	for i in 1:length(idxs)
		s *= idxs[i] * (i < length(idxs) ? "," : "")
	end
	s *= ")"
	s
end

function from_getindex(args)
	s = ""
	src = from_expr(args[1])
	s *= src * ".ARRAYELEM("
	idxs = map((i)->from_expr(i), args[2:end])
	for i in 1:length(idxs)
		s *= idxs[i] * (i < length(idxs) ? "," : "")
	end
	s *= ")"
	s
end

function from_setindex(args)
	s = ""
	src = from_expr(args[1])
	s *= src * ".ARRAYELEM("
	idxs = map((i)->from_expr(i), args[3:end])
	for i in 1:length(idxs)
		s *= idxs[i] * (i < length(idxs) ? "," : "")
	end
	s *= ") = " * from_expr(args[2])
	s
end



# unsafe_setindex! has several variations. Here we handle only one.
# For this (and other related) array indexing methods we could just do
# codegen but we would have to extend the j2c_array runtime to support
# all the other operations allowed on Julia arrays

function from_unsafe_setindex!(args)
	@assert (length(args) == 4) "Array operation unsafe_setindex! has unexpected number of arguments"
	@assert !isArrayType(args[2]) "Array operation unsafe_setindex! has unexpected format"
	src = from_expr(args[2])
	v = from_expr(args[3])
	mask = from_expr(args[4])
	"for(uint64_t i = 1; i < $(src).ARRAYLEN(); i++) {\n\t if($(mask)[i]) \n\t $(src)[i] = $(v);}\n"
end

function from_tuple(args)
	"{" * mapfoldl((a) -> from_expr(a), (a, b) -> "$a, $b", args) * "}" 
end

function from_arraysize(args)
	s = from_expr(args[1])
	if length(args) == 1
		s *= ".ARRAYLEN()"
	else
		s *= ".ARRAYSIZE(" * from_expr(args[2]) * ")"
	end
	s
end

function from_ccall(args)
	dprintln(3,"ccall args:")
	dprintln(3,"target tuple: ", args[1], " - ", typeof(args[1]))
	dprintln(3,"return type: ", args[2])
	dprintln(3,"input types tuple: ", args[3])
	dprintln(3,"inputs tuple: ", args[4:end])
	for i in 1:length(args)
		dprintln(3,args[i])
	end
	dprintln(3,"End of ccall args")
	fun = args[1]
	if isInlineable(fun, args)
		return from_inlineable(fun, args)
	end

	if isa(fun, QuoteNode)
		s = from_expr(fun)
	elseif isa(fun, Expr) && (is(fun.head, :call1) || is(fun.head, :call))
		s = canonicalize(string(fun.args[2])) 
		dprintln(3,"ccall target: ", s)
	else
		throw("Invalid ccall format...")
	end
	s *= "("
	numInputs = length(args[3].args)-1
	argsStart = 4
	argsEnd = length(args)
    if contains(s, "cblas") && contains(s, "gemm")
        s *= "(CBLAS_LAYOUT) $(from_expr(args[4])), "
        s *= "(CBLAS_TRANSPOSE) $(from_expr(args[6])), "
        s *= "(CBLAS_TRANSPOSE) $(from_expr(args[8])), "
        argsStart = 10
    end
	s *= mapfoldl((a)->from_expr(a), (a, b)-> "$a, $b", args[argsStart:2:end])
	s *= ")"
	dprintln(3,"from_ccall: ", s)
	s
end

function from_arrayset(args)
	idxs = mapfoldl((a) -> from_expr(a), (a, b) -> "$a, $b", args[3:end]) 
	src = from_expr(args[1])
	val = from_expr(args[2])
	"$src.ARRAYELEM($idxs) = $val"
end

function istupletyp(typ)                                                                               
    isa(typ, DataType) && is(typ.name, Tuple.name) 
end 

function from_getfield(args)
	dprintln(3,"from_getfield args are: ", args)
	tgt = from_expr(args[1])
    if isa(args[1], SymbolNode)
      args1typ = args[1].typ
    elseif isa(args[1], GenSym)
      args1typ = lstate.symboltable[args[1]]
    else
	  throw("Unhandled argument 1 type to getfield")
    end
	#if istupletyp(args1typ) && isPrimitiveJuliaType(eltype(args1typ))
	if istupletyp(args1typ)
		eltyp = toCtype(eltype(args1typ))
		return "(($eltyp *)&$(tgt))[" * from_expr(args[2]) * " - 1]"
	end
	throw(string("Unhandled call to getfield ",args1typ, " ", eltype(args1typ)))
	#=
	mod, tgt = resolveCallTarget(args[1], args[2:end])
	if mod == "Intrinsics"
		return from_expr(tgt)
	elseif isInlineable(tgt, args[2:end])
		return from_inlineable(tgt, args[2:end])
	end
	from_expr(mod) * "." * from_expr(tgt)
	=#
end

function from_nfields(args)
	dprintln(3,"Args are: ", args)
	dprintln(3,"Args type = ", typeof(args))
	string(nfields(args[1].typ))
end

function from_steprange_last(args)
  start = from_expr(args[1])
  step = from_expr(args[2])
  stop = from_expr(args[3])
  return "("*stop*"-("*stop*"-"*start*")%"*step*")"
end

function from_arrayalloc(args)
	dprintln(3,"Array alloc args:")
	map((i)->dprintln(3,args[i]), 1:length(args))
	dprintln(3,"----")
	dprintln(3,"Parsing array type: ", args[4])
	typ, dims = parseArrayType(args[4])
	dprintln(3,"Array alloc typ = ", typ)
	dprintln(3,"Array alloc dims = ", dims)
	typ = toCtype(typ)
	dprintln(3,"Array alloc after ctype conversion typ = ", typ)
	shp = []
	for i in 1:dims
		push!(shp, from_expr(args[6+(i-1)*2]))
	end
	shp = foldl((a, b) -> "$a, $b", shp)
	return "j2c_array<$typ>::new_j2c_array_$(dims)d(NULL, $shp);\n" 
end

function from_builtins_comp(f, args)
    tgt = string(f)
    return eval(parse("from_$cmd()"))
end

function from_array_ptr(args)
    return "$(from_expr(args[4])).data"
end

function from_sizeof(args)
    return "sizeof($(toCtype(args[1])))"
end

function from_builtins(f, args)
	tgt = string(f)
	if tgt == "getindex" || tgt == "getindex!"
		return from_getindex(args)
	elseif tgt == "setindex" || tgt == "setindex!"
		return from_setindex(args)
	elseif tgt == "top"
		return ""
	elseif tgt == "box"
		return from_box(args)
	elseif tgt == "arrayref"
		return from_getindex(args)
	elseif tgt == "tupleref"
		return from_tupleref(args)
	elseif tgt == "tuple"
		return from_tuple(args)
	elseif tgt == "arraylen"
		return from_arraysize(args)
	elseif tgt == "arraysize"
		return from_arraysize(args)
	elseif tgt == "ccall"
		return from_ccall(args)
	elseif tgt == "arrayset"
		return from_arrayset(args)
	elseif tgt == ":jl_alloc_array_1d" || tgt == ":jl_new_array"
		return from_arrayalloc(args)
	elseif tgt == ":jl_alloc_array_2d"
		return from_arrayalloc(args)
    elseif tgt == ":jl_array_ptr"
        return from_array_ptr(args)
	elseif tgt == "getfield"
		return from_getfield(args)
	elseif tgt == "unsafe_arrayref"
		return from_getindex(args) 
	elseif tgt == "safe_arrayref"
		return from_safegetindex(args) 
	elseif tgt == "unsafe_arrayset" || tgt == "safe_arrayset"
		return from_setindex(args)
	elseif tgt == "_unsafe_setindex!"
		return from_unsafe_setindex!(args) 
	elseif tgt == "nfields"
		return from_nfields(args)
    elseif tgt == "sizeof"
        return from_sizeof(args)
  elseif tgt =="steprange_last"
    return from_steprange_last(args)
	end

	
	dprintln(3,"Compiling ", string(f))
	throw("Unimplemented builtin")
end

function from_box(args)
	s = ""
	typ = args[1]
	val = args[2]
	s *= from_expr(val)
	s
end

function from_intrinsic(f :: ANY, args)
	intr = string(f)
	dprintln(3,"Intrinsic ", intr, ", args are ", args)

	if intr == "mul_int"
		return "($(from_expr(args[1]))) * ($(from_expr(args[2])))"
	elseif intr == "neg_int"
		return "-" * "(" * from_expr(args[1]) * ")"
	elseif intr == "mul_float"
		return "($(from_expr(args[1]))) * ($(from_expr(args[2])))"
	elseif intr == "urem_int"
		return "($(from_expr(args[1]))) % ($(from_expr(args[2])))"
	elseif intr == "add_int"
		return "($(from_expr(args[1]))) + ($(from_expr(args[2])))"
	elseif intr == "or_int"
		return "($(from_expr(args[1]))) | ($(from_expr(args[2])))"
	elseif intr == "xor_int"
		return "($(from_expr(args[1]))) ^ ($(from_expr(args[2])))"
	elseif intr == "and_int"
		return "($(from_expr(args[1]))) & ($(from_expr(args[2])))"
	elseif intr == "sub_int"
		return "($(from_expr(args[1]))) - ($(from_expr(args[2])))"
	elseif intr == "slt_int"
		return "($(from_expr(args[1]))) < ($(from_expr(args[2])))"
	elseif intr == "sle_int"
		return "($(from_expr(args[1]))) <= ($(from_expr(args[2])))"
	elseif intr == "lshr_int"
		return "($(from_expr(args[1]))) >> ($(from_expr(args[2])))"
	elseif intr == "shl_int"
		return "($(from_expr(args[1]))) << ($(from_expr(args[2])))"
	elseif intr == "checked_ssub"
		return "($(from_expr(args[1]))) - ($(from_expr(args[2])))"
	elseif intr == "checked_sadd"
		return "($(from_expr(args[1]))) + ($(from_expr(args[2])))"
	elseif intr == "srem_int"
		return "($(from_expr(args[1]))) % ($(from_expr(args[2])))"
	#TODO: Check if flip semantics are the same as Julia codegen.
	# For now, we emit unary negation
	elseif intr == "flipsign_int"
        return "-" * "(" * from_expr(args[1]) * ")"
	elseif intr == "check_top_bit"
		typ = typeof(args[1])
		if !isPrimitiveJuliaType(typ)
			if hasfield(args[1], :typ)
				typ = args[1].typ
			end
		end
		nshfts = 8*sizeof(typ) - 1
		oprnd = from_expr(args[1])
        return oprnd * " >> " * string(nshfts) * " == 0 ? " * oprnd * " : " * oprnd
	elseif intr == "select_value"
		return "(" * from_expr(args[1]) * ")" * " ? " *
		"(" * from_expr(args[2]) * ") : " * "(" * from_expr(args[3]) * ")"
	elseif intr == "not_int"
		return "!" * "(" * from_expr(args[1]) * ")"
	elseif intr == "ctpop_int"
		return "__builtin_popcount" * "(" * from_expr(args[1]) * ")"
	elseif intr == "add_float"
		return "($(from_expr(args[1]))) + ($(from_expr(args[2])))"
	elseif intr == "lt_float"
		return "($(from_expr(args[1]))) < ($(from_expr(args[2])))"
	elseif intr == "ne_float"
		return "($(from_expr(args[1]))) != ($(from_expr(args[2])))"
	elseif intr == "le_float"
		return "($(from_expr(args[1]))) <= ($(from_expr(args[2])))"
	elseif intr == "neg_float"
		return "-($(from_expr(args[1])))"
	elseif intr == "abs_float"
		return "fabs(" * from_expr(args[1]) * ")"
	elseif intr == "sqrt_llvm"
		return "sqrt(" * from_expr(args[1]) * ")"
	elseif intr == "sub_float"
		return "($(from_expr(args[1]))) - ($(from_expr(args[2])))"
	elseif intr == "div_float"
		return "(" * from_expr(args[1]) * ")" * 
			" / " * "(" * from_expr(args[2]) * ")"
	elseif intr == "sitofp"
		return from_expr(args[1]) * from_expr(args[2])
	elseif intr == "fptrunc" || intr == "fpext"
		dprintln(3,"Args = ", args)
		return "(" * toCtype(eval(args[1])) * ")" * from_expr(args[2])
	elseif intr == "trunc_llvm" || intr == "trunc"
		return from_expr(args[1]) * "trunc(" * from_expr(args[2]) * ")"
	elseif intr == "floor_llvm" || intr == "floor"
		return "floor(" * from_expr(args[1]) * ")"
	elseif intr == "ceil_llvm" || intr == "ceil"
		return "ceil(" * from_expr(args[1]) * ")"
	elseif intr == "rint_llvm" || intr == "rint"
		return "round(" * from_expr(args[1]) * ")"
	elseif f == :(===)
		return "(" * from_expr(args[1]) * " == " * from_expr(args[2]) * ")"
	elseif f == "pow"
		return "pow(" * from_expr(args[1]) * ", " * from_expr(args[2]) * ")"
	elseif f == "powf"
		return "powf(" * from_expr(args[1]) * ", " * from_expr(args[2]) * ")"
	elseif intr == "nan_dom_err"
		dprintln(3,"nan_dom_err is: ")
		for i in 1:length(args)
			dprintln(3,args[i])
		end
		#return "assert(" * "isNan(" * from_expr(args[1]) * ") && !isNan(" * from_expr(args[2]) * "))"
		return from_expr(args[1])
    elseif intr in ["checked_trunc_uint", "checked_trunc_sint"]
        return "$(from_expr(args[1])) $(from_expr(args[2]))"
	else
		dprintln(3,"Intrinsic ", intr, " is known but no translation available")
		throw("Unhandled Intrinsic...")
	end
end

function from_inlineable(f, args)
	dprintln(3,"Checking if ", f, " can be inlined")
	dprintln(3,"Args are: ", args)
	if has(_operators, string(f))
		return "(" * from_expr(args[1]) * string(f) * from_expr(args[2]) * ")"
	elseif has(_builtins, string(f))
		return from_builtins(f, args)
	elseif has(_Intrinsics, string(f))
		return from_intrinsic(f, args)
	else
		throw("Unknown Operator or Method encountered: " * string(f))
	end
end

function isInlineable(f, args)
	if has(_operators, string(f)) || has(_builtins, string(f)) || has(_Intrinsics, string(f))
		return true
	end
	return false
end

function arrayToTuple(a)
	ntuple((i)->a[i], length(a))
end

function from_symbol(ast)
    if ast == :Inf
        return "INFINITY"
    end
	hasfield(ast, :name) ? canonicalize(string(ast.name)) : canonicalize(ast)
end

function from_symbolnode(ast)
	canonicalize(string(ast.name))
end

function from_linenumbernode(ast)
	""
end

function from_labelnode(ast)
	"label" * string(ast.label) * " : "
end

function from_ref(args)
    "&$(from_expr(args[1]))"
end

function from_call1(ast::Array{Any, 1})
	dprintln(3,"Call1 args")
	s = ""
	for i in 2:length(ast)
		s *= from_expr(ast[i])
		dprintln(3,ast[i])
	end
	dprintln(3,"Done with call1 args")
	s
end

function isPendingCompilation(list, tgt)
	for i in 1:length(list)
		ast, name, typs = lstate.worklist[i]
		if name == tgt
			return true
		end
	end
	return false
end

function resolveCallTarget(args::Array{Any, 1})
	dprintln(3,"Trying to resolve target with args: ", args)
	M = ""
	t = ""
	s = ""
	#case 0:
	f = args[1]
	if isa(f, Symbol) && isInlineable(f, args[2:end])
		return M, string(f), from_inlineable(f, args[2:end])
	elseif isa(f, Symbol) && is(f, :call)
		#This means, we have a Base.call - if f is not a Function, this is translated to f(args)
		arglist = mapfoldl((a)->from_expr(a), (a,b)->"$a, $b", args[3:end])
		if isa(args[2], DataType)
			t = "{" * arglist * "}" 
		else
			t = from_expr(args[2]) * "(" * arglist * ")"	
		end
	elseif isa(f,Expr) && (is(f.head,:call) || is(f.head,:call1))
        if length(f.args) == 3 && isa(f.args[1], TopNode) && is(f.args[1].name,:getfield) && isa(f.args[3],QuoteNode)
			s = f.args[3].value
			if isa(f.args[2],Module)
				M = f.args[2]
			end
		end
		dprintln(3,"Case 0: Returning M = ", M, " s = ", s, " t = ", t)
	#case 1:
	elseif isa(args[1], TopNode) && is(args[1].name, :getfield) && isa(args[3], QuoteNode)
		dprintln(3,"Case 1: args[3] is ", args[3])
		s = args[3].value
		if isa(args[2], Module)
			M = args[2]
			dprintln(3,"Case 1: Returning M = ", M, " s = ", s, " t = ", t)
		else
			#case 2:
			M = ""
			s = ""
			t = from_expr(args[2]) * "." * from_expr(args[3])
			#M, _s = resolveCallTarget([args[2]])
			dprintln(3,"Case 1: Returning M = ", M, " s = ", s, " t = ", t)
		end
	elseif isa(args[1], TopNode) && is(args[1].name, :getfield) && hasfield(args[1], :head) && is(args[1].head, :call)
		return resolveCallTarget(args[1])
		
	elseif isdefined(:GetfieldNode) && isa(args[1],GetfieldNode) && isa(args[1].value,Module)
        M = args[1].value; s = args[1].name; t = ""

	elseif isdefined(:GlobalRef) && isa(args[1],GlobalRef) && isa(args[1].mod,Module)
        M = args[1].mod; s = args[1].name; t = ""

	# case 3:
	elseif isa(args[1], TopNode) && isInlineable(args[1].name, args[2:end])
		t = from_inlineable(args[1].name, args[2:end])
		dprintln(3,"Case 3: Returning M = ", M, " s = ", s, " t = ", t)
	end
	dprintln(3,"In resolveCallTarget: Returning M = ", M, " s = ", s, " t = ", t)
	return M, s, t
end

#=
function pattern_match_call(ast::Array{Any, 1})
	# math functions 
	libm_math_functions = Set([:sin, :cos, :tan, :asin, :acos, :acosh, :atanh, :log, :log2, :log10, :lgamma, :log1,:asinh,:atan,:cbrt,:cosh,:erf,:exp,:expm1,:sinh,:sqrt,:tanh])

	dprintln(3,"pattern matching ",ast)
	s = ""
	if( length(ast)==2 && typeof(ast[1])==Symbol && in(ast[1],libm_math_functions) && typeof(ast[2])==SymbolNode && (ast[2].typ==Float64 || ast[2].typ==Float32))
	  dprintln(3,"FOUND ", ast[1])
	  s = string(ast[1])*"("*from_expr(ast[2].name)*");"
	end
	s
end
=#

function from_call(ast::Array{Any, 1})

	dprintln(3,"Compiling call: ast = ", ast, " args are: ")
	for i in 1:length(ast)
		dprintln(3,"Arg ", i, " = ", ast[i], " type = ", typeof(ast[i]))
	end
	# Try and find the target of the call
	mod, fun, t = resolveCallTarget(ast)

	# resolveCallTarget will eagerly try to translate the call
	# if it can. If it does, then we are done.
	if !isempty(t)
		return t;
	end

	if fun == ""
		fun = ast[1]
	end

	args = ast[2:end]
	dprintln(3,"fun is: ", fun)
	dprintln(3,"call Args are: ", args)

	if isInlineable(fun, args)
		dprintln(3,"Doing with inlining ", fun, "(", args, ")")
		fs = from_inlineable(fun, args)
		return fs
	end
	dprintln(3,"Not inlinable")
	funStr = "_" * string(fun)	

	# If we have previously compiled this function
	# we fallthru and simply emit the call.
	# Else we lookup the function Symbol and enqueue it
	# TODO: This needs to specialize on types
	skipCompilation = has(lstate.compiledfunctions, funStr) ||
		isPendingCompilation(lstate.worklist, funStr)
	if(fun == :println)
		s =  "std::cout << "
		argTyps = []
		for a in 2:length(args)
			s *= from_expr(args[a]) * (a < length(args) ? "<<" : "")
			if !skipCompilation
				# Attempt to find type
				if typeAvailable(args[a])
					push!(argTyps, args[a].typ)
				elseif isPrimitiveJuliaType(typeof(args[a]))
					push!(argTyps, typeof(args[a]))
				elseif haskey(lstate.symboltable, args[a])
					push!(argTyps, lstate.symboltable[args[a]])
				end
			end
		end
		s *= "<< std::endl;"
		return s
	end


	s = ""
	map((i)->dprintln(3,i[2]), lstate.worklist)
	map((i)->dprintln(3,i), lstate.compiledfunctions)
	s *= "_" * from_expr(fun) * "("
	argTyps = []
	for a in 1:length(args)
		s *= from_expr(args[a]) * (a < length(args) ? "," : "")
		if !skipCompilation
			# Attempt to find type
			if typeAvailable(args[a])
				push!(argTyps, args[a].typ)
			elseif isPrimitiveJuliaType(typeof(args[a]))
				push!(argTyps, typeof(args[a]))
			elseif haskey(lstate.symboltable, args[a])
				push!(argTyps, lstate.symboltable[args[a]])
			end
		end
	end
	s *= ")"
	dprintln(3,"Finished translating call : ", s)
	dprintln(3,ast[1], " : ", typeof(ast[1]), " : ", hasfield(ast[1], :head) ? ast[1].head : "")
	if !skipCompilation && (isa(fun, Symbol) || isa(fun, Function) || isa(fun, TopNode) || isa(fun, GlobalRef))
		#=
		dprintln(3,"Worklist is: ")
		for i in 1:length(lstate.worklist)
			ast, name, typs = lstate.worklist[i]
			dprintln(3,name);
		end
		dprintln(3,"Compiled Functions are: ")
		for i in 1:length(lstate.compiledfunctions)
			name = lstate.compiledfunctions[i]
			dprintln(3,name);
		end
		=#
		dprintln(3,"Inserting: ", fun, " : ", "_" * canonicalize(fun), " : ", arrayToTuple(argTyps))
		insert(fun, mod, "_" * canonicalize(fun), arrayToTuple(argTyps))
	end
	s
end

# Generate return statements. The way we handle multiple return
# values is different depending on whether this return statement is in 
# the entry point or not. If it is, then the multiple return values are 
# pushed into spreallocated slots in the argument list (ret0, ret1...)
# If not, just return a tuple.

function from_return(args)
	global inEntryPoint
	dprintln(3,"Return args are: ", args)
	retExp = ""
    if length(args) == 0
        return "return"
    elseif inEntryPoint
		arg1 = args[1]
		arg1_typ = Any
		if typeAvailable(arg1)
			arg1_typ = arg1.typ
		elseif isa(arg1, GenSym)
			arg1_typ = lstate.symboltable[arg1]
		end
		if istupletyp(arg1_typ)
			retTyps = arg1_typ.parameters
			for i in 1:length(retTyps)
				retExp *= "*ret" * string(i-1) * " = " * from_expr(arg1) * ".f" * string(i-1) * ";\n"
			end
		else
			retExp = "*ret0 = " * from_expr(arg1) * ";\n"
		end
		return retExp * "return"
	else
		return "return " * from_expr(args[1])
	end
end


function from_gotonode(ast, exp = "")
	labelId = ast.label
	s = ""
	dprintln(3,"Compiling goto: ", exp, " ", typeof(exp))
	if isa(exp, Expr) || isa(exp, SymbolNode) || isa(exp, Symbol) || isa(exp, GenSym)
		s *= "if (!(" * from_expr(exp) * ")) "
	end
	s *= "goto " * "label" * string(labelId)
	s
end

function from_gotoifnot(args)
	exp = args[1]
	labelId = args[2]
	s = ""
	dprintln(3,"Compiling gotoifnot: ", exp, " ", typeof(exp))
	if isa(exp, Expr) || isa(exp, SymbolNode) || isa(exp, Symbol) || isa(exp, GenSym)
		s *= "if (!(" * from_expr(exp) * ")) "
	end
	s *= "goto " * "label" * string(labelId)
	s
end
#=
function from_goto(exp, labelId)
	s = ""
	dprintln(3,"Compiling goto: ", exp, " ", typeof(exp))
	if isa(exp, Expr) || isa(exp, SymbolNode) || isa(exp, Symbol)
		s *= "if (!(" * from_expr(exp) * ")) "
	end
	s *= "goto " * "label" * string(labelId)
	s
end
=#
function from_getfieldnode(ast)
	throw("Unexpected node: " * string(ast))
end

function from_globalref(ast)
	mod = ast.mod
	name = ast.name
	dprintln(3,"Name is: ", name, " and its type is:", typeof(name))
	from_expr(name)
end

function from_topnode(ast)
	canonicalize(ast)
end

function from_quotenode(ast)
	from_symbol(ast)
end

function from_line(args)
	""
end

function from_parforend(args)
	global lstate
	s = ""
	parfor = args[1]
	lpNests = parfor.loopNests
	for i in 1:length(lpNests)
		s *= "}\n"	
	end
	s *= lstate.ompdepth <=1 ? "}\n}/*parforend*/\n" : "" # end block introduced by private list
	dprintln(3,"Parforend = ", s)
	lstate.ompdepth -= 1
	s
end

function loopNestCount(loop)
	"(((" * from_expr(loop.upper) * ") + 1 - (" * from_expr(loop.lower) * ")) / (" * from_expr(loop.step) * "))"
end

#TODO: Implement task mode support here
function from_insert_divisible_task(args)
	inserttasknode = args[1]
	dprintln(3,"Ranges: ", inserttasknode.ranges)
	dprintln(3,"Args: ", inserttasknode.args)
	dprintln(3,"Task Func: ", inserttasknode.task_func)
	dprintln(3,"Join Func: ", inserttasknode.join_func)
	dprintln(3,"Task Options: ", inserttasknode.task_options)
	dprintln(3,"Host Grain Size: ", inserttasknode.host_grain_size)
	dprintln(3,"Phi Grain Size: ", inserttasknode.phi_grain_size)
	throw("Task mode is not supported yet")	
end

function from_loopnest(ivs, starts, stops, steps)
	vecclause = (vectorizationlevel == VECFORCE) ? "#pragma simd\n" : ""
	mapfoldl(
        (i) ->
            (i == length(ivs) ? vecclause : "") *
            "for ( $(ivs[i]) = $(starts[i]); $(ivs[i]) <= (int64_t)$(stops[i]); $(ivs[i]) += $(steps[i])) {\n",
            (a, b) -> "$a $b",
            1:length(ivs)
    )
end

# If the parfor body is too complicated then DomainIR or ParallelIR will set
# instruction_count_expr = nothing

# Meaning of num_threads_mode
# mode = 1 uses static insn count if it is there, but doesn't do dynamic estimation and fair core allocation between levels in a loop nest.
# mode = 2 does all of the above
# mode = 3 in addition to 2, also uses host minimum (0) and Phi minimum (10)

function from_parforstart(args)
	global lstate
	num_threads_mode = ParallelIR.num_threads_mode

	dprintln(3,"args: ",args);

	parfor = args[1]
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
	ivs = map((a)->from_expr(a.indexVariable), lpNests)
	starts = map((a)->from_expr(a.lower), lpNests)
	stops = map((a)->from_expr(a.upper), lpNests)
	steps = map((a)->from_expr(a.step), lpNests)

	dprintln(3,"ivs ",ivs);
	dprintln(3,"starts ", starts);
	dprintln(3,"stops ", stops);
	dprintln(3,"steps ", steps);

	# Generate the actual loop nest
	loopheaders = from_loopnest(ivs, starts, stops, steps)

	lstate.ompdepth += 1
	if lstate.ompdepth > 1
		return loopheaders
	end
	
	s = ""

	# Generate initializers and OpenMP clauses for reductions
	rds = parfor.reductions
	rdvars = rdinis = rdops = ""
	if !isempty(rds)
		rdvars = map((a)->from_expr(a.reductionVar), rds)
		rdinis = map((a)->from_expr(a.reductionVarInit), rds)
		rdops  = map((a)->string(a.reductionFunc), rds)
	end
	rdsprolog = rdsclause = ""
	for i in 1:length(rds)
		rdsprolog *= "$(rdvars[i]) = $(rdinis[i]);\n"
		rdsclause *= "reduction($(rdops[i]) : $(rdvars[i])) " 
	end
	

	# Check if there are private vars and emit the |private| clause
	dprintln(3,"Private Vars: ")
	dprintln(3,"-----")
	dprintln(3,private_vars)
	dprintln(3,"-----")
	privatevars = isempty(private_vars) ? "" : "private(" * mapfoldl((a) -> canonicalize(a), (a,b) -> "$a, $b", private_vars) * ")"

	
	lcountexpr = ""
	for i in 1:length(lpNests)
		lcountexpr *= "(((" * starts[i] * ") + 1 - (" * stops[i] * ")) / (" * steps[i] * "))" * (i == length(lpNests) ? "" : " * ")
	end
	preclause = ""
	nthreadsclause = ""
	instruction_count_expr = parfor.instruction_count_expr
	if num_threads_mode == 1
		if instruction_count_expr != nothing
			insncount = from_expr(instruction_count_expr)
			preclause = "unsigned _vnfntc = computeNumThreads(((unsigned)" * insncount * ") * (" * lcountexpr * "));\n";
			nthreadsclause = "num_threads(_vnfntc) "
		end
	elseif num_threads_mode == 2
		if instruction_count_expr != nothing
			insncount = from_expr(instruction_count_expr)
			preclause = "J2cParRegionThreadCount j2c_block_region_thread_count(std::min(unsigned($lcountexpr ),computeNumThreads(((unsigned) $insncount ) * ( $lcountexpr ))),__LINE__,__FILE__);\n"
		else
			preclause = "J2cParRegionThreadCount j2c_block_region_thread_count(" * lcountexpr * ",__LINE__,__FILE__);\n";
		end
		nthreadsclause = "num_threads(j2c_block_region_thread_count.getUsed()) "
	elseif num_threads_mode == 3
		if instruction_count_expr != nothing
			insncount = from_expr(instruction_count_expr)
			preclause = "J2cParRegionThreadCount j2c_block_region_thread_count(std::min(unsigned($lcountexpr),computeNumThreads(((unsigned) $insncount) * ($lcountexpr))),__LINE__,__FILE__, 0, 10);\n";
		else
			preclause = "J2cParRegionThreadCount j2c_block_region_thread_count($lcountexpr,__LINE__,__FILE__, 0, 10);\n";
		end
		nthreadsclause = "if(j2c_block_region_thread_count.runInPar()) num_threads(j2c_block_region_thread_count.getUsed()) ";
	end

	s *= rdsprolog * "{\n$preclause #pragma omp parallel $nthreadsclause $privatevars\n{\n"
	s *= "#pragma omp for private(" * mapfoldl((a)->a, (a, b)->"$a, $b", ivs) * ") $rdsclause\n"
	s *= loopheaders
	s
end

function from_parforstart_serial(args)
	parfor = args[1]
	lpNests = parfor.loopNests
	global lstate
	s = ""

	ivs = map((a)->from_expr(a.indexVariable), lpNests)
	starts = map((a)->from_expr(a.lower), lpNests)
	stops = map((a)->from_expr(a.upper), lpNests)
	steps = map((a)->from_expr(a.step), lpNests)

	s *= "{\n{\n" * mapfoldl(
			(i) -> "for ( $(ivs[i]) = $(starts[i]); $(ivs[i]) <= $(stops[i]); $(ivs[i]) += $(steps[i])) {\n",
			(a, b) -> "$a $b",
			1:length(lpNests))
	s
end

# TODO: Should simple objects be heap allocated ?
# For now, we stick with stack allocation
function from_new(args)
	s = ""
	typ = args[1] #type of the object
	if isa(typ, DataType)
		objtyp, ptyps = parseParametricType(typ)
		ctyp = canonicalize(objtyp) * mapfoldl((a) -> canonicalize(a), (a, b) -> a * b, ptyps)
		s = ctyp * "{"
		s *= mapfoldl((a) -> from_expr(a), (a, b) -> "$a, $b", args[2:end]) * "}" 
	elseif isa(typ.args[1], TopNode) && typ.args[1].name == :getfield
		typ = getfield(typ.args[2], typ.args[3].value)
		objtyp, ptyps = parseParametricType(typ)
		ctyp = canonicalize(objtyp) * (isempty(ptyps) ? "" : mapfoldl((a) -> canonicalize(a), (a, b) -> a * b, ptyps))
		s = ctyp * "{"
		s *= (isempty(args[4:end]) ? "" : mapfoldl((a) -> from_expr(a), (a, b) -> "$a, $b", args[4:end])) * "}" 
	end
	s
end

function body(ast)
	ast.args[3]
end

function from_loophead(args)
	iv = from_expr(args[1])
	decl = "uint64_t"
	if haskey(lstate.symboltable, args[1])
		decl = toCtype(lstate.symboltable[args[1]]) 
	end
	start = from_expr(args[2])
	stop = from_expr(args[3])
	"for($decl $iv = $start; $iv <= $stop; $iv += 1) {\n"
end

function from_loopend(args)
	"}\n"
end


function from_expr(ast::Expr)
	s = ""
	head = ast.head
	args = ast.args
	typ = ast.typ

	if head == :block
		dprintln(3,"Compiling block")
		s *= from_exprs(args)

	elseif head == :body
		dprintln(3,"Compiling body")
		s *= from_exprs(args)

	elseif head == :new
		dprintln(3,"Compiling new")
		s *= from_new(args)

	elseif head == :lambda
		dprintln(3,"Compiling lambda")
		s *= from_lambda(ast, args)

	elseif head == :(=)
		dprintln(3,"Compiling assignment")
		s *= from_assignment(args)

    elseif head == :(&)
        dprintln(3, "Compiling ref")
        s *= from_ref(args)

	elseif head == :call
		dprintln(3,"Compiling call")
		s *= from_call(args)

	elseif head == :call1
		dprintln(3,"Compiling call1")
		s *= from_call1(args)

	elseif head == :return
		dprintln(3,"Compiling return")
		s *= from_return(args)

	elseif head == :line
		s *= from_line(args)

	elseif head == :gotoifnot
		dprintln(3,"Compiling gotoifnot : ", args)
		s *= from_gotoifnot(args)

	elseif head == :parfor_start
		s *= from_parforstart(args)

	elseif head == :parfor_end
		s *= from_parforend(args)

	elseif head == :insert_divisible_task
		s *= from_insert_divisible_task(args)

	elseif head == :boundscheck
		# Nothing
	elseif head == :assert
		# Nothing
	elseif head == :simdloop
		# Nothing

	# For now, we ignore meta nodes.
	elseif head == :meta
		# Nothing

	elseif head == :loophead
		s *= from_loophead(args)

	elseif head == :loopend
		s *= from_loopend(args)

	else
		dprintln(3,"Unknown head in expression: ", head)
		throw("Unknown head")
	end
	s
end

function from_expr(ast::Any)
	dprintln(3,"Compiling expression: ", ast)
	s = ""
	asttyp = typeof(ast)
	dprintln(3,"With type: ", asttyp)

	if isPrimitiveJuliaType(asttyp)
		#s *= "(" * toCtype(asttyp) * ")" * string(ast)
        if asttyp == Char
            s *= "'$(string(ast))'"
        else
		    s *= string(ast)
        end
	elseif isPrimitiveJuliaType(ast)
		s *= "(" * toCtype(ast) * ")"
	elseif asttyp == Expr
		s *= from_expr(ast)
	elseif asttyp == Symbol
		s *= from_symbol(ast)
	elseif asttyp == SymbolNode
		s *= from_symbolnode(ast)
	elseif asttyp == LineNumberNode
		s *= from_linenumbernode(ast)
	elseif asttyp == LabelNode
		s *= from_labelnode(ast)
	elseif asttyp == GotoNode
		s *= from_gotonode(ast)
	elseif asttyp == TopNode
		s *= from_topnode(ast)
	elseif asttyp == QuoteNode
		s *= from_quotenode(ast)
	elseif isdefined(:NewvarNode) && asttyp == NewvarNode
		s *= from_newvarnode(ast)
	elseif isdefined(:GetfieldNode) && asttyp == GetfieldNode
		s *= from_getfieldnode(ast)
	elseif isdefined(:GlobalRef) && asttyp == GlobalRef
		s *= from_globalref(ast)
	elseif asttyp == GenSym
		s *= "GenSym" * string(ast.id)
	else
		#s *= dispatch(lstate.adp, ast, ast)
		dprintln(3,"Unknown node type encountered: ", ast, " with type: ", asttyp)
		throw("Fatal Error: Could not translate node")
	end
	s
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

function from_varargpack(vargs)
	args = vargs[1]
	vsym = canonicalize(args[1])
	vtyps = args[2]
	toCtype(vtyps) * " " * vsym * " = " * 
		"{" * mapfoldl((i) -> vsym * string(i), (a, b) -> "$a, $b", 1:length(vtyps.types)) * "};"
end

function from_formalargs(params, vararglist, unaliased=false)
	s = ""
	ql = unaliased ? "__restrict" : ""
	dprintln(3,"Compiling formal args: ", params)
	dumpSymbolTable(lstate.symboltable)
	for p in 1:length(params)
		dprintln(3,"Doing param $p: ", params[p])
		dprintln(3,"Type is: ", typeof(params[p]))
		if get(lstate.symboltable, params[p], false) != false
			ptyp = toCtype(lstate.symboltable[params[p]])
			s *= ptyp * ((isArrayType(lstate.symboltable[params[p]]) ? "&" : "")
				* (isArrayType(lstate.symboltable[params[p]]) ? " $ql " : " ")
				* canonicalize(params[p])
				* (p < length(params) ? ", " : ""))
		# We may have a varags expression
		elseif isa(params[p], Expr)
			assert(isa(params[p].args[1], Symbol))
			dprintln(3,"varargs type: ", params[p], lstate.symboltable[params[p].args[1]])
			varargtyp = lstate.symboltable[params[p].args[1]]
			for i in 1:length(varargtyp.types)
				vtyp = varargtyp.types[i]
				cvtyp = toCtype(vtyp)
				s *= cvtyp * ((isArrayType(vtyp) ? "&" : "")
				* (isArrayType(vtyp) ? " $ql " : " ")
				* canonicalize(params[p].args[1]) * string(i)
				* (i < length(varargtyp.types) ? ", " : ""))
			end
		else
			throw("Could not translate formal argument: " * string(params[p]))
		end
	end
	dprintln(3,"Formal args are: ", s)
	s
end

function from_newvarnode(args...)
	""
end

function from_callee(ast::Expr, functionName::ASCIIString)
	dprintln(3,"Ast = ", ast)
	dprintln(3,"Starting processing for $ast")
	typ = toCtype(body(ast).typ)
	dprintln(3,"Return type of body = $typ")
	params	=	ast.args[1]
	env		=	ast.args[2]
	bod		=	ast.args[3]
	dprintln(3,"Body type is ", bod.typ)
	f = Dict(ast => functionName)
	bod = from_expr(ast)
	args = from_formalargs(params, [], false)
	dumpSymbolTable(lstate.symboltable)
	s::ASCIIString = "$typ $functionName($args) { $bod } "
	s
end



function isScalarType(typ)
	!isArrayType(typ) && !isCompositeType(typ)
end

# Creates an entrypoint that dispatches onto host or MIC.
# For now, emit host path only
function createEntryPointWrapper(functionName, params, args, jtyp)
  if length(params) > 0
    params = mapfoldl((a)->canonicalize(a), (a,b) -> "$a, $b", params) * ", "
  else
    params = ""
  end
	actualParams = params * foldl((a, b) -> "$a, $b",
		[(isScalarType(jtyp[i]) ? "" : "*") * "ret" * string(i-1) for i in 1:length(jtyp)])
	wrapperParams = "int run_where"
  if length(args) > 0
    wrapperParams *= ", $args"
  end
	allocResult = ""
	retSlot = ""
	for i in 1:length(jtyp)
		delim = i < length(jtyp) ? ", " : ""
		retSlot *= toCtype(jtyp[i]) * 
			(isScalarType(jtyp[i]) ? "" : "*") * "* __restrict ret" * string(i-1) * delim
		if isArrayType(jtyp[i])
			typ = toCtype(jtyp[i])
			allocResult *= "*ret" * string(i-1) * " = new $typ();\n"
		end
	end
	#printf(\"Starting execution of cgen generated code\\n\");
	#printf(\"End of execution of cgen generated code\\n\");
	s::ASCIIString =
	"extern \"C\" void _$(functionName)_($wrapperParams, $retSlot) {\n
		$allocResult
		$functionName($actualParams);
	}\n"
	s
end

# This is the entry point to cgen from the PSE driver
function from_root(ast::Expr, functionName::ASCIIString, isEntryPoint = true)
	global inEntryPoint
	inEntryPoint = isEntryPoint
	global lstate
	emitunaliasedroots = false
	if isEntryPoint
		#adp = ASTDispatcher()
		lstate = LambdaGlobalData()
		# If we are forcing vectorization then we will not emit the unaliased versions
        # of the roots
		emitunaliasedroots = (vectorizationlevel == VECDEFAULT ? true : false)
	end
	dprintln(3,"Ast = ", ast)
	dprintln(3,"functionName = ", functionName)
	dprintln(3,"Starting processing for $ast")
	params	=	ast.args[1]
	aparams	=	lambdaparams(ast)
	env		=	ast.args[2]
	bod		=	ast.args[3]
	dprintln(3,"Processing body: ", bod)
	returnType = bod.typ
	typ = returnType

	# Translate the body
	bod = from_expr(ast)


	for i in 1:length(params)
		dprintln(3,"Param ", i, " is ", params[i], " with type ", typeof(params[i]))
	end
	# Find varargs expression if present
	vararglist = []
	for k in params
		if isa(k, Expr) # If k is a vararg expression
			dprintln(3,"vararg expr: ", k, k.args[1], k.head)
			if isa(k.args[1], Symbol) && haskey(lstate.symboltable, k.args[1])
				push!(vararglist, (k.args[1], lstate.symboltable[k.args[1]]))
				dprintln(3,lstate.symboltable[k.args[1]])
				dprintln(3,vararglist)
			end
		end
	end
	
	# Translate arguments
	args = from_formalargs(params, vararglist, false)

	# If emitting unaliased versions, get "restrict"ed decls for arguments
	argsunal = emitunaliasedroots ? from_formalargs(params, vararglist, true) : ""

	if DEBUG_LVL>=3 
		dumpSymbolTable(lstate.symboltable)
	end

	if !isempty(vararglist)
		bod = from_varargpack(vararglist) * bod
	end

	hdr = ""
	wrapper = ""
	if istupletyp(returnType)
	    returnType = tuple(returnType.parameters...)
	else
		returnType = (returnType,)
	end
	hdr = from_header(isEntryPoint)

	# Create an entry point that will be called by the Julia code.
	if isEntryPoint
		wrapper = (emitunaliasedroots ? createEntryPointWrapper(functionName * "_unaliased", params, argsunal, returnType) : "") * createEntryPointWrapper(functionName, params, args, returnType)
		rtyp = "void"
		retargs = foldl((a, b) -> "$a, $b",
		[toCtype(returnType[i]) * " * __restrict ret" * string(i-1) for i in 1:length(returnType)])

		if length(args) > 0
			args *= ", " * retargs
			argsunal *= ", "*retargs
		else
			args = retargs
			argsunal = retargs
		end
	else
		rtyp = toCtype(typ)
	end
	s::ASCIIString = "$rtyp $functionName($args)\n{\n$bod\n}\n"
	s *= (isEntryPoint && emitunaliasedroots) ? "$rtyp $(functionName)_unaliased($argsunal)\n{\n$bod\n}\n" : ""
	forwarddecl::ASCIIString = isEntryPoint ? "" : "$rtyp $functionName($args);\n"
	if inEntryPoint
		inEntryPoint = false
	end
	push!(lstate.compiledfunctions, functionName)
	c = hdr * forwarddecl * from_worklist() * s * wrapper
	if isEntryPoint
		resetLambdaState(lstate)
	end	
	c
end

function insert(func::Any, mod::Any, name, typs)
	if mod == ""
		insert(func, name, typs)
		return
	end
	target = resolveFunction(func, mod, typs)
	dprintln(3,"Resolved function ", func, " : ", name, " : ", typs)
	insert(target, name, typs)
end

function insert(func::TopNode, name, typs)
	target = eval(func)
	dprintln(3,"Resolved function ", func, " : ", name, " : ", typs, " target ", target)
	insert(target, name, typs)
end

function insert(func::SymbolNode, name, typs)
	target = eval(func)
	dprintln(3,"Resolved function ", func, " : ", name, " : ", typs, " target ", target)
	insert(target, name, typs)
end

function insert(func::Symbol, name, typs)
	target = resolveFunction(func, typs)
	dprintln(3,"Resolved function ", func, " : ", name, " : ", typs, " target ", target)
	insert(target, name, typs)
end

function insert(func::Function, name, typs)
	global lstate
	#ast = code_typed(func, typs; optimize=true)
	ast = ParallelAccelerator.Driver.code_typed(func, typs)
	if !has(lstate.compiledfunctions, name)
		push!(lstate.worklist, (ast, name, typs))
	end
end

# Translate function nodes in breadth-first order
function from_worklist()
	s = ""
	si = ""
	global lstate
	while !isempty(lstate.worklist)
		a, fname, typs = splice!(lstate.worklist, 1)
		dprintln(3,"Checking if we compiled ", fname, " before")
		dprintln(3,lstate.compiledfunctions)
		if has(lstate.compiledfunctions, fname)
			dprintln(3,"Yes, skipping compilation...")
			continue
		end
		dprintln(3,"No, compiling it now")
		empty!(lstate.symboltable)
		empty!(lstate.ompprivatelist)
		if isa(a, Symbol)
			a = ParallelAccelerator.Driver.code_typed(a, typs)
		end
		dprintln(3,"============ Compiling AST for ", fname, " ============") 
		dprintln(3,a)
		length(a) >= 1 ? dprintln(3,a[1].args) : ""
		dprintln(3,"============ End of AST for ", fname, " ============") 
		si = length(a) >= 1 ? from_root(a[1], fname, false) : ""
		dprintln(3,"============== C++ after compiling ", fname, " ===========")
		dprintln(3,si)
		dprintln(3,"============== End of C++ for ", fname, " ===========")
		dprintln(3,"Adding ", fname, " to compiledFunctions")
		dprintln(3,lstate.compiledfunctions)
		dprintln(3,"Added ", fname, " to compiledFunctions")
		dprintln(3,lstate.compiledfunctions)
		s *= si
	end
	s
end

#
# Utility methods to write, compile and link generated code
#
import Base.write
function writec(s, outfile_name=nothing; with_headers=false)
    if outfile_name == nothing
        outfile_name = generate_new_file_name()
    end
    if with_headers
        s = from_header(true) * "extern \"C\" {\n" * s * "\n}"
    end
    packageroot = getPackageRoot()
    cgenOutput = "$packageroot/deps/generated/$(outfile_name).cpp"
    cf = open(cgenOutput, "w")
    write(cf, s)
    dprintln(3,"Done committing cgen code")
    close(cf)
    return outfile_name
end

function getCompileCommand(full_outfile_name, cgenOutput)
  # return an error if this is not overwritten with a valid compiler
  compileCommand = `echo "invalid backend compiler"`

  packageroot = getPackageRoot()
  otherArgs = "-DJ2C_REFCOUNT_DEBUG -DDEBUGJ2C"
  Opts = "-O3"
  if backend_compiler == USE_ICC
    vecOpts = (vectorizationlevel == VECDISABLE ? "-no-vec" : "")
    # Generate dyn_lib
    compileCommand = `icc $Opts -g $vecOpts -qopenmp -fpic -c -o $full_outfile_name $otherArgs $cgenOutput`
  elseif backend_compiler == USE_GCC
    compileCommand = `gcc $Opts -g -fopenmp -fpic -c -o $full_outfile_name $otherArgs $cgenOutput`
  end

  return compileCommand
end

function compile(outfile_name)
  global use_bcpp
  packageroot = getPackageRoot()

  cgenOutput = "$packageroot/deps/generated/$outfile_name.cpp"

  if use_bcpp == 1
    cgenOutputTmp = "$packageroot/deps/generated/$(outfile_name)_tmp.cpp"
    run(`cp $cgenOutput $cgenOutputTmp`)
    # make cpp code readable
    beautifyCommand = `bcpp $cgenOutputTmp $cgenOutput`
    run(beautifyCommand)
  end

  full_outfile_name = `$packageroot/deps/generated/$outfile_name.o`
  compileCommand = getCompileCommand(full_outfile_name, cgenOutput)
  println(compileCommand)
  run(compileCommand)	
end

function getLinkCommand(outfile_name, lib)
  # return an error if this is not overwritten with a valid compiler
  linkCommand = `echo "invalid backend linker"`

  packageroot = getPackageRoot()

  if backend_compiler == USE_ICC
    linkCommand = `icc -g -shared -qopenmp -o $lib $packageroot/deps/generated/$outfile_name.o`
  elseif backend_compiler == USE_GCC
    linkCommand = `gcc -g -shared -fopenmp -o $lib $packageroot/deps/generated/$outfile_name.o`
  end

  return linkCommand 
end


function link(outfile_name)
  packageroot = getPackageRoot()
  lib = "$packageroot/deps/generated/lib$outfile_name.so.1.0"
  linkCommand = getLinkCommand(outfile_name, lib)
  println(linkCommand)
  run(linkCommand)
  dprintln(3,"Done cgen linking")
  return lib
end

# When in standalone mode, this is the entry point to cgen.
function generate(func::Function, typs; init_lstate=false)
  global lstate
  if init_lstate
      lstate = LambdaGlobalData()
  end
  name = string(func.env.name)
  insert(func, name, typs)
  return from_worklist()
end
end # cgen module
