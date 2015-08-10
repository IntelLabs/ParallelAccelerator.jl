# A prototype Julia to C++ generator
# jaswanth.sreeram@intel.com
# 

module cgen
using ..ParallelIR
using CompilerTools
export generate, from_root, writec, compile, link
import IntelPSE, ..getPackageRoot

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
		debugp("ERROR: Unexpected node: ", node, " with type = ", tgt)
		throw("Could not dispatch node")
	end
	debugp("Dispatching to call: ", a.dt[tgt], " with args: ", args)
	getfield(a.m, a.dt[tgt])(args)
end
=#

type LambdaGlobalData
	#adp::ASTDispatcher
	ompprivatelist::Array{Any, 1}
	globalUDTs::Dict{Any, Any}
	symboltable::Dict{Any, Any}
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
			Char	=>	"char"
	)

		#new(ASTDispatcher(), [], Dict(), Dict(), [], [])
		new([], Dict(), Dict(), [], [], _j, 0)
	end
end



# Globals
# verbose = true
verbose = false 
inEntryPoint = false
lstate = nothing
#packageroot = nothing

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
_builtins = ["getindex", "setindex", "arrayref", "top", "box", 
			"unbox", "tuple", "arraysize", "arraylen", "ccall",
			"arrayset", "getfield", "unsafe_arrayref", "unsafe_arrayset",
			"safe_arrayref", "safe_arrayset", "tupleref",
			"call1", ":jl_alloc_array_1d", ":jl_alloc_array_2d"]

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
		"trunc", "ceil_llvm", "ceil", "pow", "powf"
]

tokenXlate = Dict(
	'*' => "star",
	'/' => "slash",
	'-' => "minus",
	'!' => "bang",
	'.' => "dot"
)

replacedTokens = Set("(,)#")
scrubbedTokens = Set("{}:")

#### End of globals ####

function __init__()
	packageroot = joinpath(dirname(@__FILE__), "..")
end

function debugp(a...)
	verbose && println(a...)
end


function getenv(var::String)
  ENV[var]
end

function from_header(isEntryPoint::Bool)
	s = from_UDTs()
	isEntryPoint ? from_includes() * s : s
end

function from_includes()
	packageroot = getPackageRoot()
	reduce(*, "", (
		"#include <omp.h>\n",
		"#include <stdint.h>\n",
		"#include <math.h>\n",
		"#include <stdio.h>\n",
		"#include <iostream>\n",
		"#include \"$packageroot/src/intel-runtime/include/pse-runtime.h\"\n",
		"#include \"$packageroot/src/intel-runtime/include/j2c-array.h\"\n",
		"#include \"$packageroot/src/intel-runtime/include/j2c-array-pert.h\"\n",
		"#include \"$packageroot/src/intel-runtime/include/pse-types.h\"\n")
	)
end

function from_UDTs()
	global lstate
	debugp("UDT Table is: ")
	debugp(lstate.globalUDTs)
	isempty(lstate.globalUDTs) ? "" : mapfoldl((a) -> (lstate.globalUDTs[a] == 1 ? from_decl(a) : ""), (a, b) -> "$a; $b", keys(lstate.globalUDTs))
end


function from_decl(k::Tuple)
	debugp("from_decl: Defining tuple type ", k)
	s = "typedef struct {\n"
	for i in 1:length(k)
		s *= toCtype(k[i]) * " " * "f" * string(i-1) * ";\n"
	end
	s *= "} Tuple_" * mapfoldl((a) -> canonicalize(a), (a, b) -> "$(a)_$(b)", k) * ";\n"
	if haskey(lstate.globalUDTs, k)
		lstate.globalUDTs[k] = 0
	end
	s
end

function from_decl(k::DataType)
	debugp("from_decl: Defining data type ", k)
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
		s *= "} Tuple" * mapfoldl((a) -> canonicalize(a), (a, b) -> "$(a)_$(b)", ptyp) * ";\n"
		return s
	end
	throw("Could not translate Julia Type: ", k)
	return ""
end

function from_decl(k)
	debugp("from_decl: Defining type ", k)
	return toCtype(lstate.symboltable[k]) * " " * canonicalize(k) * ";\n"
end

function isCompositeType(t)
	# TODO: Expand this to real UDTs
	b = isa(t, Tuple) || is(t, UnitRange{Int64}) || is(t, StepRange{Int64, Int64})
	b |= isa(t, DataType) && t.name == Tuple.name
	debugp("Is ", t, " a composite type: ", b)
	b
end

function from_lambda_old(args)
	s = ""
	params	=	length(args) != 0 ? args[1] : []
	env		=	length(args) >=2 ? args[2] : []
	locals	=	length(env) > 0 ? env[1] : []
	vars	=	length(env) >=2 ? env[2] : []
	typ		=	length(env) >= 4 ? env[4] : []
	decls = ""
	debugp("locals are: ", locals)
	debugp("vars are: ", vars)
	debugp("Typ is ", typ)
	global lstate
	debugp("ompPrivateList is: ", lstate.ompprivatelist)
	for k in 1:length(vars)
		v = vars[k]
		lstate.symboltable[v[1]] = v[2]
		# If we have user defined types, record them
		if in(v[1], locals) && (v[3] & 32 != 0)
			push!(lstate.ompprivatelist, v[1])
		end 
	end
	debugp("Done with ompvars")
	bod = from_expr(args[3])
	debugp("lambda locals = ", locals)
	debugp("lambda params = ", params)
	debugp("symboltable here = ")
	dumpSymbolTable(lstate.symboltable)
	
	for k in keys(lstate.symboltable)
		if (has(locals, k) && !has(params, k)) || (!has(locals, k) && !has(params, k))
			debugp("About to check for composite type: ", lstate.symboltable[k])
			if isCompositeType(lstate.symboltable[k])
				#globalUDTs, from_decl(_symbolTable[k])
				if !haskey(lstate.globalUDTs, lstate.symboltable[k])
					lstate.globalUDTs[lstate.symboltable[k]] = 1
				end
			end
			decls *= toCtype(lstate.symboltable[k]) * " " * canonicalize(k) * ";\n"
			#end
		end
	end
	decls * bod
end

function from_lambda(ast, args)
	s = ""
	linfo = CompilerTools.LambdaHandling.lambdaExprToLambdaInfo(ast)
	params = linfo.input_params
	vars = linfo.var_defs
	gensyms = linfo.gen_sym_typs

	decls = ""
	global lstate
	debugp("ompPrivateList is: ", lstate.ompprivatelist)
	for k in keys(vars)
		v = vars[k] # v is a VarDef
		lstate.symboltable[k] = v.typ
		# If we have user defined types, record them
		if !in(k, params) && (v.desc & 32 != 0)
			push!(lstate.ompprivatelist, k)
		end 
	end
	debugp("Emitting gensyms")
	for k in 1:length(gensyms)
		debugp("gensym: ", k, " is of type: ", gensyms[k])
		lstate.symboltable["GenSym" * string(k-1)] = gensyms[k]
	end
	debugp("Done with ompvars")
	bod = from_expr(args[3])
	debugp("lambda params = ", params)
	debugp("lambda vars = ", vars)
	debugp("symboltable here = ")
	dumpSymbolTable(lstate.symboltable)
	
	for k in keys(lstate.symboltable)
		if !in(k, params) #|| (!in(k, locals) && !in(k, params))
			debugp("About to check for composite type: ", lstate.symboltable[k])
			if isCompositeType(lstate.symboltable[k])
				if !haskey(lstate.globalUDTs, lstate.symboltable[k])
					lstate.globalUDTs[lstate.symboltable[k]] = 1
				end
			end
			debugp("Adding ", k, " to decls with type ", lstate.symboltable[k]) 
			decls *= toCtype(lstate.symboltable[k]) * " " * canonicalize(k) * ";\n"
		end
	end
	decls * bod
end


function from_exprs(args::Array)
	s = ""
	debugp("[From Exprs] : ", args)
	for a in args
		debugp("Doing arg a = ", a)
		se = from_expr(a)
		s *= se * (!isempty(se) ? ";\n" : "")
	end
	s
end


function dumpSymbolTable(a::Dict{Any, Any})
	debugp("SymbolTable: ")
	for x in keys(a)
		debugp(x, " ==> ", a[x])
	end
end

function dumpDecls(a::Array{Dict{Any, ASCIIString}})
	for x in a
		for k in keys(x)
			debugp(x[k], " ", k)
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
	lhs = args[1]
	rhs = args[2]
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
			debugp("Unknown type in assignment: ", args)
			throw("FATAL error....exiting")
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

function isArrayType(typ)
	isa(typ, DataType) && ((typ.name == Array.name) || (typ.name == BitArray.name))
end

function toCtype(typ::Tuple)
	return "Tuple_" * mapfoldl((a) -> canonicalize(a), (a, b) -> "$(a)_$(b)", typ)
end

function toCtype(typ)
	debugp("Converting type: ", typ, " to ctype")
	if haskey(lstate.jtypes, typ)
		debugp("Found simple type: ", typ, " returning ctype: ", lstate.jtypes[typ])
		return lstate.jtypes[typ]
	elseif isArrayType(typ)	
		atyp, dims = parseArrayType(typ)
		debugp("Found array type: ", atyp, " with dims: ", dims)
		atyp = toCtype(atyp)
		debugp("Atyp is: ", atyp)
		assert(dims >= 0)
		return " j2c_array< $(atyp) > "
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
	s
end

function parseParametricType(typ)
	assert(isa(typ, DataType))
	return typ.name, typ.parameters
end

function parseParametricType_s(typ)
	debugp("Parsing parametric type: ", typ)
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
	debugp("Parsed type Base:", baseTyp, " and Parameter:", pTyps[1])
	return baseTyp, pTyps
end


function toCName(a)
	return string(a)
end

function from_tupleref(args)
	# TODO generate std::tuples instead of structs
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

function from_tuple(args)
	debugp(args)
	s = "{"
	for i in 1:length(args)
		debugp("Arg: ", i, " : ", args[i], " is type ", typeof(args[i])) 
	end
	s *= mapfoldl((a) -> from_expr(a), (a, b) -> "$a, $b", args) * "}" 
	debugp("Returning: ", s)
	s
end

function from_arraysize(args)
	s = from_expr(args[1])
	if length(args) == 1
		s *= ".ARRAYLEN()"
	else
		s *= ".ARRAYSIZE(" * from_expr(args[2]) * ")"
	end
	debugp("Returning ", s)
	s
end

function from_ccall(args)
	debugp("ccall args:")
	debugp("target tuple: ", args[1], " - ", typeof(args[1]))
	debugp("return type: ", args[2])
	debugp("input types tuple: ", args[3])
	debugp("inputs tuple: ", args[4:end])
	for i in 1:length(args)
		debugp(args[i])
	end
	debugp("End of ccall args")
	fun = args[1]
	if isInlineable(fun, args)
		return from_inlineable(fun, args)
	end

	if isa(fun, QuoteNode)
		s = from_expr(fun)
	elseif isa(fun, Expr) && (is(fun.head, :call1) || is(fun.head, :call))
		s = canonicalize(string(fun.args[2])) #* "/*" * from_expr(fun.args[3]) * "*/"
		debugp("ccall target: ", s)
	else
		throw("Invalid ccall format...")
	end
	debugp("Length is ", length(args) )
	s *= "("
	debugp("Ccall args start: ", round(Int, (length(args)-1)/2)+2)
	debugp("Ccall args end: ", length(args))
	numInputs = length(args[3].args)-1
	argsStart = 4
	argsEnd = length(args)
	debugp("Emitting args:")
	for i in argsStart:2:argsEnd
		debugp(args[i])
	end
	debugp("End of ccall args:")
	s *= mapfoldl((a)->from_expr(a), (a, b)-> "$a, $b", args[argsStart:2:end])
	s *= ")"
	debugp("from_ccall: ", s)
	s
end

function from_arrayset(args)
	debugp("arrayset args are: ", args, length(args))
	idxs = mapfoldl((a) -> from_expr(a), (a, b) -> "$a, $b", args[3:end]) 
	src = from_expr(args[1])
	val = from_expr(args[2])
	"$src.ARRAYELEM($idxs) = $val"
end

function from_getfield(args)
	debugp("Getfield, args are: ", length(args))
	mod, tgt = resolveCallTarget(args[1], args[2:end])
	if mod == "Intrinsics"
		return from_expr(tgt)
	elseif isInlineable(tgt, args[2:end])
		return from_inlineable(tgt, args[2:end])
	end
	from_expr(mod) * "." * from_expr(tgt)
end

#=
function from_getfield(args)
	debugp("Getfield, args are: ", length(args))
	#assert(length(args) == 2)
	rcvr = from_expr(args[1])
	if rcvr == "Intrinsics"
		return from_expr(args[2]) 
	end
	rcvr * "." * from_expr(args[2])
end
=#
function from_arrayalloc(args)
	typ, dims = parseArrayType(args[4])
	typ = toCtype(typ)
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

function from_builtins(f, args)
	debugp("from_builtins:");
	debugp("function is: ", f)
	debugp("Args are: ", args)
	tgt = string(f)
	if tgt == "getindex"
		return from_getindex(args)
	elseif tgt == "setindex"
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
	elseif tgt == ":jl_alloc_array_1d"
		return from_arrayalloc(args)
	elseif tgt == ":jl_alloc_array_2d"
		return from_arrayalloc(args)
	elseif tgt == "getfield"
		debugp("f is: ", f)
		debugp("args are: ", args)
		return from_getfield(args)
	elseif tgt == "unsafe_arrayref"
		return from_getindex(args) 
	elseif tgt == "safe_arrayref"
		return from_safegetindex(args) 
	elseif tgt == "unsafe_arrayset" || tgt == "safe_arrayset"
		return from_setindex(args)
	end
	
	debugp("Compiling ", string(f))
	throw("Unimplemented builtin")
end

function from_box(args)
	s = ""
	typ = args[1]
	val = args[2]
	s *= from_expr(val)
	s
end

function from_intrinsic(f, args)
	intr = string(f)
	debugp("Intrinsic ", intr, ", args are ", args)

	if intr == "mul_int"
		return from_expr(args[1]) * " * " * from_expr(args[2])
	elseif intr == "mul_float"
		return from_expr(args[1]) * " * " * from_expr(args[2])
	elseif intr == "add_int"
		return from_expr(args[1]) * " + " * from_expr(args[2])
	elseif intr == "or_int"
		return from_expr(args[1]) * " | " * from_expr(args[2])
	elseif intr == "xor_int"
		return from_expr(args[1]) * " ^ " * from_expr(args[2])
	elseif intr == "and_int"
		return from_expr(args[1]) * " & " * from_expr(args[2])
	elseif intr == "sub_int"
		return from_expr(args[1]) * " - " * from_expr(args[2])
	elseif intr == "slt_int"
		return from_expr(args[1]) * " < " * from_expr(args[2])
	elseif intr == "sle_int"
		return from_expr(args[1]) * " <= " * from_expr(args[2])
	elseif intr == "select_value"
		return "(" * from_expr(args[1]) * ")" * " ? " *
		"(" * from_expr(args[2]) * ") : " * "(" * from_expr(args[3]) * ")"
	elseif intr == "not_int"
		return "!" * "(" * from_expr(args[1]) * ")"
	elseif intr == "add_float"
		return from_expr(args[1]) * " + " * from_expr(args[2])
	elseif intr == "lt_float"
		return from_expr(args[1]) * " < " * from_expr(args[2])
	elseif intr == "ne_float"
		return from_expr(args[1]) * " != " * from_expr(args[2])
	elseif intr == "le_float"
		return from_expr(args[1]) * " <= " * from_expr(args[2])
	elseif intr == "neg_float"
		return "-" * from_expr(args[1])
	elseif intr == "abs_float"
		return "fabs(" * from_expr(args[1]) * ")"
	elseif intr == "sqrt_llvm"
		return "sqrt(" * from_expr(args[1]) * ")"
	elseif intr == "sub_float"
		return from_expr(args[1]) * " - " * from_expr(args[2])
	elseif intr == "div_float"
		return "(" * from_expr(args[1]) * ")" * 
			" / " * "(" * from_expr(args[2]) * ")"
	elseif intr == "sitofp"
		return from_expr(args[1]) * from_expr(args[2])
	elseif intr == "fptrunc" || intr == "fpext"
		return from_expr(args[1]) * from_expr(args[2])
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
		debugp("nan_dom_err is: ")
		for i in 1:length(args)
			debugp(args[i])
		end
		#return "assert(" * "isNan(" * from_expr(args[1]) * ") && !isNan(" * from_expr(args[2]) * "))"
		return from_expr(args[1])
	else
		debugp("Intrinsic ", intr, " is known but no translation available")
		throw("Unhandled Intrinsic...")
	end
end

function from_inlineable(f, args)
	debugp("Checking if ", f, " can be inlined")
	debugp("Args are: ", args)
	if has(_operators, string(f))
		return "(" * from_expr(args[1]) * string(f) * from_expr(args[2]) * ")"
	elseif has(_builtins, string(f))
		return from_builtins(f, args)
	elseif has(_Intrinsics, string(f))
		return from_intrinsic(f, args)
	else
		throw("Unknown Operator or Method encountered: ", string(f))
	end
end

function isInlineable(f, args)
	debugp("Checking Isinlineable of function f=", f, " with args = ", args)
	debugp("String representation of function: ", string(f), " : ", has(_builtins, string(f)))
	debugp("AST: ", f)
	debugp("Args = ", args, " length = ", length(args))
	if has(_operators, string(f)) || has(_builtins, string(f)) || has(_Intrinsics, string(f))
		return true
	end
	debugp("Typeof: ", typeof(f), " : ", typeof(f) == TopNode ? string(f.name) : "None")
	return false
end

function arrayToTuple(a)
	ntuple((i)->a[i], length(a))
end

function from_symbol(ast)
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

function from_call1(ast::Array{Any, 1})
	debugp("Call1 args")
	s = ""
	for i in 2:length(ast)
		s *= from_expr(ast[i])
		debugp(ast[i])
	end
	debugp("Done with call1 args")
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
	debugp("Trying to resolve target with args: ", args)
	M = ""
	t = ""
	s = ""
	debugp("Length of args is: ", length(args))
	for i in 1:length(args)
		debugp("Resolve: arg ", i , " is ", args[i], " type is: ", typeof(args[i]))
		debugp("Head is: ", hasfield(args[i], :head) ? args[i].head : "")
	end
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
		debugp("Case 0: Returning M = ", M, " s = ", s, " t = ", t)
	#case 1:
	elseif isa(args[1], TopNode) && is(args[1].name, :getfield) && isa(args[3], QuoteNode)
		debugp("Case 1: args[3] is ", args[3])
		s = args[3].value
		if isa(args[2], Module)
			M = args[2]
			debugp("Case 1: Returning M = ", M, " s = ", s, " t = ", t)
		else
			#case 2:
			M = ""
			s = ""
			t = from_expr(args[2]) * "." * from_expr(args[3])
			#M, _s = resolveCallTarget([args[2]])
			debugp("Case 1: Returning M = ", M, " s = ", s, " t = ", t)
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
		debugp("Case 3: Returning M = ", M, " s = ", s, " t = ", t)
	end
	debugp("In resolveCallTarget: Returning M = ", M, " s = ", s, " t = ", t)
	return M, s, t
end

#=
function pattern_match_call(ast::Array{Any, 1})
	# math functions 
	libm_math_functions = Set([:sin, :cos, :tan, :asin, :acos, :acosh, :atanh, :log, :log2, :log10, :lgamma, :log1,:asinh,:atan,:cbrt,:cosh,:erf,:exp,:expm1,:sinh,:sqrt,:tanh])

	debugp("pattern matching ",ast)
	s = ""
	if( length(ast)==2 && typeof(ast[1])==Symbol && in(ast[1],libm_math_functions) && typeof(ast[2])==SymbolNode && (ast[2].typ==Float64 || ast[2].typ==Float32))
	  debugp("FOUND ", ast[1])
	  s = string(ast[1])*"("*from_expr(ast[2].name)*");"
	end
	s
end
=#

function from_call(ast::Array{Any, 1})

	debugp("Compiling call: ast = ", ast, " args are: ")
	for i in 1:length(ast)
		debugp("Arg ", i, " = ", ast[i], " type = ", typeof(ast[i]))
	end
	mod, fun, t = resolveCallTarget(ast)
	if !isempty(t)
		return t;
	end
	if fun == ""
		fun = ast[1]
	end
	args = ast[2:end]
	debugp("fun is: ", fun)
	debugp("call Args are: ", args)
	if isInlineable(fun, args)
		debugp("Doing with inlining ", fun, "(", args, ")")
		fs = from_inlineable(fun, args)
		return fs
	end
	debugp("Not inlinable")
	funStr = "_" * string(fun)	
	# If we have previously compiled this function
	# we fallthru and simply emit the call.
	# Else we lookup the function Symbol and enqueue it
	# TODO: This needs to specialize on types
	skipCompilation = has(lstate.compiledfunctions, funStr) ||
		isPendingCompilation(lstate.worklist, funStr)
	s = ""
	debugp("Worklist: ")
	map((i)->debugp(i[2]), lstate.worklist)
	debugp("Compiled functionslist: ", lstate.compiledfunctions)
	map((i)->debugp(i), lstate.compiledfunctions)
	debugp("Translating call, target is ", funStr, " args are : ", args)
	s *= "_" * from_expr(fun) * "("
	debugp("After call translation, target is ", s)
	argTyps = []
	for a in 1:length(args)
		debugp("Doing arg: ", a)
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
	debugp("Finished translating call : ", s)
	debugp(ast[1], " : ", typeof(ast[1]), " : ", hasfield(ast[1], :head) ? ast[1].head : "")
	if !skipCompilation && (isa(fun, Symbol) || isa(fun, Function))
		debugp("Worklist is: ")
		for i in 1:length(lstate.worklist)
			ast, name, typs = lstate.worklist[i]
			debugp(name);
		end
		debugp("Compiled Functions are: ")
		for i in 1:length(lstate.compiledfunctions)
			name = lstate.compiledfunctions[i]
			debugp(name);
		end
		debugp("Inserting: ", fun, " : ", "_" * canonicalize(fun), " : ", arrayToTuple(argTyps))
		insert(fun, mod, "_" * canonicalize(fun), arrayToTuple(argTyps))
	end
	s
end

function from_return(args)
	global inEntryPoint
	debugp("Return args are: ", args)
	retExp = ""
	if inEntryPoint
		if typeAvailable(args[1]) && isa(args[1].typ, Tuple)
			retTyps = args[1].typ
			for i in 1:length(retTyps)
				retExp *= "*ret" * string(i-1) * " = " * from_expr(args[1]) * ".f" * string(i-1) * ";\n"
			end
		else
			retExp = "*ret0 = " * from_expr(args[1]) * ";\n"
		end
		return retExp * "return"
	else
		return "return " * from_expr(args[1])
	end
end


function from_gotonode(ast, exp = "")
	labelId = ast.label
	s = ""
	debugp("Compiling goto: ", exp, " ", typeof(exp))
	if isa(exp, Expr) || isa(exp, SymbolNode) || isa(exp, Symbol)
		s *= "if (!(" * from_expr(exp) * ")) "
	end
	s *= "goto " * "label" * string(labelId)
	s
end

function from_gotoifnot(args)
	exp = args[1]
	labelId = args[2]
	s = ""
	debugp("Compiling gotoifnot: ", exp, " ", typeof(exp))
	if isa(exp, Expr) || isa(exp, SymbolNode) || isa(exp, Symbol)
		s *= "if (!(" * from_expr(exp) * ")) "
	end
	s *= "goto " * "label" * string(labelId)
	s
end
#=
function from_goto(exp, labelId)
	s = ""
	debugp("Compiling goto: ", exp, " ", typeof(exp))
	if isa(exp, Expr) || isa(exp, SymbolNode) || isa(exp, Symbol)
		s *= "if (!(" * from_expr(exp) * ")) "
	end
	s *= "goto " * "label" * string(labelId)
	s
end
=#
function from_getfieldnode(ast)
	throw("Unexpected node: ", ast)	
end

function from_topnode(ast)
	from_symbol(ast)
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
	debugp("Parforend = ", s)
	lstate.ompdepth -= 1
	s
end

function loopNestCount(loop)
	"(((" * from_expr(loop.upper) * ") + 1 - (" * from_expr(loop.lower) * ")) / (" * from_expr(loop.step) * "))"
end


function from_insert_divisible_task(args)
	inserttasknode = args[1]
	debugp("Ranges: ", inserttasknode.ranges)
	debugp("Args: ", inserttasknode.args)
	debugp("Task Func: ", inserttasknode.task_func)
	debugp("Join Func: ", inserttasknode.join_func)
	debugp("Task Options: ", inserttasknode.task_options)
	debugp("Host Grain Size: ", inserttasknode.host_grain_size)
	debugp("Phi Grain Size: ", inserttasknode.phi_grain_size)
	throw("Task mode is not supported yet")	
end

function from_loopnest(ivs, starts, stops, steps)
	mapfoldl(
		(i) -> "for ( $(ivs[i]) = $(starts[i]); $(ivs[i]) <= $(stops[i]); $(ivs[i]) += $(steps[i])) {\n",
		(a, b) -> "$a $b",
		1:length(ivs))
end

# If the parfor body is too complicated then DIR or PIR will set
# instruction_count_expr = nothing

# Meaning of num_threads_mode
# mode = 1 uses static insn count if it is there, but doesn't do dynamic estimation and fair core allocation between levels in a loop nest.
# mode = 2 does all of the above
# mode = 3 in addition to 2, also uses host minimum (0) and Phi minimum (10)

function from_parforstart(args)
	global lstate
	num_threads_mode = ParallelIR.num_threads_mode

	println("args: ",args);

	parfor = args[1]
	lpNests = parfor.loopNests
    private_vars = parfor.private_vars

	# Translate metadata for the loop nests
	ivs = map((a)->from_expr(a.indexVariable), lpNests)
	starts = map((a)->from_expr(a.lower), lpNests)
	stops = map((a)->from_expr(a.upper), lpNests)
	steps = map((a)->from_expr(a.step), lpNests)

	println("ivs ",ivs);
	println("starts ", starts);
	println("stops ", stops);
	println("steps ", steps);

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
	#privatevars = isempty(lstate.ompprivatelist) ? "" : "private(" * mapfoldl((a) -> canonicalize(a), (a,b) -> "$a, $b", lstate.ompprivatelist) * ")"
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
	typ = args[1] #type of the object
	@assert isa(typ, DataType) "typ:$typ is not DataType"
	objtyp, ptyps = parseParametricType(typ)
	ctyp = canonicalize(objtyp) * mapfoldl((a) -> canonicalize(a), (a, b) -> a * b, ptyps)
	s = ctyp * "{"
	s *= mapfoldl((a) -> from_expr(a), (a, b) -> "$a, $b", args[2:end]) * "}" 
	s
end

function body(ast)
	ast.args[3]
end

function from_loophead(args)
	iv = from_expr(args[1])
	start = from_expr(args[2])
	stop = from_expr(args[3])
	"for($iv = $start; $iv <= $stop; $iv += 1) {\n"
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
		debugp("Compiling block")
		s *= from_exprs(args)

	elseif head == :body
		debugp("Compiling body")
		s *= from_exprs(args)

	elseif head == :new
		debugp("Compiling new")
		s *= from_new(args)

	elseif head == :lambda
		debugp("Compiling lambda")
		s *= from_lambda(ast, args)

	elseif head == :(=)
		debugp("Compiling assignment")
		s *= from_assignment(args)

	elseif head == :call
		debugp("Compiling call")
		s *= from_call(args)

	elseif head == :call1
		debugp("Compiling call1")
		s *= from_call1(args)

	elseif head == :return
		debugp("Compiling return")
		s *= from_return(args)

	elseif head == :line
		s *= from_line(args)

	elseif head == :gotoifnot
		debugp("Compiling gotoifnot : ", args)
		s *= from_gotoifnot(args)

	elseif head == :parfor_start
		s *= from_parforstart(args)

	elseif head == :parfor_end
		s *= from_parforend(args)

	elseif head == :insert_divisible_task
		s *= from_insert_divisible_task(args)

	elseif head == :boundscheck
		# Nothing

	elseif head == :loophead
		s *= from_loophead(args)

	elseif head == :loopend
		s *= from_loopend(args)

	else
		debugp("Unknown head in expression: ", head)
		throw("Unknown head")
	end
	s
end

function from_expr(ast::Any)
	debugp("Compiling expression: ", ast)
	s = ""
	asttyp = typeof(ast)
	debugp("With type: ", asttyp)

	if isPrimitiveJuliaType(asttyp)
		#s *= "(" * toCtype(asttyp) * ")" * string(ast)
		s *= string(ast)
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
		s *= from_getfieldnode(ast)
	elseif asttyp == GenSym
		s *= "GenSym" * string(ast.id)
	else
		#s *= dispatch(lstate.adp, ast, ast)
		debugp("Unknown node type encountered: ", ast, " with type: ", asttyp)
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

function from_formalargs(params, unaliased=false)
	s = ""
	ql = unaliased ? "__restrict" : ""
	debugp("Compiling formal args: ", params)
	for p in 1:length(params)
		if haskey(lstate.symboltable, params[p])
			s *= (toCtype(lstate.symboltable[params[p]])
				* (isArrayType(lstate.symboltable[params[p]]) ? "&" : "")
				* (isArrayType(lstate.symboltable[params[p]]) ? " $ql " : " ")
				* canonicalize(params[p])
				* (p < length(params) ? ", " : ""))
		end
	end
	debugp("Formal args are: ", s)
	s
end

function from_newvarnode(args...)
	""
end

function from_callee(ast::Expr, functionName::ASCIIString)
	debugp("Ast = ", ast)
	verbose && debugp("Starting processing for $ast")
	typ = toCtype(body(ast).typ)
	verbose && debugp("Return type of body = $typ")
	params	=	ast.args[1]
	env		=	ast.args[2]
	bod		=	ast.args[3]
	debugp("Body type is ", bod.typ)
	f = Dict(ast => functionName)
	bod = from_expr(ast)
	args = from_formalargs(params)
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

function from_root(ast::Expr, functionName::ASCIIString, isEntryPoint = true)
	global inEntryPoint
	inEntryPoint = isEntryPoint
	global lstate
	if isEntryPoint
		#adp = ASTDispatcher()
		lstate = LambdaGlobalData()
	end
	debugp("Ast = ", ast)
	verbose && debugp("Starting processing for $ast")
	params	=	ast.args[1]
	env		=	ast.args[2]
	bod		=	ast.args[3]
	debugp("Processing body: ", bod)
	returnType = bod.typ
	typ = returnType
	bod = from_expr(ast)
	args = from_formalargs(params)
	argsunal = from_formalargs(params, true)
	dumpSymbolTable(lstate.symboltable)
	hdr = ""
	wrapper = ""
	if !isa(returnType, Tuple)
		returnType = (returnType,)
	end
	hdr = from_header(isEntryPoint)
	if isEntryPoint
		wrapper = createEntryPointWrapper(functionName * "_unaliased", params, argsunal, returnType) * 
			createEntryPointWrapper(functionName, params, args, returnType)
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
	s::ASCIIString = "$rtyp $functionName($args)\n{\n$bod\n}\n" * (isEntryPoint ? "$rtyp $(functionName)_unaliased($argsunal)\n{\n$bod\n}\n" : "")
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

function insert(func::Symbol, mod::Any, name, typs)
	if mod == ""
		insert(func, name, typs)
		return
	end
	target = resolveFunction(func, mod, typs)
	debugp("Resolved function ", func, " : ", name, " : ", typs)
	insert(target, name, typs)
end

function insert(func::Symbol, name, typs)
	target = resolveFunction(func, typs)
	debugp("Resolved function ", func, " : ", name, " : ", typs, " target ", target)
	insert(target, name, typs)
end

function insert(func::Function, name, typs)
	global lstate
	#ast = code_typed(func, typs; optimize=true)
	ast = code_typed(func, typs)
	if !has(lstate.compiledfunctions, name)
		push!(lstate.worklist, (ast, name, typs))
	end
end

function from_worklist()
	s = ""
	si = ""
	global lstate
	while !isempty(lstate.worklist)
		#a, fname, typs = pop!(lstate.worklist)
		a, fname, typs = splice!(lstate.worklist, 1)
		debugp("Checking if we compiled ", fname, " before")
		debugp(lstate.compiledfunctions)
		if has(lstate.compiledfunctions, fname)
			continue
		end
		debugp("No, compiling it now")
		empty!(lstate.symboltable)
		empty!(lstate.ompprivatelist)
		if isa(a, Symbol)
			a = code_typed(a, typs; optimize=true)
		end
		debugp("============ Compiling AST for ", fname, " ============") 
		debugp(a)
		length(a) >= 1 ? debugp(a[1].args) : ""
		debugp("============ End of AST for ", fname, " ============") 
		si = length(a) >= 1 ? from_root(a[1], fname, false) : ""
		debugp("============== C++ after compiling ", fname, " ===========")
		debugp(si)
		debugp("============== End of C++ for ", fname, " ===========")
		debugp("Adding ", fname, " to compiledFunctions")
		debugp(lstate.compiledfunctions)
		debugp("Added ", fname, " to compiledFunctions")
		debugp(lstate.compiledfunctions)
		s *= si
	end
	s
end
import Base.write
function writec(s)
	packageroot = getPackageRoot()
	cgenOutput = "$packageroot/src/intel-runtime/tmp.cpp"
	cf = open(cgenOutput, "w")
	write(cf, s)
	debugp("Done committing cgen code")
	close(cf)
end

function compile()
	packageroot = getPackageRoot()
	
	cgenOutputTmp = "$packageroot/src/intel-runtime/tmp.cpp"
	cgenOutput = "$packageroot/src/intel-runtime/out.cpp"

	# make cpp code readable
	beautifyCommand = `bcpp $cgenOutputTmp $cgenOutput`
	run(beautifyCommand)

	iccOpts = "-O3"
	otherArgs = "-DJ2C_REFCOUNT_DEBUG -DDEBUGJ2C"
	# Generate dyn_lib
	#compileCommand = `icc $iccOpts -qopenmp -fpic -c -o $package_root/intel-runtime/out.o $cgenOutput $otherArgs -I$package_root/src/arena_allocator -qoffload-attribute-target=mic`
	compileCommand = `icc $iccOpts -qopenmp -fpic -c -o $packageroot/src/intel-runtime/out.o $cgenOutput $otherArgs`
	run(compileCommand)	
end

function link()
	packageroot = getPackageRoot()
	#linkCommand = `icc -shared -Wl,-soname,libout.so.1 -o $package_root/src/intel-runtime/libout.so.1.0 $package_root/src/intel-runtime/out.o -lc $package_root/src/intel-runtime/lib/libintel-runtime.so`
	linkCommand = `icc -shared -qopenmp -o $packageroot/src/intel-runtime/libout.so.1.0 $packageroot/src/intel-runtime/out.o`
	run(linkCommand)
	debugp("Done cgen linking")
end

# When in standalone mode, this is the entry point to cgen
function generate(func::Function, typs)
	name = string(func.env.name)
	insert(func, name, typs)
	return from_worklist()
end
end # cgen module
