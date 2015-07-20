# A prototype Julia to C++ generator
# jaswanth.sreeram@intel.com

module cgen
using ..ParallelIR
using ..IntelPSE
export generate, from_root, writec, compile, link
import IntelPSE.getPackageRoot

verbose = true
_symPre = 0
_jtypesToctypes = Dict(
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

# These are primitive operators on scalars and arrays
_operators = ["*", "/", "+", "-", "<", ">"]
# These are primitive "methods" for scalars and arrays
_builtins = ["getindex", "setindex", "arrayref", "top", "box", 
			"unbox", "tuple", "arraysize", "arraylen", "ccall",
			"arrayset", "getfield", "unsafe_arrayref", "unsafe_arrayset",
			"call1", ":jl_alloc_array_1d", ":jl_alloc_array_2d"]

# Intrinsics
_Intrinsics = [
		"===",
		"box", "unbox",
        #arithmetic
        "neg_int", "add_int", "sub_int", "mul_int", "sle_int",
        "sdiv_int", "udiv_int", "srem_int", "urem_int", "smod_int",
        "neg_float", "add_float", "sub_float", "mul_float", "div_float",
		"rem_float", "sqrt_llvm", "fma_float", "muladd_float",
        "fptoui", "fptosi", "uitofp", "sitofp", "not_int",
		"nan_dom_err", "lt_float", "slt_int", "abs_float", "select_value",
		"fptrunc", "fpext"
]

function debugp(a...)
	#verbose && println(a...)
end

inEntryPoint = false
ompPrivateList = []

function getenv(var::String)
  ENV[var]
end

#= function getPackageRoot()
	package_root = getenv("JULIA_ROOT")
	len_root     = endof(package_root)
	if(package_root[len_root] == '/')
		package_root = package_root[1:len_root-1]
  	end
	package_root
end =#

globalUDTs = Dict()

function from_header(isEntryPoint::Bool)
	s = from_UDTs()
	isEntryPoint ? from_includes() * s : s
end

function from_includes()
	package_root = getPackageRoot()
	reduce(*, "", (
		"#include <omp.h>\n",
		"#include <stdint.h>\n",
		"#include <math.h>\n",
		"#include <stdio.h>\n",
		"#include <iostream>\n",
		"#include \"$package_root/src/intel-runtime/include/pse-runtime.h\"\n",
		"#include \"$package_root/src/intel-runtime/include/j2c-array.h\"\n",
		"#include \"$package_root/src/intel-runtime/include/j2c-array-pert.h\"\n",
		"#include \"$package_root/src/intel-runtime/include/pse-types.h\"\n")
	)
end

function from_UDTs()
	global globalUDTs
	isempty(globalUDTs) ? "" : mapfoldl((a) -> from_decl(a), (a, b) -> "$a; $b", keys(globalUDTs))
end


function from_decl(k::Tuple)
	debugp("from_decl: Defining tuple type ", k)
	s = "typedef struct {\n"
	for i in 1:length(k)
		s *= toCtype(k[i]) * " " * "f" * string(i-1) * ";\n"
	end
	s *= "} Tuple_" * mapfoldl((a) -> canonicalize(a), (a, b) -> "$(a)_$(b)", k) * ";\n"
	s
end

function from_decl(k::DataType)
	debugp("from_decl: Defining data type ", k)
	if is(k, UnitRange{Int64})
		btyp, ptyp = parseParametricType(k)
		s = "typedef struct {\n\t"
		s *= toCtype(ptyp[1]) * " start;\n"
		s *= toCtype(ptyp[1]) * " stop;\n"
		s *= "} " * canonicalize(k) * ";\n"
		return s
	end
	return ""
end

function from_decl(k)
	debugp("from_decl: Defining type ", k)
	return toCtype(_symbolTable[k]) * " " * canonicalize(k) * ";\n"
end

function isCompositeType(t)
	# TODO: Expand this to real UDTs
	b = isa(t, Tuple) || is(t, UnitRange{Int64})
	debugp("Is ", t, " a composite type: ", b)
	b
end

function from_lambda(args)
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
	global ompPrivateList
	global globalUDTs
	debugp("ompPrivateList is: ", ompPrivateList)
	for k in 1:length(vars)
		v = vars[k]
		_symbolTable[v[1]] = v[2]
		# If we have user defined types, record them
		if in(v[1], locals) && (v[3] & 32 != 0)
			push!(ompPrivateList, v[1])
		end 
	end
	debugp("Done with ompvars")
	bod = from_expr(args[3])
	debugp("lambda locals = ", locals)
	debugp("lambda params = ", params)
	debugp("symboltable here = ")
	dumpSymbolTable(_symbolTable)
	
	for k in keys(_symbolTable)
		if (has(locals, k) && !has(params, k)) || (!has(locals, k) && !has(params, k))
			#decls *= from_decl(k)
			#if isa(_symbolTable[k], Tuple)
			#if !isPrimitiveJuliaType(_symbolTable[k])
			debugp("About to check for composite type: ", _symbolTable[k])
			if isCompositeType(_symbolTable[k])
				#globalUDTs, from_decl(_symbolTable[k])
				globalUDTs[_symbolTable[k]] = 1
			end
			decls *= toCtype(_symbolTable[k]) * " " * canonicalize(k) * ";\n"
			#end
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

_symbolTable = Dict{Any, Any}()
_compiledFunctions = []
_worklist = []

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
	return has(names(a), f)
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
			_symbolTable[lhs] = rhs.typ
		elseif haskey(_symbolTable, rhs)
			_symbolTable[lhs] = _symbolTable[rhs]
		elseif isPrimitiveJuliaType(typeof(rhs))
			_symbolTable[lhs] = typeof(rhs)
		elseif isPrimitiveJuliaType(typeof(rhsO))
			_symbolTable[lhs] = typeof(rhs0)
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
	haskey(_jtypesToctypes, t)
end

function isArrayType(typ)
	#ndims(typ) > 0
	#TODO: use parseParametricType instead
	startswith(string(typ), "Array")
end

function toCtype(typ::Tuple)
	return "Tuple_" * mapfoldl((a) -> canonicalize(a), (a, b) -> "$(a)_$(b)", typ)
end

function toCtype(typ)
	debugp("Converting type: ", typ, " to ctype")
	if haskey(_jtypesToctypes, typ)
		debugp("Found simple type: ", typ, " returning ctype: ", _jtypesToctypes[typ])
		return _jtypesToctypes[typ]
	elseif isArrayType(typ)	
		atyp, dims = parseArrayType(typ)
		debugp("Found array type: ", atyp, " with dims: ", dims)
		atyp = toCtype(atyp)
		debugp("Atyp is: ", atyp)
		assert(dims >= 0)
		return "j2c_array<$(atyp)>"
	elseif in(:parameters, names(typ)) && length(typ.parameters) != 0
		# For parameteric types, for now assume we have equivalent C++
		# implementations
		btyp, ptyps = parseParametricType(typ)
		#return toCtype(b) * toCtype(p[1]) * " * "
		return canonicalize(btyp) * mapfoldl((a) -> canonicalize(a), (a, b) -> a * b, ptyps)
	else
		return canonicalize(typ)
	end
end

function genName()
	global _symPre
	_symPre += 1
	"anon" * string(_symPre)
end

tokenXlate = Dict(
	'*' => "star",
	'/' => "slash",
	'-' => "minus",
	'!' => "bang",
	'.' => "dot"
)

function canonicalize_re(tok)
	debugp("Canonicalizing ", tok, " : ", string(tok)) 
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

_replacedTokens = Set("(,)#")
_scrubbedTokens = Set("{}:")
function canonicalize(tok)
	global _replacedTokens
	global _scrubbedTokens
	s = string(tok)
	debugp("Canonicalizing ", tok, " : ", string(tok)) 
	s = replace(s, _scrubbedTokens, "")
	s = replace(s, r"^[^a-zA-Z]", "_")
	s = replace(s, _replacedTokens, "p")
	debugp("Canonicalized token = ", s) 
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
		#s *= from_expr(args[i]) * (i < length(args) ? ", " : "")
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
	debugp("Done with ccall args")
	fun = args[1]
	#s = from_expr(fun)	
	if isInlineable(fun, args)
		return from_inlineable(fun, args)
	end

	if isa(fun, QuoteNode)
		s = from_expr(fun)
	elseif isa(fun, Expr) && ( is(fun.head, :call1) || is(fun.head, :call))
		s = string(fun.args[2]) #* "/*" * from_expr(fun.args[3]) * "*/"
	else
		throw("Invalid ccall format...")
	end
	debugp("Length is ", length(args) )
	s *= "("
	debugp("Ccall args start: ", round(Int, (length(args)-1)/2)+2)
	debugp("Ccall args end: ", length(args))
	numInputs = length(args[3].args)-1
	argsStart = round(Int, (length(args)-1)/2)+2
	argsEnd = argsStart + numInputs - 1
	#for i in round(Int, (length(args)-1)/2)+2:length(args)
	for i in argsStart : argsEnd
		s *= from_expr(args[i]) * (i < argsEnd ? ", " : "")
	end
	s *= ")"
	debugp("from_ccall: ", s)
	s
end

function from_arrayset(args)
	debugp("arrayset args are: ", args, length(args))
	#idxs = map((i) -> from_expr(i), args[3:end])
	idxs = mapfoldl((a) -> from_expr(a), (a, b) -> "$a, $b", args[3:end]) 
	src = from_expr(args[1])
	val = from_expr(args[2])
	#src, val, idx = map((i)->from_expr(i), args)
	#s = "$(src).SETELEM($(idx), $(val))"
	"$src.ARRAYELEM($idxs) = $val"
end

function from_getfield(args)
	debugp("Getfield, args are: ", length(args))
	#assert(length(args) == 2)
	mod, tgt = resolveCallTarget(args[1], args[2:end])
	#rcvr = from_expr(args[1])
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
	elseif tgt == "unsafe_arrayset"
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

	#from_expr(args[1]) * @eval("from_" * string($f) * "( " * $args) * from_expr(args[2])
	if intr == "mul_int"
		s = from_expr(args[1]) * " * " * from_expr(args[2])
		return s
	elseif intr == "mul_float"
		s = from_expr(args[1]) * " * " * from_expr(args[2])
		return s;
	elseif intr == "add_int"
		return from_expr(args[1]) * " + " * from_expr(args[2])
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
		return "~" * "(" * from_expr(args[1]) * ")"
	elseif intr == "add_float"
		return from_expr(args[1]) * " + " * from_expr(args[2])
	elseif intr == "lt_float"
		return from_expr(args[1]) * " < " * from_expr(args[2])
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
	elseif f == :(===)
		return "(" * from_expr(args[1]) * " == " * from_expr(args[2]) * ")"
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
	ntuple(length(a), (i)->a[i])
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
		ast, name, typs = _worklist[i]
		if name == tgt
			return true
		end
	end
	return false
end
#=
function resolveCallTarget(ast)
	debugp("In call, ast is: ", ast)
	topn = false
	if typeof(ast[1]) == TopNode
		#fun = from_top(ast[1])
		fun = ast[1].name
		topn = true
		debugp("Got topnode, setting target to ", fun)
	else
		fun = ast[1]
	end
end
=#
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


function from_call(ast::Array{Any, 1})
	# pattern match math functions to avoid overheads
	s = pattern_match_call(ast)
	if(s != "")
		return s;
	end

	debugp("Compiling call: ast = ", ast, " args are: ")
	for i in 1:length(ast)
		debugp("Arg ", i, " = ", ast[i], " type = ", typeof(ast[i]))
	end
	#mod, fun = resolveCallTarget(ast[1], ast[2:end])
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
	skipCompilation = has(_compiledFunctions, funStr) ||
		isPendingCompilation(_worklist, funStr)
	#compiledPreviously = has(_compiledFunctions, string(fun)) || has(_worklist, string(fun))
	s = ""
	debugp("Worklist: ", _worklist)
	debugp("Compiled functionslist: ", _compiledFunctions)
	map((i)->debugp(i), _compiledFunctions)
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
			end
		end
	end
	s *= ")"
	debugp("Finished translating call : ", s)
	debugp(ast[1], " : ", typeof(ast[1]), " : ", hasfield(ast[1], :head) ? ast[1].head : "")
	if !skipCompilation && (isa(fun, Symbol) || isa(fun, Function))
		debugp("Inserting: ", fun, " : ", "_" * canonicalize(fun), " : ", arrayToTuple(argTyps))
		insert(fun, "_" * canonicalize(fun), arrayToTuple(argTyps))
	end
	s
end

function from_return(args)
	global inEntryPoint
	debugp("Return args are: ", args)
	#debugp("Type of ret are: ", retTyps)
	retExp = ""
	if inEntryPoint
		if typeAvailable(args[1]) && isa(args[1].typ, Tuple)
			retTyps = args[1].typ
			for i in 1:length(retTyps)
			#retExp *= "*ret" * string(i-1) * " = " * from_expr(args[1].args[i+1]) * ";\n"
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
	s = ""
	parfor = args[1]
	lpNests = parfor.loopNests
	for i in 1:length(lpNests)
		s *= "}\n"	
	end
	s *= "}\n}/*parforend*/\n" # end block introduced by private list
	debugp("Parforend = ", s)
	s
end

function loopNestCount(loop)
	"(((" * from_expr(loop.upper) * ") + 1 - (" * from_expr(loop.lower) * ")) / (" * from_expr(loop.step) * "))"
end


# If the parfor body is too complicated then DIR or PIR will set
# instruction_count_expr = nothing

# Meaning of num_threads_mode
# mode = 1 uses static insn count if it is there, but doesn't do dynamic estimation and fair core allocation between levels in a loop nest.
# mode = 2 does all of the above
# mode = 3 in addition to 2, also uses host minimum (0) and Phi minimum (10)

function from_parforstart(args)
	num_threads_mode = ParallelIR.num_threads_mode

	parfor = args[1]
	lpNests = parfor.loopNests
	global ompPrivateList
	s = ""
	privatevars = isempty(ompPrivateList) ? "" : "private(" * mapfoldl((a) -> canonicalize(a), (a,b) -> "$a, $b", ompPrivateList) * ")"

	ivs = map((a)->from_expr(a.indexVariable), lpNests)
	starts = map((a)->from_expr(a.lower), lpNests)
	stops = map((a)->from_expr(a.upper), lpNests)
	steps = map((a)->from_expr(a.step), lpNests)

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

	s *= "{\n$preclause #pragma omp parallel $nthreadsclause $privatevars\n{\n"
	s *= "#pragma omp for private($(ivs[1]))\n"
	s *= mapfoldl(
			(i) -> "for ( $(ivs[i]) = $(starts[i]); $(ivs[i]) <= $(stops[i]); $(ivs[i]) += $(steps[i])) {\n",
			(a, b) -> "$a $b",
			1:length(lpNests))
	s
end
#=

	debugp("Got Parfor start, args = ", args)
	debugp("insns count Expr is: ", parfor.instruction_count_expr)
	debugp("After translation, insns count Expr is: ", from_expr(parfor.instruction_count_expr))
	lpNests = parfor.loopNests
	debugp("In parforstart, ompPrivateList is: ", ompPrivateList)
	s *= isempty(ompPrivateList) ? "" : "#pragma omp parallel private(" * privateVars * ")\n{\n"
	for i in 1:length(lpNests)
		iv = from_expr(lpNests[i].indexVariable)
		start = from_expr(lpNests[i].lower)
		stop = from_expr(lpNests[i].upper)
		step = from_expr(lpNests[i].step)
		s *= (i == 1) ? "#pragma omp for private($iv)\n" : ""
		s *= "for ($iv =  $start;$iv <= $stop; ($iv) += $step) {\n"
	end
	debugp("Parforstart = ", s)
	s
end
=#

#gnumthreads_mode = 2
#=
function from_parforstart(args)
	
	if(gnumthreads_mode != 0)
		return from_parforstart(args, global_num_threads_mode)
	s = ""
	parfor = args[1]
	debugp("Got Parfor start, args = ", args)
	debugp("insns count Expr is: ", parfor.instruction_count_expr)
	debugp("After translation, insns count Expr is: ", from_expr(parfor.instruction_count_expr))
	global ompPrivateList
	debugp("In parforstart, ompPrivateList is: ", ompPrivateList)
	privateVars = isempty(ompPrivateList) ? "" : mapfoldl((a) -> canonicalize(a), (a,b) -> "$a, $b", ompPrivateList)
	s *= isempty(ompPrivateList) ? "" : "#pragma omp parallel private(" * privateVars * ")\n{\n"
	for i in 1:length(lpNests)
		iv = from_expr(lpNests[i].indexVariable)
		start = from_expr(lpNests[i].lower)
		stop = from_expr(lpNests[i].upper)
		step = from_expr(lpNests[i].step)
		s *= (i == 1) ? "#pragma omp for private($iv)\n" : ""
		s *= "for ($iv =  $start;$iv <= $stop; ($iv) += $step) {\n"
	end
	debugp("Parforstart = ", s)
	s
end
=#

# TODO: Should simple objects be heap allocated ?
# For now, we stick with stack allocation
function from_new(args)
	typ = args[1] #type of the object
	assert(isa(typ, DataType))
	objtyp, ptyps = parseParametricType(typ)
	#ctyp = objtyp * join(ptyps)
	#ctyp = canonicalize(objtyp) * mapfoldl((a) -> toCtype(a), (a, b) -> a * b, ptyps)
	ctyp = canonicalize(objtyp) * mapfoldl((a) -> canonicalize(a), (a, b) -> a * b, ptyps)
	#s = "new " * ctyp * "("
	s = ctyp * "{"
	#for a in 2:length(args)
	#	s *= from_expr(args[a]) * (a < length(args) ? ", " : "")
	#end
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
		s *= from_lambda(args)

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
		s *= "(" * toCtype(asttyp) * ")" * string(ast)
	elseif isPrimitiveJuliaType(ast)
		s *= "(" * toCtype(ast) * ")"
	else
		s *= dispatch(adp, ast, ast)
	end
	s
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
		if haskey(_symbolTable, params[p])
			s *= (toCtype(_symbolTable[params[p]])
				* (isArrayType(_symbolTable[params[p]]) ? "&" : "")
				* (isArrayType(_symbolTable[params[p]]) ? " $ql " : " ")
				* canonicalize(params[p])
				* (p < length(params) ? ", " : ""))
		end
	end
	debugp("Formal args are: ", s)
	debugp("Packege root is: ", IntelPSE.getPackageRoot())
	s
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
	dumpSymbolTable(_symbolTable)
	s::ASCIIString = "$typ $functionName($args) { $bod } "
	s
end

type ASTDispatcher
	nodes::Array{Any, 1}
	dt::Dict{Any, Any}
	m::Module
	function ASTDispatcher()
		d = Dict{Any, Any}()
		n = [Expr, Symbol, SymbolNode, LineNumberNode, LabelNode,
			GotoNode, TopNode, QuoteNode,
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

adp = ASTDispatcher()

function isScalarType(typ)
	!isArrayType(typ) && !isCompositeType(typ)
end

# Creates an entrypoint that dispatches onto host or MIC.
# For now, emit host path only
function createEntryPointWrapper(functionName, params, args, jtyp)
	actualParams = 
		mapfoldl((a)->canonicalize(a), (a,b) -> "$a, $b", params) * 
		", " * foldl((a, b) -> "$a, $b",
		[(isScalarType(jtyp[i]) ? "" : "*") * "ret" * string(i-1) for i in 1:length(jtyp)])
	wrapperParams = "int run_where, $args"
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
	debugp("Ast = ", ast)
	verbose && debugp("Starting processing for $ast")
	params	=	ast.args[1]
	env		=	ast.args[2]
	bod		=	ast.args[3]
	returnType = bod.typ
	typ = returnType
	#f = Dict(ast => functionName)
	bod = from_expr(ast)
	args = from_formalargs(params)
	argsunal = from_formalargs(params, true)
	dumpSymbolTable(_symbolTable)
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
		retargs = ", " * foldl((a, b) -> "$a, $b",
		[toCtype(returnType[i]) * " * __restrict ret" * string(i-1) for i in 1:length(returnType)])

		args *= retargs
		argsunal *= retargs
	else
		
		rtyp = toCtype(typ)
	end
	s::ASCIIString = "$rtyp $functionName($args)\n{\n$bod\n}\n" * (isEntryPoint ? "$rtyp $(functionName)_unaliased($argsunal)\n{\n$bod\n}\n" : "")
	if inEntryPoint
		inEntryPoint = false
	end
	hdr * from_worklist() * s * wrapper
end

function insert(func::Symbol, name, typs)
	target = resolveFunction(func, typs)
	debugp("Resolved function ", func, " : ", name, " : ", typs, " target ", target)
	insert(target, name, typs)
end

function insert(func::Function, name, typs)
	#ast = code_typed(func, typs; optimize=true)
	ast = code_typed(func, typs)
	if !has(_compiledFunctions, name)
		push!(_worklist, (ast, name, typs))
	end
end

function from_worklist()
	s = ""
	si = ""
	while !isempty(_worklist)
		a, fname, typs = pop!(_worklist)
		if has(_compiledFunctions, fname)
			continue
		end
		empty!(_symbolTable)
		empty!(globalUDTs)
		empty!(ompPrivateList)
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
		debugp(_compiledFunctions)
		push!(_compiledFunctions, fname)
		s *= si
	end
	s
end
import Base.write
function writec(s::ASCIIString)
	package_root = getPackageRoot()
	cgenOutput = "$package_root/src/intel-runtime/tmp.cpp"
	cf = open(cgenOutput, "w")
	write(cf, s)
	debugp("Done committing cgen code")
	close(cf)
end

function compile()
	package_root = getPackageRoot()
	
	cgenOutputTmp = "$package_root/src/intel-runtime/tmp.cpp"
	cgenOutput = "$package_root/src/intel-runtime/out.cpp"

	# make cpp code readable
	beautifyCommand = `bcpp $cgenOutputTmp $cgenOutput`
	run(beautifyCommand)

	iccOpts = "-O3 --offload-mode=none"
#	iccOpts = "-O3"
	otherArgs = "-DJ2C_REFCOUNT_DEBUG -DDEBUGJ2C"
	# Generate dyn_lib
	#compileCommand = `icc $iccOpts -qopenmp -fpic -c -o $package_root/intel-runtime/out.o $cgenOutput $otherArgs -I$package_root/src/arena_allocator -qoffload-attribute-target=mic`
	compileCommand = `icc $iccOpts -qopenmp -fpic -c -o $package_root/src/intel-runtime/out.o $cgenOutput $otherArgs --offload-mode=none`
	run(compileCommand)	
end

function link()
	package_root = getPackageRoot()
	linkCommand = `icc -shared -Wl,-soname,libout.so.1 -o $package_root/src/intel-runtime/libout.so.1.0 $package_root/src/intel-runtime/out.o -lc $package_root/src/intel-runtime/lib/libintel-runtime.so`
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
