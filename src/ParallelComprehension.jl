#require("intel-pse.jl")
#importall IntelPSE
#ParallelIR.set_debug_level(3)
#ParallelIR.PIRSetFuseLimit(-1)
#IntelPSE.set_debug_level(3)
#ccall(:set_j2c_verbose, Void, (Cint,), 9)

#require("/home/etotoni/.julia/v0.4/CompilerTools/src/ast_walk.jl")
import CompilerTools.AstWalker

export @replace_comprehensions
#####	macros for replacing array comprehensions
#####	to cartesianarray calls
macro parallelize1D(comp, typ)
	compreFun = gensym("compreFun")
	compreTyp = gensym("compreTyp")

	@eval function $compreFun($(comp.args[2].args[1]))
					$(comp.args[1])
				end
	return :(
			cartesianarray($compreFun, $typ, 
				($(comp.args[2].args[2].args[2]),))
			)
end

macro parallelize1D_CT(comp)
	compreFun = gensym("compreFun")
	compreTyp = gensym("compreTyp")
	return :(
		(function()
			function $compreFun($(comp.args[2].args[1]))
				$(comp.args[1])
			end
			#$compreTyp = code_typed($compreFun, 
			#	(Int64))
			#cartesianarray($compreFun, $(compreTyp)[1].args[3].typ, 
			cartesianarray($compreFun, Int64, 
				($(comp.args[2].args[2].args[2])))
		end)())
end

macro parallelize2D_CT(comp)
	compreFun = gensym("compreFun")
	compreTyp = gensym("compreTyp")
	return :(
		(function()
			function $compreFun($(comp.args[2].args[1]), $(comp.args[3].args[1]))
				$(comp.args[1])
			end
			#$compreTyp = code_typed($compreFun, 
			#	(Int64, Int64))
			#cartesianarray($compreFun, $(compreTyp)[1].args[3].typ, 
			cartesianarray($compreFun, Int64, 
				($(comp.args[2].args[2].args[2]), $(comp.args[3].args[2].args[2])))
		end)())
end
macro parallelize2D(comp, typ)
	compreFun = gensym("compreFun")
	compreTyp = gensym("compreTyp")
	@eval function $compreFun($(comp.args[2].args[1]), $(comp.args[3].args[1]))
		$(comp.args[1])
	end
	return :(
			#$compreTyp = code_typed($compreFun, 
			#	(Int64, Int64))
			#cartesianarray($compreFun, $(compreTyp)[1].args[3].typ, 
			cartesianarray($compreFun, $typ, 
				($(comp.args[2].args[2].args[2]), $(comp.args[3].args[2].args[2])))
		)
end
#=
macro parallelize3D(comp)
	compreFun = gensym("compreFun")
	compreTyp = gensym("compreTyp")
	@eval	function $compreFun($(comp.args[2].args[1]),
					$(comp.args[3].args[1]), 
					$(comp.args[4].args[1]))
					$(comp.args[1])
				end
	return :(
			$compreTyp = code_typed($compreFun, 
				(Int64, Int64, Int64))
			cartesianarray($compreFun,
				$(compreTyp)[1].args[3].typ, 
				($(comp.args[2].args[2].args[2]), $(comp.args[3].args[2].args[2]), 
					$(comp.args[4].args[2].args[2])))
		)
end
macro parallelize3D_CT(comp)
	compreFun = gensym("compreFun")
	compreTyp = gensym("compreTyp")
	return :(
		(function()
			function $compreFun($(comp.args[2].args[1]),
				$(comp.args[3].args[1]), 
				$(comp.args[4].args[1]))
				$(comp.args[1])
			end
			$compreTyp = code_typed($compreFun, 
				(Int64, Int64, Int64))
			cartesianarray($compreFun,
				$(compreTyp)[1].args[3].typ, 
				($(comp.args[2].args[2].args[2]), $(comp.args[3].args[2].args[2]), 
					$(comp.args[4].args[2].args[2])))
		end)())
end
=#
macro rank(comp)
	return :(length(comp.args)-1)
end

macro parallelize(comp, typ)
	(isa(comp, Expr) && comp.head == :comprehension) || throw("comprehension expected")
	cRank = @rank(comp)
	compreFun = gensym("compreFun")
  block = quote
    function $compreFun()
      $(comp.args[1])
    end
  end
  curr_body = block.args[2].args[2].args
  # We insert into the body of the function logic to transform the coordinates
  # based on the range specified.  Cartesian array expects the size of the
  # array in the initializer, so the function should offset the coordinates
  # passed in based on the start of the range
  # For example, if the comprehension is [i for i in 3:5], the initialization
  # function should increment i by 2 because the cartesian array will start at
  # index 1
  # TODO: This fails in parallel-ir
  # new_body = Array(Any, length(curr_body) + length(comp.args) - 1)
  # new_body[1:length(comp.args)-1] = [:($(var.args[1]) += $(var.args[2].args[1]) - 1) 
  #                                    for var in comp.args[2:end]]
  # new_body[length(comp.args):end] = curr_body
  # block.args[2].args[2].args = new_body
  orig = block.args[2].args[1].args[1]

  # We add an argument to the initialization function for every variable in our
  # comprehension, allowing us to support n-dimensional comprehensions.
  block.args[2].args[1].args = Array(Any, length(comp.args))
  block.args[2].args[1].args[1] = orig
  block.args[2].args[1].args[2:end] = [var.args[1] for var in comp.args[2:end]]

	@eval $block
	constructor = :(
			cartesianarray($compreFun, $typ, 
				())
			)

  # The dimensions of the cartesian array are the end of the range minus the
  # start of the range plus one.
  constructor.args[4].args = [:($(range.args[2].args[2]) - $(range.args[2].args[1]) + 1)
                              for range in comp.args[2:end]]
  return constructor
end

#### Test program

#=
function process_node(node)
  if !isa(node, Expr)
    return node
  elseif node.head == :comprehension
    comp = node
	cRank = @rank(comp)
	if(cRank == 1)
		return :(@parallelize1D($comp))
	elseif(cRank == 2)
		return :(@parallelize2D($comp))
	elseif(@rank(comp) == 3)
		return :(@parallelize3D($comp))
	else
		return comp
	end
  elseif node.head == :(=)
    node.args[2] = process_node(node.args[2])
  elseif node.head == :return
    node.args[1] = process_node(node.args[1])
  end
  return node
end
=#
#AstWalker.set_debug_level(10)
function infer_comprehension_type(comp)
  eval(quote 
    function _tmp()
      $comp
    end
  end)
  a = code_typed(_tmp, ())
  # code_typed allows us to get the inferred type of the comprehension, we use
  # string processing + Symbol + Eval to extract the actual element type from
  # the resulting Array.
  # TODO: There's probably a better way to do this
  # a - :function
  block = a[1].args[end]
  ret_statement = block.args[end]
  ret_value = ret_statement.args[1]
  typ_str = string(ret_value.typ)
  # typ_str = "Array{T, N}"
  vals = split(typ_str, "{")
  # vals = ["Array", "T, N}]"
  return eval(convert(Symbol, split(vals[2], ",")[1]))  # T
end

function process_node(node, state, top_level_number, is_top_level, read)
	if !isa(node,Expr)
    return nothing
  end
  if node.head == :typed_comprehension
    typ = node.args[1]
    # Transform into untyped :comprehension because type will be passed as a
    # parameter the same way we handle untyped :comprehensions
    node.head = :comprehension
    node.args = node.args[2:end]
  elseif (node.head == :comprehension)
    typ = infer_comprehension_type(node)
  end
  if (node.head == :comprehension)
    return :(@parallelize($node, $typ))
  end
end

macro replace_comprehensions(func)
#  println("------------- Before ---------------")
#  println(func)
#  println("------------------------------------")
	AstWalker.AstWalk(func, process_node, nothing)
#	println("done walk")
#=	if !(isa(func, Expr) && func.head == :function)
    throw("replace_comprehensions expects a function")
  end
  map!(process_node, func.args[2].args)
  =#
#  println("------------- After ----------------")
#  println(func)
#  println("------------------------------------")
  return eval(func)
end

#=
@replace_comprehensions function render()
	#c3 = @parallelize [i*j*k for i in 1:4, j in 1:3, k = 1:5]
	#println("Expanded:")
	#ex = macroexpand( @parallelize [i*j for i in 1:4, j in 1:2])
	#println(ex)
	#println("End Expanded")
	#c3 = @parallelize2D [i*j for i in 1:4, j in 1:2]
	c3 = [i for i in 3:9]
	return c3
end
IntelPSE.offload(render, ())
function main()
	c = render()
	println(c)
end
main()
=#
