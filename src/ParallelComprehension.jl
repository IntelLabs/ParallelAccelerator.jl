import CompilerTools.AstWalker

export @replace_comprehensions

macro rank(comp)
	return :(length(comp.args)-1)
end

function parallelize(comp, typ)
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

	constructor = :(cartesianarray($compreFun, $typ, ())::Array{$typ, $(length(comp.args[2:end]))})

  # The dimensions of the cartesian array are the end of the range minus the
  # start of the range plus one.
  constructor.args[1].args[4].args = [range.args[2].args[2] - range.args[2].args[1] + 1
                                      for range in comp.args[2:end]]
  return constructor
end

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
    return parallelize(node, typ)
  end
end

macro replace_comprehensions(func)
	AstWalker.AstWalk(func, process_node, nothing)
  println(func)
  return eval(func)
end

using IntelPSE
IntelPSE.set_debug_level(3)
# IntelPSE.DomainIR.set_debug_level(3)
# IntelPSE.ParallelIR.set_debug_level(3)
@replace_comprehensions function test()
  a = Float64[i for i in 1:4]
  return a
end

test2 = IntelPSE.offload(test, ())

function main()
  a = test2()
  println(a)
end

main()
