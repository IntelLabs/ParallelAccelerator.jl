require("intel-pse.jl")
importall IntelPSE
ParallelIR.set_debug_level(3)
ParallelIR.PIRSetFuseLimit(-1)
IntelPSE.set_debug_level(3)
ccall(:set_j2c_verbose, Void, (Cint,), 9)

#####	macros for replacing array comprehensions
#####	to cartesianarray calls
macro parallelize1D(comp)
	compreFun = gensym("compreFun")
	compreTyp = gensym("compreTyp")

	@eval function $compreFun($(comp.args[2].args[1]))
					$(comp.args[1])
				end
	return :(
			cartesianarray($compreFun, Int64, 
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
macro parallelize2D(comp)
	compreFun = gensym("compreFun")
	compreTyp = gensym("compreTyp")
	@eval function $compreFun($(comp.args[2].args[1]), $(comp.args[3].args[1]))
		$(comp.args[1])
	end
	return :(
			#$compreTyp = code_typed($compreFun, 
			#	(Int64, Int64))
			#cartesianarray($compreFun, $(compreTyp)[1].args[3].typ, 
			cartesianarray($compreFun, Int64, 
				($(comp.args[2].args[2].args[2]), $(comp.args[3].args[2].args[2])))
		)
end
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
macro rank(comp)
	return :(length(comp.args)-1)
end

macro parallelize(comp)
	(isa(comp, Expr) && comp.head == :comprehension) || throw("comprehension expected")
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
end

#### Test program

function render()
	#c3 = @parallelize [i*j*k for i in 1:4, j in 1:3, k = 1:5]
	#println("Expanded:")
	#ex = macroexpand( @parallelize [i*j for i in 1:4, j in 1:2])
	#println(ex)
	#println("End Expanded")
	#c3 = @parallelize2D [i*j for i in 1:4, j in 1:2]
	c3 = @parallelize1D [i for i in 1:4]
	return c3
end
IntelPSE.offload(render, ())
function main()
	c = render()
	println(c)
end
main()

