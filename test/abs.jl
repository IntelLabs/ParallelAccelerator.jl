module AbsTest
using ParallelAccelerator

ParallelAccelerator.DomainIR.set_debug_level(4)
ParallelAccelerator.ParallelIR.set_debug_level(4)
ParallelAccelerator.cgen.set_debug_level(4)
ParallelAccelerator.set_debug_level(4)

function example(x)
    abs(x)
end

function test1()
    ParallelAccelerator.accelerate(AbsTest.example, (Array{Float64,2},))
end

function test2()
    ParallelAccelerator.accelerate(AbsTest.example, (Int,))
end

end
