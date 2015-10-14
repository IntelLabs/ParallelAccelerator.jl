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
    example_acc = ParallelAccelerator.accelerate(AbsTest.example, (Array{Float64,2},))
    A = ones(10, 10)
    return example_acc(A)
end

function test2()
    ParallelAccelerator.accelerate(AbsTest.example, (Int,))
    return example_acc(3)
end

end
