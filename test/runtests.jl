using ParallelAccelerator
using Base.Test

ParallelAccelerator.DomainIR.set_debug_level(4)
ParallelAccelerator.ParallelIR.set_debug_level(4)
ParallelAccelerator.cgen.set_debug_level(4)
ParallelAccelerator.set_debug_level(4)

include("example1.jl")
using Example1

@test Example1.main() == 10000

### Tests that illustrate known bugs.

include("abs.jl")
using AbsTest
# KeyError: Union{} not found
@test_throws KeyError ParallelAccelerator.accelerate(AbsTest.example, (Array{Float64,2},))

# ErrorException("failed process: Process(`icc [...]
@test_throws ErrorException ParallelAccelerator.accelerate(AbsTest.example, (Int64,))
