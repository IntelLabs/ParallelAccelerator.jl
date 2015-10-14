using ParallelAccelerator
using Base.Test


@test Example1.main() == 10000

### Tests that illustrate known bugs.

include("abs.jl")
using AbsTest
# KeyError: Union{} not found
@test_throws KeyError AbsTest.test1()

# ErrorException("failed process: Process(`icc [...]
@test_throws ErrorException AbsTest.test2()
