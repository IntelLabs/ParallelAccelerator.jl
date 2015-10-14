using ParallelAccelerator
using Base.Test

include("abs.jl")
using AbsTest

### Working tests.

@test AbsTest.test1() == ones(10, 10)

### Tests that illustrate known bugs.

@test_throws ErrorException AbsTest.test2() == 3
