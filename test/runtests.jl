using ParallelAccelerator
using Base.Test

include("abs.jl")
using AbsTest

### Working tests.

@test AbsTest.test1() == ones(10, 10)

@test AbsTest.test2() == 3

### Tests that illustrate known bugs.

@test AbsTest.test3() == 3
