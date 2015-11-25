#=
Copyright (c) 2015, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
=#

using ParallelAccelerator
using Base.Test

include("abs.jl")
using AbsTest
println("Testing abs()...")
@test AbsTest.test1() == 3
@test AbsTest.test2() == 3.0
@test AbsTest.test3() == 3.0
@test AbsTest.test4() == ones(10, 10)
@test AbsTest.test5() == ones(10, 10).+0.1
@test AbsTest.test6() == ones(10, 10)
println("Done testing abs().")

include("rand.jl")
using RandTest
println("Testing rand()...")
@test all(RandTest.test1() .<= ones(2,3).*2.0) && all(RandTest.test1() .>= zeros(2,3)) 
@test size(RandTest.test2())==(2,3)
println("Done testing rand()...")


include("BitArray.jl")
using BitArrayTest 
println("Testing BitArrays...")
@test BitArrayTest.test1() == [1.1; 0.1; 0.1; 1.9]
@test BitArrayTest.test2() == [1.1; 0.1; 0.1; 1.9]
@test BitArrayTest.test3() == [1.1; 9.0; 10.0; 1.9]
println("Done testing BitArrays.")

include("seq.jl")
using SeqTest 
println("Testing sequential code...")
@test_approx_eq SeqTest.test1() [9.0 12.0; 10.0 20.0]
println("Done testing sequential code.")

include("cat.jl")
using CatTest 
println("Testing cat...")
@test_approx_eq CatTest.test1() [3.0; 1.0]
@test_approx_eq CatTest.test2() [2.0 2.5; 1.5 11.0]
@test_approx_eq CatTest.test3() [2.0 2.5; 1.5 11.0]
println("Done testing cat.")

include("ranges.jl")
using RangeTest
println("Testing range related features.")
rt1 = RangeTest.test1()
@test ndims(rt1) == 1
@test rt1 == [2.2; 6.6]
rt2 = RangeTest.test2()
@test ndims(rt2) == 2
@test rt2 == reshape([2.2 6.6], 2, 1)

include("aug_assign.jl")

# Examples.  We're not including them all here, because it would take
# too long, but just including black-scholes and opt-flow seems like a
# good compromise that exercises much of ParallelAccelerator.

include("../examples/black-scholes/black-scholes.jl")
include("../examples/opt-flow/opt-flow.jl")

# Delete file left behind by opt-flow.
dir = dirname(@__FILE__)
img_file = joinpath(dir, "out.flo")
rm(img_file)
