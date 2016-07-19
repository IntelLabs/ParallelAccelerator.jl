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

module MiscTest
using ParallelAccelerator

#ParallelAccelerator.DomainIR.set_debug_level(4)
#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(4)
#ParallelAccelerator.set_debug_level(4)
#using CompilerTools
#CompilerTools.LivenessAnalysis.set_debug_level(3)
#CompilerTools.LambdaHandling.set_debug_level(3)

@acc function for_ret()
    for f = 1:10
    end
end

function test1()
    return for_ret()
end

@acc function for_ret1(n)
    for f = 1:n
    end
end

function test2()
    return for_ret1(10)
end

@acc function opt_At_mul_B!(X, W)
     X' * W
end

function test_At_mul_B(m::Int, k::Int, n::Int)
    W = Array(Float64, m, k)   
    X = Array(Float64, m, n)
    fill!(W, 3)
    fill!(X, 5)
    opt_At_mul_B!(X, W)
end

function test3()
  all(Bool[ x == 150.0 for x in MiscTest.test_At_mul_B(10,10,10) ])
end

@acc function f()
    W = zeros(Int, 5, 5)
    s = Int[sum(W[:,j]) for j in 1:5]
end

function test4()
    f()
end

@acc function const_array_init()
    Int[1,2,3] .+ Int[4,5,6]
end

function test5()
    const_array_init() == [5,7,9]
end

@acc function mod_rem_test(x, y)
    Int[ mod(x, y), rem(x, y) ]
end

@acc function end_test_numeric(a)
    a[3:4] .+ 1
end

function test7()
    end_test_numeric([1;2;3;4]) == [4;5]
end

@acc function end_test(a)
    a[3:end] .+ 1
end

function test6()
    end_test([1;2;3;4]) == [4;5]
end

end

using Base.Test
println("Testing miscellaneous features...")
@test MiscTest.test1() == nothing
@test MiscTest.test2() == nothing
@test MiscTest.test3() 
@test MiscTest.test4() == [0.0, 0.0, 0.0, 0.0, 0.0]
@test MiscTest.test5() 
#@test MiscTest.test6() 
@test MiscTest.test7() 
@test MiscTest.mod_rem_test(7,3) == [1, 1]
@test MiscTest.mod_rem_test(7,-3) == [-2, 1]
@test MiscTest.mod_rem_test(-7,3) == [2, -1]
@test MiscTest.mod_rem_test(-7,-3) == [-1, -1]
println("Done testing miscellaneous features...")

