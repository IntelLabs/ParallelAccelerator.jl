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

module ReductionTest
using ParallelAccelerator

#ParallelAccelerator.DomainIR.set_debug_level(4)
#ParallelAccelerator.ParallelIR.set_debug_level(4)
#ParallelAccelerator.CGen.set_debug_level(4)
#ParallelAccelerator.set_debug_level(4)

@acc function sum_A(col::Int)
    A = ones(Int, col, col, col)
    sum(A)
end

@acc function sum_A_1(col::Int)
    A = ones(Int, col, col, col)
    B = sum(A, 1)
    C = sum(B, 2)
end

@acc function sum_A_cond_1(col::Int)
    A = ones(Int, col)
    A[1] = -1
    sum(A[A .> 0])
end

@acc function sum_A_cond_2(col::Int)
    A = ones(Int, col, col)
    A[1,1] = -1
    B = [ i for i = 1:col ]
    C = B .> 2
    cartesianarray(Int, (col,)) do j
        sum(A[C, j:j])
    end
end

@acc function sum_A_range_1(col::Int)
    A = ones(Int, col, col)
    A[1,2] = 0
    sum(A[:, 2])
end

function test1()
    return sum_A(5) == @noacc sum_A(5)
end

function test2()
    return sum_A_1(5) == @noacc sum_A_1(5)
end

function test3()
    return sum_A_cond_1(5) == @noacc sum_A_cond_1(5)
end

function test4()
    return sum_A_cond_2(5) == @noacc sum_A_cond_2(5)
end

function test5()
    return sum_A_range_1(5) == @noacc sum_A_range_1(5)
end

end

using ReductionTest
println("Testing reductions...")

@test ReductionTest.test1() 
@test ReductionTest.test2()
@test ReductionTest.test3()
@test ReductionTest.test4()
@test ReductionTest.test5()

println("Done testing reductions.")

