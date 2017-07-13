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

module MapReduceTest
using ParallelAccelerator

#ParallelAccelerator.DomainIR.set_debug_level(3)
#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)

@acc function map_1(A)
    a = map(x -> x + 1, A) 
    b = map(x -> x - 1, A)
    c = a .* b
    return c
end

@acc function map_2(A, B)
    map((x, y) -> x * y, A, B)
end

@acc function map_3(A)
    map!(x -> x - 1, A) 
    return A
end

@acc function map_4(A, B)
    map!(x -> x - 1, A, B) 
    sum(A.+B)
end

@acc function reduce_1(A)
    reduce((x,y) -> x + y, 0, A)
end

@acc function reduce_2(A)
    reduce((x,y) -> x + y, 0, map(x -> x * x, A))
end

@acc function sum_A(col)
    A = ones(Int, col, col, col)
    sum(A)
end

@acc function sum_A_1(col)
    A = ones(Int, col, col, col)
    B = sum(A, 1)
    C = sum(B, 2)
end

@acc function sum_A_cond_1(col)
    A = ones(Int, col)
    A[1] = -1
    sum(A[A .> 0])
end

@acc function sum_A_cond_2(col)
    A = ones(Int, col, col)
    A[1,1] = -1
    B = [ i for i = 1:col ]
    C = B .> 2
    cartesianarray(Tuple{Int}, (col,)) do j
        sum(A[C, j:j])
    end
end

@acc function sum_A_range_1(col)
    A = ones(Int, col, col)
    A[1,2] = 0
    sum(A[:, 2])
end

function test1()
    A = Int[x for x = 1:10]
    map_1(A) == @noacc map_1(A)
end

function test2()
    A = Int[x for x = 1:10]
    map_2(A, A) == @noacc map_2(A, A)
end

function test3()
    A = Int[x for x = 1:10]
    map_3(A) 
    A[1] == 0 && A[10] == 9
end

function test4()
    A = Int[0 for x = 1:10]
    B = Int[x for x = 1:10]
    map_4(A, B) == @noacc map_4(A, B)
end

function test5()
    A = Int[x for x = 1:10]
    reduce_1(A) == sum(A)
end

function test6()
    A = Int[x for x = 1:10]
    reduce_2(A) == sum(abs2, A)
end

function test7()
    return sum_A(5) == @noacc sum_A(5)
end

function test8()
    return sum_A_1(5) == @noacc sum_A_1(5)
end

function test9()
    return sum_A_cond_1(5) == @noacc sum_A_cond_1(5)
end

function test10()
    return sum_A_cond_2(5) == @noacc sum_A_cond_2(5)
end

function test11()
    return sum_A_range_1(5) == @noacc sum_A_range_1(5)
end

end

using Base.Test
using MapReduceTest
println("Testing map and reduce...")

@test MapReduceTest.test1() 
@test MapReduceTest.test2()
@test MapReduceTest.test3()
@test MapReduceTest.test4()
@test MapReduceTest.test5()
@test MapReduceTest.test6() 
@test MapReduceTest.test7()
@test MapReduceTest.test8()
@test MapReduceTest.test9()
@test MapReduceTest.test10()
@test MapReduceTest.test11()

println("Done testing map and reduce.")

