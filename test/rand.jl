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

module RandTest
using ParallelAccelerator

#ParallelAccelerator.DomainIR.set_debug_level(4)
#ParallelAccelerator.ParallelIR.set_debug_level(4)
#ParallelAccelerator.CGen.set_debug_level(4)
#ParallelAccelerator.set_debug_level(4)

@acc function simple_rand(size1::Int64, size2::Int64)
    A = rand(size1,size2)
    C = A.*2.0
    return C
end

@acc function simple_randn(size1::Int64, size2::Int64)
    A = randn(size1,size2)
    C = A.*2.0
    return C
end

@acc function tuple_rand()
    y = [1.0 2.0 3.0 ;1.1 2.1 3.1]
    y += 0.1*rand(size(y))
    return y
end

@acc function tuple_randn()
    y = [1.0 2.0 3.0 ;1.1 2.1 3.1]
    y += 0.1*randn(size(y))
    return y
end

function test1()
    return simple_rand(2,3)
end

function test2()
    return simple_randn(2,3)
end

function test3()
    return tuple_rand()
end

function test4()
    return tuple_randn()
end


end

using Base.Test
println("Testing rand()...")
@test all(RandTest.test1() .<= ones(2,3).*2.0) && all(RandTest.test1() .>= zeros(2,3)) 
@test size(RandTest.test2())==(2,3)
@test size(RandTest.test3())==(2,3)
@test size(RandTest.test4())==(2,3)
println("Done testing rand()...")

