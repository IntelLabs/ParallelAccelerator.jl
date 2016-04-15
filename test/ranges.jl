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

module RangesTest
using ParallelAccelerator

#ParallelAccelerator.DomainIR.set_debug_level(4)
#ParallelAccelerator.ParallelIR.set_debug_level(4)
#ParallelAccelerator.CGen.set_debug_level(4)
#ParallelAccelerator.set_debug_level(4)


@acc function singulartest1(A::Array{Float64,2})
    return A[:,1] .* 2.0
end

@acc function rangetest1(A::Array{Float64,2})
    return A[:,1:1] .* 2.0
end

@acc function reduce_col(col::Int)
    A = rand(10^5,10);
    x = rand(10^5);
    sum((A[:, col] .- x[:]) .* (A[:, col] .- x[:]))
end

@acc function reduce_col2(col::Int)
    A = rand(10^5,10);
    x = rand(10^5);
    r = 1:size(A, 1)
    sum((A[r, col] .- x[r]) .* (A[r, col] .- x[r]))
end

function test1()
    return reduce_col(3)
end

function test2()
    return reduce_col2(3)
end

function test3()
    return singulartest1([1.1 2.2; 3.3 4.4])
end

function test4()
    return rangetest1([1.1 2.2; 3.3 4.4])
end

end

using Base.Test
println("Testing ranges...")

@test RangesTest.test1() > 1.66e3
@test RangesTest.test2() > 1.66e3
@test RangesTest.test3() == [2.2; 6.6]
@test ndims(RangesTest.test4()) == 2

println("Done testing ranges.")

