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

module BitArrayTest
using ParallelAccelerator

#ParallelAccelerator.DomainIR.set_debug_level(4)
#ParallelAccelerator.ParallelIR.set_debug_level(4)
#ParallelAccelerator.CGen.set_debug_level(4)
#ParallelAccelerator.set_debug_level(4)


@acc function bitarrtest(A::Array{Float64,1})
    pcond = A .> 2.0 
    A[pcond] = 0.1 
    return A
end

@acc function bitarrtest2(A::Array{Float64,1})
    A[A.>2.0] = 0.1 
    return A
end

@acc function bitarrtest3(A::Array{Float64,1},B::Array{Float64,1},C::Array{Float64,1})  
    pcond = A .> C
    A[pcond] = B[pcond]
    return A
end

@acc function bitarrtest4(A::Array{Float64,1}, C::Float64)
    pcond = A .> C
    return sum(A[pcond])
end

function test1()
    return bitarrtest([1.1; 2.2; 3.2; 1.9])
end

function test2()
    return bitarrtest2([1.1; 2.2; 3.2; 1.9])
end

function test3()
    return bitarrtest3([1.1; 2.2; 3.2; 1.9], [9.0; 9.0; 10.0; 11.0], [2.0; 2.0; 2.0; 2.0])
end

function test4()
    return bitarrtest4([1.1; 2.2; 3.2; 1.9], 2.0)
end

end

using Base.Test

println("Testing BitArrays...")
@test BitArrayTest.test1() == [1.1; 0.1; 0.1; 1.9]
@test BitArrayTest.test2() == [1.1; 0.1; 0.1; 1.9]
@test BitArrayTest.test3() == [1.1; 9.0; 10.0; 1.9]
@test BitArrayTest.test4() == 2.2 + 3.2
println("Done testing BitArrays.")


