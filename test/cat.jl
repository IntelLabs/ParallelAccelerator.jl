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

module CatTest
using ParallelAccelerator

#ParallelAccelerator.DomainIR.set_debug_level(4)
#ParallelAccelerator.ParallelIR.set_debug_level(4)
#ParallelAccelerator.CGen.set_debug_level(4)
#ParallelAccelerator.set_debug_level(4)
#using CompilerTools
#CompilerTools.OptFramework.set_debug_level(4)

@acc function cat1(a::Array{Float64,1})
    d = 3.0 
    a1 = a[1]
    a3 = a[3]
    C = [a1/d, a3/d] 
    return C
end

@acc function cat2(d::Float64)
    C= [2.0/d 3.0/d; 1.0/d 20.0/d]
    return C.+1.0
end

@acc function cat3(d::Float64)
    C= Float64[2.0/d 3.0/d 4.0/d; 1.0/d 20.0/d 30.0/d]
    return C.+1.0
end

function test1()
    return cat1([9.0; 2.0; 3.0]) 
end

function test2()
    return cat2(2.0) 
end

function test3()
    return cat3(2.0) 
end

end

using Base.Test
println("Testing cat...")
@test CatTest.test1() ≈ [3.0; 1.0]
@test CatTest.test2() ≈ [2.0 2.5; 1.5 11.0]
@test CatTest.test3() ≈ [2.0 2.5 3.0; 1.5 11.0 16.0]
println("Done testing cat.")

