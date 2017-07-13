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

module ParForTest

using ParallelAccelerator
ParallelAccelerator.DomainIR.set_debug_level(3)
ParallelAccelerator.ParallelIR.set_debug_level(3)
ParallelAccelerator.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)
#using CompilerTools
#CompilerTools.OptFramework.set_debug_level(3)

@acc function parfor1(n)
 A = Array{Int}(n, n)
 @par for i in 1:n, j in 1:n
    A[i,j] = i * j
 end
 return sum(A)
end

@acc function parfor2(n)
 A = Array{Int}(n, n)
 s::Int = 0
 m::Int = 0
 @par s(+) m(+) for i in 1:n, j = 1:n
    A[i,j] = i * j
    s = s + A[i,j]
    m = m + 1
 end
 return s * m
end

@acc function parfor3(n)
 A::Array{Int,2} = Array{Int}(n, n)
 s::Array{Int,1} = zeros(Int, n)
 m::Int = 0
 @par s(.+) m(+) for i in 1:n
    for j = 1:n
      A[j,i] = i * j
    end
    s = s .+ A[:,i]
    m = m + 1
 end
 return s .* m
end

function test1()
  parfor1(10) == @noacc parfor1(10) 
end

function test2()
  parfor2(10) == @noacc parfor2(10) 
end

function test3()
  parfor3(10) == @noacc parfor3(10) 
end

end

using Base.Test
println("Testing parfor support via @par macro...")
@test ParForTest.test1() 
@test ParForTest.test2()
@test ParForTest.test3()
println("Done testing parfor.") 
