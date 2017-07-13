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

module APILibTest

using ParallelAccelerator
#importall ParallelAccelerator.API.Lib

#using CompilerTools
#CompilerTools.ReadWriteSet.set_debug_level(3)
#CompilerTools.AstWalker.set_debug_level(3)
#CompilerTools.OptFramework.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)
#ParallelAccelerator.DomainIR.set_debug_level(3)
#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)

@acc minMax(A) = (indmin(A), indmax(A))

@acc vecLen(A) = sqrt(sum(abs2, A))

@acc diagTest(A) = sum(diag(diagm(A)))

@acc traceTest(A) = trace(diagm(A)) 

@acc testEye(A) = eye(A) .* A

@acc testRepMat(A) = sum(repmat(A, 2, 3), 1)

@acc function testRepMat2(A, B) 
   (l, m, n) = size(B)
   C = repmat(A, 1, 1, n) ./ B
   D = 1.0 ./ maximum(C, 2)
   return D
end

function test1()
  A = rand(5)
  minMax(A) == (@noacc minMax(A)) && minMax(A) == (Base.indmin(A), Base.indmax(A))
end

function test2()
  A = rand(5)
  abs(vecLen(A) - (@noacc vecLen(A))) < 1.0e-10 &&
  abs(vecLen(A) - sqrt(Base.sum(abs2, A))) < 1.0e-10
end

function test3()
  A = rand(5)
  abs(diagTest(A) - sum(A)) < 1.0e-10
end

function test4()
  A = rand(5)
  abs(diagTest(A) - sum(A)) < 1.0e-10
end

function test5()
  A = rand(5, 5)
  abs(sum(testEye(A) .- Base.eye(A) .* A)) < 1.0e-10
end

function test6()
  A = rand(4, 3)
  abs(sum(testRepMat(A) .- sum(repmat(A, 2, 3), 1))) < 1.0e-10
end

function test7()
    A = testRepMat2(ones(3,3), ones(3,3,3))
    abs((sum(A) - 9.0)) < 1.0e-10
end

end

using Base.Test
println("Testing parallel library functions...")
@test APILibTest.test1()
@test APILibTest.test2()
@test APILibTest.test3()
@test APILibTest.test4()
@test APILibTest.test5()
@test APILibTest.test6()
@test APILibTest.test7() 
println("Done testing parallel library functions.") 

