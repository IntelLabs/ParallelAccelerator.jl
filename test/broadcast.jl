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

module BroadcastTest
using ParallelAccelerator
using CompilerTools

#ParallelAccelerator.DomainIR.set_debug_level(3)
#ParallelAccelerator.ParallelIR.set_debug_level(4)
#ParallelAccelerator.CGen.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)
#CompilerTools.LambdaHandling.set_debug_level(3)

@acc function twoargs(A, B)
    broadcast(*, A, B)
end

@acc function threeargs(A, B, C)
    broadcast(*, A, B, C)
end

function test1()
    A = [1, 2, 3]
    B = [4, 5, 6]
    twoargs(A, B) == @noacc twoargs(A, B)
end

function test2()
    A = [1, 2, 3]
    B = 3
    twoargs(A, B) == @noacc twoargs(A, B)
end

function test3()
    A = [1, 2]
    B = [1 2 3; 4 5 6]
    C = twoargs(A, B) 
    println("test3 C = ", C)
    C == @noacc twoargs(A, B)
end

function test4()
    A = [1 2 3]
    B = [1 2 3; 4 5 6]
    twoargs(A, B) == @noacc twoargs(A, B)
end

function test5()
    A = [1 2 3]
    B = [1 2 3; 4 5 6]
    C = 1
    threeargs(A, B, C) == @noacc threeargs(A, B, C)
end

function test6()
    A = [1 2 3]
    B = [1 2 3; 4 5 6]
    C = [1, 2]
    threeargs(A, B, C) == @noacc threeargs(A, B, C)
end

end

using Base.Test

println("Testing broadcast...")
@test BroadcastTest.test1() 
@test BroadcastTest.test2()
@test BroadcastTest.test3()
@test BroadcastTest.test4()
#@test BroadcastTest.test5()
#@test BroadcastTest.test6()
println("Done testing broadcast.")


