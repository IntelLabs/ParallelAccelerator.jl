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

module VCatTest
using ParallelAccelerator

#ParallelAccelerator.DomainIR.set_debug_level(4)
#ParallelAccelerator.ParallelIR.set_debug_level(4)
#ParallelAccelerator.CGen.set_debug_level(4)
#ParallelAccelerator.set_debug_level(4)
#using CompilerTools
#CompilerTools.OptFramework.set_debug_level(4)

@acc function cat1(a,b)
    C = vcat(a,b)
    return C
end

function test1()
    return cat1([1.,2.,3.],[4.,5.,6.]) 
end

function test2()
    return cat1([1. 2. 3.; 6. 7. 8.],[4. 5. 6.]) 
end

end

using Base.Test
println("Testing vcat...")
@test_approx_eq VCatTest.test1() [1.,2.,3.,4.,5.,6.]
@test_approx_eq VCatTest.test2() [1. 2. 3.; 6. 7. 8.; 4. 5. 6.]
println("Done testing vcat.")

