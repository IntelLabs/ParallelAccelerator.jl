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

module ComplexTest
using ParallelAccelerator

#ParallelAccelerator.DomainIR.set_debug_level(3)
#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(4)
#ParallelAccelerator.set_debug_level(4)

# test input, constant, and output
@acc function complex_test1(x::Complex{Float64})
    x + (1.0 + 2.0im)
end

# test array input, constant, and output
@acc function complex_test2(x::Array{Complex{Float64}, 1})
    x .+ (1.0 + 2.0im)
end

# test rand and reduction
@acc function complex_test3(n::Int)
    x = rand(Complex128, n)
    sum(x)
end

function test1()
    x = complex_test1(2.0 + 1.0im) 
    println(" test1 returns: ", x)
    return x == (3.0 + 3.0im)
end

function test2()
    return complex_test2(Complex{Float64}[1.0 + 2.0im, 2.0 + 1.0im]) == (Complex{Float64}[2.0 + 4.0im, 3.0 + 3.0im])
end

function test3()
    x = complex_test3(10)
    return true
end

end

using Base.Test

println("Testing complex number support...")
@test ComplexTest.test1() 
@test ComplexTest.test2()
# test3 is no longer supported by OpenMP after we switch from C's _Complex to C++'s std::complex
# test3 should work for the native threading backend though.
if ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE
    @test ComplexTest.test3()
end
println("Done testing complex number support.")

