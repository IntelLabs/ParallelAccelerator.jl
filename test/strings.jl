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

module StringTest
using ParallelAccelerator

#ParallelAccelerator.DomainIR.set_debug_level(4)
#ParallelAccelerator.ParallelIR.set_debug_level(4)
#ParallelAccelerator.CGen.set_debug_level(4)
#ParallelAccelerator.set_debug_level(4)

@acc function f1()
    x = "hello"
    y = x[3]
    return UInt8(y)
end

@acc function f2()
    x = "hello"
    return length(x)
end

@acc function f3(x)
    println(x, " world!")
    return "OK"
end

@acc function f4(x)
    return x * "bar"
end

@acc function f5(x)
    string(x) * "bar"
end

function test1()
    f1() == 108 # ASCII 'l' is 108
end

function test2()
    f2() == 5
end

function test3()
    f3("Hello")  == "OK"
end

function test4()
    f4("foo")  == "foobar"
end

function test5()
    f5(1)  == "1bar"
end

end

using Base.Test

println("Testing strings...")
@test StringTest.test1() 
@test StringTest.test2() 
@test StringTest.test3() 
@test StringTest.test4() 
@test StringTest.test5() 
println("Done testing strings...")

