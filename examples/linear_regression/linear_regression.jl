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

using ParallelAccelerator
using DocOpt

@acc function linear_regression(iterations)
    D = 10  # Number of features
    N = 10000 # number of instances
    p = 3 # number of functions

    labels = rand(p,N)
    points = rand(D,N)
    w = zeros(p,D)
    alphaN = 0.01/N

    for i in 1:iterations
       w -= alphaN*((w*points)-labels)*points' 
    end
    w
end

function main()
    doc = """linear_regression.jl

linear regression statistical method.

Usage:
  linear_regression.jl -h | --help
  linear_regression.jl [--iterations=<iterations>]

Options:
  -h --help                  Show this screen.
  --iterations=<iterations>  Specify a number of iterations [default: 50].
"""
    arguments = docopt(doc)

    iterations = parse(Int, arguments["--iterations"])

    srand(0)

    println("iterations = ",iterations)

    tic()
    linear_regression(2)
    println("SELFPRIMED ", toq())

    tic()
    W = linear_regression(iterations)
    time = toq()
    println("result = ", W)
    println("rate = ", iterations / time, " iterations/sec")
    println("SELFTIMED ", time)

end

main()

