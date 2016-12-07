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

@acc function logistic_regression(iterations)
    D = 10  # Number of features
    N = 10000 # number of instances

    labels = reshape(rand(N),1,N)
    points = rand(D,N)
    w = reshape(2.0.*rand(D)-1.0,1,D)

    for i in 1:iterations
       w -= ((1.0./(1.0.+exp(-labels.*(w*points))).-1.0).*labels)*points'
    end
    w
end

function main()
    doc = """logistic_regression.jl

Logistic regression statistical method.

Usage:
  logistic_regression.jl -h | --help
  logistic_regression.jl [--iterations=<iterations>]

Options:
  -h --help                  Show this screen.
  --iterations=<iterations>  Specify a number of iterations [default: 50].
"""
    arguments = docopt(doc)

    iterations = parse(Int, arguments["--iterations"])

    srand(0)

    println("iterations = ",iterations)

    tic()
    logistic_regression(2)
    println("SELFPRIMED ", toq())

    tic()
    W = logistic_regression(iterations)
    time = toq()
    println("result = ", W)
    println("rate = ", iterations / time, " iterations/sec")
    println("SELFTIMED ", time)

end

main()

