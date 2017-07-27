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
if VERSION >= v"0.6.0-pre"
using SpecialFunctions
end

#ParallelAccelerator.DomainIR.set_debug_level(3)
#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)
#using CompilerTools
#CompilerTools.OptFramework.set_debug_level(3)
#CompilerTools.LambdaHandling.set_debug_level(3)
#CompilerTools.LivenessAnalysis.set_debug_level(3)

@acc begin

@inline function cndf2(in)
    out = 0.5 .+ 0.5 .* erf.(0.707106781 .* in)
    return out
end

function blackscholes(sptprice, strike, rate, volatility, time)
    logterm = log10.(sptprice ./ strike)
    powterm = .5 .* volatility .* volatility
    den = volatility .* sqrt.(time)
    d1 = (((rate .+ powterm) .* time) .+ logterm) ./ den
    d2 = d1 .- den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike .* exp.(- rate .* time)
    c1 = futureValue .* NofXd2
    call = sptprice .* NofXd1 .- c1
    put  = call .- futureValue .+ sptprice
end

end

function run(iterations)
    sptprice   = Float64[ 42.0 for i = 1:iterations ]
    initStrike = Float64[ 40.0 + (i / iterations) for i = 1:iterations ]
    rate       = Float64[ 0.5 for i = 1:iterations ]
    volatility = Float64[ 0.2 for i = 1:iterations ]
    time       = Float64[ 0.5 for i = 1:iterations ]

    tic()
    put = blackscholes(sptprice, initStrike, rate, volatility, time)
    t = toq()
    println("checksum: ", sum(put))
    return t
end

function main()
    doc = """black-scholes.jl

Black-Scholes option pricing model.

Usage:
  black-scholes.jl -h | --help
  black-scholes.jl [--iterations=<iterations>]

Options:
  -h --help                  Show this screen.
  --iterations=<iterations>  Specify a number of iterations [default: 10000000].
"""
    arguments = docopt(doc)

    iterations = parse(Int, arguments["--iterations"])

    srand(0)

    println("iterations = ",iterations)

    tic()
    blackscholes(Float64[], Float64[], Float64[], Float64[], Float64[])
    println("SELFPRIMED ", toq())

    time = run(iterations)
    println("rate = ", iterations / time, " opts/sec")
    println("SELFTIMED ", time)

end

main()
