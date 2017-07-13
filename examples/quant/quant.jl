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

const spot = 100.0
const strike = 110.0
const vol = 0.25
const maturity = 2.0

function det2x2(a::Array{Float64,2})
  a1 = a[1]
  b1 = a[2]
  a2 = a[3]
  b2 = a[4]
  a1 * b2 - a2 * b1
end

function det3x3(a::Array{Float64,2})
  a1 = a[1]
  b1 = a[2]
  c1 = a[3]
  a2 = a[4]
  b2 = a[5]
  c2 = a[6]
  a3 = a[7]
  b3 = a[8]
  c3 = a[9]
  a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1
end

function inv2x2(a::Array{Float64,2})
  d = det2x2(a)
  Float64[a[4] (-a[3]); (-a[2]) a[1]] ./ d
end

function inv3x3(a::Array{Float64,2})
  d = det3x3(a)
  a11 = a[1]
  a21 = a[2]
  a31 = a[3]
  a12 = a[4]
  a22 = a[5]
  a32 = a[6]
  a13 = a[7]
  a23 = a[8]
  a33 = a[9]
  Float64[det2x2(Float64[a22 a23; a32 a33])/d det2x2(Float64[a13 a12; a33 a32])/d det2x2(Float64[a12 a13; a22 a23])/d;
          det2x2(Float64[a23 a21; a33 a31])/d det2x2(Float64[a11 a13; a31 a33])/d det2x2(Float64[a13 a11; a23 a21])/d;
          det2x2(Float64[a21 a22; a31 a32])/d det2x2(Float64[a12 a11; a32 a31])/d det2x2(Float64[a11 a12; a21 a22])/d ]
end

function init(numPaths::Int, numSteps::Int)
    # Pre-compute some values
    dt = maturity / numSteps
    volSqrtdt = vol * sqrt(dt)
    fwdFactor = vol^2 * dt * -0.5
    vsqrtdt_log2e = volSqrtdt * log2(exp(1.0))
    fwdFactor_log2e = fwdFactor * log2(exp(1.0))

    # Storage
    # Compute all per-path asset values for each time step
    # NOTE: hand-hoisting obscures the basic formula.
    # Can the compiler do this optimization?
    asset = Array{Array{Float64,1}}(numSteps+1)
    asset[1] = Array{Float64}(numPaths)
    fill!(asset[1], spot)
    for s=2:numSteps+1
        asset[s] = asset[s-1] .* exp2.(fwdFactor_log2e .+ vsqrtdt_log2e .* randn(numPaths))
    end
    return asset
end

@acc begin

function dotp(a, b)
  sum(a .* b)
end

function model(strike, numPaths, numSteps, asset)
    avgPathFactor = 1.0/numPaths
    putpayoff = max(strike .- asset[numSteps+1], 0.0)
    cashFlowPut = putpayoff * avgPathFactor
    # Now go back in time using regression
    for s = numSteps:-1:2
        curasset = asset[s]
        vcond = curasset .<= strike
        valPut0 = curasset .* 0.0 # zeros(numPaths)
        valPut0[vcond] = 1.0
        valPut1 = valPut0 .* curasset # assetval
        valPut2 = valPut1 .* valPut1  # assetval ^ 2
        valPut3 = valPut2 .* valPut1  # assetval ^ 3
        valPut4 = valPut2 .* valPut2  # assetval ^ 4
        # compute the regression
        ridgecorrect = 0.01
        sum0 = sum(valPut0)
        sum1 = sum(valPut1)
        sum2 = sum(valPut2)
        sum3 = sum(valPut3)
        sum4 = sum(valPut4)
        sum0 += ridgecorrect
        sum1 += ridgecorrect
        sum2 += ridgecorrect
        sum3 += ridgecorrect
        sum4 += ridgecorrect
        sqrtest = Float64[sum0 sum1 sum2;
                          sum1 sum2 sum3;
                          sum2 sum3 sum4]
        invsqr = inv3x3(sqrtest)
        vp0 = dotp(cashFlowPut, valPut0)
        vp1 = dotp(cashFlowPut, valPut1)
        vp2 = dotp(cashFlowPut, valPut2)
        (betaPut0, betaPut1, betaPut2) = [vp0 vp1 vp2] * invsqr
        regImpliedCashFlow = valPut0 .* betaPut0 .+ valPut1 .* betaPut1 .+ valPut2 .* betaPut2
        payoff = valPut0 .* (strike .- curasset)
        pcond = payoff .> regImpliedCashFlow
        cashFlowPut[pcond] = payoff[pcond]
    end

    amerpayoffput = sum(cashFlowPut)
    finalputpayoff = sum(putpayoff)

    return amerpayoffput/numPaths, finalputpayoff/numPaths
end

end

function main()
    doc = """quant.jl

Quantitative option pricing model.

Usage:
  quant.jl -h | --help
  quant.jl [--assets=<assets>] [--iterations=<iterations>]

Options:
  -h --help                  Show this screen.
  --assets=<assets>          Specify the number of assets to simulate [default: 524288].
  --iterations=<iterations>  Specify a number of iterations [default: 256].
"""
    arguments = docopt(doc)

    paths = parse(Int, arguments["--assets"])
    steps = parse(Int, arguments["--iterations"])

    srand(0)

    println("assets = ", paths)
    println("iterations = ", steps)

    asset = init(paths, steps)
    tic()
    model(strike, 1, 1, asset)
    compiletime = toq()
    println("SELFPRIMED ", compiletime)

    tic()
    amerpayoffput, finalputpayoff = model(strike, paths, steps, asset)
    selftimed = toq()
    println("European Put Option Price: ", finalputpayoff)
    println("American Put Option Price: ", amerpayoffput)
    println("SELFTIMED ", selftimed)

end

main()
