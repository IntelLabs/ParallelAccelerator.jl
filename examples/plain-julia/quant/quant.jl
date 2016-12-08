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

using DocOpt

const spot = 100.0
const strike = 110.0
const vol = 0.25
const maturity = 2.0

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
    asset = Array(Array{Float64,1}, numSteps+1)
    asset[1] = Array(Float64, numPaths)
    fill!(asset[1], spot)
    for s=2:numSteps+1
        asset[s] = asset[s-1] .* exp2(fwdFactor_log2e .+ vsqrtdt_log2e .* randn(numPaths))
    end
    return asset
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
        invsqr = inv(sqrtest)
        vp0 = dot(cashFlowPut, valPut0)
        vp1 = dot(cashFlowPut, valPut1)
        vp2 = dot(cashFlowPut, valPut2)
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
