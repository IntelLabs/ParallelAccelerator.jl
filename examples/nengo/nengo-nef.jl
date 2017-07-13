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

# This code was ported from a Python implementation available at
# http://nengo.ca/docs/html/nef_algorithm.html
# Original Python implementation is copyright (c) the author.

using ParallelAccelerator
using CompilerTools
using DocOpt
if "--plot" in ARGS
    using PyPlot
end

const Float = Float64

const dt = Float(0.001)    # simulation time step
const end_t = Float(10.0)  # simulation duration
const t_rc = Float(0.02)   # membrane RC time constant
const t_ref = Float(0.002) # refractory period
const t_pstc = Float(0.1)  # post-synaptic time constant
const N_samples = 100      # number of sample points to use when finding decoders
const rate_A = [25, 75]    # range of maximum firing rates for population A
const rate_B = [50, 100]   # range of maximum firing rates for population B

# the input to the system over time
input(t)=sin(t)

# the function to compute between A and B
funcAB(x)=x*x

uniform(low, high) = rand() * (high - low) + low

function generate_gain_and_bias(count, intercept_low, intercept_high, rate_low, rate_high)
    gain = Array{Float}(count)
    bias = Array{Float}(count)
    for i in 1:count
        # desired intercept (x value for which the neuron starts firing
        intercept = uniform(intercept_low, intercept_high)
        # desired maximum rate (firing rate when x is maximum)
        rate = uniform(rate_low, rate_high)

        # this algorithm is specific to LIF neurons, but should
        #  generate gain and bias values to produce the desired
        #  intercept and rate
        z = 1.0 / (1-exp((t_ref-(1.0/rate))/t_rc))
        g = (1 - z)/(intercept - 1.0)
        b = 1 - g*intercept
        gain[i] = g
        bias[i] = b
    end
    return gain, bias
end

# a simple leaky integrate-and-fire model, scaled so that v=0 is resting
#  voltage and v=1 is the firing threshold
@acc @inline function run_neurons(input, v, ref)
    w = v .+ (input .- v) .* dt ./ t_rc
    wMask = w .>= 0.0
    v[wMask] = w[wMask]
    v[w .< 0.0] = 0.0
    refMask = ref .> 0.0
    v[refMask] = 0.0
    ref[refMask] = (ref .- dt)[refMask]
    spikes = v .> 1.0
    v[spikes] = 0.0
    ref[spikes] = t_ref
    return spikes
end

# measure the spike rate of a whole population for a given represented value x
function compute_response(x, encoder, gain, bias, time_limit=0.5)
    N = length(encoder)          # number of neurons
    v = zeros(Float, N)          # voltage
    ref = zeros(Float, N)        # refractory period

    # compute input corresponding to x
    input = Array{Float}(N)
    for i in 1:N
        input[i] = x*encoder[i]*gain[i]+bias[i]
        v[i]=uniform(0,1)  # randomize the initial voltage level
    end

    count = zeros(Int, N)    # spike count for each neuron

    # feed the input into the population for a given amount of time
    t = 0
    while t<time_limit
        spikes=run_neurons(input, v, ref)
        for i in 1:length(spikes)
            if spikes[i] count[i]+=1 end
        end
        t += dt
    end
    return [convert(Float, c)/time_limit for c in count]  # return the spike rate (in Hz)
end

# compute the tuning curves for a population
function compute_tuning_curves(encoder, gain, bias)
    N = length(encoder)
    # generate a set of x values to sample at
    x_values = Float[i*2.0/N_samples - 1.0 for i in 0:N_samples-1]

    # build up a matrix of neural responses to each input (i.e. tuning curves)
    A = Array{Float}(N, N_samples)
    for i in 1:length(x_values)
        x = x_values[i]
        response = compute_response(x, encoder, gain, bias)
        A[:, i] = response
    end
    return x_values, A
end

# compute decoders
function compute_decoder(encoder, gain, bias, func=x->x)
    # get the tuning curves
    x_values, A = compute_tuning_curves(encoder, gain, bias)

    # get the desired decoded value for each sample point
    # result is N_samples * 1
    value = Float[ func(x) for x in x_values, i in 1:1 ]
    println("size(A) = ", size(A))
    println("size(value) = ", size(value))
    # find the optimum linear decoder
    At = transpose(A)
    # Gamma is N_samples * N_samples
    Gamma = A * At
    # Upsilon is N_samples * 1
    Upsilon = A * value
    Ginv = pinv(Gamma)
    println("size(Ginv) = ", size(Ginv))
    println("size(Upsilon) = ", size(Upsilon))
    decoder = Ginv * Upsilon ./ dt
    return decoder
end

@acc function simulate(NT, inputs, weights, pstc_scale, decoder_B, gain_B, bias_B, encoder_A, gain_A, bias_A, N_A, N_B)

    v_A = zeros(Float, N_A)       # voltage for population A
    ref_A = zeros(Float, N_A)     # refractory period for population A
    input_A = zeros(Float, N_A)   # input for population A

    v_B = zeros(Float, N_B)       # voltage for population B
    ref_B = zeros(Float, N_B)     # refractory period for population B
    input_B = zeros(Float, N_B)   # input for population B
    outputs = zeros(Float, NT)    # outputs
    output = 0.0                  # the decoded output value from population B
    for i = 1:NT
        # call the input function to determine the input value
        x = inputs[i]
        # convert the input value into an input for each neuron
        input_A = x .* encoder_A .* gain_A .+ bias_A

        # run population A and determine which neurons spike
        spikes_A=run_neurons(input_A, v_A, ref_A)
        # for each neuron that spikes, increase the input current
        #  of all the neurons it is connected to by the synaptic
        #  connection weight
        for j = 1:N_A
            if spikes_A[j]
                input_B += weights[:,j] .* pstc_scale
            end
        end

        # compute the total input into each neuron in population B
        #  (taking into account gain and bias)
        total_B = gain_B .* input_B .+ bias_B
        input_B = input_B .* (1.0 - pstc_scale)

        # run population B and determine which neurons spike
        spikes_B=run_neurons(total_B, v_B, ref_B)
        # for each neuron in B that spikes, update our decoded value
        #  (also applying the same post-synaptic filter)
        output = output * (1.0-pstc_scale) + sum(decoder_B[spikes_B, 1]) * pstc_scale
        outputs[i] = output
    end
    return outputs
end

function main()
    doc = """nengo-nef.jl

Nengo neural engineering framework.

Usage:
  nengo-nef.jl -h | --help
  nengo-nef.jl [--plot] [--n_a=<n_a>] [--n_b=<n_b>]

Options:
  -h --help    Show this screen.
  --plot       Enable PyPlot drawing.
  --n_a=<n_a>  Number of neurons in first population [default: 500].
  --n_b=<n_b>  Number of neurons in second population [default: 400].
"""
    arguments = docopt(doc)

    N_A = parse(Int, arguments["--n_a"])
    N_B = parse(Int, arguments["--n_b"])
    plotting = arguments["--plot"]

    # gc_enable(false)

    srand(0)

    #################################################
    # Step 1: Initialization
    #################################################

    # create random encoders for the two populations
    encoder_A = Int[(rand(Bool) ? -1 : 1) for i in 1:N_A]
    encoder_B = Int[(rand(Bool) ? -1 : 1) for i in 1:N_B]

    # random gain and bias for the two populations
    gain_A, bias_A = generate_gain_and_bias(N_A, -1, 1, rate_A[1], rate_A[2])
    gain_B, bias_B = generate_gain_and_bias(N_B, -1, 1, rate_B[1], rate_B[2])

    # find the decoders for A and B
    decoder_A=compute_decoder(encoder_A, gain_A, bias_A, funcAB)
    decoder_B=compute_decoder(encoder_B, gain_B, bias_B)

    # compute the weight matrix
    weights = encoder_B * transpose(decoder_A)

    #################################################
    # Step 2: Running the simulation
    #################################################

    # scaling factor for the post-synaptic filter
    pstc_scale=1.0-exp(-dt/t_pstc)

    # for storing simulation data to plot afterward
    NT = convert(Int, end_t / dt)

    times=Float[ dt * t for t = 0:NT-1 ]
    inputs=Float[ input(x) for x in times ]
    ideal=Float[ funcAB(x) for x in inputs ]

    tic()
    simulate(0, Float[], zeros(Float,0,0), Float(0.0), zeros(Float,0,0), Float[], Float[], Int[], Float[], Float[], N_A, N_B)
    println("SELFPRIMED ", toq())

    println("Starting simulation")
    tic()
    outputs=simulate(NT, inputs, weights, Float(pstc_scale), decoder_B, gain_B, bias_B, encoder_A, gain_A, bias_A, N_A, N_B)
    println("SELFTIMED ", toq())

    # gc_enable(true)

    if plotting
        #################################################
        # Step 3: Plot the results
        #################################################
        x,A = compute_tuning_curves(encoder_A, gain_A, bias_A)
        x,B = compute_tuning_curves(encoder_B, gain_B, bias_B)
        PyPlot.figure()
        PyPlot.plot(x, transpose(A))
        PyPlot.title("Tuning curves for population A")
        PyPlot.savefig("A.png")

        PyPlot.figure()
        PyPlot.plot(x, transpose(B))
        PyPlot.title("Tuning curves for population B")
        PyPlot.savefig("B.png")

        PyPlot.figure()
        PyPlot.plot(times, inputs, label = "input")
        PyPlot.plot(times, ideal, label= "ideal")
        PyPlot.plot(times, outputs, label = "output")
        PyPlot.title("Simulation results")
        PyPlot.legend()
        PyPlot.savefig("result.png")
    end

    println("checksum: ", sum(outputs))
    return outputs

end

main()
