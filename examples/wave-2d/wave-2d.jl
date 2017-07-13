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

# This code was ported from a MATLAB implementation available at
# http://www.piso.at/julius/index.php/projects/programmierung/13-2d-wave-equation-in-octave
# with permission of the author.
# Original MATLAB implementation is copyright (c) 2014 Julius Piso.

using ParallelAccelerator
using DocOpt
if "--demo" in ARGS
    using Winston
end

@acc function runWaveStencil(p, c, f, r, t, n, s2, s4, s)
    runStencil(p, c, f, 1, :oob_skip) do p, c, f
        f[0, 0] = 2 * c[0, 0] - p[0, 0] + r * r * (c[-1, 0] + c[1, 0] + c[0, -1] + c[0, 1] - 4*c[0, 0])
    end

    # Dynamic source
    f[s2+s4-1:s2+s4+1, 1:2] = 1.5 * sin.(t*n)
    f[s2-s4-1:s2-s4+1, 1:2] = 1.5 * sin.(t*n)
    f[2:s-1, 1:2] = 1.0

    # Transparent boundary handling
    f[2:s-1, 1:1] = (2.0 * c[2:s-1, 1:1] + (r-1.0) * p[2:s-1, 1:1] + 2.0*r*r*(c[2:s-1, 2:2] + c[3:s, 1:1] + c[1:s-2, 1:1] - 3.0 * c[2:s-1, 1:1])) / (1.0+r) # Y:1
    f[2:s-1, s:s] = (2.0 * c[2:s-1, s:s] + (r-1.0) * p[2:s-1, s:s] + 2.0*r*r*(c[2:s-1, s-1:s-1] + c[3:s, s:s] + c[1:s-2, s:s] - 3.0 * c[2:s-1, s:s])) / (1.0+r) # Y:s
    f[1:1, 2:s-1] = (2.0 * c[1:1, 2:s-1] + (r-1.0) * p[1:1, 2:s-1] + 2.0*r*r*(c[2:2, 2:s-1] + c[1:1, 3:s] + c[1:1, 1:s-2] - 3.0 * c[1:1, 2:s-1])) / (1.0+r) # X:1
    f[s:s, 2:s-1] = (2.0 * c[s:s, 2:s-1] + (r-1.0) * p[s:s, 2:s-1] + 2.0*r*r*(c[s-1:s-1, 2:s-1] + c[s:s, 3:s] + c[s:s, 1:s-2] - 3.0 * c[s:s, 2:s-1])) / (1.0+r) # Y:s

    return 0
end

function prime_wave2d()
    speed = 10         # Propagation speed
    s = 16             # Array size (spatial resolution of the simulation)

    s2 = div(s, 2)
    s4 = div(s, 4)
    s8 = div(s, 8)
    s16 = div(s, 16)

    p = zeros(s, s) # past
    c = zeros(s, s) # current
    f = zeros(s, s) # future

    dt = 0.0001     # Time resolution of the simulation
    dx = 0.01       # Distance between elements
    r = speed * dt / dx

    n = 300

    for i = s2 - s16 : s2 + s16
        # Initial conditions
        c[i, s2 - s16 : s2 + s16] = - 2 * cos.(0.5 * 2 * pi / (s8) * (s2 - s16 : s2 + s16)) * cos.(0.5 * 2 * pi / (s8) * i)
        p[i, 1:s] = c[i, 1:s]
    end

    runWaveStencil(p, c, f, r, 0.0, n, s2, s4, s)
end

function wave2d(demo::Bool)

    speed = 10         # Propagation speed
    s = 512            # Array size (spatial resolution of the simulation)

    if demo
        stopTime = 0.1 # Time step at which to stop the main loop
    else
        stopTime = 0.05
    end

    s2 = div(s, 2)
    s4 = div(s, 4)
    s8 = div(s, 8)
    s16 = div(s, 16)

    p = zeros(s, s) # past
    c = zeros(s, s) # current
    f = zeros(s, s) # future

    dt = 0.0001     # Time resolution of the simulation
    dx = 0.01       # Distance between elements
    r = speed * dt / dx

    n = 300

    for i = s2 - s16 : s2 + s16
        # Initial conditions
        c[i, s2 - s16 : s2 + s16] = - 2 * cos.(0.5 * 2 * pi / (s8) * (s2 - s16 : s2 + s16)) * cos.(0.5 * 2 * pi / (s8) * i)
        p[i, 1:s] = c[i, 1:s]
    end

    # Main loop
    for t = 0 : dt : stopTime
        # Wave equation
        runWaveStencil(p, c, f, r, t, n, s2, s4, s)

        # Transparent corner handling
        f[1:1, 1:1] = (2 * c[1:1, 1:1] + (r-1) * p[1:1, 1:1] + 2*r*r* (c[2:2, 1:1] + c[1:1, 2:2] - 2*c[1:1, 1:1])) / (1+r) # X:1; Y:1
        f[s:s, s:s] = (2 * c[s:s, s:s] + (r-1) * p[s:s, s:s] + 2*r*r* (c[s-1:s-1, s:s] + c[s:s, s-1:s-1] - 2*c[s, s])) / (1+r) # X:s; Y:s
        f[1:1, s:s] = (2 * c[1:1, s:s] + (r-1) * p[1:1, s:s] + 2*r*r* (c[2:2, s:s] + c[1:1, s-1:s-1] - 2*c[1:1, s:s])) / (1+r) # X:1; Y:s
        f[s:s, 1:1] = (2 * c[s:s, 1:1] + (r-1) * p[s:s, 1:1] + 2*r*r* (c[s-1:s-1, 1:1] + c[s:s, 2:2] - 2*c[s:s, 1:1])) / (1+r) # X:s; Y:1

        # Rotate buffers for next iteration
        tmp = p
        p = c
        c = f
        f = tmp

        if demo
            if mod(t/dt, 10) == 0
                plot = Winston.imagesc(c)
                Winston.display(plot)
            end
        end
    end
end

function main()
    doc = """wave-2d.jl

Two-dimensional wave equation solver.

Usage:
  wave-2d.jl -h | --help
  wave-2d.jl [--demo]

Options:
  -h --help  Show this screen.
  --demo     Use settings that look good for the purposes of a demo, and plot the results using Winston.
"""
    arguments = docopt(doc)
    demo = arguments["--demo"]

    tic()
    prime_wave2d()
    println("SELFPRIMED ", toq())

    tic()
    wave2d(demo)
    println("SELFTIMED ", toq())
end

main()
