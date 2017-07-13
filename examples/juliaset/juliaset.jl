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
# http://www.albertostrumia.it/Fractals/FractalMatlab/Julia/juliaSH.m.
# Original MATLAB implementation is copyright (c) the author.

using ParallelAccelerator
using DocOpt
if "--demo" in ARGS
    using Winston
end

include("ndgrid.jl") # provides the meshgrid function

function juliaset(demo::Bool)
    col = 128             # color depth
    m = 1000              # image size
    cx = 0                # center X
    cy = 0                # center Y
    l = 1.5               # span

    if demo
        iters = 50
        zoomAmount = 0.9
    else
        iters = 10
        zoomAmount = 0.6
    end

    # The c constant.
    c = -.745429 + .11308*im

    for zoom = 1:iters

        # `x` and `y` are two 1000-element arrays representing the x
        # and y axes: [-1.5, -1.497, ..., 0, ..., 1.497, 1.5] on the
        # first iteration of this loop.
        x = linspace(cx-l, cx+l, m)
        y = linspace(cy-l, cy+l, m)

        # `X` and `Y` are two arrays containing, respectively, the x-
        # and y-coordinates of each point on a 1000x1000 grid.
        (X, Y) = meshgrid(x, y)

        # Let `Z` represent the complex plane: a 1000x1000 array of
        # numbers each with a real and a complex part.
        Z = X + im*Y

        # Iterate the Julia set computation (squaring each element of
        # Z and adding c) for `col` steps.
        W = iterate(col, Z, c)

        # Mask out the NaN values (overflow).
        minval = minimum(W)
        W[isnan.(W)] = minval - minval/10

        if demo
            # Plot the result as a "heat map"-style plot.  The values
            # of the elements of W determine the "color" of the
            # corresponding point.
            plot = Winston.imagesc(W)
            Winston.display(plot)
        end

        # Zoom into the next frame, shrinking the distance that `x`
        # and `y` will cover.
        l = l * zoomAmount
    end

end

@acc function iterate(col, Z, c)
    for k = 1:col
        Z = Z.*Z .+ c
    end
    return exp(-abs(Z))
end

function main()
    doc = """juliaset.jl

Julia set fractal visualization.

Usage:
  juliaset.jl -h | --help
  juliaset.jl [--demo]

Options:
  -h --help  Show this screen.
  --demo     Use settings that look good for the purposes of a demo, and plot the results using Winston.
"""
    arguments = docopt(doc)
    demo = arguments["--demo"]

    tic()
    iterate(0, Array{Complex{Float64}}(0,2), Complex(0.0))
    time = toq()
    println("SELFPRIMED ", time)

    tic()
    juliaset(demo)
    time = toq()
    println("SELFTIMED ", time)
end

main()
