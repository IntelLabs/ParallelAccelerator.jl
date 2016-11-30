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
# http://exolete.com/code/lbm/
# with permission of the author.
# Original MATLAB implementation is copyright (c) 2006 Iain Haslam.

using ParallelAccelerator
using DocOpt

function computeF(density, c_squ, u_squ, t, u, v)
    t * density * (1 + u / c_squ + 0.5 * (v / c_squ) * (v / c_squ) - u_squ / (2 * c_squ))
end

@acc function boltzmann_kernel(numactivenodes, F, G, ON, UX, X1)
    avu = 1.
    prevavu = 1.
    ts = 0
    deltaU = 1e-7
    t1 = 4/9
    t2 = 1/9
    t3 = 1/36
    c_squ = 1/3
    omega = 1.0
    while (ts<4000 && 1e-10<abs((prevavu-avu)/avu)) || ts<100
        ux_sum = 0.
        runStencil(F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8], F[9],
                   G[1], G[2], G[3], G[4], G[5], G[6], G[7], G[8], G[9], ON, UX, X1,
                   1, :oob_wraparound) do F1, F2, F3, F4, F5, F6, F7, F8, F9, G1, G2, G3, G4, G5, G6, G7, G8, G9, ON, UX, X1
                       # Propagate
                       f4 = F4[1, -1]
                       f3 = F3[0,-1]
                       f2 = F2[-1,-1]
                       f5 = F5[1,0]
                       f1 = F1[-1,0]
                       f6 = F6[1,1]
                       f7 = F7[0,1]
                       f8 = F8[-1,1]
                       f9 = F9[0,0]
                       density = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
                       ux = (f1 + f2 + f8 - f4 - f5 - f6) / density
                       uy = (f2 + f3 + f4 - f6 - f7 - f8) / density
                       if X1[0,0] == 1
                           ux = ux + deltaU # Increase inlet pressure
                       end
                       if ON[0,0] == 1
                           ux = 0.
                           uy = 0.
                           density = 0.
                       end
                       UX[0,0] = ux
                       u_squ = ux * ux + uy * uy
                       u_c2  = ux + uy
                       u_c4  = uy - ux
                       u_c6  = - u_c2
                       u_c8  = - u_c4
                       # Calculate equilibrium distribution: stationary
                       feq9 = t1 * density * (1. - u_squ / (2. * c_squ))
                       # nearest-neighbours
                       feq1 = computeF(density, c_squ, u_squ, t2, ux, ux)
                       feq3 = computeF(density, c_squ, u_squ, t2, uy, uy)
                       feq5 = computeF(density, c_squ, u_squ, t2, -ux, ux)
                       feq7 = computeF(density, c_squ, u_squ, t2, -uy, uy)
                       # next-nearest neighbours
                       feq2 = computeF(density, c_squ, u_squ, t3, u_c2, u_c2)
                       feq4 = computeF(density, c_squ, u_squ, t3, u_c4, u_c4)
                       feq6 = computeF(density, c_squ, u_squ, t3, u_c6, u_c6)
                       feq8 = computeF(density, c_squ, u_squ, t3, u_c8, u_c8)
                       if ON[0,0] == 1
                           G5[0,0] = f1
                           G6[0,0] = f2
                           G7[0,0] = f3
                           G8[0,0] = f4
                           G1[0,0] = f5
                           G2[0,0] = f6
                           G3[0,0] = f7
                           G4[0,0] = f8
                       else
                           G1[0,0] = omega * feq1 + (1. - omega) * f1
                           G2[0,0] = omega * feq2 + (1. - omega) * f2
                           G3[0,0] = omega * feq3 + (1. - omega) * f3
                           G4[0,0] = omega * feq4 + (1. - omega) * f4
                           G5[0,0] = omega * feq5 + (1. - omega) * f5
                           G6[0,0] = omega * feq6 + (1. - omega) * f6
                           G7[0,0] = omega * feq7 + (1. - omega) * f7
                           G8[0,0] = omega * feq8 + (1. - omega) * f8
                       end
                       G9[0,0] = omega * feq9 + (1. - omega) * f9
                       return
                   end
        X = G
        G = F
        F = X
        prevavu = avu
        avu = sum(UX) / numactivenodes
        ts = ts + 1
    end
    return F
end

function boltzmann(nx, ny)
    density = 1.0
    F  = [ fill!(zeros(nx, ny), density/9) for i = 1:9 ]
    G  = deepcopy(F)
    X1 = [ x == 1 ? 1 : 0 for x = 1:nx, y = 1:ny ]
    UX = zeros(nx, ny)
    ON = zeros(Int, nx, ny)
    ON[rand(nx, ny) .> 0.7] = 1 # extremely porous random domain
    numactivenodes = sum(1 .- ON)
    boltzmann_kernel(numactivenodes, F, G, ON, UX, X1)
end

function main()
    doc = """boltzmann.jl

2D lattice Boltzmann fluid flow model.

Usage:
  boltzmann.jl -h | --help
  boltzmann.jl [--nx=<nx>] [--ny=<ny>]

Options:
  -h --help  Show this screen.
  --nx=<nx>  Specify size of simulation grid in the x dimension [default: 200].
  --ny=<ny>  Specify size of simulation grid in the y dimension [default: 200].
"""
    arguments = docopt(doc)
    nx = parse(Int, arguments["--nx"])
    ny = parse(Int, arguments["--ny"])

    srand(0)

    println("grid size: ", nx, " x ", ny)

    tic()
    F = boltzmann(1, 1)
    println("SELFPRIMED ", toq())

    tic()
    F = boltzmann(nx, ny)
    println("SELFTIMED ", toq())

    println("checksum: ", sum(F[1]), " ", sum(F[2]), " ", sum(F[3]), " ", sum(F[4]), " ", sum(F[5]), " ", sum(F[6]), " ", sum(F[7]), " ", sum(F[8]), " ", sum(F[9]))
end

main()
