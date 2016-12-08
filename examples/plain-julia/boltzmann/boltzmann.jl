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

using DocOpt

function boltzmann_kernel(nx, ny, numactivenodes, F, ON, UX, UY)
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
        # Propagate
        F[4]=F[4][[2:nx; 1], [ny; 1:ny-1]]
        F[3]=F[3][:, [ny; 1:ny-1]]
        F[2]=F[2][[nx; 1:nx-1], [ny; 1:ny-1]]
        F[5]=F[5][[2:nx; 1], :]
        F[1]=F[1][[nx; 1:nx-1], :]
        F[6]=F[6][[2:nx; 1], [2:ny; 1]]
        F[7]=F[7][:, [2:ny; 1]]
        F[8]=F[8][[nx; 1:nx-1], [2:ny; 1]]
        DENSITY=sum(F)
        UX=(sum(F[[1;2;8]])-sum(F[[4;5;6]]))./DENSITY
        UY=(sum(F[[2;3;4]])-sum(F[[6;7;8]]))./DENSITY
        UX[1,1:ny]=UX[1,1:ny].+deltaU # Increase inlet pressure
        UX[ON]=0.
        UY[ON]=0.
        DENSITY[ON]=0.
        U_SQU=UX.^2+UY.^2
        U_C2=UX+UY
        U_C4=-UX+UY
        U_C6=-U_C2
        U_C8=-U_C4
        FEQ = Array(Any, 9)
        # Calculate equilibrium distribution: stationary
        FEQ[9]=t1.*DENSITY.*(1.-U_SQU./(2*c_squ))
        # nearest-neighbours
        FEQ[1]=t2.*DENSITY.*(1.+UX/c_squ+0.5*(UX./c_squ).^2-U_SQU./(2*c_squ))
        FEQ[3]=t2.*DENSITY.*(1.+UY/c_squ+0.5*(UY./c_squ).^2-U_SQU./(2*c_squ))
        FEQ[5]=t2.*DENSITY.*(1.-UX/c_squ+0.5*(UX./c_squ).^2-U_SQU./(2*c_squ))
        FEQ[7]=t2.*DENSITY.*(1.-UY/c_squ+0.5*(UY./c_squ).^2-U_SQU./(2*c_squ))
        # next-nearest neighbours
        FEQ[2]=t3.*DENSITY.*(1.+U_C2/c_squ+0.5*(U_C2./c_squ).^2-U_SQU./(2*c_squ))
        FEQ[4]=t3.*DENSITY.*(1.+U_C4/c_squ+0.5*(U_C4./c_squ).^2-U_SQU./(2*c_squ))
        FEQ[6]=t3.*DENSITY.*(1.+U_C6/c_squ+0.5*(U_C6./c_squ).^2-U_SQU./(2*c_squ))
        FEQ[8]=t3.*DENSITY.*(1.+U_C8/c_squ+0.5*(U_C8./c_squ).^2-U_SQU./(2*c_squ))
        FEQ = omega .* FEQ .+ (1. - omega) .* F
        FEQ[1][ON] .= F[5][ON]
        FEQ[2][ON] .= F[6][ON]
        FEQ[3][ON] .= F[7][ON]
        FEQ[4][ON] .= F[8][ON]
        FEQ[5][ON] .= F[1][ON]
        FEQ[6][ON] .= F[2][ON]
        FEQ[7][ON] .= F[3][ON]
        FEQ[8][ON] .= F[4][ON]
        F = FEQ
        prevavu = avu
        avu = sum(UX) / numactivenodes
        ts = ts + 1
    end
    return F
end

function boltzmann(nx, ny)
    density = 1.0
    F  = [ fill!(zeros(nx, ny), density/9) for i = 1:9 ]
    UX = zeros(nx, ny)
    UY = zeros(nx, ny)
    ON = rand(nx, ny) .> 0.7 # extremely porous random domain
    println("length of ON = ", sum(ON))
    OFF = ~ON
    numactivenodes = sum(OFF)
    boltzmann_kernel(nx, ny, numactivenodes, F, ON, UX, UY)
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
