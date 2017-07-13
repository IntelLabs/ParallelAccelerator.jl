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

#CompilerTools.OptFramework.set_debug_level(3)
#CompilerTools.LambdaHandling.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)
#ParallelAccelerator.DomainIR.set_debug_level(3)
#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)

@acc function compute(src, dst, N)
  runStencil(dst, src, N, :oob_skip) do b, a
    b[0,0,0] = (a[0,0,-1] + a[0,0,1] + a[0,-1,0] + a[0,1,0] + a[-1,0,0] + a[1,0,0]) / 6
    return a, b
  end
  return isodd(N) ? dst : src
end

function initialize(x, y, z)
  src = Array{Float32}(x, y, z)
  rand!(src)
  dst = copy(src)
  return src, dst
end

function checksum(src)
  local z, y, x
  (x, y, z) = size(src)
  sum(src) / (x * y * z)
end

function main()
    doc = """laplace-3d.jl

Laplace 6-point 3D stencil.

Usage:
  laplace-3d.jl -h | --help
  laplace-3d.jl [--size=<size>] [--iterations=<iterations>]

Options:
  -h --help                  Show this screen.
  --size=<size>              Specify a 3d array size (<size> x <size> x <size>) [default: 300].
  --iterations=<iterations>  Specify a number of iterations [default: 100].
"""
    arguments = docopt(doc)

    size = parse(Int, arguments["--size"])
    iterations = parse(Int, arguments["--iterations"])

    srand(0)

    function laplace_3d(iterations)
        local src, dst
        println("Run laplace-3d with size ", size, "x", size, "x", size, " for ", iterations, " iterations.")
        (src, dst) = initialize(size, size, size)
        tic()
        compute(src, dst, 0)
        println("SELFPRIMED ", toq())

        tic()
        dst = compute(src, dst, iterations)
        println("SELFTIMED ", toq())
        println("checksum: ", checksum(dst))
    end

    laplace_3d(iterations)

end

main()
