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
using Images

@acc function harrisCornerDetect(Iin)
  (w, h) = size(Iin)

  Ix  = Array(Float32, w, h)
  Iy  = Array(Float32, w, h)
  Sxx = Array(Float32, w, h)
  Syy = Array(Float32, w, h)
  Sxy = Array(Float32, w, h)

  runStencil(Ix, Iin, :oob_dst_zero) do b, a
    b[0,0] = ((a[-1,-1] * -1.0f0) + (a[-1,0] * -2.0f0) + (a[-1,1] * -1.0f0) + a[1,-1] + (a[1,0] * 2.0f0) + a[1,1]) / 12.0f0
  end
  runStencil(Iy, Iin, :oob_dst_zero) do b, a
    b[0,0] = ((a[-1,-1] * -1.0f0) + (a[0,-1] * -2.0f0) + (a[1,-1] * -1.0f0) + a[-1,1] + (a[0,1] * 2.0f0) + a[1,1]) / 12.0f0
  end

  Ixx = Ix .* Ix
  Iyy = Iy .* Iy
  Ixy = Ix .* Iy

  runStencil(Sxx, Ixx, :oob_dst_zero) do b, a
    b[0,0] = (a[-1,-1] + a[-1,0] + a[-1,1] + a[0,-1] + a[0,0] + a[0,1] + a[1,-1] + a[1,0] + a[1,1])
  end
  runStencil(Syy, Iyy, :oob_dst_zero) do b, a
    b[0,0] = (a[-1,-1] + a[-1,0] + a[-1,1] + a[0,-1] + a[0,0] + a[0,1] + a[1,-1] + a[1,0] + a[1,1])
  end
  runStencil(Sxy, Ixy, :oob_dst_zero) do b, a
    b[0,0] = (a[-1,-1] + a[-1,0] + a[-1,1] + a[0,-1] + a[0,0] + a[0,1] + a[1,-1] + a[1,0] + a[1,1])
  end

  det    = (Sxx .* Syy) .- (Sxy .* Sxy)
  trace  = Sxx .+ Syy
  harris = det .- (0.04f0 .* trace .* trace)

  return harris
end

function main()
    doc = """harris.jl

Harris corner detection.

Usage:
  harris.jl -h | --help
  harris.jl [--img-file=<img-file>]

Options:
  -h --help              Show this screen.
  --img-file=<img-file>  Specify a path to an input (grayscale) image file [default: ../example.jpg].
"""
    arguments = docopt(doc)

    dir = dirname(@__FILE__)
    img_file = joinpath(dir, arguments["--img-file"])

    (fname, ext) = splitext(img_file)
    out_file = string(fname, "-corners", ".jpg")

    println("input file = ", img_file)
    println("output file = ", out_file)

    function harris(input_fname, output_fname)
        local img = convert(Matrix{Float32}, load(input_fname))
        tic()
        res = harrisCornerDetect(Matrix{Float32}(0, 0))
        println("SELFPRIMED ", toq())

        tic()
        res = harrisCornerDetect(img)
        selftimed = toq()
        res = map((x,y) -> x > 0.001f0 ? 1.0f0 : (y / 2.0f0), res, img)
        save(output_fname, ufixed8sc(convert(Image, res)))
        println("checksum: ", sum(res))
        println("SELFTIMED ", selftimed)
    end

    harris(img_file, out_file)

end

main()
