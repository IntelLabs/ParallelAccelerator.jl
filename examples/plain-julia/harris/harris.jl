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

# Harris Corner Detection
# Original port by Todd A. Anderson from PolyMage version.

using DocOpt
using Images

# Compute and return the application of a stencil to an input 2D array
# shape specifies the offsets from output pixel of the desired points of source
# kf takes these values in same order and computes value of output pixel
# boundary is just zeros (where any point doesn't exist)
function run(src, shape, kf)
  (s1, s2) = size(src)
  min1 = 1
  max1 = s1
  min2 = 1
  max2 = s1
  npts = length(shape)
  for p in 1:npts
    (o1, o2) = shape[p]
    min1 = max(min1, 1-o1)
    max1 = min(max1, s1-o1)
    min2 = max(min2, 1-o2)
    max2 = min(max2, s2-o2)
  end
  dst = Array(Float32, s1, s2)
  args = Array(Float32, npts)
  for i1 in 1:s1
    for i2 in 1:s2
      if i1>=min1 && i1<=max1 && i2>=min2 && i2<=max2
        for p in 1:npts
          (o1, o2) = shape[p]
          args[p] = src[i1+o1, i2+o2]
        end
        dst[i1, i2] = kf(args)
      else
        dst[i1, i2] = 0.0f0
      end
    end
  end
  return dst
end

function harrisCornerDetect(Iin)
  (w, h) = size(Iin)

  Ix_args(args) = ((args[1] * -1.0f0) + (args[2] * -2.0f0) + (args[3] * -1.0f0) + args[4] + (args[5] * 2.0f0) + args[6]) / 12.0f0
  Ix = run(Iin, [(-1,-1), (-1,0), (-1,1), (1,-1), (1,0), (1,1)], Ix_args)
  Iy_args(args) = ((args[1] * -1.0f0) + (args[2] * -2.0f0) + (args[3] * -1.0f0) + args[4] + (args[5] * 2.0f0) + args[6]) / 12.0f0
  Iy = run(Iin, [(-1,-1), (0,-1), (1,-1), (-1,1), (0,1), (1,1)], Iy_args)

  Ixx = Ix .* Ix
  Iyy = Iy .* Iy
  Ixy = Ix .* Iy

  S_args(args) = sum(args)
  Sxx = run(Ixx, [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)], S_args)
  Syy = run(Iyy, [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)], S_args)
  Sxy = run(Ixy, [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)], S_args)

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
  --img-file=<img-file>  Specify a path to an input (grayscale) image file [default: ../../example.jpg].
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
        res = harrisCornerDetect(Matrix{Float32}())
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
