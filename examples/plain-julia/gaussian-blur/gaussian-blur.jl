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
using Images

function blur(img, iterations)
    w, h = size(img)
    for i = 1:iterations
      img[3:w-2,3:h-2] =
           img[3-2:w-4,3-2:h-4] * 0.003  + img[3-1:w-3,3-2:h-4] * 0.0133 + img[3:w-2,3-2:h-4] * 0.0219 + img[3+1:w-1,3-2:h-4] * 0.0133 + img[3+2:w,3-2:h-4] * 0.0030 +
           img[3-2:w-4,3-1:h-3] * 0.0133 + img[3-1:w-3,3-1:h-3] * 0.0596 + img[3:w-2,3-1:h-3] * 0.0983 + img[3+1:w-1,3-1:h-3] * 0.0596 + img[3+2:w,3-1:h-3] * 0.0133 +
           img[3-2:w-4,3+0:h-2] * 0.0219 + img[3-1:w-3,3+0:h-2] * 0.0983 + img[3:w-2,3+0:h-2] * 0.1621 + img[3+1:w-1,3+0:h-2] * 0.0983 + img[3+2:w,3+0:h-2] * 0.0219 +
           img[3-2:w-4,3+1:h-1] * 0.0133 + img[3-1:w-3,3+1:h-1] * 0.0596 + img[3:w-2,3+1:h-1] * 0.0983 + img[3+1:w-1,3+1:h-1] * 0.0596 + img[3+2:w,3+1:h-1] * 0.0133 +
           img[3-2:w-4,3+2:h-0] * 0.003  + img[3-1:w-3,3+2:h-0] * 0.0133 + img[3:w-2,3+2:h-0] * 0.0219 + img[3+1:w-1,3+2:h-0] * 0.0133 + img[3+2:w,3+2:h-0] * 0.0030
    end
    return img
end

function main()
    doc = """gaussian-blur.jl

Gaussian blur image processing.

Usage:
  gaussian-blur.jl -h | --help
  gaussian-blur.jl [--img-file=<img-file>] [--iterations=<iterations>]

Options:
  -h --help                  Show this screen.
  --img-file=<img-file>      Specify a path to an input image file [default: ../../example.jpg].
  --iterations=<iterations>  Specify a number of iterations [default: 100].
"""
    arguments = docopt(doc)

    dir = dirname(@__FILE__)
    img_file = joinpath(dir, arguments["--img-file"])
    iterations = parse(Int, arguments["--iterations"])

    (fname, ext) = splitext(img_file)
    out_file = string(fname, "-blur", ".jpg")

    println("input file = ", img_file)
    println("iterations = ", iterations)
    println("output file = ", out_file)

    function gaussian_blur(input_fname, output_fname, iterations)
        local img = convert(Array{Float32, 2}, load(input_fname))
        tic()
        blur(img, 0)
        println("SELFPRIMED ", toq())

        tic()
        img = blur(img, iterations)
        println("SELFTIMED ", toq())
        save(output_fname, ufixed8sc(convert(Image, img)))
    end

    gaussian_blur(img_file, out_file, iterations)

end

main()
