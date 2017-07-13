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

include("image.jl")
using .Image

#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)
ParallelAccelerator.ParallelIR.PIRNumThreadsMode(2)

@acc begin
# Block Jacobi preconditioner
# Used in singleScaleOpticalFlow below
# Needs to be toplevel function for use with ParallelAccelerator
@inline function blockJacobiPreconditioner(Ix, Iy, ru, rv, lam)
    a = Ix.*Ix.+4.0f0*lam
    b = Ix.*Iy
    c = Iy.*Iy.+4.0f0*lam
    invdet = 1.0f0./(a.*c-b.*b)
    rru = (c.*ru-b.*rv).*invdet
    rrv = (a.*rv-b.*ru).*invdet
    rru, rrv
end

# Solve optical flow problem at one scale
# Flow problem is formulated as the solution of a linear system where matrix is sparse
# Use an iterative method to
# Pre: size(i1)==size(i2)
function singleScaleOpticalFlow(i1::Matrix{Float32}, i2::Matrix{Float32}, lam::Float32, ni::Int)
# Returns two Matrix{Float32} size=size(i1)
  (w, h) = size(i1)
  # Compute the gradient of i1
  # This is basically two stencils, one for x and one for y
  # The stencil is [1 8 0 -8 -1]/12 and leave zeros in the boundary
  Ix = Array{Float32}(w, h)
  runStencil(Ix, i1, 1, :oob_dst_zero) do b, a
    b[0,0] = (a[-2,0] + 8.0f0 * a[-1,0] - 8.0f0 * a[1,0] - a[2,0])/12.0f0
  end
  Iy = Array{Float32}(w, h)
  runStencil(Iy, i1, :oob_dst_zero) do b, a
    b[0,0] = (a[0,-2] + 8.0f0 * a[0,-1] - 8.0f0 * a[0,1] - a[0,2])/12.0f0
  end
  # Compute the time partial derivative
  It = i1-i2

  # Initialise for iterative solve
  xu = zeros(Float32, w, h)
  xv = zeros(Float32, w, h)
  ru = -Ix.*It
  rv = -Iy.*It
  zu, zv = blockJacobiPreconditioner(Ix, Iy, ru, rv, lam)
  pu, pv = zu, zv
  rsold = sum(ru.*zu)+sum(rv.*zv)

  # Iterate to solution
  for i in 1:ni
    # The matrix vector product is equivalent to a stencil
    # Want:
    #   -lam for (-1,0), (+1,0), (0,-1), and (0,+1)
    #   Ix*Iy for (0,0) of the other array
    #   Ix^2+4*lam and Iy^2+4*lam for (0,0) respectively (Ix for u, Iy for v)
    # At boundaries just include terms that exist
    Apu = Array{Float32}(w, h)
    Apv = Array{Float32}(w, h)
    runStencil(Apu, Apv, pu, pv, Ix, Iy, :oob_src_zero) do Apu, Apv, pu, pv, Ix, Iy
      ix = Ix[0,0]
      iy = Iy[0,0]
      Apu[0,0] = ix * (ix*pu[0,0] + iy*pv[0,0]) + lam*(4.0f0*pu[0,0]-(pu[-1,0]+pu[1,0]+pu[0,-1]+pu[0,1]))
      Apv[0,0] = iy * (ix*pu[0,0] + iy*pv[0,0]) + lam*(4.0f0*pv[0,0]-(pv[-1,0]+pv[1,0]+pv[0,-1]+pv[0,1]))
    end
    pTAp = sum(pu.*Apu)+sum(pv.*Apv)
    alpha = rsold/pTAp
    xu += alpha*pu
    xv += alpha*pv
    ru -= alpha*Apu
    rv -= alpha*Apv
    zu, zv = blockJacobiPreconditioner(Ix, Iy, ru, rv, lam)
    rsnew = sum(ru.*zu)+sum(rv.*zv)
    beta = rsnew/rsold
    pu = zu + beta*pu
    pv = zv + beta*pv
    rsold = rsnew
  end

  return xu, xv
end

# Solve optical flow problem at all scales
# Pre: size(i1)==size(i2)
@fastmath function multiScaleOpticalFlow(i1::Matrix{Float32}, i2::Matrix{Float32}, lam::Float32, ni::Int, ns::Int)
# Returns two Matrix{Float32} size=size(i1)
  (w, h) = size(i1)
  if ns==1
    return singleScaleOpticalFlow(i1, i2, lam, ni)
  end
  scale = Float32((Float64(w)/50.0)^(-1.0/Float32(ns)))
  local u::Array{Float32,2}, v::Array{Float32,2} # initialised in the first iteration of the loop
  i = ns
  while i >= 0
    cw = round(Int, Float32(w)*scale^Float32(i))
    ch = round(Int, Float32(h)*scale^Float32(i))
    # println("cw,ch=",cw,",",ch)
    # println("Doing scale: ", cw, "x", ch)
    # Downsample the images
    si1 = Image.downSample(i1, cw, ch)
    si2 = Image.downSample(i2, cw, ch)
    # Interpolate the current flow and warp image 2 by current flow
    if i==ns
      wi = si2
      u = zeros(Float32, cw, ch)
      v = zeros(Float32, cw, ch)
    else
      # Coarsest scale, no flow yet, use zeros, warp is image 2
      u, v = Image.interpolateFlow(u, v, cw, ch)
      wi = Image.warpMotion(si2, u, v, si1)
    end
    # Compute the flow from image 1 to warp
    # println("checksum si1 wi: ", sum(si1), " ", sum(wi))
    du, dv = singleScaleOpticalFlow(si1, wi, lam, (i==0 ? ni*5 : ni))
    # Update the flow
    u += du
    v += dv
    i -= 1
  end
  return u, v
end

function multiFrameOpticalFlow(frames, nframes)
  cartesianarray(Tuple{Matrix{Float32}, Matrix{Float32}}, (nframes - 1,)) do i
    multiScaleOpticalFlow(frames[i], frames[i+1], Float32(0.025), 100, 44)
  end
end

end

function main()
    doc = """opt-flow.jl

Horn-Schunck multi-frame optical flow estimator.

Usage:
  opt-flow.jl -h | --help
  opt-flow.jl [--image-name-prefix=<image-name-prefix>] [--image-name-suffix=<image-name-suffix>] [--num-frames=<num-frames>]

Options:
  -h --help                     Show this screen.
  --image-name-prefix=<image-name-prefix>  Specify an image filename prefix [default: small_0].
  --image-name-suffix=<image-name-suffix>  Specify an image filename prefix [default: .dat].
  --num-frames=<num-frames>                Specify a number of frames (at least 2) [default: 2].

Assumes that image files are in the same directory as opt-flow.jl.
"""
    arguments = docopt(doc)

    fname_prefix = arguments["--image-name-prefix"]
    fname_suffix = arguments["--image-name-suffix"]
    nframes = parse(Int, arguments["--num-frames"])

    images_dir = dirname(@__FILE__)

    filenames = [ string(fname_prefix, string(i < 10 ? "0" : "", i), fname_suffix) for i = 1:nframes ]
    filepaths = [ joinpath(images_dir, i) for i in filenames ]

    println("nframes = ", nframes)
    assert(nframes > 1)
    frames = Array{Matrix{Float32}}(nframes)
    for i = 1:nframes
        frames[i] = Image.readImage(filepaths[i])
        if i > 1 && size(frames[i])!=size(frames[i-1])
            write(STDERR, "images are different sizes")
            exit(-1)
        end
    end

    println("filenames = ", filenames)
    println("checksums = ", map(sum, frames))
    println("Image size: ", size(frames[1], 1), "x", size(frames[1], 2))

    tic()
    multiFrameOpticalFlow(frames, 1)
    println("SELFPRIMED ", toq())

    tic()
    u,v = multiFrameOpticalFlow(frames, nframes)
    selftimed = toq()
    Image.writeFlo(u[1], v[1], "out.flo")
    println("checksum: ", sum(u[1]), " ", sum(v[1]))
    println("SELFTIMED ", selftimed)
end

main()
