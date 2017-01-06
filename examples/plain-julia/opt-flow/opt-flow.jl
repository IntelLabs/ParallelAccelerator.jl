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

include("image.jl")
using .Image

# Solve optical flow problem at one scale
# Flow problem is formulated as the solution of a linear system where matrix is sparse
# Use an iterative method to
# Pre: size(i1)==size(i2)
function singleScaleOpticalFlow(i1::Matrix{Float32}, i2::Matrix{Float32}, lam::Float32, ni::Int)
# Returns Matrix{Float32} size=(size(i1,1),size(i1,2),2)
  (h, w) = size(i1)
  # Compute the gradient of i1
  # This is basically two stencils, one for x and one for y
  # The stencil is [1 8 0 -8 -1]/12 and leave zeros in the boundary
  Ix = zeros(Float32, h, w)
  Iy = zeros(Float32, h, w)
  for y in 1:h
    for x in 1:w
      if x>2 && x<w-1
        Ix[y,x] = (-i1[y,x+2]-8.0f0*i1[y,x+1]+8.0f0*i1[y,x-1]+i1[y,x-2])/12.0f0
      end
      if y>2 && y<h-1
        Iy[y,x] = (-i1[y+2,x]-8.0f0*i1[y+1,x]+8.0f0*i1[y-1,x]+i1[y-2,x])/12.0f0
      end
    end
  end
  # Compute the time partial derivative
  It = i1-i2

  # Block Jacobi preconditioner
  function blockJacobiPreconditioner(r)
    a = Ix.*Ix.+4.0f0*lam
    b = Ix.*Iy
    c = Iy.*Iy.+4.0f0*lam
    invdet = 1.0f0./(a.*c-b.*b)
    res = zeros(Float32, h, w, 2)
    res[:,:,1] = (c.*r[:,:,1]-b.*r[:,:,2]).*invdet
    res[:,:,2] = (a.*r[:,:,2]-b.*r[:,:,1]).*invdet
    return res
  end

  # Matrix vector product
  function mvm(x)
    # This operation is equivalent to a stencil
    # Want:
    #   -lam for (-1,0,0), (+1,0,0), (0,-1,0), and (0,+1,0)
    #   Ix*Iy for (0,0,flip)
    #   Ix^2+4*lam and Iy^2+4*lam for (0,0,0) respectively (Ix for u, Iy for v)
    # At boundaries just include terms that exist
    y = zeros(Float32, h, w, 2)
    for i in 1:h
      for j in 1:w
        ur = Ix[i,j]*(Ix[i,j]*x[i,j,1]+Iy[i,j]*x[i,j,2])+4.0f0*lam*x[i,j,1]
        vr = Iy[i,j]*(Ix[i,j]*x[i,j,1]+Iy[i,j]*x[i,j,2])+4.0f0*lam*x[i,j,2]
        if i>1
          ur += -lam*x[i-1,j,1]
          vr += -lam*x[i-1,j,2]
        end
        if j>1
          ur += -lam*x[i,j-1,1]
          vr += -lam*x[i,j-1,2]
        end
        if i<h
          ur += -lam*x[i+1,j,1]
          vr += -lam*x[i+1,j,2]
        end
        if j<w
          ur += -lam*x[i,j+1,1]
          vr += -lam*x[i,j+1,2]
        end
        y[i,j,1] = ur
        y[i,j,2] = vr
      end
    end
    return y
  end

  # Initial for iterative solve
  x = zeros(Float32, h, w, 2)
  r = zeros(Float32, h, w, 2)
  r[:,:,1] = -Ix.*It
  r[:,:,2] = -Iy.*It
  z = blockJacobiPreconditioner(r)
  p = z
  rsold = sum(r.*z)

  # Iterate to solution
  # Use preconditioned conjugate gradient algorithm
  for i in 1:ni
    Ap = mvm(p)
    pTAp = sum(p.*Ap)
    alpha = rsold/pTAp
    x += alpha*p
    r -= alpha*Ap
    z = blockJacobiPreconditioner(r)
    rsnew = sum(r.*z)
    beta = rsnew/rsold
    p = z + beta*p
    rsold = rsnew
  end

  return x
end

# Solve optical flow problem at all scales
# Pre: size(i1)==size(i2)
function multiScaleOpticalFlow(i1::Matrix{Float32}, i2::Matrix{Float32}, lam::Float32, ni::Int, ns::Int)
# Returns Matrix{Float32} size=(size(i1,1),size(i1,2),2)
  (h, w) = size(i1)
  if ns==1
    return singleScaleOpticalFlow(i1, i2, lam, ni)
  end
  scale = Float32((Float64(w)/50.0)^(-1.0/ns))
  local u # initialised in the first iteration of the loop
  for i in ns:-1:0
    cw = round(Int, w*scale^Float32(i))
    ch = round(Int, h*scale^Float32(i))
    println("Doing scale: ", cw, "x", ch)
    # Downsample the images
    si1 = Image.downSample(i1, cw, ch)
    si2 = Image.downSample(i2, cw, ch)
    # Interpolate the current flow and warp image 2 by current flow
    if i==ns
      # Coarsest scale, no flow yet, use zeros, warp is image 2
      u = zeros(Float32, ch, cw, 2)
      wi = si2
    else
      u = Image.interpolateFlow(u, cw, ch)
      wi = Image.warpMotion(si2, u, si1)
    end
    # Compute the flow from image 1 to warp
    if i==0
      ni *= 5
    end
    du = singleScaleOpticalFlow(si1, wi, lam, ni)
    # Update the flow
    u += du
  end
  return u
end

function main()
    doc = """opt-flow.jl

Horn-Schunck optical flow estimator.

Usage:
  opt-flow.jl -h | --help
  opt-flow.jl [--image-path-1=<image-path-1>] [--image-path-2=<image-path-2>]

Options:
  -h --help                      Show this screen.
  --image-path-1=<image-path-1>  Specify a path to the first image [default: ../../opt-flow/small_001.dat].
  --image-path-2=<image-path-2>  Specify a path to the second image [default: ../../opt-flow/small_002.dat].
"""
    arguments = docopt(doc)

    dir = dirname(@__FILE__)
    img_file_1 = joinpath(dir, arguments["--image-path-1"])
    img_file_2 = joinpath(dir, arguments["--image-path-2"])

    i1 = Image.readImage(img_file_1)
    i2 = Image.readImage(img_file_2)
    if size(i1)!=size(i2)
        write(STDERR, "images are different sizes")
        exit(-1)
    end

    println("filenames = ", img_file_1, " ", img_file_2)
    println("Image size: ", size(i1, 1), "x", size(i1, 2))

    tic()
    multiScaleOpticalFlow(i1, i2, Float32(0.025), 100, 1)
    println("SELFPRIMED ", toq())

    tic()
    u = multiScaleOpticalFlow(i1, i2, Float32(0.025), 100, 44)
    selftimed = toq()
    Image.writeFlo(u, "out.flo")
    println("SELFTIMED ", selftimed)
end

main()
