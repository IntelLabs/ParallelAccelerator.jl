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

# Some image manipulation functions

module Image

using ParallelAccelerator
export downSample, interpolateFlow, warpMotion, readImage, writeFlo

@acc begin

function downsample_inner(x, y, sx, sy, a)
  x0 = sx * Float32(x - 1)
  y0 = sy * Float32(y - 1)
  x1 = sx * Float32(x) - 1f-3
  y1 = sy * Float32(y) - 1f-3
  c = 0f0
  for i = round(Int, floor(x0)):round(Int, ceil(x1)) - 1
    for j = round(Int, floor(y0)):round(Int, ceil(y1)) - 1
      ii = Float32(i)
      jj = Float32(j)
      p = ii < x0 ? (ii + 1f0 - x0) : (ii > x1 - 1f0 ? (x1 - ii) : 1f0)
      q = jj < y0 ? (jj + 1f0 - y0) : (jj > y1 - 1f0 ? (y1 - jj) : 1f0)
      c += p * q * a[i + 1, j + 1]
    end
  end
  c / (sx * sy)
end

# Down sample an image to a smaller size
#   We can think of the image as dividing up into a grid of equal sized pixels
#   The pixels of the desired smaller image overlap the pixels of the larger original image
#   We want the value of the output pixel to be the average of the input pixels overlapped weighted by
#   area of overlap
# Pre: size(oi)>=(nw,nh)
function downSample(a, nw, nh)
  (w, h) = size(a)
  sx::Float32 = Float32(w) / Float32(nw)
  sy::Float32 = Float32(h) / Float32(nh)
  na = [ downsample_inner(x, y, sx, sy, a) for x = 1:nw, y = 1:nh ]
  return na
end

# Linearly interpolate between array values
# Pre: 1<=x<size(a,1) && 1<=y<size(a,2)
function interpolate(a, x, y)
# Returns Float32
  xx = round(Int, floor(x))
  yy = round(Int, floor(y))
  #println("x,y=",x,",",y)
  alpha = x-Float32(xx)
  beta  = y-Float32(yy)
  return Float32(beta*(alpha*a[xx+1,yy+1] + (1f0-alpha)*a[xx,yy+1]) +
         (1f0-beta)*(alpha*a[xx+1,yy] + (1f0-alpha)*a[xx,yy]))
end

# Given a flow at a coarser grain, interpolate it to a finer grain
# Pre: size(ou)==size(ov)<=(nh,nw)
function interpolateFlow(ou, ov, nw, nh)
# Returns two Matrix{Float32} size=(nh,nw)
  (ow, oh) = size(ou)
  if ow==nw && oh==nh
    return ou, ov
  end
  sx::Float32 = Float32(ow-1) / Float32(nw)
  sy::Float32 = Float32(oh-1) / Float32(nh)
  nu = [ interpolate(ou, Float32(x-1)*sx+1f0, Float32(y-1)*sy+1f0)*sx for x = 1:nw, y = 1:nh ]
  nv = [ interpolate(ou, Float32(x-1)*sx+1f0, Float32(y-1)*sy+1f0)*sx for x = 1:nw, y = 1:nh ]
  return nu, nv
end

function warpMotion_inner(u,v,i,ii,w,h,x,y)
  nx = Float32(x)+u[x,y]
  ny = Float32(y)+v[x,y]
  (nx<1f0 || nx>=float(w) || ny<1f0 || ny>=float(h)) ? ii[x,y] : interpolate(i, nx, ny)
end

# Given a flow and an image, move the image according to the flow
# Pre: size(i)==size(u)==size(v)==size(ii)
function warpMotion(i, u, v, ii)
# Returns Matrix{Float32} size=size(i)
  (w::Int, h::Int) = size(i)
  [ warpMotion_inner(u,v,i,ii,w,h,x,y) for x = 1:w, y = 1:h ]
end

end

function readImage(fn)
  f = open(fn, "r")
  w = read(f, Int32)
  h = read(f, Int32)
  img = zeros(Float32, w, h)
  for i in 1:h
    for j in 1:w
      img[j,i] = Float32(read(f, UInt8))/255.0
    end
  end
  close(f)
  return img
end

function writeFlo(u, v, fn)
  w = size(u, 1)
  h = size(u, 2)
  f = open(fn, "w")
  write(f, w)
  write(f, h)
  for i in 1:h
    for j in 1:w
      write(f, u[j, i])
      write(f, v[j, i])
    end
  end
  close(f)
end

end
