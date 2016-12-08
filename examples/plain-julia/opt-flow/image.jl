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

export downSample, interpolateFlow, warpMotion, readImage, writeFlo

# Down sample an image to a smaller size
#   We can think of the image as dividing up into a grid of equal sized pixels
#   The pixels of the desired smaller image overlap the pixels of the larger original image
#   We want the value of the output pixel to be the average of the input pixels overlapped weighted by
#   area of overlap
# Pre: size(oi)>=(nw,nh)
function downSample(oi::Matrix{Float32}, nw::Int, nh::Int)
# Returns Matrix{Float32} size=(nh,nw)
  (oh, ow) = size(oi)
  if ow==nw && oh==nh
    return oi
  end
  fx = Float32(ow)/Float32(nw)   # width of old pixels per new pixel
  fy = Float32(oh)/Float32(nh)   # height of old pixels per new pixel
  ni = zeros(Float32, nh, nw)
  for y in 1:nh
    for x in 1:nw
      r = 0.0f0
      fi = Float32(y-1)*fy     # start vertical in old pixels of new pixel
      fli = Float32(y)*fy      # end vertical in old pixels of new pixel
      li = round(Int, floor(fli))+1  # end vertical index in old pixels of new pixel
      fj = Float32(x-1)*fx     # start horozontal in old pixels of new pixel
      flj = Float32(x)*fx      # end horozontal in old pixels of new pixel
      lj = round(Int, floor(flj))+1  # end horozontal index in old pixels of new pixel
      for i in round(Int, floor(fi))+1:min(li, oh)
        for j in round(Int, floor(fj))+1:min(lj, ow)
          c = 1.0f0  # weight each full overlapped old pixel by one, adjusted below
          if i-fi<1  # adjust for start vertical overlap
            c = i-fi
          end
          if j-fj<1  # adjust for start horozontal overlap
            c *= j-fj
          end
          if i==li   # adjust for end vertical overlap
            c *= fli-li+1
          end
          if j==lj   # adjust for end horozontal overlap
            c *= flj-lj+1
          end
          r += c*oi[i,j]
        end
      end
      ni[y,x] = r/(fx*fy)  # adjust for total area of old pixels overlapped by new pixel
    end
  end
  return ni
end

# Linearly interpolate between array values
# Pre: 1<=x<=size(a,2) && 1<=y<=size(a,1) && 1<=j<=size(a,3)
function interpolate(a::Array{Float32, 3}, x::Float32, y::Float32, j)
# Returns Float32
  xx = round(Int, floor(x))
  yy = round(Int, floor(y))
  alpha = x-xx
  beta = y-yy
  if alpha>0.0f0
    if beta>0.0f0
      return        beta *(alpha*a[yy+1,xx+1,j]+(1.0f0-alpha)*a[yy+1,xx,j])+
             (1.0f0-beta)*(alpha*a[yy,  xx+1,j]+(1.0f0-alpha)*a[yy,  xx,j])
    else
      return alpha*a[yy,xx+1,j]+(1.0f0-alpha)*a[yy,xx,j]
    end
  else
    if beta>0.0f0
      return beta*a[yy+1,xx,j]+(1.0f0-beta)*a[yy,xx,j]
    else
      return a[yy,xx,j]
    end
  end
end

# Linearly interpolate between array values
# Pre: 1<=x<=size(a,2) && 1<=y<=size(a,1)
function interpolate(a::Matrix{Float32}, x::Float32, y::Float32)
# Returns Float32
  xx = round(Int, floor(x))
  yy = round(Int, floor(y))
  alpha = x-xx
  beta = y-yy
  if alpha>0.0f0
    if beta>0.0f0
      return        beta *(alpha*a[yy+1,xx+1]+(1.0f0-alpha)*a[yy+1,xx])+
             (1.0f0-beta)*(alpha*a[yy,  xx+1]+(1.0f0-alpha)*a[yy,  xx])
    else
      return alpha*a[yy,xx+1]+(1.0f0-alpha)*a[yy,xx]
    end
  else
    if beta>0.0f0
      return beta*a[yy+1,xx]+(1.0f0-beta)*a[yy,xx]
    else
      return a[yy,xx]
    end
  end
end

# Given a flow at a coarser grain, interpolate it to a finer grain
# Pre: size(ou,1)<=nh && size(ou,2)<=nw && size(ou,3)=2
function interpolateFlow(ou::Array{Float32, 3}, nw::Int, nh::Int)
# Returns Array{Float32, 3} size=(nh,nw,2)
  oh = size(ou, 1)
  ow = size(ou, 2)
  if ow==nw && oh==nh
    return ou
  end
  nu = zeros(Float32, nh, nw, 2)
  s = Float32(Float64(nw)/Float64(ow))
  for y in 1:nh
    for x in 1:nw
      nx = Float32(x-1)/Float32(nw)*Float32(ow)+1
      ny = Float32(y-1)/Float32(nh)*Float32(oh)+1
      if nx>=ow
        nx = Float32(ow)
      end
      if ny>=oh
        ny = Float32(oh)
      end
      nu[y,x,1] = interpolate(ou, nx, ny, 1)/s
      nu[y,x,2] = interpolate(ou, nx, ny, 2)/s
    end
  end
  return nu
end

# Given a flow and an image, move the image according to the flow
# Pre: size(i,1)==size(u,1)==size(ii,1) && size(i,2)=size(u,2)==size(ii,2) && size(u,3)=2
function warpMotion(i::Matrix{Float32}, u::Array{Float32, 3}, ii::Matrix{Float32})
# Returns Matrix{Float32} size=size(i)
  (h, w) = size(i)
  wi = zeros(Float32, h, w)
  for y in 1:h
    for x in 1:w
      nx = x+u[y,x,1]
      ny = y+u[y,x,2]
      wi[y,x] = (nx<1 || nx>w || ny<1 || ny>h ? ii[y,x] : interpolate(i, nx, ny))
    end
  end
  return wi
end

function readImage(fn::AbstractString)
  f = open(fn, "r")
  w = read(f, UInt32)
  h = read(f, UInt32)
  img = zeros(Float32, h, w)
  for i in 1:h
    for j in 1:w
      img[i,j] = Float32(read(f, UInt8))/255.0
    end
  end
  close(f)
  return img
end

function writeFlo(u::Array{Float32, 3}, fn::AbstractString)
  h = size(u, 1)
  w = size(u, 2)
  f = open(fn, "w")
  write(f, w)
  write(f, h)
  for i in 1:h
    for j in 1:w
      write(f, u[i, j, 1])
      write(f, u[i, j, 2])
    end
  end
  close(f)
end

end
