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

module TestKmeans
using ParallelAccelerator
using CompilerTools

#CompilerTools.OptFramework.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)
#ParallelAccelerator.DomainIR.set_debug_level(3)
#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)

    
@acc function kmeans(iterations::Int64, points, numCenter)
    D,N = size(points) # number of features, instances
    centroids = Float64[convert(Float64,i+j) for i in 1:D, j in 1:numCenter]

    for l in 1:iterations
        dist::Array{Array{Float64,1},1} = [ Float64[sqrt(sum((points[:,i]-centroids[:,j]).^2)) for j in 1:numCenter] for i in 1:N]
        labels::Array{Int,1} = [indmin(dist[i]) for i in 1:N]
        centroids::Array{Float64,2} = [ sum(points[j,labels.==i])/sum(labels.==i) for j in 1:D, i in 1:numCenter]
    end 
    return centroids
end

end

using Base.Test

points = [1. 2. 3. 4.; 4. 3. 2. 1.; 4. 5. 6. 7.]

println("Testing kmeans...")
@test_approx_eq TestKmeans.kmeans(10, points, 2) [1.5  3.5; 3.5  1.5; 4.5  6.5]
println("Done testing kmeans...")


