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

@acc function kmeans(numCenter, iterNum, points)
    D = size(points,1) # number of features
    N = size(points,2) # number of instances
    centroids = rand(D, numCenter)

    for l in 1:iterNum
        dist = [ Float64[sqrt(sum((points[:,i].-centroids[:,j]).^2)) for j in 1:numCenter] for i in 1:N]
        labels = Int[indmin(dist[i]) for i in 1:N]
        centroids = Float64[ sum(points[j,labels.==i])/sum(labels.==i) for j in 1:D, i in 1:numCenter]
    end 
    return centroids
end

function main()
    doc = """K-means clustering algorithm.

Usage:
  k-means.jl -h | --help
  k-means.jl [--iterations=<iterations>] [--size=<size>] [--centers=<centers>]

Options:
  -h --help                  Show this screen.
  --size=<size>              Specify number of points; defaults to 50000.
  --iterations=<iterations>  Specify number of iterations; defaults to 30.
  --centers=<centers>        Specify number of centers; defaults to 5.

"""
    arguments = docopt(doc)

    if (arguments["--size"] != nothing)
        size = parse(Int, arguments["--size"])
    else
        size = 50000
    end


    if (arguments["--iterations"] != nothing)
        iterations = parse(Int, arguments["--iterations"])
    else
        iterations = 30
    end

    if (arguments["--centers"] != nothing)
        numCenter = parse(Int, arguments["--centers"])
    else
        numCenter = 5
    end

    D = 20
    srand(0)

    println("iterations = ", iterations)
    println("centers= ", numCenter) 
    println("number of points = ", size) 
    points = rand(D,size)

    tic()
    kmeans(numCenter, 2, rand(D,100))
    time = toq()
    println("SELFPRIMED ", time)

    tic()
    centroids_out = kmeans(numCenter, iterations, points)
    time = toq()
    println("result = ", centroids_out)
    println("rate = ", iterations / time, " iterations/sec")
    println("SELFTIMED ", time)

end

main()
