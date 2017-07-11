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
using Base.Test

include("lib.jl")
include("parfor.jl")
include("mapreduce.jl")
include("abs.jl")
include("const_promote.jl")
include("rand.jl")
include("BitArray.jl")
include("range.jl")
include("seq.jl")
include("cat.jl")
include("hcat.jl")
include("vcat.jl")
include("ranges.jl")
#include("misc.jl")
include("aug_assign.jl")
include("complex.jl")
include("print.jl")
include("strings.jl")
include("test_lr.jl")
include("test_kmeans.jl")
include("gemv_test.jl")
include("transpose_test.jl")
include("vecnorm_test.jl")
include("broadcast.jl")

# Examples.  We're not including them all here, because it would take
# too long, but just including a few seems like a good compromise that
# exercises much of ParallelAccelerator.

module TestBlackScholes
using Base.Test
include("../examples/black-scholes/black-scholes.jl")
# black-scholes should have only 1 allocation and 1 parfor after optimization
if !(ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE)
@test ParallelAccelerator.get_num_acc_allocs()==1 && ParallelAccelerator.get_num_acc_parfors()==1
end
end

module TestPi
using Base.Test
include("../examples/pi/pi.jl")
# pi should have no allocation and 1 parfor after optimization
if !(ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE)
@test ParallelAccelerator.get_num_acc_allocs()==0 && ParallelAccelerator.get_num_acc_parfors()==1
end
end

module TestOptFlow
using Base.Test
include("../examples/opt-flow/opt-flow.jl")
end

module TestKMeans
using Base.Test
include("../examples/k-means/k-means.jl")
# k-means should have 3 top-level parfors after optimization
if !(ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE)
@test ParallelAccelerator.get_num_acc_parfors()==3
end
end

#module TestLR
#using Base.Test
#include("../examples/logistic_regression/logistic_regression.jl")
# logistic_regression should have 5 top-level parfors after optimization
#if !(ParallelAccelerator.getPseMode() == ParallelAccelerator.THREADS_MODE)
#@test ParallelAccelerator.get_num_acc_parfors()==5
#end
#end

# Delete file left behind by opt-flow.
dir = pwd()
img_file = joinpath(dir, "out.flo")
rm(img_file)
