using ParallelAccelerator
using Base.Test

include("example1.jl")
using Example1

@test Example1.main() == 10000
