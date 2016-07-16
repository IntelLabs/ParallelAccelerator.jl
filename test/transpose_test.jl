module TestTranspose
using ParallelAccelerator

ParallelAccelerator.set_debug_level(3)
ParallelAccelerator.DomainIR.set_debug_level(3)
ParallelAccelerator.ParallelIR.set_debug_level(3)
ParallelAccelerator.CGen.set_debug_level(3)

@acc transpose_t(A) = A'

function test()
    A = [1. 2. 3.; 4. 5. 6.]
    B = transpose_t(A)
    return B
end

end

using Base.Test
println("testing transpose...")
@test_approx_eq TestTranspose.test() [1. 4.; 2. 5.; 3. 6.]
println("Done testing transpose.")

