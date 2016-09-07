module TestTranspose
using ParallelAccelerator

@acc transpose_t(A) = A'

function test()
    A = [1. 2. 3.; 4. 5. 6.]
    B = transpose_t(A)
    return B
end

@acc function test2()
    A = [1 2 3; 4 5 6]
    B = transpose(A)
    return B
end


end

using Base.Test
println("testing transpose...")
@test_approx_eq TestTranspose.test() [1. 4.; 2. 5.; 3. 6.]
@test TestTranspose.test2()==[1 4; 2 5; 3 6]
println("Done testing transpose.")

