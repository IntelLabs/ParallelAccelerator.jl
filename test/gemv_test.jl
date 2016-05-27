module TestGemv
using ParallelAccelerator

@acc gemv_t(A,y) = A*y
@acc gemv_t2(A,y) = A'*y

function test()
    A = [1. 2. 3.; 4. 5. 6.]
    y = [1.,2.,3.]
    y2 = [1.,2.]
    z = gemv_t(A,y)
    return z
end

function test2()
    A = [1. 2. 3.; 4. 5. 6.]
    y = [1.,2.]
    z = gemv_t2(A,y)
    return z
end

end

using Base.Test
println("testing gemv...")
@test_approx_eq TestGemv.test() [14.0,32.0]
@test_approx_eq TestGemv.test2() [9.0,12.0,15.0]
println("Done testing gemv.")



