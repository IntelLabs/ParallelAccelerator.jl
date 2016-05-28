module TestVecnorm
using ParallelAccelerator

@acc norm_t(y) = norm(y,1)
@acc norm_t2(y) = norm(y,2)

function test()
    y = [1.,2.,3.]
    z = norm_t(y)
    return z
end

function test2()
    y = [1.,2.,3.]
    z = norm_t2(y)
    return z
end

end

using Base.Test
println("testing vecnorm...")
@test_approx_eq TestVecnorm.test() 6.0
@test_approx_eq TestVecnorm.test2() 3.7416573867739413 
println("Done testing vecnorm.")



