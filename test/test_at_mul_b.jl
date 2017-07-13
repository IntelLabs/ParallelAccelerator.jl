using ParallelAccelerator

#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)

@acc function opt_At_mul_B!(U, X, W)
	 #At_mul_B!(U, X, W)  # u <- w'x
	 U = X' * W
end

function main(m::Int, k::Int, n::Int)
    U = Array{Float64}(n, k)
    W = Array{Float64}(m, k)   
    X = Array{Float64}(m, n)
    fill!(W, 3)
    fill!(X, 5)
    opt_At_mul_B!(U, X, W)
    println("done")

end

main(100, 100, 200)
