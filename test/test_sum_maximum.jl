using ParallelAccelerator

#ParallelAccelerator.ParallelIR.set_debug_level(3)

@acc function find_chg(k,m,W,Wp)	 
	W_tmp = [abs(W[i,j] - Wp[i,j]) for i in 1:m, j in 1:k]
	s = [sum(W_tmp[:,j]) for j in 1:k]
	chg = maximum(s)
end

function main(m::Int, k::Int)
    W  = Array{Float64}(m, k)   
    Wp = Array{Float64}(m, k)
    fill!(W, 3)
    fill!(Wp, 5)
    chg = find_chg(k,m,W,Wp)

    println("chg = ", chg)

end

main(100, 100)
