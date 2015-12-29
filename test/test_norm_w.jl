using ParallelAccelerator

@acc function normalizeW(W::DenseMatrix{Float64}, k::Int)
# normalize each column
#=
    for j = 1:k
        w = view(W,:,j)
        scale!(w, 1.0 / sqrt(sumabs2(w)))
	#s = sqrt(sumabs2(W[:,j]))
	#W[:,j] /= s
    end
=#
    s = [sqrt(sumabs2(W[:,j])) for j in 1:k]
    scale!(W,s)
end

function main(m::Int, k::Int, n::Int)
    W  = Array(Float64, m, k)   
    fill!(W, 3)
    normalizeW(W, k)
    println("done")

end

main(100, 100, 200)