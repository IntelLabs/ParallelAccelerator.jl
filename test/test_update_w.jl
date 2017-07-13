using ParallelAccelerator

@acc function opt_update_w( k, m, W, Y, E1 )
	 for j = 1:k
            #w = view(W,:,j)
            #y = view(Y,:,j)
            e1 = E1[j]
            for i = 1:m
                #w[i] = y[i] - e1 * w[i]
		W[i,j] = Y[i,j] - e1 * W[i,j]
            end
        end
end

function main(m::Int, k::Int)
    Y  = Array{Float64}(m, k)    # to store E{x g(w'x)} for components
    E1 = Array{Float64}(k)       # store E{g'(w'x)} for components
    W  = Array{Float64}(m, k)    # to store E{x g(w'x)} for components

    fill!(W, 3)
    fill!(Y, 9)
    fill!(E1, 2)
    opt_update_w(k,m,W,Y,E1)

    return W

end

main(100, 100)
println("Done testing test_update_w.")
