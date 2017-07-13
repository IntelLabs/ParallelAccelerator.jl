using ParallelAccelerator

immutable newType
    w::Vector{Float64}
    b::Float64
end
@acc function evaluate(f::newType, X::AbstractMatrix)
    if f.b != 0
	X = X + f.b
    end
    return X
end

function main(m::Int, n::Int)
    X = Array{Float64}(m, n)
    W = Array{Float64}(m)
    fill!(X, 5)
    fill!(W, 3)
    b = 1.0
    f = newType(W,b)
    evaluate(f, X)
    println("done")

end

main(100, 200)
