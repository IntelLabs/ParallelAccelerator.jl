# A toy example demonstrating how to use ParallelAccelerator.offload().

module Example1

using ParallelAccelerator

function example(n::Int)
    for i = 1 : 10000
        n = n + 1
    end

    return n
end

tic()
example_pa = ParallelAccelerator.offload(example, (Int,))
time = toq()
println("ParallelAccelerator.offload time: ", time)

function main()

    tic()
    ret = example_pa(0)
    time = toq()
    println("SELFTIMED ", time)

    return ret

end

export main

end
