# A toy example demonstrating how to use ParallelAccelerator.accelerate().

module Example1

using ParallelAccelerator

ParallelAccelerator.set_debug_level(3)

function example(n::Int)
    for i = 1 : 10000
        n = n + 1
    end

    return n
end

tic()
example_pa = ParallelAccelerator.accelerate(example, (Int,))
time = toq()
println("ParallelAccelerator.accelerate time: ", time)

function main()

    tic()
    ret = example_pa(0)
    time = toq()
    println("SELFTIMED ", time)

    return ret

end

export main

end
