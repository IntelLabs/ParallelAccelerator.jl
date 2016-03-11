.. _examples:

*********
Examples
*********

The ``examples/`` subdirectory has a few example programs demonstrating
how to use ParallelAccelerator. You can run them at the command line.
For instance::

    $ julia ~/.julia/v0.4/ParallelAccelerator/examples/laplace-3d/laplace-3d.jl
    Run laplace-3d with size 300x300x300 for 100 iterations.
    SELFPRIMED 1.626313704
    SELFTIMED 2.427061371
    checksum: 0.49989453


The *SELFTIMED* line in the printed output shows the running time,
while the *SELFPRIMED* line shows the time it takes to compile the
accelerated code and run it with a small "warm-up" input.

For the first function you accelerate in a given Julia session, you
might notice a much longer time being reported for *SELFPRIMED*.  This
delay (about 20 seconds on an 8-core desktop machine) is the time it
takes for Julia to load the ParallelAccelerator package itself.  See
`Speeding up package load time via userimg.jl`_ for a workaround.

Pass the ``--help`` option to see usage information for each example::

    julia ~/.julia/v0.4/ParallelAccelerator/examples/laplace-3d/laplace-3d.jl -- --help laplace-3d.jl
    laplace-3d.jl

    Laplace 6-point 3D stencil.

    Usage:
      laplace-3d.jl -h | --help
      laplace-3d.jl [--size=<size>] [--iterations=<iterations>]

    Options:
      -h --help                  Show this screen.
      --size=<size>              Specify a 3d array size (<size> x <size> x <size>); defaults to 300.
      --iterations=<iterations>  Specify a number of iterations; defaults to 100.


You can also run the examples at the ``julia>`` prompt::

    julia> include("$(homedir())/.julia/v0.4/ParallelAccelerator/examples/laplace-3d/laplace-3d.jl")
    Run laplace-3d with size 300x300x300 for 100 iterations.
    SELFPRIMED 1.634722587
    SELFTIMED 2.404315047
    checksum: 0.49989453


Some of the examples require additional Julia packages.  The 
```REQUIRE`` file <https://github.com/IntelLabs/ParallelAccelerator.jl/blob/master/REQUIRE>`_ in our repository lists all registered packages that
examples depend on.

