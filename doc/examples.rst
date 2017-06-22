.. _examples:

*********
Examples
*********

After running ``Pkg.test("ParallelAccelerator")``, a good next step is
to try out some example programs from the ParallelAccelerator
``examples/`` subdirectory.  These can be run either from the command
line or from the Julia prompt.  For instance, to run the
``black-scholes`` example from the Julia prompt::

    julia> include("$(homedir())/.julia/v0.5/ParallelAccelerator/examples/black-scholes/black-scholes.jl")

On your first run, you should see output similar to the following
(this was on an 8-core Linux machine with 8 GB of RAM, using ICC and
MKL)::

    iterations = 10000000
    SELFPRIMED 31.675743032
    checksum: 2.0954821257116845e8
    rate = 2.268548657657791e7 opts/sec
    SELFTIMED 0.44081047

The *SELFTIMED* line in the printed output shows the running time,
while the *SELFPRIMED* line shows the time it takes to compile the
accelerated code and run it with a small "warm-up" input.

For the first function you accelerate in a given Julia session, you
might notice a long time being reported for *SELFPRIMED*.  This delay
(about 25 to 30 seconds on an 8-core desktop machine) is the time it
takes for Julia to load the ParallelAccelerator package itself.  If
you run the function again, you'll notice that ``SELFPRIMED`` gets
smaller.  Here's an example interaction:

    julia> include("$(homedir())/.julia/v0.5/ParallelAccelerator/examples/black-scholes/black-scholes.jl")
    iterations = 10000000
    SELFPRIMED 31.675743032
    checksum: 2.0954821257116845e8
    rate = 2.268548657657791e7 opts/sec
    SELFTIMED 0.44081047
    
    julia> include("$(homedir())/.julia/v0.5/ParallelAccelerator/examples/black-scholes/black-scholes.jl")
    iterations = 10000000
    SELFPRIMED 1.62395378
    checksum: 2.0954821257116845e8
    rate = 2.3933823208592944e7 opts/sec
    SELFTIMED 0.417818746
    
    julia>

Notice that ``SELFPRIMED`` dropped from almost 32 seconds to about 1.6
seconds. This is because, for the second run, the ParallelAccelerator
package was already loaded. The remaining 1.6 seconds is mostly the
time it took for the ParallelAccelerator compiler to compile the
accelerated code.

It's instructive to compare the running time of the
ParallelAccelerator code with the plain Julia version:

    julia> include("$(homedir())/.julia/v0.5/ParallelAccelerator/examples/plain-julia/black-scholes/black-scholes.jl")
    iterations = 10000000
    SELFPRIMED 0.000573466
    checksum: 2.0954821257116845e8
    rate = 3.640941015492507e6 opts/sec
    SELFTIMED 2.746542709

This time, ``SELFPRIMED`` is almost 0 (because there is no package to
be loaded and no work for the ParallelAccelerator compiler to do), but
``SELFTIMED`` is rather larger than it was under ParallelAccelerator::
2.7 seconds.

You can also run example programs from the command line::

    $ julia ~/.julia/v0.5/ParallelAccelerator/examples/black-scholes/black-scholes.jl

Pass the ``--help`` option to see usage information for each example::

    $ julia ~/.julia/v0.5/ParallelAccelerator/examples/black-scholes/black-scholes.jl -- --help
    black-scholes.jl

    Black-Scholes option pricing model.

    Usage:
      black-scholes.jl -h | --help
      black-scholes.jl [--iterations=<iterations>]

    Options:
      -h --help                  Show this screen.
      --iterations=<iterations>  Specify a number of iterations [default: 10000000].

The ``examples/`` subdirectory contains many more examples.  Most
examples also have a corresponding plain Julia version available in
``examples/plain-julia/``.  Some examples require additional Julia
packages to be installed.
