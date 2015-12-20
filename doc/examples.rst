## Examples

The `examples/` subdirectory has a few example programs demonstrating
how to use ParallelAccelerator. You can run them at the command line.
For instance:

``` .bash
$ julia ~/.julia/v0.4/ParallelAccelerator/examples/laplace-3d/laplace-3d.jl
Run laplace-3d with size 300x300x300 for 100 iterations.
SELFPRIMED 18.663935711
SELFTIMED 1.527286803
checksum: 0.49989778
```

The `SELFTIMED` line in the printed output shows the running time,
while the `SELFPRIMED` line shows the time it takes to compile the
accelerated code and run it with a small "warm-up" input.

Pass the `--help` option to see usage information for each example:

``` .bash
$ julia ~/.julia/v0.4/ParallelAccelerator/examples/laplace-3d/laplace-3d.jl -- --help
laplace-3d.jl

Laplace 6-point 3D stencil.

Usage:
  laplace-3d.jl -h | --help
    laplace-3d.jl [--size=<size>] [--iterations=<iterations>]

    Options:
      -h --help                  Show this screen.
        --size=<size>              Specify a 3d array size (<size> x <size> x <size>); defaults to 300.
          --iterations=<iterations>  Specify a number of iterations; defaults to 100.
          ```

          You can also run the examples at the `julia>` prompt:

          ```
          julia> include("$(homedir())/.julia/v0.4/ParallelAccelerator/examples/laplace-3d/laplace-3d.jl")
          Run laplace-3d with size 300x300x300 for 100 iterations.
          SELFPRIMED 18.612651534
          SELFTIMED 1.355707121
          checksum: 0.49989778
          ```

Some of the examples require additional Julia packages.  The top-level
`REQUIRE` file in this repository lists all registered packages that
examples depend on.

