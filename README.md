# ParallelAccelerator

[![Build Status](https://magnum.travis-ci.com/IntelLabs/ParallelAccelerator.jl.svg?token=149Z9PxxcSTNz1n9bRpz&branch=master)](https://magnum.travis-ci.com/IntelLabs/ParallelAccelerator.jl)

This is the ParallelAccelerator Julia package, part of the High
Performance Scripting project at Intel Labs.

## Prerequisites

  * Install Julia v0.4.0.  Go to http://julialang.org/downloads/ and
    download the appropriate version listed under "Current Release".
    Then check that you can run Julia and get to a `julia>` prompt.
    You will know you're running the correct version if when you run
    it, you see `Version 0.4.0`.
  * You will need a C++ compiler: either `gcc` or `icpc`.

## Installation

At the `julia>` prompt, run these commands:

``` .julia
Pkg.clone("https://github.com/IntelLabs/CompilerTools.jl.git")        # Install the CompilerTools package on which this package depends.
Pkg.clone("https://github.com/IntelLabs/ParallelAccelerator.jl.git")  # Install this package.
Pkg.build("ParallelAccelerator")                                      # Build the C++ runtime component of the package.
Pkg.test("CompilerTools")                                             # Run CompilerTools tests.
Pkg.test("ParallelAccelerator")                                       # Run ParallelAccelerator tests.
```

For the two `Pkg.clone` commands, you will be prompted for your GitHub
username and password.
 
If all of the above succeeded, you should be ready to use
`ParallelAccelerator`.  We're in the process of documenting how to use
this package.  For now, a good place to look for examples is the code
in the
[ParallelAcceleratorBenchmarks](https://github.com/IntelLabs/ParallelAcceleratorBenchmarks)
repo.  For example, the `black-scholes/src` directory in that repo
contains the file `blackscholes-pse.jl`, which you can include at the
Julia REPL:

```
julia> include("blackscholes-pse.jl")
```

Each directory in `ParallelAcceleratorBenchmarks` has a README with
some more information about each workload.  Caveat: some of the
workloads require installing additional Julia packages.
