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
ParallelAccelerator.  We're in the process of documenting how to use this
package.  For now, you can look at a few programs included in the `examples/`
sub-directoy. You can run them either at commandline or in Julia REPL:

```
julia> include("black-scholes.jl")
```

Each directory in `examples/` has a README with some more information about
each workload.  Caveat: some of the workloads require installing additional
Julia packages.

## Basic Usage

To start using ParallelAccelerator in you own program, first import the 
package by `using ParallelAccelerator`, and then put `@acc` macro before
the function you want to accelerate. A trivial example is given below:

``` .julia
julia> using ParallelAccelerator

julia> @acc f(x) = x .+ x .* x
f (generic function with 1 method)

julia> f([1,2,3,4,5])
5-element Array{Int64,1}:
  2
  6
 12
 20
 30
```

You can also use `@acc begin ... end`, and put more than one functions in the block
to have all of them accelerated. The `@acc` macro only works for top-level definitions.

## How It Works

## Advanced Usage

### Array Operations

### Stencil

### Parallel Comprehension 

### Faster compilation via user.img

## Limitations 

...what is known not to work... etc.

### How to file 

