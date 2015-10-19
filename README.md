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
  * You will need a C/C++ compiler: either `gcc`/`g++` or
    `icc`/`icpc`.  We recommend GCC 4.8.4 or later and ICC 15.0.3 or
    later.  At package build time, ParallelAccelerator will check to
    see if you have ICC installed.  If so, ParallelAccelerator will
    use it.  Otherwise, it will use GCC.

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

You can also use `@acc begin ... end`, and put multiple functions in the block
to have all of them accelerated. The `@acc` macro only works for top-level 
definitions.

## How It Works

ParallelAccelerator is essentially a domain-specific compiler written in Julia
that discovers and exploits the implicit parallelism in source programs that 
use parallel programming patterns such as *map, reduction, comprehension, and
stencil*. For example, Julia array operators such as `.+, .-, .*, ./` are 
translated by ParallelAccelerator internally into a *map* operation over all
elements of input arrays.  For the most part, these patterns are already 
present in standard Julia, so programmers can use ParallelAccelerator to 
run the same Julia program without (significantly) modifying its source code. 

The `@acc` macro provided by ParallelAccelerator first intercepts Julia
functions at macro level, and performs a set of substitutions to capture the
set of implicitly parallel operations that we are targeting, and point them to
those supplied in the `ParallelAccelerator.API` module. It then creates a proxy
function that when called with concrete arguments (and their types) will try to
compile the original function to an optimized form. So the first time
calling an accelerated function would incur some compilation time, but all
subsequent calls to the same function will not.

ParallelAccelerator performs aggressive optimizations when it is safe to do so.
For example, it automatically infers equivalence relation among array
variables, and will fuse adjacent parallel loops into a single loop. Eventually
all parallel patterns are lowered into explicit parallel `for` loops internally
represented as part of Julia's typed AST. 

Finally, these parallel loops are then translated into a C program with OpenMP
pragmas, and ParallelAccelerator will use an external C/C++ compiler to compile
it into binary form before loading it back into Julia as a dynamic library for
execution. The translation to C currently imposes certain constraints (see
details below), and as a consequence we can only run user programs that meet such
constraints. 

## Advanced Usage

As mentioned above, ParallelAccelerator aims to optimize implicitly parallel
Julia programs that are safe to parallelize. It also tries to be non-invasive, 
which means a user function or program should continue with as expected
even when only a part of it is accelerated. It is still important to know what
exactly are accelerated and what are not, however, and we encourage user to
write program using high-level array operations that are amenable to domain
specific analysis and optimizations, rather than explicit for-loops with
unrestricted mutations or unknown side-effects. 

### Array Operations

### Stencil

### Parallel Comprehension 

### Faster compilation via userimg.jl

It is possible to embed a binary/compiled version of the ParallelAccelerator compiler and CompilerTools
into a Julia executable.  This has the potential to greatly reduce the time it takes for our compiler
to accelerate a given program.  To use this feature, start the Julia REPL and do the following:

importall ParallelAccelerator
ParallelAccelerator.embed()

This version of embed() tries to embed ParallelAccelerator into the Julia version used by the current REPL.

If you want to target a different Julia distribution, you can alternatively use the following
version of embed.

ParallelAccelerator.embed("<your/path/to/the/root/of/the/julia/distribution>")

This "embed" function takes a path and is expected to point to the root directory of a Julia source
distribution.  embed performs some simple checks to try to verify this fact.  Then, embed will try to
create a file base/userimg.jl that will tell the Julia build how to embed the compiled version into
the Julia executable.  Then, embed runs make in the Julia root directory to create the embedded
Julia version.

If there is already a userimg.jl file in the base directory then a new file is created called
ParallelAccelerator_userimg.jl and it then becomes the user's responsibility to merge that with the
existing userimg.jl and run make if they want this faster compile time.

After the call to embed finishes and you try to exit the Julia REPL, you may receive an exception
like: ErrorException("ccall: could not find function git_libgit2_shutdown in library libgit2").
This error does not effect the embedding process but we are working towards a solution to this
minor issue.

## Limitations 

...what is known not to work... etc.

## Comments, Suggestions, and Bug Reports



