# ParallelAccelerator

[![Build Status](https://travis-ci.org/IntelLabs/ParallelAccelerator.jl.svg?branch=master)](https://travis-ci.org/IntelLabs/ParallelAccelerator.jl)

This is the ParallelAccelerator Julia package, part of the High
Performance Scripting project at Intel Labs. 

## Prerequisites

  * **Julia v0.4.0.** If you don't have it yet, there are various ways
    to install Julia:
      * Go to http://julialang.org/downloads/ and download the
        appropriate version listed under "Current Release".
      * On Ubuntu or Debian variants, `sudo add-apt-repository
        ppa:staticfloat/juliareleases -y && sudo apt-get update -q &&
        sudo apt-get install julia -y` should work.
      * On OS X with Homebrew, `brew update && brew tap
        staticfloat/julia && brew install julia` should work.
    Check that you can run Julia and get to a `julia>` prompt.  You
    will know you're running the correct version if when you run it,
    you see `Version 0.4.0`.
  * **Either `gcc`/`g++` or `icc`/`icpc`.** We recommend GCC 4.8.4 or
    later and ICC 15.0.3 or later.  At package build time,
    ParallelAccelerator will check to see if you have ICC installed.
    If so, ParallelAccelerator will use it.  Otherwise, it will use
    GCC. Clang with GCC front-end (default on Mac OS X) also works.
  * Platforms we have tested on so far include Ubuntu 14.04, CentOS
    6.6, and Mac OS X Yosemite with both GCC and ICC.

## Installation

At the `julia>` prompt, run these commands:

``` .julia
Pkg.clone("https://github.com/IntelLabs/CompilerTools.jl.git")        # Install the CompilerTools package on which this package depends.
Pkg.clone("https://github.com/IntelLabs/ParallelAccelerator.jl.git")  # Install this package.
Pkg.build("ParallelAccelerator")                                      # Build the C++ runtime component of the package.
Pkg.test("CompilerTools")                                             # Run CompilerTools tests.
Pkg.test("ParallelAccelerator")                                       # Run ParallelAccelerator tests.
```
 
If all of the above succeeded, you should be ready to use
ParallelAccelerator.

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

ParallelAccelerator is essentially a domain-specific compiler written in Julia.
It performs additional analysis and optimization on top of the Julia compiler.
ParallelAccelerator discovers and exploits the implicit parallelism in source programs that
use parallel programming patterns such as *map, reduce, comprehension, and
stencil*. For example, Julia array operators such as `.+, .-, .*, ./` are
translated by ParallelAccelerator internally into data-parallel  *map* operations over all
elements of input arrays.  For the most part, these patterns are already
present in standard Julia, so programmers can use ParallelAccelerator to run
the same Julia program without (significantly) modifying the source code. 

The `@acc` macro provided by ParallelAccelerator first intercepts Julia
functions at macro level and substitutes the set of implicitly parallel
operations that we are targeting. `@acc` points them to those supplied in the
`ParallelAccelerator.API` module. It then creates a proxy function that when
called with concrete arguments (and known types) will try to compile the
original function to an optimized form. Therefore, there is some compilation
time the first time an accelerated function is called. The subsequent
calls to the same function will not have compilation time overhead.

ParallelAccelerator performs aggressive optimizations when they are safe depending on the program structure.
For example, it will automatically infer size equivalence relations among array
variables and skip array bounds check whenever it can safely do so.   Eventually all
parallel patterns are lowered into explicit parallel `for` loops which are internally
represented at the level of Julia's typed AST. Aggressive loop fusion will
try to combine adjacent loops into one and eliminate temporary array objects
that store intermediate results.

Finally, functions with parallel for loops are translated into a C program with
OpenMP pragmas, and ParallelAccelerator will use an external C/C++ compiler to
compile it into binary form before loading it back into Julia as a dynamic
library for execution. This step of translating Julia to C currently imposes
certain constraints (see details below), and therefore we can only run user
programs that meet such constraints. 

## Advanced Usage

As mentioned above, ParallelAccelerator aims to optimize implicitly parallel
Julia programs that are safe to parallelize. It also tries to be non-invasive,
which means a user function or program should continue to work as expected even
when only a part of it is accelerated. It is still important to know what
parts are accelerated, however. As a general guideline,
we encourage users to write program using high-level array operations rather
than writing explicit for-loops which can have unrestricted mutations or unknown
side-effects. High-level operations are more amenable to analysis and
optimization provided by ParallelAccelerator.

To help user verify program correctness, the optimizations of ParallelAccelerator
can be turned off by setting environment variable `PROSPECT_MODE=none` before
running the julia program. Doing so will still trigger certain useful macro 
translations (such as `runStencil`, see below), but no optimizations or
Julia-to-C translation will take place. Users can also use `@noacc`
at the function call site to use the original version of the function.


### Map and Reduce

Array operations that work uniformly on all elements of input arrays and
produce an output array of equal size are called `point-wise` operations.
`Point-wise` binary operations in Julia usually have a `.` prefix in the
operator name. These operations are translated internally into data-parallel *map* operations by
ParallelAccelerator. The following are recognized by `@acc` as *map*
operations:

* Unary functions: `-, +, acos, acosh, angle, asin, asinh, atan, atanh, cbrt,
cis, cos, cosh, exp10, exp2, exp, expm1, lgamma, log10, log1p, log2, log,
sin, sinh, sqrt, tan, tanh, abs, copy, erf`

* Binary functions: `-, +, .+, .-, .*, ./, .\, .%, .>, .<, .<=, .>=, .==, .<<,
.>>, .^, div, mod, rem, &, |, $`

Array assignments are also recognized and converted into *in-place map*
operations.  Expressions like `a = a .+ b` will be turned into an *in-place map*
that takes two inputs arrays, `a` and `b`, and updates `a` in-place. 

Array operations that compute a single result by repeating an associative
and commutative operator on all input array elements are called *reduce* operations.
The following are recognized by `@acc` as `reduce` operations: 

```
minimum, maximum, sum, prod, any, all
```

We also support range operations to a limited extent. For example, `a[r] =
b[r]` where `r` is either a `BitArray` or `UnitRange` (e.g., `1:s`) is
internally converted to parallel operations when the ranges can be inferred
statically to be *compatible*. However, such support is still
experimental, and occasionally ParallelAccelerator will complain about not
being able to optimize them. We are working on improving this feature
to provide more coverage and better error messages.

### Parallel Comprehension 

Array comprehensions in Julia are in general also parallelizable, because 
unlike general loops, their iteration variables have no inter-dependencies. 
So the `@acc` macro will turn them into an internal form that we call
`cartesianarray`:

```
A = Type[ f (x1, x2, ...) for x1 in r1, x2 in r2, ... ]
```
becomes
```
cartesianarray((i1,i2,...) -> begin x1 = r1[i1]; x2 = r2[i2]; f(x1,x2,...) end,
             Type,
             (length(r1), length(r2), ...))
```

This `cartesianarray` function is also exported by `ParallelAccelerator` and
can be directly used by the user. Both the above two forms are acceptable
programs, and equivalent in semantics, they both produce a N-dimensional array
whose element is of `Type`, where `N` is the number of `x`s and `r`s, and
currently only up-to-3 dimensions are supported.

It should be noted, however, not all comprehensions are safe to parallelize.
For example, if the function `f` above reads and writes to a variable outside of comprehension, 
then making it run in parallel can produce non-deterministic
result. Therefore, it is the responsibility of the user to avoid using `@acc` such situations arise.

Another difference between parallel comprehension and the afore-mentioned *map*
operation is that array indexing operations in the body of a parallel
comprehension remain explicit and therefore should go through
necessary bounds-checking to ensure safety. On the other hand, in all *map* operations such
bounds-checking is skipped.

### Stencil

Stencils are commonly found in scientific computing and image processing. A stencil
computation is one that computes new values for all elements of an array based
on the current values of their neighboring elements. Since Julia's base library
does not provide such an API, ParallelAccelerator exports a general
`runStencil` interface to help with stencil programming:

```
runStencil(kernel :: Function, buffer1, buffer2, ..., 
           iteration :: Int, boundaryHandling :: Symbol)
```

As an example, the following (taken from Gaussian Blur example) computes a
5x5 stencil computation (note the use of Julia's `do` syntax that lets
user write a lambda function):

```
runStencil(buf, img, iterations, :oob_skip) do b, a
       b[0,0] =
            (a[-2,-2] * 0.003  + a[-1,-2] * 0.0133 + a[0,-2] * 0.0219 + a[1,-2] * 0.0133 + a[2,-2] * 0.0030 +
             a[-2,-1] * 0.0133 + a[-1,-1] * 0.0596 + a[0,-1] * 0.0983 + a[1,-1] * 0.0596 + a[2,-1] * 0.0133 +
             a[-2, 0] * 0.0219 + a[-1, 0] * 0.0983 + a[0, 0] * 0.1621 + a[1, 0] * 0.0983 + a[2, 0] * 0.0219 +
             a[-2, 1] * 0.0133 + a[-1, 1] * 0.0596 + a[0, 1] * 0.0983 + a[1, 1] * 0.0596 + a[2, 1] * 0.0133 +
             a[-2, 2] * 0.003  + a[-1, 2] * 0.0133 + a[0, 2] * 0.0219 + a[1, 2] * 0.0133 + a[2, 2] * 0.0030)
       return a, b
    end
```

It take two input arrays, `buf` and `img`, and performs an iterative stencil
loop (ISL) of given `iterations`. The stencil kernel is specified by a lambda
function that takes two arrays `a` and `b` (that correspond to `buf` and
`img`), and computes the value of the output buffer using relative indices
as if a cursor is traversing all array elements. `[0,0]` represents
the current cursor position. The `return` statement in this lambda reverses
the position of `a` and `b` to specify a buffer rotation that should happen
in-between the stencil iterations. The use of `runStencil` will assume
all input and output buffers are of the same dimension and size.

Stencil boundary handling can be specified as one of the following symbols:

* `:oob_skip`. Writing to output is skipped when input indexing is out-of-bound.
* `:oob_wraparound`. Indexing is `wrapped-around` at the array boundaries so they are always safe.
* `:oob_dst_zero`. Writing 0 to output array when any of the input indexing is out-of-bound.
* `:oob_src_zero`. Assume 0 is returned by a read operation when indexing is out-of-bound.

Just like parallel comprehension, accessing the variables outside is allowed
in a stencil body. However, accessing outside array values is
not supported, and reading/writing the same outside variable can cause
non-determinism. 

All arrays that need to be relatively indexed can be specified as
input buffers. `runStencil` does not impose any implicit  buffer rotation
order and the user can choose not to rotate buffers in `return`. There 
can be multiple output buffers as well. Finally, the call to `runStencil` does 
not have any return value, and inputs are rotated for `iteration - 1` times if rotation is specified.

ParallelAccelerator exports a naive Julia implementation of `runStencil` that
runs without using `@acc`. Its purpose is mostly for correctness checking.
When `@acc` is being used with environment variable `PROSPECT_MODE=none`,
instead of parallelizing the stencil computation  `@acc` will expand the call
to `runStencil` to a fast sequential implementation.

### Faster compilation via userimg.jl

It is possible to embed a binary/compiled version of the ParallelAccelerator
compiler and CompilerTools into a Julia executable. This has the potential to
greatly reduce the time it takes for our compiler to accelerate a given
program. For this feature, the user needs to have the Julia source code 
and should be able to rebuild Julia. Hence,
Julia installations using ready binaries are not suitable for this purpose.

To use this feature, start the Julia REPL and do the following:

```
importall ParallelAccelerator
ParallelAccelerator.embed()
```

This version of `embed()` tries to embed ParallelAccelerator into the Julia
version used by the current REPL.

If you want to target a different Julia distribution, you can alternatively use
the following version of embed.

```
ParallelAccelerator.embed("<your/path/to/the/root/of/the/julia/distribution>")
```

This `embed` function takes a path which is expected to point to the root
directory of a Julia source distribution.  `embed` performs some simple checks to
try to verify this fact.  Then, embed will try to create a file
`base/userimg.jl` that will tell the Julia build how to embed the compiled
version into the Julia executable.  Finally, `embed` runs `make` in the Julia root
directory to create the embedded Julia version.

If there is already a `userimg.jl` file in the base directory then a new file is
created called `ParallelAccelerator_userimg.jl` and it then becomes the user's
responsibility to merge that with the existing `userimg.jl` and run `make`.

After the call to `embed` finishes, when trying to exit the Julia REPL, you may
receive an exception like: ErrorException(`"ccall: could not find function
git_libgit2_shutdown in library libgit2"`).  This error does not effect the
embedding process but we are working towards a solution to this minor issue.

## Limitations 

One of the most notable limitation is that ParallelAccelerator currently
tries to compile Julia to C, which puts some constraints on what
can be successfully compiled and run:

1. We only support a limited subset of Julia language features currently.
   This includes basic numbers and dense array types, a subset of math 
   functions, and basic control flow structures. Notably, we do not support 
   String types and custom data types such as records and unions, since their 
   translation to C is difficult. There is also no support for exceptions, 
   I/O operations (only very limited `println`), and arbitrary ccalls.

2. We do not support calling Julia functions from C in the optimized
   function. What this implies is that we transitively convert 
   every Julia function in the call chain to C. If any of them is not 
   translated properly, the target function with `@acc` will fail to compile. 

3. We do not support Julia's `Any` type in C, mostly to
   defend against erroneous translation. If the AST of a Julia function
   contains a variable with `Any` type, our Julia-to-C translator will give up
   compiling the function. This is indeed more limiting than it sounds, because
   Julia does not annotate all expressions in a typed AST with complete type 
   information. For example, this happens for some expressions that call Julia's 
   own intrinsics. We are working on supporting more of them if we can derive 
   the actual type to be not `Any`, but this is still a work-in-progress.

At the moment ParallelAccelerator only supports the Julia-to-C back-end. We
are working on alternatives such as Julia's upcoming threading implementation 
that hopefully can alleviate the above mentioned
restrictions without sacrificing much of the speed brought by quality C
compilers and parallel runtime such as OpenMP.  

Apart from the constraints imposed by Julia-to-C translation, our current 
implementation of ParallelAccelerator has some other limitations:

1. We currently support a limited subset of Julia functions available in the `Base` library.
   However, not all Julia functions in the Base library
   are supported yet and using them may or may not work in ParallelAccelerator.
   For supported functions, we rely on capturing operator names to resolve array related functions and operators
   to our API module. This prevents them from being inlined by Julia
   which helps our translation. For unsupported functions such as `mean(x)`,
   Julia's typed AST for the program
   that contains `mean(x)` becomes a lowered call that is basically the
   the low-level sequential implementation which cannot be
   handled by ParallelAccelerator. Of course, adding support
   for functions like `mean` is not a huge effort, and we are still in 
   the process of expanding the coverage of supported APIs.

2.  ParallelAccelerator relies heavily on full type information being available
    in Julia's typed AST in order to work properly. Although we do not require
    user functions to be explicitly typed, it is in general a good practice to
    ensure the function that is being accelerated can pass Julia's type inference
    without leaving any parameters or internal variables with an `Any` type. 
    There is currently no facility to help users understand whether something
    is being optimized or silently rejected. We plan to provide 
    better report on what is going on under the hood.

## Comments, Suggestions, and Bug Reports

Performance tuning is a hard problem, especially in 
high performance scientific programs. ParallelAccelerator
is but a first step of bringing reasonable parallel performance to a
productivity language by utilizing both domain specific and general compiler
optimization techniques without much effort from the user. However,
eventually there will be bottlenecks and not all optimizations work in
favor of each other. ParallelAccelerator is still a proof-of-concept
at this stage and we hope to hear from all users. We welcome bug reports and code contributions. 
Please feel free to use our system, fork the project, contact us by email, and
file bug reports on the issue tracker.


