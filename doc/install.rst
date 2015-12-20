.. _install

## Installation

At the `julia>` prompt, run these commands:

``` .julia
Pkg.add("ParallelAccelerator")          # Install this package and its dependencies.
```

Since we add features and fix bugs very frequently, we
recommend switching to our `master` branch.

``` .julia
Pkg.checkout("ParallelAccelerator")     # Switch to master branch 
Pkg.checkout("CompilerTools")           # Switch to master branch 
Pkg.build("ParallelAccelerator")        # Build the C++ runtime component of the package.
Pkg.test("CompilerTools")               # Run CompilerTools tests.
Pkg.test("ParallelAccelerator")         # Run ParallelAccelerator tests.
```
 
If all of the above succeeded, you should be ready to use
ParallelAccelerator.

