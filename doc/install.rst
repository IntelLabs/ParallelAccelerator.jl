.. _install:

*********
Installation
*********

At the ``julia>`` prompt, run these commands::

    Pkg.add("ParallelAccelerator")          # Install this package and its dependencies.

Since we add features and fix bugs very frequently, we
recommend switching to our `master` branch::

    Pkg.checkout("ParallelAccelerator")     # Switch to master branch 
    Pkg.checkout("CompilerTools")           # Switch to master branch 
    Pkg.build("ParallelAccelerator")        # Build the C runtime component of the package, and configure the package for your environment.
    Pkg.test("CompilerTools")               # Run CompilerTools tests.
    Pkg.test("ParallelAccelerator")         # Run ParallelAccelerator tests.
 
If all of the above succeeded, you should be ready to use
ParallelAccelerator.

It is a good idea to run ``Pkg.test("ParallelAccelerator")`` to make
sure it runs in your environment.  If you install a new C/C++ compiler
or BLAS library after installing ParallelAccelerator, be sure to run
``Pkg.build("ParallelAccelerator")`` and
``Pkg.test("ParallelAcclerator")`` again to pick up the changes and
confirm that they worked.



