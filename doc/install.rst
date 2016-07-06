.. _install:

*********
Installation
*********
General
*********

At the ``julia>`` prompt, run these commands::

    Pkg.add("ParallelAccelerator")          # Install this package and its dependencies.

Since we add features and fix bugs very frequently, we
recommend switching to our `master` branch::

    Pkg.checkout("ParallelAccelerator")     # Switch to master branch 
    Pkg.checkout("CompilerTools")           # Switch to master branch 
    Pkg.build("ParallelAccelerator")        # Build the C++ runtime component of the package.
    Pkg.test("CompilerTools")               # Run CompilerTools tests.
    Pkg.test("ParallelAccelerator")         # Run ParallelAccelerator tests.
 
If all of the above succeeded, you should be ready to use
ParallelAccelerator.

Windows specific installation notes
*********
Under Microsoft Windows, follow the same installation procedure as outlined above, but note that ``Pkg.add("ParallelAccelerator")`` and ``Pkg.build("ParallelAccelerator")`` will throw errors because Windows cannot process the bash script used for building the runtime component. 

Solution:
    * Download and install MSYS2 from https://msys2.github.io/.
    * Open an MSYS2 console and do::
    
        pacman -S gcc
        cd /c/path/to/julia/ParallelAccelerator/deps/
        ./build.sh
    
Pkg.test("ParallelAccelerator") should now succeed, but will issue a warning that it cannot find a BLAS installation. 
