.. _prerequisites:

*********
 Prerequisites
*********

The bare minimum you need to use ParallelAccelerator is:

  * A \*nix OS, ideally Linux.
  * A recent installation of Julia.
  * A C/C++ compiler, ideally `ICC <https://software.intel.com/en-us/intel-parallel-studio-xe/try-buy>`_.  For best results, the compiler should support OpenMP.  The package build process will check for the presence of OpenMP support.

To make ParallelAccelerator run to its full potential, you'll also need:

  * A `BLAS <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_ library, ideally `Intel MKL <https://software.intel.com/en-us/mkl>`_.  The package build process will check if a BLAS library is available.

More details about each of these dependencies are below.

Operating System
----------------

Platforms we have tested on include Ubuntu 16.04, Ubuntu 14.04, CentOS 6.6, and macOS Sierra, and macOS Yosemite.  Ubuntu is the platform on which ParallelAccelerator has been most thoroughly used and tested, and is likely to be the best supported.

Julia Version
-------------

ParallelAccelerator has been tested with various v0.5.x and v0.4.x versions of Julia.  It will not work with versions of Julia prior to v.0.4.0.

C/C++ Compiler
--------------

ParallelAccelerator depends on an external C/C++ compiler.  This dependency is necessary because ParallelAcclerator works by compiling Julia to C++ using its "CGen" backend.  We have tested ParallelAccelerator with various versions of ICC and GCC.

At package build time, ParallelAccelerator will check to see if ICC is available, and if so, it will default to using ICC; otherwise, it will use GCC.  If neither is installed, the build will fail.  On macOS, the default ``gcc`` (which actually uses Clang as a back-end) will work; however, it does not support OpenMP, so macOS users may prefer to install a standard GCC.

If the C/C++ compiler you use doesn't support OpenMP, ParallelAccelerator will print ``OpenMP is not used.`` messages at runtime.

BLAS Library
------------

If you do not have a BLAS library available, then as ParallelAccelerator runs, you may see the following message:

    WARNING: MKL and OpenBLAS not found. Matrix multiplication might be slow.
         Please install MKL or OpenBLAS and rebuild ParallelAccelerator for better performance.

For best results, we recommend installing `the Intel Math Kernel Library (MKL) <https://software.intel.com/en-us/mkl>`_.  As an alternative, we provide the below instructions for installing and using `OpenBLAS <http://www.openblas.net/>`_ on macOS and Ubuntu.  However, ParallelAccelerator uses a few features that are supported in MKL but not in OpenBLAS, as noted in `issue #147 <https://github.com/IntelLabs/ParallelAccelerator.jl/issues/147>`_.

  * To install OpenBLAS on Ubuntu and use it with ParallelAccelerator:

    - Run ``sudo apt-get install libopenblas-dev``, which should also install the necessary dependencies.
    - Remember to run ``Pkg.build("ParallelAccelerator")`` at the Julia prompt to rebuild ParallelAccelerator.  The build should now detect that a BLAS library is installed.
  * To install OpenBLAS on macOS (using Homebrew) and use it with ParallelAccelerator:

    - Run ``brew install homebrew/science/openblas``.  This should install OpenBLAS in ``/usr/local/opt/openblas``.
    - Set the following environment variables:

      + ``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/opt/openblas/lib``
      + ``export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/opt/openblas/include/``
    - Remember to run ``Pkg.build("ParallelAccelerator")`` at the Julia prompt to rebuild ParallelAccelerator.  The build should now detect that a BLAS library is installed.
