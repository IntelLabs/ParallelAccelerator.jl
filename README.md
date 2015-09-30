# ParallelAccelerator

[![Build Status](https://magnum.travis-ci.com/IntelLabs/ParallelAccelerator.jl.svg?token=149Z9PxxcSTNz1n9bRpz&branch=master)](https://magnum.travis-ci.com/IntelLabs/ParallelAccelerator.jl)

This is the ParallelAccelerator Julia package, part of the High
Performance Scripting project at Intel Labs.

## Prerequisites

  * Install a *nightly build* of Julia.  See "Nightly builds" at the
    bottom of http://julialang.org/downloads/ .  The most recently
    released version of Julia (0.3.11) is not new enough to support
    all of ParallelAccelerator's features.
  * Install the CompilerTools package, following the instructions
    [here](https://github.com/IntelLabs/CompilerTools.jl#compilertools).
  * You will need a C++ compiler: either `gcc` or `icpc`.

## Installation

Once you have completed the above steps, run Julia and then run the
command

    Pkg.clone("https://github.com/IntelLabs/ParallelAccelerator.jl.git")

at the `julia>` prompt.  You will be prompted for your GitHub username
and password.

Next, run:

    Pkg.build("ParallelAccelerator")

This will build the C++ runtime component of the package.

## Running

Once the packages are installed, you can try out examples from the
[ParallelAcceleratorBenchmarks](https://github.com/IntelLabs/ParallelAcceleratorBenchmarks)
repo.  See the README files for each benchmark for instructions on how
to run.

<!-- Installation on Mac OS X: -->

<!-- install Intel Compiler -->
<!-- install bcpp with Homebrew -->

