.. _introduction:

*********
Introduction
*********

*ParallelAccelerator* is a Julia package for speeding up
compute-intensive Julia programs.  In particular, Julia code that
makes heavy use of high-level array operations is a good candidate for
speeding up with ParallelAccelerator.

With the ``@acc`` macro that ParallelAccelerator provides, users may
specify parts of a program to accelerate.  ParallelAccelerator
compiles these parts of the program to fast native code.  It
automatically eliminates overheads such as array bounds checking when
it is safe to do so.  It also parallelizes and vectorizes many
data-parallel operations.

ParallelAccelerator is part of the High Performance Scripting (HPS)
project at Intel Labs.

Quick install (requires Julia 0.5.x or 0.4.x)::

        Pkg.add("ParallelAccelerator") 

Read on for more detailed installation and usage information.
