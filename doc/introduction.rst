.. _introduction:

*********
Introduction
*********


ParallelAccelerator package is a compiler framework that aggressively 
optimizes compute-intensive Julia programs on top of the Julia compiler.
It automatically eliminates overheads such as array bounds checking when
it is safe to eliminate them. It also parallelizes and vectorizes many 
data-parallel operations. Users can just add @acc macro before 
their functions to accelerate them.

ParallelAccelerator is part of the High Performance Scripting (HPS) project at Intel Labs.

Quick install (requires Julia 0.4)::

        Pkg.add("ParallelAccelerator") 
