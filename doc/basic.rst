.. _basic:

*********
Basic Usage
*********

To start using ParallelAccelerator in your own program, first import
the package with ``using ParallelAccelerator``, and then put the ``@acc``
macro before the function you want to accelerate. A trivial example is
given below::

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

You can also use ``@acc begin ... end``, and put multiple functions in the block
to have all of them accelerated. The ``@acc`` macro only works for top-level 
definitions.

