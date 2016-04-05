.. _advanced:

*********
Advanced Usage
*********

As mentioned above, ParallelAccelerator aims to optimize implicitly parallel
Julia programs that are safe to parallelize. It also tries to be non-invasive,
which means a user function or program should continue to work as expected even
when only a part of it is accelerated. It is still important to know what
parts are accelerated, however. As a general guideline,
we encourage users to write programs using high-level array operations rather
than writing explicit for-loops which can have unrestricted mutations or unknown
side-effects. High-level operations are more amenable to analysis and
optimization provided by ParallelAccelerator.

To help user verify program correctness, the optimizations of ParallelAccelerator
can be turned off by setting environment variable ``PROSPECT_MODE=none`` before
running Julia.  Programs that use ParallelAccelerator will still run
(including those that use ``runStencil``, described below), but no optimizations or
Julia-to-C translation will take place. Users can also use ``@noacc``
at the function call site to use the original version of the function.


Map and Reduce
--------------

Array operations that work uniformly on all elements of input arrays and
produce an output array of equal size are called `point-wise` operations.
`Point-wise` binary operations in Julia usually have a `.` prefix in the
operator name. These operations are translated internally into data-parallel *map* operations by
ParallelAccelerator. The following are recognized by ``@acc`` as *map*
operations:

* Unary functions: ``-``, ``+``, ``acos``, ``acosh``, ``angle``,
  ``asin``, ``asinh``, ``atan``, ``atanh``, ``cbrt``, ``cis``,
  ``cos``, ``cosh``, ``exp10``, ``exp2``, ``exp``, ``expm1``,
  ``lgamma``, ``log10``, ``log1p``, ``log2``, ``log``, ``sin``,
  ``sinh``, ``sqrt``, ``tan``, ``tanh``, ``abs``, ``copy``, ``erf``

* Binary functions: ``-``, ``+``, ``.+``, ``.-``, ``.*``, ``./``,
  ``.\``, ``.%``, ``.>``, ``.<``, ``.<=``, ``.>=``, ``.==``, ``.<<``,
  ``.>>``, ``.^``, ``div``, ``mod``, ``rem``, ``&``, ``|``, ``$``,
  ``min``, ``max``

Array assignments are also recognized and converted into *in-place map*
operations.  Expressions like ``a = a .+ b`` will be turned into an *in-place map*
that takes two inputs arrays, ``a`` and ``b``, and updates ``a`` in-place. 

Array operations that compute a single result by repeating an associative
and commutative operator on all input array elements are called *reduce* operations.
The following are recognized by ``@acc`` as ``reduce`` operations:
``minimum``, ``maximum``, ``sum``, ``prod``, ``any``, ``all``.


We also support range operations to a limited extent. For example, ``a[r] =
b[r]`` where ``r`` is either a ``BitArray`` or ``UnitRange`` (e.g., ``1:s``) is
internally converted to parallel operations when the ranges can be inferred
statically to be *compatible*. However, such support is still
experimental, and occasionally ParallelAccelerator will complain about not
being able to optimize them. We are working on improving this feature
to provide more coverage and better error messages.

Parallel Comprehension 
--------------

Array comprehensions in Julia are in general also parallelizable, because 
unlike general loops, their iteration variables have no inter-dependencies. 
So the ``@acc`` macro will turn them into an internal form that we call
``cartesianarray``::

    A = Type[ f (x1, x2, ...) for x1 in r1, x2 in r2, ... ]

becomes::

    cartesianarray((i1,i2,...) -> begin x1 = r1[i1]; x2 = r2[i2]; f(x1,x2,...) end,
             Type,(length(r1), length(r2), ...))

This ``cartesianarray`` function is also exported by ``ParallelAccelerator`` and
can be directly used by the user. Both the above two forms are acceptable
programs, and equivalent in semantics.  They both produce a N-dimensional array
whose element is of ``Type``, where ``N`` is the number of *x* and *r* variables, and
currently only up-to-3 dimensions are supported.

It should be noted, however, that not all comprehensions are safe to parallelize.
For example, if the function ``f`` above reads and writes to a variable outside of the comprehension, 
then making it run in parallel can produce a non-deterministic
result. Therefore, it is the responsibility of the user to avoid using ``@acc`` in such situations.

Another difference between parallel comprehension and the aforementioned *map*
operation is that array indexing operations in the body of a parallel
comprehension remain explicit and therefore should go through
necessary bounds-checking to ensure safety. On the other hand, in all *map* operations such
bounds-checking is skipped.

Stencil
------
      
Stencils are commonly found in scientific computing and image processing. A stencil
computation is one that computes new values for all elements of an array based
on the current values of their neighboring elements. Since Julia's base library
does not provide such an API, ParallelAccelerator exports a general
``runStencil`` interface to help with stencil programming::
                          
    runStencil(kernel :: Function, buffer1, buffer2, ..., 
                   iteration :: Int, boundaryHandling :: Symbol)
                                     

As an example, the following (taken from
`our Gaussian blur example <https://github.com/IntelLabs/ParallelAccelerator.jl/blob/master/examples/gaussian-blur/gaussian-blur.jl>`_)
performs a 5x5 stencil computation (note the use of Julia's ``do``-block syntax that lets
the user write a lambda function)::

    runStencil(buf, img, iterations, :oob_skip) do b, a
        b[0,0] =
          (a[-2,-2] * 0.003  + a[-1,-2] * 0.0133 + a[0,-2] * 0.0219 + a[1,-2] * 0.0133 + a[2,-2] * 0.0030 +
           a[-2,-1] * 0.0133 + a[-1,-1] * 0.0596 + a[0,-1] * 0.0983 + a[1,-1] * 0.0596 + a[2,-1] * 0.0133 +
           a[-2, 0] * 0.0219 + a[-1, 0] * 0.0983 + a[0, 0] * 0.1621 + a[1, 0] * 0.0983 + a[2, 0] * 0.0219 +
           a[-2, 1] * 0.0133 + a[-1, 1] * 0.0596 + a[0, 1] * 0.0983 + a[1, 1] * 0.0596 + a[2, 1] * 0.0133 +
           a[-2, 2] * 0.003  + a[-1, 2] * 0.0133 + a[0, 2] * 0.0219 + a[1, 2] * 0.0133 + a[2, 2] * 0.0030)
        return a, b
    end


It takes two input arrays, `buf` and `img`, and performs an iterative stencil
loop (ISL) of the number of iterations given by `iterations`.
The stencil kernel is specified by a lambda
function that takes two arrays `a` and `b` (that correspond to `buf` and
`img`), and computes the value of the output buffer using relative indices
as if a cursor is traversing all array elements. `[0,0]` represents
the current cursor position. The `return` statement in this lambda reverses
the position of `a` and `b` to specify a buffer rotation that should happen
in between the stencil iterations. ``runStencil`` assumes that
all input and output buffers are of the same dimension and size.

Stencil boundary handling can be specified as one of the following symbols:

* ``:oob_skip``: Writing to output is skipped when input indexing is out-of-bound.
* ``:oob_wraparound``: Indexing is "wrapped around" at the array boundaries so they are always safe.
* ``:oob_dst_zero``: Write 0 to the output array when any of the input indices is out-of-bounds.
* ``:oob_src_zero``. Assume 0 is returned by a read operation when indexing is out-of-bounds.

Just as with parallel comprehension, accessing the variables outside the body
of the ``runStencil`` lambda expression is allowed.
However, accessing outside array values is
not supported, and reading/writing the same outside variable can cause
non-determinism. 
All arrays that need to be relatively indexed can be specified as
input buffers. ``runStencil`` does not impose any implicit buffer rotation
order, and the user can choose not to rotate buffers in ``return``. There 
can be multiple output buffers as well. Finally, the call to ``runStencil`` does 
not have any return value, and inputs are rotated ``iterations - 1`` times if rotation is specified.
ParallelAccelerator exports a naive Julia implementation of ``runStencil`` that
runs without using ``@acc``. Its purpose is mostly for correctness checking.
When ``@acc`` is being used with environment variable ``PROSPECT_MODE=none``,
instead of parallelizing the stencil computation  ``@acc`` will expand the call
to ``runStencil`` to a fast sequential implementation.

