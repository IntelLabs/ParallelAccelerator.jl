.. _compiletime:

*********
Speeding up package load time via userimg.jl
********

It is possible to embed a binary, compiled version of the ParallelAccelerator
package into a Julia executable. This has the potential to
greatly reduce the time it takes for Julia to load ParallelAccelerator.
To use this feature, one needs to have the Julia source code
and should be able to rebuild Julia.  Hence, users with
Julia installations using pre-built binaries will not be able to use it.

Note that this approach will speed up the `using ParallelAccelerator`
statement, but it will merely delay the compilation of most of the
ParallelAccelerator package until an accelerated function is called.
This is probably *not* the behavior that most users want.

To use this feature, start the Julia REPL and run the following::

    importall ParallelAccelerator
    ParallelAccelerator.embed()


The ``embed`` function tries to embed ParallelAccelerator into the Julia
version used by the current REPL.

If you want to target a different Julia distribution, you can alternatively use
the following version of ``embed``::

    ParallelAccelerator.embed("your/path/to/the/root/of/the/julia/distribution")


This version of ``embed`` takes a path which is expected to point to the root
directory of a Julia source distribution.  ``embed`` performs some simple checks to
try to verify this fact.  Then, ``embed`` will try to create a file
``base/userimg.jl`` that will tell the Julia build how to embed the compiled
version into the Julia executable.  Finally, ``embed`` runs ``make`` in the Julia root
directory to create the embedded Julia version.

If there is already a ``userimg.jl`` file in the base directory, then a new file is
created called ``ParallelAccelerator_userimg.jl`` and it then becomes the user's
responsibility to merge that file with the existing ``userimg.jl`` and run ``make``.


