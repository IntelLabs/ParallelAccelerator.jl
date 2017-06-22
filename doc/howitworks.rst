.. _howitworks:

*********
How It Works
*********

ParallelAccelerator is essentially a domain-specific compiler written in Julia.
It performs additional analysis and optimization on top of the Julia compiler.
ParallelAccelerator discovers and exploits the implicit parallelism in source programs that
use parallel programming patterns such as *map*, *reduce*, *comprehension*, and
*stencil*. For example, Julia array operators such as ``.+``, ``.-``, ``.*``, and ``./`` are
translated by ParallelAccelerator internally into data-parallel *map* operations over all
elements of input arrays.  For the most part, these patterns are already
present in standard Julia, so programmers can use ParallelAccelerator to run
the same Julia program without (significantly) modifying the source code. 

The ``@acc`` macro provided by ParallelAccelerator first intercepts Julia
functions at the macro level and substitutes the set of implicitly parallel
operations that we are targeting. ``@acc`` points them to those supplied in the
``ParallelAccelerator.API`` module. It then creates a proxy function that when
called with concrete arguments (and known types) will try to compile the
original function to an optimized form. Therefore, there is some compilation
time the first time an accelerated function is called. The subsequent
calls to the same function will not have compilation time overhead.

ParallelAccelerator performs aggressive optimizations when they are safe depending on the program structure.
For example, it will automatically infer size equivalence relations among array
variables and skip array bounds check whenever it can safely do so.   Eventually all
parallel patterns are lowered into explicit parallel *for* loops which are internally
represented at the level of Julia's typed AST. Aggressive loop fusion will
try to combine adjacent loops into one and eliminate temporary array objects
that store intermediate results.

Finally, functions with parallel for loops are translated into a C program with
OpenMP pragmas, and ParallelAccelerator will use an external C/C++ compiler to
compile it into binary form before loading it back into Julia as a dynamic
library for execution. This step of translating Julia to C currently imposes
certain limitations, and therefore we can only run user
programs that meet such limitations.

To learn more about how ParallelAccelerator works under the hood, see
`our ECOOP 2017 paper`
<http://2017.ecoop.org/event/ecoop-2017-papers-parallelizing-julia-with-a-non-invasive-dsl>`_.

