.. _limitations:

*********
Limitations 
*********

Currently, ParallelAccelerator tries to compile Julia to C++, which puts some constraints on what
can be successfully compiled and run:

1. We only support a limited subset of Julia language features currently.
   This includes basic numbers and dense array types, a subset of math 
   functions, and basic control flow structures. Notably, we do not support 
   ``String`` types and custom data types such as records and unions, since their 
   translation to C is difficult. There is also no support for exceptions, 
   I/O operations (only very limited ``println``), and arbitrary ``ccall``.
   We also do not support keyword arguments.

2. We do not support calling Julia functions from C++ in the optimized
   function. What this implies is that we transitively convert 
   every Julia function in the call chain to C++. If any of them is not 
   translated properly, the target function with ``@acc`` will fail to compile. 

3. We do not support Julia's ``Any`` type in C++, mostly to
   defend against erroneous translation. If the AST of a Julia function
   contains a variable with ``Any`` type, our Julia-to-C++ translator will give up
   compiling the function. This is indeed more limiting than it sounds, because
   Julia does not annotate all expressions in a typed AST with complete type 
   information. For example, this happens for some expressions that call Julia's 
   own intrinsics. We are working on supporting more of them if we can derive 
   the actual type to be not ``Any``, but this is still a work in progress.
                                                
At the moment ParallelAccelerator only supports the Julia-to-C++ back-end. We
are working on alternatives that make use of Julia's upcoming threading implementation 
that hopefully can alleviate the above mentioned
restrictions, without sacrificing much of the speed brought by quality C++
compilers and parallel runtimes such as OpenMP.
                                                
Apart from the constraints imposed by Julia-to-C++ translation, our current 
implementation of ParallelAccelerator has some other limitations:
                 
1. We currently support a limited subset of Julia functions available in the ``Base`` library.
   However, not all Julia functions in ``Base``
   are supported yet, and using them may or may not work in ParallelAccelerator.
   For supported functions, we rely on capturing operator names to resolve array related functions and operators
   to our API module. This prevents them from being inlined by Julia
   which helps our translation. For unsupported functions such as ``mean``,
   Julia's typed AST for the program
   that contains ``mean`` becomes a lowered call that is basically
   the low-level sequential implementation which cannot be
   handled by ParallelAccelerator. Of course, adding support
   for functions like ``mean`` is not a huge effort, and we are still in 
   the process of expanding the coverage of supported APIs.

2. ParallelAccelerator relies heavily on full type information being available
   in Julia's typed AST in order to work properly. Although we do not require
   user functions to be explicitly typed, it is in general a good practice to
   ensure the function that is being accelerated can pass Julia's type inference
   without leaving any arguments or internal variables with an ``Any`` type. 
   There is currently no facility to help users understand whether something
   is being optimized or silently rejected. In the future, we plan to provide 
   such functionality to give users better insight into what is going on under the hood.

