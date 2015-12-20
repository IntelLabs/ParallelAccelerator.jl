### Faster compilation via userimg.jl

It is possible to embed a binary/compiled version of the ParallelAccelerator
compiler and CompilerTools into a Julia executable. This has the potential to
greatly reduce the time it takes for our compiler to accelerate a given
program. For this feature, the user needs to have the Julia source code 
and should be able to rebuild Julia. Hence,
Julia installations using ready binaries are not suitable for this purpose.

To use this feature, start the Julia REPL and do the following:

```
importall ParallelAccelerator
ParallelAccelerator.embed()
```

This version of `embed()` tries to embed ParallelAccelerator into the Julia
version used by the current REPL.

If you want to target a different Julia distribution, you can alternatively use
the following version of embed.

```
ParallelAccelerator.embed("<your/path/to/the/root/of/the/julia/distribution>")
```

This `embed` function takes a path which is expected to point to the root
directory of a Julia source distribution.  `embed` performs some simple checks to
try to verify this fact.  Then, embed will try to create a file
`base/userimg.jl` that will tell the Julia build how to embed the compiled
version into the Julia executable.  Finally, `embed` runs `make` in the Julia root
directory to create the embedded Julia version.

If there is already a `userimg.jl` file in the base directory then a new file is
created called `ParallelAccelerator_userimg.jl` and it then becomes the user's
responsibility to merge that with the existing `userimg.jl` and run `make`.

After the call to `embed` finishes, when trying to exit the Julia REPL, you may
receive an exception like: ErrorException(`"ccall: could not find function
git_libgit2_shutdown in library libgit2"`).  This error does not effect the
embedding process but we are working towards a solution to this minor issue.


