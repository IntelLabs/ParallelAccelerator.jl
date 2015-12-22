# ParallelAccelerator

[![Join the chat at https://gitter.im/IntelLabs/ParallelAccelerator.jl](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/IntelLabs/ParallelAccelerator.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/IntelLabs/ParallelAccelerator.jl.svg?branch=master)](https://travis-ci.org/IntelLabs/ParallelAccelerator.jl)
[![Coverage Status](https://coveralls.io/repos/IntelLabs/ParallelAccelerator.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/IntelLabs/ParallelAccelerator.jl?branch=master)

ParallelAccelerator package is a compiler framework that
aggressively optimizes compute-intensive Julia programs on top of the Julia compiler.
It automatically eliminates overheads such as array bounds 
checking when it is safe to eliminate them.
It also parallelizes and vectorizes many data-parallel operations.
Users can just add `@acc` macro before their functions to accelerate them.

ParallelAccelerator is part of the High
Performance Scripting (HPS) project at Intel Labs.

Quick install (requires Julia 0.4):
``` .julia
Pkg.add("ParallelAccelerator") 
```

## Resources

- **Documentation:** <http://parallelacceleratorjl.readthedocs.org/>
- **Mailing List:** <http://groups.google.com/group/julia-hps/>
- **Chat Room:** <https://gitter.im/IntelLabs/ParallelAccelerator.jl>
- **GitHub Issues:** <https://github.com/IntelLabs/ParallelAccelerator.jl/issues>
