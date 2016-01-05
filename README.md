# ParallelAccelerator

[![Join the chat at https://gitter.im/IntelLabs/ParallelAccelerator.jl](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/IntelLabs/ParallelAccelerator.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/IntelLabs/ParallelAccelerator.jl.svg?branch=master)](https://travis-ci.org/IntelLabs/ParallelAccelerator.jl)
[![Coverage Status](https://coveralls.io/repos/IntelLabs/ParallelAccelerator.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/IntelLabs/ParallelAccelerator.jl?branch=master)

ParallelAccelerator is a compiler framework, implemented as a Julia package, that
aggressively optimizes compute-intensive Julia programs.
It automatically eliminates overheads such as array bounds 
checking when it is safe to eliminate them.
It also parallelizes and vectorizes many data-parallel operations.

Users can annotate functions with ParallelAccelerator's provided
`@acc` macro.  At runtime, ParallelAccelerator compiles
`@acc`-annotated functions to optimized native code.

ParallelAccelerator is part of the High
Performance Scripting (HPS) project at Intel Labs.

Quick install (requires Julia 0.4):
``` .julia
Pkg.add("ParallelAccelerator") 
```

## Resources

- **Documentation:** <http://parallelacceleratorjl.readthedocs.org/>
- **Presentations:**
  - [Slides](http://www.slideshare.net/ChristianPeel/ehsan-parallel-acceleratordec2015) and [audio](https://soundcloud.com/christian-peel/ehsan-totoni-on-parallelacceleratorjl) of a talk at [Bay Area Julia Users](http://www.meetup.com/Bay-Area-Julia-Users/events/226531171/), December 17, 2015.
  - [Video](https://www.youtube.com/watch?v=O6PN-kpbNTw) of a talk at [SPLASH-I](http://2015.splashcon.org/event/splash2015-splash-i-lindsey-kuper-talk), October 29, 2015.
- **Mailing List:** <http://groups.google.com/group/julia-hps/>
- **Chat Room:** <https://gitter.im/IntelLabs/ParallelAccelerator.jl>
- **GitHub Issues:** <https://github.com/IntelLabs/ParallelAccelerator.jl/issues>
