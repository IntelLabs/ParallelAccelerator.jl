.. _prerequisites:

*********
 Prerequisites
*********

To use ParallelAccelerator, you need the following:

  * **Julia v0.5.x or v0.4.x.** ParallelAccelerator will not work with versions
    of Julia prior to v.0.4.0.  If you don't have an up-to-date
    version of Julia installed yet, there are various ways to do so:
      * Go to http://julialang.org/downloads/ and download the
        appropriate version listed under "Current Release".
      * On Ubuntu or Debian variants, ``sudo add-apt-repository
        ppa:staticfloat/juliareleases -y && sudo apt-get update -q &&
        sudo apt-get install julia -y`` should work.
      * On OS X with Homebrew, ``brew update && brew tap
        staticfloat/julia && brew install julia`` should work.

    Check that you can run Julia and get to a ``julia>`` prompt.  You
    will know you're running the correct version if when you run it,
    you see ``Version 0.5.x`` or ``Version 0.4.x`` where ``x`` is 0 or greater.
  * **A C/C++ compiler: either GCC or ICC.** We recommend GCC 4.8.4 or
    later and ICC 15.0.3 or later.  At package build time,
    ParallelAccelerator will check to see if you have ICC installed.
    If so, ParallelAccelerator will default to ICC.  Otherwise, it will use
    GCC. Clang with GCC front-end (default on Mac OS X) also works.
  * Platforms we have tested on so far include Ubuntu 14.04, CentOS
    6.6, and Mac OS X Yosemite with both GCC and ICC.
