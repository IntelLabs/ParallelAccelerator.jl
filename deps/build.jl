#=
Copyright (c) 2015, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
=#

using Compat

println("ParallelAccelerator: build.jl begin.")
println("ParallelAccelerator: Building j2c-array shared library")

ld_library_path = ""
dyld_library_path = ""

if haskey(ENV, "DYLD_LIBRARY_PATH")
    dyld_library_path = ENV["DYLD_LIBRARY_PATH"]
end

if haskey(ENV, "LD_LIBRARY_PATH")
    ld_library_path = ENV["LD_LIBRARY_PATH"]
end

if Compat.is_windows()
    println("Installing ParallelAccelerator for Windows.")

    builddir = dirname(Base.source_path())
    println("Build directory is ", builddir)

    import WinRPM

    println("Installing gcc-c++.")
    WinRPM.install("gcc-c++";yes=true)
    WinRPM.install("gcc";yes=true)
    WinRPM.install("headers";yes=true)

    gpp = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","bin","g++")
    RPMbindir = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","bin")
    incdir = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","include")

    push!(Base.Libdl.DL_LOAD_PATH,RPMbindir)
    ENV["PATH"]=ENV["PATH"]*";"*RPMbindir

    println("Installed gcc is version ")
    run(`$gpp --version`)

    println("Building libj2carray.dll.")
    run(`$gpp -g -shared -std=c++11 -I $incdir -o $builddir\\libj2carray.dll -lm $builddir\\j2c-array.cpp`)

    conf_file = builddir * "\\generated\\config.jl"
    cf = open(conf_file, "w")
    println(cf, "backend_compiler = USE_MINGW")
    println(cf, "openblas_lib = \"\"")
    #println(cf, "openblas_lib = \"", Base.Libdl.find_library([string("libopenblas64_")]), "\"")
    println(cf, "mkl_lib = \"", Base.Libdl.find_library([string("libmkl")]), "\"")
    
    try
        run(`bcpp`)
        println(cf, "use_bcpp = 1")
    catch some_exception
        println("bcpp not found and will not be used.")
    end
        
    try
        btest = open("blas_test.cpp","w")
        println(btest,"#include <cblas.h>\nint main(){return 0;}")
        close(btest)
        run(pipeline(`$gpp -I $incdir blas_test.cpp`, stdout=DevNull, stderr=DevNull))
        println(cf, "sys_blas = 1")
    catch some_exception
        println(cf, "sys_blas = 0")
    end

    close(cf) 
else
    run(`./build.sh $dyld_library_path $ld_library_path`)
end
println("ParallelAccelerator: build.jl done.")
