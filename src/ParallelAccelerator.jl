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

#VERSION >= v"0.4.0-dev" && __precompile__()

module ParallelAccelerator

import CompilerTools.DebugMsg
DebugMsg.init()

importall CompilerTools
using CompilerTools.OptFramework

#######################################################################
# Constants and functions below this line are not exported, and should 
# be used either fully qualified, or explicitly imported.
#######################################################################

const OFF_MODE = 0
const HOST_MODE = 1
const OFFLOAD1_MODE = 2
const OFFLOAD2_MODE = 3
const TASK_MODE = 4
const THREADS_MODE = 5

num_acc_allocs = 0
num_acc_parfors = 0

"""
(internal) Sets the number of allocations left in the accelerated function after optimizations. 
"""
function set_num_acc_allocs(allocs::Int)
    global num_acc_allocs = allocs
    return
end

"""
(internal) Sets the number of parfors generated for the accelerated function after optimizations.
"""
function set_num_acc_parfors(parfors::Int)
    global num_acc_parfors = parfors
    return
end


"""
Returns the number of allocations left in the accelerated function after optimizations. 
"""
function get_num_acc_allocs()
    return num_acc_allocs
end

"""
Returns the number of parfors generated for the accelerated function after optimizations.
"""
function get_num_acc_parfors()
    return num_acc_parfors
end

"""
Generate a file path to the directory above the one containing this source file.
This should be the root of the package.
"""
function getPackageRoot()
  joinpath(dirname(@__FILE__), "..")
end

const USE_ICC = 0
const USE_GCC = 1
const USE_MINGW = 2
const NONE = 3

use_bcpp = 0
backend_compiler = USE_ICC
mkl_lib = ""
openblas_lib = ""
sys_blas = 0
openmp_supported = 0
package_root = getPackageRoot()

#config file overrides backend_compiler variable
if isfile("$package_root/deps/generated/config.jl")
  include("$package_root/deps/generated/config.jl")
end

function getMklLib()
  return mkl_lib
end

function getOpenblasLib()
  return openblas_lib
end

function getSysBlas()
  return sys_blas
end

function getUseBcpp()
  return use_bcpp
end

function getBackendCompiler()
  return backend_compiler
end

cached_mode = nothing

"""
Return internal mode number by looking up environment variable "PROSPECT_MODE".
"""
function getPseMode()
    global cached_mode
    if cached_mode == nothing
        if haskey(ENV,"PROSPECT_MODE")
            env_mode = ENV["PROSPECT_MODE"]
            if (env_mode == "cgen" || env_mode == "offload1" || env_mode == "offload2" || env_mode == "task") && backend_compiler == NONE
                println("ParallelAccelerator backend CGen requested but no C compiler is installed...")
                println("...reverting to Julia native threading backend.")
                ENV["PROSPECT_MODE"] = "threads"
            end
        end

        if haskey(ENV,"PROSPECT_MODE")
           mode = ENV["PROSPECT_MODE"]
        else
           mode = "threads"
        end

        if mode == "none" || mode == "off"
          cached_mode = OFF_MODE 
        elseif mode == "cgen"
          cached_mode = HOST_MODE 
        elseif mode == "offload1"
          cached_mode = OFFLOAD1_MODE
        elseif mode == "offload2"
          cached_mode = OFFLOAD2_MODE
        elseif mode == "task"
          cached_mode = TASK_MODE
        elseif mode == "threads"
          cached_mode = THREADS_MODE
        else
          error(string("Unknown PROSPECT_MODE = ", mode))
        end
    end

    return cached_mode
end

const NO_TASK_MODE = 0
const HOST_TASK_MODE = 1
const PHI_TASK_MODE = 2
const DYNAMIC_TASK_MODE = 3


"""
Return internal mode number by looking up environment variable "PROSPECT_TASK_MODE".
If not specified, it defaults to NO_TASK_MODE, or DYNAMIC_TASK_MODE when 
getPseMode() is TASK_MODE.
"""
function getTaskMode()
  if haskey(ENV,"PROSPECT_TASK_MODE")
     mode = ENV["PROSPECT_TASK_MODE"]
  else
    if getPseMode() == TASK_MODE
      mode = "dynamic"
    else
      mode = "none"
    end
  end
  if mode == "none" || mode == "off"
    NO_TASK_MODE
  elseif mode == "task"
    HOST_TASK_MODE
  elseif mode == "phi"
    PHI_TASK_MODE
  elseif mode == "dynamic"
    DYNAMIC_TASK_MODE
  else
    error(string("Unknown PROSPECT_TASK_MODE = ", mode))
  end
end

type UnsupportedFeature <: Exception
  text :: AbstractString
end

"""
Call this function if you want to embed binary-code of ParallelAccelerator into your Julia build to speed-up @acc compilation time.
It will attempt to add a userimg.jl file to your Julia distribution and then re-build Julia.
"""
function embed(julia_root)
  # See if the specified julia_root path exists.
  if !ispath(julia_root)
    println("The specified path to the Julia source code, ", julia_root, ", was not valid.")
    return nothing
  end

  # Do a check to see if this is likely to be a Julia distribution by looking for a base directory.
  base_dir = string(julia_root, "/base")
  if !ispath(base_dir)
    println("The specified path to the Julia source code, ", julia_root, ", does not seem to be a Julia source distribution; base dir not found.")
    return nothing
  end

  # The contents of the userimg.jl file.
  current_file = @__FILE__
  userimg_contents = string("""
# This is a dummy use of the ParallelAccelerator module.  It will
# cause a brief delay once, when Julia compiles, in exchange for *not*
# having a delay every time the package is used.

Base.reinit_stdio()

include("$current_file")

# This is to force the target of CGen's atexit hook to transitively compile so that
# compilation will not be attempted atexit time during Julia's build process.
ParallelAccelerator.CGen.CGen_finalize()

module __UserimgDummyModule__

using ParallelAccelerator
const dottimes=GlobalRef(ParallelAccelerator.API, :*)
const dotplus=GlobalRef(ParallelAccelerator.API, :-)
const runStencil=GlobalRef(ParallelAccelerator.API, :runStencil)

@eval __userimg_dummy_fn__(A::Array{Float64,2}, B::Array{Float64,2}) = begin
    \$runStencil((a, b) -> a[0,0] = b[0,0], A, B, 1, :oob_skip)
    \$dotplus(2, \$dottimes(A, B))
end

ParallelAccelerator.accelerate(__userimg_dummy_fn__, (Array{Float64,2},Array{Float64,2},))

end
""")

  userimg_file = string(base_dir, "/userimg.jl")
  # If there is already a userimg.jl file then just report the need for a manual merge.
  if isfile(userimg_file)
    println("The Julia source tree already seems to have a userimg.jl file.")
    println("A file ParallelAccelerator_userimg.jl has been created in ", base_dir, ".")
    println("Please manually merge this file with userimg.jl and then re-compile Julia.")

    outfile = string(base_dir, "/ParallelAccelerator_userimg.jl")
    f = open(outfile, "w")
    println(f, userimg_contents)
    close(f)
  else
    # There is no existing userimg.jl file so create one with the userimg_contents.
    f = open(userimg_file, "w")
    println(f, userimg_contents)
    close(f)

    curwd = pwd()    # Get the current directory so we can switch back to that later.
    cd(julia_root)   # Switch the the root of the julia distribution where we can run make.
    run(`make`)      # Re-build Julia.
    cd(curwd)        # Go back to the initial working directory.
  end

  return nothing
end

"""
This version of embed tries to use JULIA_HOME to find the root of the source distribution.
It then calls the version above specifying the path.
"""
function embed()
  embed(joinpath(JULIA_HOME, "..", ".."))
end

# a hack to make accelerate function and DomainIR mutually recursive.
_accelerate(function_name, signature) = accelerate(function_name, signature, 0)

# data types for declaring size array for data sources
# will be matched in CGen to proper C arrays
type H5SizeArr_t
end
type SizeArr_t
end

function show_backtrace()
    bt = backtrace()
    s = sprint(io->Base.show_backtrace(io, bt))
    println(s)
end

using Compat

include("api.jl")
include("domain-ir.jl")
include("parallel-ir.jl")
include("j2c-array.jl")
include("cgen.jl")
#include("callgraph.jl")
include("comprehension.jl")
include("driver.jl")

importall .Driver

"""
Called when the package is loaded to do initialization.
"""
function __init__()
    package_root = getPackageRoot()
    if Compat.is_linux()
        ld_env_key = "LD_LIBRARY_PATH"
    elseif Compat.is_apple()
        ld_env_key = "DYLD_LIBRARY_PATH"
    elseif Compat.is_windows()
        ld_env_key = "PATH"
    else
        throw(string("System is not linux, apple, or windows...giving up."))
    end
    prefix = ""
    # See if the LD_LIBRARY_PATH environment varible is already set.
    if haskey(ENV, ld_env_key)
        # Prepare the current value to have a new path added to it.
        prefix = ENV[ld_env_key] * ":"
    end
    # Add the bin directory off the package root to the LD_LIBRARY_PATH.
    ENV[ld_env_key] = string(prefix, package_root, "bin")

    # Must add LD_LIBRARY_PATH for MIC
    if getPseMode() == OFFLOAD1_MODE || getPseMode() == OFFLOAD2_MODE || getPseMode() == TASK_MODE
        ENV["LD_LIBRARY_PATH"]=string(CGen.generated_file_dir, ":", ENV["LD_LIBRARY_PATH"])
        #ENV["MIC_LD_LIBRARY_PATH"]=string(generated_file_dir, ":", ENV["MIC_LD_LIBRARY_PATH"])
    end

    if getPseMode() == OFF_MODE
      addOptPass(runStencilMacro, PASS_MACRO)
      #addOptPass(cleanupAPI, PASS_MACRO)
    else 
      addOptPass(captureOperators, PASS_MACRO)
      addOptPass(toCartesianArray, PASS_MACRO)
      addOptPass(expandParMacro, PASS_MACRO)
      addOptPass(extractCallGraph, PASS_TYPED)
      addOptPass(toDomainIR, PASS_TYPED)
      addOptPass(toParallelIR, PASS_TYPED)
      addOptPass(toFlatParfors, PASS_TYPED)
      if getPseMode() == THREADS_MODE
        addOptPass(toJulia, PASS_TYPED)
      else
        addOptPass(toCGen, PASS_TYPED)
      end
    end
end

import .API.@par
import .API.runStencil
import .API.cartesianarray
import .API.parallel_for
export accelerate, @acc, @noacc, @par, runStencil, cartesianarray, parallel_for

end
