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

#import Base.deepcopy_internal
#
#@doc """
#Overload deepcopy_internal() to just return a Module instead of trying to duplicate it.
#"""
#deepcopy_internal(x :: Module, stackdict::ObjectIdDict) = x

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

@doc """
Return internal mode number by looking up environment variable "PROSPECT_MODE".
"""
function getPseMode()
  if haskey(ENV,"PROSPECT_MODE")
     mode = ENV["PROSPECT_MODE"]
  else
     mode = "host"
  end
  if mode == "none" || mode == "off"
    OFF_MODE 
  elseif mode == "host"
    HOST_MODE 
  elseif mode == "offload1"
    OFFLOAD1_MODE
  elseif mode == "offload2"
    OFFLOAD2_MODE
  elseif mode == "task"
    TASK_MODE
  elseif mode == "threads"
    THREADS_MODE
  else
    error(string("Unknown PROSPECT_MODE = ", mode))
  end
end

const NO_TASK_MODE = 0
const HOST_TASK_MODE = 1
const PHI_TASK_MODE = 2
const DYNAMIC_TASK_MODE = 3

function isDistributedMode()
    mode = "0"
    if haskey(ENV,"HPS_DISTRIBUTED_MEMORY")
        mode = ENV["HPS_DISTRIBUTED_MEMORY"]
    end
    return mode=="1"
end

@doc """
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

@doc """
Generate a file path to the directory above the one containing this source file.
This should be the root of the package.
"""
function getPackageRoot()
  joinpath(dirname(@__FILE__), "..")
end

type UnsupportedFeature <: Exception
  text :: AbstractString
end

@doc """
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
  userimg_contents = string("Base.reinit_stdio()\ninclude(\"", @__FILE__, "\")\ntmp_f(A,B)=begin runStencil((a, b) -> a[0,0] = b[0,0], A, B, 1, :oob_skip); A.*B.+2 end\nParallelAccelerator.accelerate(tmp_f,(Array{Float64,1},Array{Float64,1},))\n")

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

@doc """
This version of embed tries to use JULIA_HOME to find the root of the source distribution.
It then calls the version above specifying the path.
"""
function embed()
  embed(joinpath(JULIA_HOME, "..", ".."))
end

# a hack to make accelerate function and DomainIR mutually recursive.
_accelerate(function_name, signature) = accelerate(function_name, signature, 0)

include("api.jl")
include("domain-ir.jl")
include("parallel-ir.jl")
include("distributed-ir.jl")
include("j2c-array.jl")
include("cgen.jl")
include("callgraph.jl")
include("comprehension.jl")
include("driver.jl")

importall .Driver

@doc """
Called when the package is loaded to do initialization.
"""
function __init__()
    package_root = getPackageRoot()
    @linux_only ld_env_key = "LD_LIBRARY_PATH"
    @osx_only   ld_env_key = "DYLD_LIBRARY_PATH"
    @windows_only ld_env_key = "PATH"
    prefix = ""
    # See if the LD_LIBRARY_PATH environment varible is already set.
    if haskey(ENV, ld_env_key)
        # Prepare the current value to have a new path added to it.
        prefix = ENV[ld_env_key] * ":"
    end
    # Add the bin directory off the package root to the LD_LIBRARY_PATH.
    ENV[ld_env_key] = string(prefix, package_root, "bin")

    if getPseMode() == OFF_MODE
      addOptPass(runStencilMacro, PASS_MACRO)
      #addOptPass(cleanupAPI, PASS_MACRO)
    else
      addOptPass(captureOperators, PASS_MACRO)
      addOptPass(toCartesianArray, PASS_MACRO)
      addOptPass(toDomainIR, PASS_TYPED)
      addOptPass(toParallelIR, PASS_TYPED)
      if isDistributedMode()
          addOptPass(toDistributedIR, PASS_TYPED)
      end
      addOptPass(toCGen, PASS_TYPED)
    end
end

import .API.runStencil
import .API.cartesianarray
import .API.parallel_for
export accelerate, @acc, @noacc, runStencil, cartesianarray, parallel_for

end

#tmp_f(A,B)=begin runStencil((a, b) -> a[0,0] = b[0,0], A, B, 1, :oob_skip); A.*B.+2 end
#ParallelAccelerator.accelerate(tmp_f,(Array{Float64,1},Array{Float64,1},))

