#VERSION >= v"0.4.0-dev" && __precompile__()

module ParallelAccelerator

export decompose, accelerate, Optimize
export cartesianarray, runStencil, @runStencil

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
Return internal mode number by looking up environment variable "INTEL_PSE_MODE".
"""
function getPseMode()
  if haskey(ENV,"INTEL_PSE_MODE")
     mode = ENV["INTEL_PSE_MODE"]
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
    error(string("Unknown INTEL_PSE_MODE = ", mode))
  end
end

const NO_TASK_MODE = 0
const HOST_TASK_MODE = 1
const PHI_TASK_MODE = 2
const DYNAMIC_TASK_MODE = 3

@doc """
Return internal mode number by looking up environment variable "INTEL_TASK_MODE".
If not specified, it defaults to NO_TASK_MODE, or DYNAMIC_TASK_MODE when 
getPseMode() is TASK_MODE.
"""
function getTaskMode()
  if haskey(ENV,"INTEL_TASK_MODE")
     mode = ENV["INTEL_TASK_MODE"]
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
    error(string("Unknown INTEL_TASK_MODE = ", mode))
  end
end

@doc """
Generate a file path to the directory above the one containing this source file.
This should be the root of the package.
"""
function getPackageRoot()
  joinpath(dirname(@__FILE__), "..")
end

# This controls the debug print level.  0 prints nothing.  At the moment, 2 prints everything.
DEBUG_LVL=0

@doc """
Set the verbose of debugging print messages from this module.
"""
function set_debug_level(x)
    global DEBUG_LVL = x
end

@doc """
Print a debug message if specified debug level is greater than or equal to this particular message's level.
"""
function dprint(level, msgs...)
    if(DEBUG_LVL >= level)
        print(msgs...)
    end
end

@doc """
Print a debug message if specified debug level is greater than or equal to this particular message's level.
"""
function dprintln(level, msgs...)
    if(DEBUG_LVL >= level)
        println(msgs...)
    end
end

# a hack to make accelerate function and DomainIR mutually recursive.
_accelerate(function_name, signature) = accelerate(function_name, signature, 0)

include("api.jl")
include("stencil-api.jl")
include("domain-ir.jl")
include("parallel-ir.jl")
include("j2c-array.jl")
include("cgen.jl")
include("pp.jl")
include("callgraph.jl")
include("comprehension.jl")
include("driver.jl")

importall .API
using .StencilAPI
using .Driver

@doc """
Called when the package is loaded to do initialization.
"""
function __init__()
    package_root = getPackageRoot()
    @linux_only ld_env_key = "LD_LIBRARY_PATH"
    @osx_only   ld_env_key = "DYLD_LIBRARY_PATH"
    prefix = ""
    # See if the LD_LIBRARY_PATH environment varible is already set.
    if haskey(ENV, ld_env_key)
        # Prepare the current value to have a new path added to it.
        prefix = ENV[ld_env_key] * ":"
    end
    # Add the bin directory off the package root to the LD_LIBRARY_PATH.
    ENV[ld_env_key] = string(prefix, package_root, "bin")

    addOptPass(toCartesianArray, PASS_MACRO)
    addOptPass(toDomainIR, PASS_TYPED)
    addOptPass(toParallelIR, PASS_TYPED)
    addOptPass(toCGen, PASS_TYPED)
end

export @acc

end
