module IntelPSE

export decompose, offload, Optimize
export cartesianarray, runStencil, @runStencil

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

function getPackageRoot()
  joinpath(dirname(@__FILE__), "..")
end

function __init__()
    package_root = getPackageRoot()
    @linux_only ld_env_key = "LD_LIBRARY_PATH"
    @osx_only   ld_env_key = "DYLD_LIBRARY_PATH"
    prefix = ""
    if haskey(ENV, ld_env_key)
        prefix = ENV[ld_env_key] * ":"
    end
    ENV[ld_env_key] = string(prefix, package_root, "bin")
end

# This controls the debug print level.  0 prints nothing.  At the moment, 2 prints everything.
DEBUG_LVL=0

function set_debug_level(x)
    global DEBUG_LVL = x
end

# A debug print routine.
function dprint(level,msgs...)
    if(DEBUG_LVL >= level)
        print(msgs...)
    end
end

# A debug print routine.
function dprintln(level,msgs...)
    if(DEBUG_LVL >= level)
        println(msgs...)
    end
end

# a hack to make offload function and DomainIR mutually recursive.
_offload(function_name, signature) = offload(function_name, signature, 0)

include("api.jl")
include("domain-ir.jl")
include("alias-analysis.jl")
include("parallel-ir.jl")
include("j2c-array.jl")
include("cgen.jl")
include("pp.jl")
include("ParallelComprehension.jl")
include("callgraph.jl")
# The following are intel-runtime specific
include("pert.jl")
include("pse-ld.jl")
#
include("driver.jl")

importall .API
importall .Driver

end
