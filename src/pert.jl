module Pert

import ..getPackageRoot

pert_inited = false

eval(x) = Core.eval(Pert, x)

function __init__()
  package_root    = getPackageRoot()
  runtime_libpath = string(package_root, "/src/intel-runtime/lib/libintel-runtime.so")
  @eval begin
    function StartTiming(state::AbstractString)
      ccall((:StartTiming, $runtime_libpath), Void, (Ptr{UInt8},), state)
    end

    function StopTiming(state::AbstractString)
      ccall((:StopTiming, $runtime_libpath), Void, (Ptr{UInt8},), state)
    end

    function Register{T, N}(a :: Array{T, N})
      data = convert(Ptr{Void}, pointer(a))
      is_scalar = convert(Cint, 0)
      dim = convert(Cuint, N)  #ndims(a)
      sizes = [size(a)...]
      max_size = pointer(sizes)
      type_size = convert(Cuint, sizeof(T))

    #    println("data ", data, "is_scalar ", is_scalar, "dim ", dim, "sizes ", sizes, "max_size ", max_size, "type_size ", type_size);

      ccall((:pert_register_data, $runtime_libpath), Cint, (Ptr{Void}, Cint, Cuint, Ptr{Int64}, Cuint),
              data, is_scalar, dim, max_size, type_size)
    end

    function pert_shutdown(package_root)
      ccall((:pert_shutdown, $runtime_libpath), Cint, ())
      ccall((:FinalizeTiming, $runtime_libpath), Void, ()) 
    end

    function pert_init(package_root, double_buffer::Bool)
      global pert_inited
      if !pert_inited
        pert_inited = true
        ccall((:InitializeTiming, $runtime_libpath), Void, ())
        ccall((:pert_init,$runtime_libpath), Cint, (Cint,), convert(Cint, double_buffer))
        shutdown() = pert_shutdown(package_root) 
        atexit(shutdown)
      end
    end
  end
end

end
