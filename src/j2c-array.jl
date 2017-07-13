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

module J2CArray

using Compat

export to_j2c_array, from_j2c_array, j2c_array_delete, from_ascii_string
import CompilerTools.Helper.isArrayType
import ..getPackageRoot

#eval(x) = Core.eval(J2CArray, x)

function getLib()
    package_root = getPackageRoot()
    if Compat.is_windows()
        return string(package_root, "/deps/libj2carray.dll")
    else
        return string(package_root, "/deps/libj2carray.so.1.0")
    end
end
    
if !isfile(getLib())
  error("libj2carray not compiled, build with: julia -e 'Pkg.build(\"ParallelAccelerator\")'")
end

function __init__()
  dyn_lib = getLib()

  @eval begin
    # Create a new j2c array object with element size in bytes and given dimension.
    # It will share the data pointer of the given inp array, and if inp is nothing,
    # the j2c array will allocate fresh memory to hold data.
    # NOTE: when elem_bytes is 0, it means the elements must be j2c array type
#    function j2c_array_new(elem_bytes::Int, inp::Union{Array, Void}, ndim::Int, dims::Tuple)
#      # note that C interface mandates Int64 for dimension data
#      _dims = Int64[ convert(Int64, x) for x in dims ]
#      _inp = (inp === nothing) ? C_NULL : convert(Ptr{Void}, pointer(inp))
#
#      #ccall((:j2c_array_new, $dyn_lib), Ptr{Void}, (Cint, Ptr{Void}, Cuint, Ptr{UInt64}),
#      #      convert(Cint, elem_bytes), _inp, convert(Cuint, ndim), pointer(_dims))
#      ccall((:j2c_array_new, cgen_lib), Ptr{Void}, (Cint, Ptr{Void}, Cuint, Ptr{UInt64}),
#            convert(Cint, elem_bytes), _inp, convert(Cuint, ndim), pointer(_dims))
#    end

    # Array size in the given dimension.
    function j2c_array_size(arr::Ptr{Void}, dim::Int)
      l = ccall((:j2c_array_size, $dyn_lib), Cuint, (Ptr{Void}, Cuint),
                arr, convert(Cuint, dim))
      return convert(Int, l)
    end

    # Retrieve j2c array data pointer, and parameter "own" means caller will
    # handle the memory deallocation of this pointer.
    function j2c_array_to_pointer(arr::Ptr{Void}, own::Bool)
      ccall((:j2c_array_to_pointer,$dyn_lib), Ptr{Void}, (Ptr{Void}, Bool), arr, own)
    end

    # Read the j2c array element of given type at the given (linear) index.
    # If T is Ptr{Void}, treat the element type as j2c array, and the
    # returned array is merely a pointer, not a new object.
    function j2c_array_get(arr::Ptr{Void}, idx::Int, T::Type)
      nbytes = (T === Ptr{Void}) ? 0 : sizeof(T)
      _value = Array{T}(1)
      ccall((:j2c_array_get,$dyn_lib), Void, (Cint, Ptr{Void}, Cuint, Ptr{Void}),
            convert(Cint, nbytes), arr, convert(Cuint, idx), convert(Ptr{Void}, pointer(_value)))
      return _value[1]
    end

    # Set the j2c array element at the given (linear) index to the given value.
    # If T is Ptr{Void}, treat value as a pointer to j2c array.
    function j2c_array_set{T}(arr::Ptr{Void}, idx::Int, value::T)
      nbytes = (T === Ptr{Void}) ? 0 : sizeof(T)
      _value = nbytes == 0 ? value : convert(Ptr{Void}, pointer(T[ value ]))
      ccall((:j2c_array_set, $dyn_lib), Void, (Cint, Ptr{Void}, Cuint, Ptr{Void}),
            convert(Cint, nbytes), arr, convert(Cuint, idx), _value)
    end

    # Delete an j2c array object.
    # Note that this only works for scalar array or arrays of
    # objects whose derefrence will definite not trigger nested
    # deletion (either data pointer is NULL, or refcount > 1).
    # Currently there is no way to cleanly delete an nested j2c
    # array without first converting back to a julia array.
    function j2c_array_delete(arr::Ptr{Void})
      ccall((:j2c_array_delete,$dyn_lib), Void, (Ptr{Void},), arr)
    end

    # Dereference a j2c array data pointer.
    # Require that the j2c array data pointer points to either
    # a scalar array, or an array of already dereferenced array
    # (whose data pointers are NULL).
    function j2c_array_deref(arr::Ptr{Void})
      ccall((:j2c_array_deref,$dyn_lib), Void, (Ptr{Void},), arr)
    end

if VERSION >= v"0.6.0-pre"
    # convert ASCIIString to J2C_Array and to C's ASCIIString 
    function to_j2c_array(inp::AbstractString, ptr_array_dict, mapAtypeKey, j2c_array_new)
      arr = to_j2c_array(Vector{UInt8}(inp), ptr_array_dict, mapAtypeKey, j2c_array_new)
      ccall((:new_ascii_string, $dyn_lib), Ptr{Void}, (Ptr{Void}, ), arr)
    end
else
    # convert ASCIIString to J2C_Array and to C's ASCIIString 
    function to_j2c_array(inp::AbstractString, ptr_array_dict, mapAtypeKey, j2c_array_new)
      arr = to_j2c_array(inp.data, ptr_array_dict, mapAtypeKey, j2c_array_new)
      ccall((:new_ascii_string, $dyn_lib), Ptr{Void}, (Ptr{Void}, ), arr)
    end
end

    # convert C's ASCIIString to Julia's Array{UInt8,1}
    function from_ascii_string(str::Ptr{Void}, ptr_array_dict)
      data = ccall((:from_ascii_string, $dyn_lib), Ptr{Void}, (Ptr{Void}, ), str)
      arr = _from_j2c_array(data, UInt8, 1, ptr_array_dict)
      ccall((:delete_ascii_string, $dyn_lib), Void, (Ptr{Void}, ), str)
      return arr
    end
  end
end

# Convert Julia array to J2C array object.
# Note that Julia array data are not copied but shared by the J2C array
# The caller needs to make sure these arrays stay alive so long as the
# returned j2c array is alive.
function to_j2c_array{T, N}(inp :: Array{T, N}, ptr_array_dict :: Dict{Ptr{Void}, Array}, mapAtypeKey :: Dict{DataType,Int64}, j2c_array_new)
  allocation_key = mapAtypeKey[typeof(inp)]
  dims = size(inp)
  _isbits = isbits(T)
  nbytes = _isbits ? sizeof(T) : 0
  _inp = _isbits ? inp : nothing
  #arr = j2c_array_new(nbytes, _inp, N, dims)
  arr = j2c_array_new(allocation_key, _inp, N, dims)
  ptr_array_dict[convert(Ptr{Void}, pointer(inp))] = inp  # establish a mapping between pointer and the original array
  if !(_isbits)
    for i = 1:length(inp)
      obj = to_j2c_array(inp[i], ptr_array_dict, mapAtypeKey, j2c_array_new) # obj is a new j2c array
      j2c_array_set(arr, i, obj) # obj is duplicated during this set
      j2c_array_delete(obj)      # safe to delete obj without triggering free
    end
  end
  return arr
end

# Convert J2C array object to Julia array.
# Note that:
# 1. We assume the input j2c array object contains no data pointer aliases.
# 2. The returned Julia array will share the pointer to J2C array data at leaf level.
function _from_j2c_array(inp::Ptr{Void}, elem_typ::DataType, N::Int, ptr_array_dict :: Dict{Ptr{Void}, Array})
  dims = Array{Int}(N)
  len  = 1
  for i = 1:N
    dims[i] = j2c_array_size(inp, i)
    len = len * dims[i]
  end
  if isbits(elem_typ)
    array_ptr = convert(Ptr{Void}, j2c_array_to_pointer(inp, true))
    if haskey(ptr_array_dict, array_ptr)
      arr = ptr_array_dict[array_ptr]
    else
if VERSION > v"0.5.0-dev+3260"
      arr = unsafe_wrap(Array,convert(Ptr{elem_typ}, array_ptr), tuple(dims...), true)
else
      arr = pointer_to_array(convert(Ptr{elem_typ}, array_ptr), tuple(dims...), true)
end
    end
  elseif isArrayType(elem_typ)
    arr = Array{elem_typ}(dims...)
    sub_type = elem_typ.parameters[1]
    sub_dim  = elem_typ.parameters[2]
    for i = 1:len
      ptr = j2c_array_get(inp, i, Ptr{Void})
      arr[i] = _from_j2c_array(ptr, sub_type, sub_dim, ptr_array_dict)
    end
  end
  return arr
end

function from_j2c_array(inp::Ptr{Void}, elem_typ::DataType, N::Int, ptr_array_dict :: Dict{Ptr{Void}, Array})
   arr = _from_j2c_array(inp, elem_typ, N, ptr_array_dict)
   j2c_array_delete(inp)
   return arr
end 

end 
