#include <stdint.h>
#include "julia.h"
#include "j2c-array.h"

extern "C" DLLEXPORT
int j2c_array_bytesize()
{
    return sizeof(j2c_array<uintptr_t>);
}

extern "C" DLLEXPORT
void *j2c_array_new(int elem_bytes, void *data, unsigned ndim, int64_t *dims)
{
    void *a = NULL;
    switch (elem_bytes)
    {
    case 0: // special case for array of array
        a = new j2c_array<j2c_array<uintptr_t> >((j2c_array<uintptr_t>*)data, ndim, dims);
        break;
    case 1:
        a = new j2c_array<int8_t>((int8_t*)data, ndim, dims);
        break;
    case 2:
        a = new j2c_array<int16_t>((int16_t*)data, ndim, dims);
        break;
    case 4:
        a = new j2c_array<int32_t>((int32_t*)data, ndim, dims);
        break;
    case 8:
        a = new j2c_array<int64_t>((int64_t*)data, ndim, dims);
        break;
    default:
        assert(false);
        break;
    }
    return a;
}

/* own means the caller wants to own the pointer */
extern "C" DLLEXPORT
void* j2c_array_to_pointer(void *arr, bool own)
{
   if (own) {
     return ((j2c_array<void*>*)arr)->being_returned();
   }
   else {
     return ((j2c_array<void*>*)arr)->data;
   }
}

extern "C" DLLEXPORT
unsigned j2c_array_length(void *arr)
{
    return ((j2c_array<int8_t>*)arr)->ARRAYLEN();
}

extern "C" DLLEXPORT
unsigned j2c_array_size(void *arr, unsigned dim)
{
    return ((j2c_array<int8_t>*)arr)->ARRAYSIZE(dim);
}

/* In case that elem_bytes is 0, value is set to a pointer into the data of
 * the input array without any copying. */
extern "C" DLLEXPORT
void j2c_array_get(int elem_bytes, void *arr, unsigned idx, void *value)
{
    switch (elem_bytes)
    {
    case 0:
        ((j2c_array<uintptr_t>**)value)[0] = ((j2c_array<j2c_array<uintptr_t> >*)arr)->ARRAYELEMREF(idx);
        break;
    case 1:
        ((int8_t*)value)[0] = ((j2c_array<int8_t>*)arr)->ARRAYELEM(idx);
        break;
    case 2:
        ((int16_t*)value)[0] = ((j2c_array<int16_t>*)arr)->ARRAYELEM(idx);
        break;
    case 4:
        ((int32_t*)value)[0] = ((j2c_array<int32_t>*)arr)->ARRAYELEM(idx);
        break;
    case 8:
        ((int64_t*)value)[0] = ((j2c_array<int64_t>*)arr)->ARRAYELEM(idx);
        break;
    default:
        assert(false);
        break;
    }
}

/* in the case elem_bytes = 0, value is a pointer to a j2c_array object */
extern "C" DLLEXPORT
void j2c_array_set(int elem_bytes, void *arr, unsigned idx, void *value)
{
    switch (elem_bytes)
    {
    case 0:
        ((j2c_array<j2c_array<uintptr_t> >*)arr)->ARRAYELEM(idx) = *(j2c_array<uintptr_t>*)value;
        break;
    case 1:
        ((j2c_array<int8_t>*)arr)->ARRAYELEM(idx) = ((int8_t*)value)[0];
        break;
    case 2:
        ((j2c_array<int16_t>*)arr)->ARRAYELEM(idx) = ((int16_t*)value)[0];
        break;
    case 4:
        ((j2c_array<int32_t>*)arr)->ARRAYELEM(idx) = ((int32_t*)value)[0];
        break;
    case 8:
        ((j2c_array<int64_t>*)arr)->ARRAYELEM(idx) = ((int64_t*)value)[0];
        break;
    default:
        assert(false);
        break;
    }
}

/* Only delete this array object, no nested deletion */
extern "C" DLLEXPORT
void j2c_array_delete(void* a)
{
    delete ((j2c_array<int8_t>*)a);
}

/* Deref without deletion */
extern "C" DLLEXPORT
void j2c_array_deref(void* a)
{
    ((j2c_array<int8_t>*)a)->decrement();
}
