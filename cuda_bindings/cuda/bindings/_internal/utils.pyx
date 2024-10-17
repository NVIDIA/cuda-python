cimport cpython
from libc.stdint cimport intptr_t
from libcpp.utility cimport move
from cython.operator cimport dereference as deref


cdef bint is_nested_sequence(data):
    if not cpython.PySequence_Check(data):
        return False
    else:
        for i in data:
            if not cpython.PySequence_Check(i):
                return False
        else:
            return True


cdef void* get_buffer_pointer(buf, Py_ssize_t size, readonly=True) except*:
    """The caller must ensure ``buf`` is alive when the returned pointer is in use."""
    cdef void* bufPtr
    cdef int flags = cpython.PyBUF_ANY_CONTIGUOUS
    if not readonly:
        flags |= cpython.PyBUF_WRITABLE
    cdef int status = -1
    cdef cpython.Py_buffer view

    if isinstance(buf, int):
        bufPtr = <void*><intptr_t>buf
    else:  # try buffer protocol
        try:
            status = cpython.PyObject_GetBuffer(buf, &view, flags)
            assert view.len == size
            assert view.ndim == 1
        except Exception as e:
            adj = "writable " if not readonly else ""
            raise ValueError(
                 "buf must be either a Python int representing the pointer "
                f"address to a valid buffer, or a 1D contiguous {adj}"
                 "buffer, of size bytes") from e
        else:
            bufPtr = view.buf
        finally:
            if status == 0:
                cpython.PyBuffer_Release(&view)

    return bufPtr


# Cython can't infer the overload by return type alone, so we need a dummy
# input argument to help it
cdef nullable_unique_ptr[ vector[ResT] ] get_resource_ptr_(object obj, ResT* __unused):
    cdef nullable_unique_ptr[ vector[ResT] ] ptr
    cdef vector[ResT]* vec
    if cpython.PySequence_Check(obj):
        vec = new vector[ResT](len(obj))
        for i in range(len(obj)):
            deref(vec)[i] = obj[i]
        ptr.reset(vec, True)
    else:
        ptr.reset(<vector[ResT]*><intptr_t>obj, False)
    return move(ptr)

cdef int get_resource_ptr(nullable_unique_ptr[vector[ResT]] &in_out_ptr, object obj, ResT* __unused) except 0:
    cdef vector[ResT]* vec
    if cpython.PySequence_Check(obj):
        vec = new vector[ResT](len(obj))
        # set the ownership immediately to avoid
        # leaking the `vec` memory in case of exception 
        # (e.g. ResT type range overflow)
        # when populating the memory in the loop
        in_out_ptr.reset(vec, True)
        for i in range(len(obj)):
            deref(vec)[i] = obj[i]
    else:
        in_out_ptr.reset(<vector[ResT]*><intptr_t>obj, False)
    return 1


cdef nullable_unique_ptr[ vector[PtrT*] ] get_resource_ptrs(object obj, PtrT* __unused):
    cdef nullable_unique_ptr[ vector[PtrT*] ] ptr
    cdef vector[PtrT*]* vec
    if cpython.PySequence_Check(obj):
        vec = new vector[PtrT*](len(obj))
        for i in range(len(obj)):
            deref(vec)[i] = <PtrT*><intptr_t>(obj[i])
        ptr.reset(vec, True)
    else:
        ptr.reset(<vector[PtrT*]*><intptr_t>obj, False)
    return move(ptr)


cdef nested_resource[ResT] get_nested_resource_ptr(object obj, ResT* __unused):
    cdef nested_resource[ResT] res
    cdef nullable_unique_ptr[ vector[intptr_t] ] nested_ptr
    cdef nullable_unique_ptr[ vector[vector[ResT]] ] nested_res_ptr
    cdef vector[intptr_t]* nested_vec = NULL
    cdef vector[vector[ResT]]* nested_res_vec = NULL
    cdef size_t i = 0, length = 0
    cdef intptr_t addr

    if is_nested_sequence(obj):
        length = len(obj)
        nested_res_vec = new vector[vector[ResT]](length)
        nested_vec = new vector[intptr_t](length)
        for i, obj_i in enumerate(obj):
            deref(nested_res_vec)[i] = obj_i
            deref(nested_vec)[i] = <intptr_t>(deref(nested_res_vec)[i].data())
        nested_res_ptr.reset(nested_res_vec, True)
        nested_ptr.reset(nested_vec, True)
    elif cpython.PySequence_Check(obj):
        length = len(obj)
        nested_vec = new vector[intptr_t](length)
        for i, addr in enumerate(obj):
            deref(nested_vec)[i] = addr
        nested_res_ptr.reset(NULL, False)
        nested_ptr.reset(nested_vec, True)
    else:
        # obj is an int (ResT**)
        nested_res_ptr.reset(NULL, False)
        nested_ptr.reset(<vector[intptr_t]*><intptr_t>obj, False)

    res.ptrs = move(nested_ptr)
    res.nested_resource_ptr = move(nested_res_ptr)
    return move(res)


class FunctionNotFoundError(RuntimeError): pass

class NotSupportedError(RuntimeError): pass


cdef tuple get_nvjitlink_dso_version_suffix(int driver_ver):
    # applicable to both cuBLAS and cuBLASLt
    if 11000 <= driver_ver < 12000:
        return ('11', '')
    elif 12000 <= driver_ver < 13000:
        return ('12', '11', '')
    else:
        raise NotSupportedError('only CUDA 11/12 driver is supported')