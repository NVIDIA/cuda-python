from cuda.bindings cimport cydriver


cdef class Event:

    cdef:
        cydriver.CUevent _handle
        bint _timing_disabled
        bint _busy_waited
        int _device_id
        object _ctx_handle

    cpdef close(self)
