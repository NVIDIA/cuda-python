:class:`VirtualMemoryResource` now holds each physical allocation, address
reservation and mapping in a reference-counted handle, so everything a
buffer maps is released when the last buffer that maps it closes. The
physical memory was leaked on every ``allocate()`` and grow
(`#2344 <https://github.com/NVIDIA/cuda-python/issues/2344>`__), the address
range added by an in-place grow could not be freed
(`#2887 <https://github.com/NVIDIA/cuda-python/issues/2887>`__), a second
grow of a buffer failed
(`#2908 <https://github.com/NVIDIA/cuda-python/issues/2908>`__), an
unaligned grow recorded a size the driver rejects at close
(`#2907 <https://github.com/NVIDIA/cuda-python/issues/2907>`__), and a
buffer was unmapped without waiting for the work queued on its stream
(`#2886 <https://github.com/NVIDIA/cuda-python/issues/2886>`__). No
reservation is freed by hand any more
(`#2345 <https://github.com/NVIDIA/cuda-python/issues/2345>`__,
`#2388 <https://github.com/NVIDIA/cuda-python/issues/2388>`__). (#2917)
