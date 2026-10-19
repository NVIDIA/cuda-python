The behavior of :class:`VirtualMemoryResource` changes as follows.
:meth:`VirtualMemoryResource.modify_allocation` returns a new
:class:`VirtualMemoryBuffer` and leaves the buffer passed in open; the two
alias the same physical memory, which is freed when the last of them closes,
and the pointer is preserved when the driver grants the adjacent address
range. The result is always a new buffer, so closing it never closes the
buffer passed in; a request the buffer already covers returns a full alias
without a driver call, and :attr:`Buffer.size` after a grow is the aligned
total. The ``config`` argument applies to the chunk that call adds and is no
longer stored on the resource
(`#2909 <https://github.com/NVIDIA/cuda-python/issues/2909>`__); it must
keep the resource's location and ``handle_type`` and passes the constructor's
option checks. Every chunk of a buffer must be exportable the same way, so a
resource created with ``handle_type=None`` needs ``handle_type=None`` in a
per-call ``config`` as well. It never changes the access of memory that is already mapped, so a request the
buffer already covers has no effect even with ``config``. Buffers from
``allocate()`` free themselves when they close and no longer call
:meth:`VirtualMemoryResource.deallocate`, which now serves pointers wrapped
with :meth:`Buffer.from_handle`. A buffer records the stream passed to
``allocate()``, and closing it waits for the work queued on that stream
before it unmaps, because ``cuMemUnmap`` is not stream-ordered; close the
buffer explicitly, or record an idle deallocation stream, to control when
that wait happens. ``modify_allocation`` accepts only buffers this resource
returned. ``location_type="host"`` requires ``handle_type=None``, which the
driver requires. ``allocate(0)`` returns an empty buffer without a driver
call. (#2917)
