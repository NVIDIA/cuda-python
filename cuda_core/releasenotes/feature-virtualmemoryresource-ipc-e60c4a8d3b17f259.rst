:class:`VirtualMemoryResource` buffers can now be shared with other
processes the way pool-backed buffers are. :attr:`Buffer.ipc_descriptor`
exports every physical allocation that backs a :class:`VirtualMemoryBuffer`
as a POSIX file descriptor, :meth:`Buffer.from_ipc_descriptor` imports the
descriptor with a ``VirtualMemoryResource`` of the receiving process for the
device that owns the memory, and a buffer or its descriptor can be sent
through ``multiprocessing`` directly (plain ``pickle`` cannot carry file
descriptors). :attr:`VirtualMemoryResource.is_ipc_enabled` is ``True`` when
the resource's ``handle_type`` is ``"posix_fd"`` on Linux and the memory
lives on a device; host-located virtual memory is not shared yet. A grown buffer
imports as one contiguous range. A descriptor holds one file descriptor per
physical allocation and pins the memory while it exists; nothing else does.
A buffer that contains imported memory cannot be exported again. For a
transport of your own, the descriptor exposes ``fds``, ``sizes``, and
``size``, and ``VirtualMemoryIPCBufferDescriptor.from_fds()`` rebuilds it
in the receiver. Fabric handles are not shared yet.
(`#2980 <https://github.com/NVIDIA/cuda-python/issues/2980>`__) (#3028)
