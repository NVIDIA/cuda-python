# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport uintptr_t
from libc.string cimport memset
from libcpp.vector cimport vector

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport (
    Buffer,
    Buffer_check_open,
    Buffer_from_deviceptr_handle,
    MemoryResource,
)
from cuda.core._rt cimport (
    ContextHandle,
    DevicePtrHandle,
    MemAllocationHandle,
    StreamHandle,
    VaMappingHandle,
    VaReservationHandle,
    VmmRangeHandle,
    as_cu,
    create_context_bound_legacy_stream,
    create_mem_allocation_handle,
    create_va_mapping_handle,
    create_va_reservation_handle,
    create_vmm_range,
    deallocation_stream,
    deviceptr_create_ref,
    deviceptr_create_vmm,
    get_last_error,
    get_primary_context,
    mem_allocation_size,
    set_deallocation_stream,
    va_mapping_allocation,
    vmm_range,
    vmm_range_append,
    vmm_range_count,
    vmm_range_mapping,
    vmm_range_reserve,
    vmm_range_total,
)
from cuda.core._stream cimport Stream, Stream_accept, Stream_is_default_token
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN, check_or_create_options

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from cuda.core._device import Device
from cuda.core._utils.cuda_utils import driver
from cuda.core._utils.version import binding_version
from cuda.core.typing import (
    DevicePointerType,
    VirtualMemoryAccessType,
    VirtualMemoryAllocationType,
    VirtualMemoryGranularityType,
    VirtualMemoryHandleType,
    VirtualMemoryLocationType,
)

if TYPE_CHECKING:
    from cuda.core.graph import GraphBuilder

__all__ = ["VirtualMemoryBuffer", "VirtualMemoryResource", "VirtualMemoryResourceOptions"]

# Location types whose physical backing lives in host memory. Shared by
# VirtualMemoryResource.__init__ and is_host_accessible so the two cannot drift.
_HOST_LOCATION_TYPES = frozenset(
    {
        VirtualMemoryLocationType.HOST,
        VirtualMemoryLocationType.HOST_NUMA,
        VirtualMemoryLocationType.HOST_NUMA_CURRENT,
    }
)


@dataclass
class VirtualMemoryResourceOptions:
    """A configuration object for the VirtualMemoryResource
       Stores configuration information which tells the resource how to use the CUDA VMM APIs

    Attributes
    ----------
    allocation_type: :obj:`~_memory.VirtualMemoryAllocationType` | str
        Controls the type of allocation.
    location_type: :obj:`~_memory.VirtualMemoryLocationType` | str
        Controls the location of the allocation.
    handle_type: :obj:`~_memory.VirtualMemoryHandleType` | str
        Export handle type for the physical allocation. Use ``"posix_fd"`` on
        Linux if you plan to import/export the allocation. Use `None` if you
        don't need an exportable handle. Host-located allocations require
        `None`.
    gpu_direct_rdma: bool
        Hint that the allocation should be GDR-capable (if supported).
    granularity: :obj:`~_memory.VirtualMemoryGranularityType` | str
        Controls granularity query and size rounding.
    addr_hint: int
        A (optional) virtual address hint to try to reserve at. Setting it to 0 lets the CUDA driver decide.
    addr_align: int
        Alignment for the VA reservation. If `None`, use the queried granularity.
    peers: Iterable[int]
        Extra device IDs that should be granted access in addition to ``device``.
    self_access: :obj:`~_memory.VirtualMemoryAccessType` | None | str
        Access flags for the owning device.
    peer_access: :obj:`~_memory.VirtualMemoryAccessType` | None | str
        Access flags for peers.
    """

    allocation_type: VirtualMemoryAllocationType = VirtualMemoryAllocationType.PINNED
    location_type: VirtualMemoryLocationType = VirtualMemoryLocationType.DEVICE
    handle_type: VirtualMemoryHandleType = VirtualMemoryHandleType.POSIX_FD
    granularity: VirtualMemoryGranularityType = VirtualMemoryGranularityType.RECOMMENDED
    gpu_direct_rdma: bool = False
    addr_hint: int | None = 0
    addr_align: int | None = None
    peers: Iterable[int] = field(default_factory=tuple)
    self_access: VirtualMemoryAccessType = VirtualMemoryAccessType.READ_WRITE
    peer_access: VirtualMemoryAccessType = VirtualMemoryAccessType.READ_WRITE

    _a = driver.CUmemAccess_flags
    _access_flags = {  # noqa: RUF012
        VirtualMemoryAccessType.READ_WRITE: _a.CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        VirtualMemoryAccessType.READ: _a.CU_MEM_ACCESS_FLAGS_PROT_READ,
        None: 0,
    }
    _h = driver.CUmemAllocationHandleType
    _handle_types = {  # noqa: RUF012
        None: _h.CU_MEM_HANDLE_TYPE_NONE,
        VirtualMemoryHandleType.POSIX_FD: _h.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        VirtualMemoryHandleType.WIN32_KMT: _h.CU_MEM_HANDLE_TYPE_WIN32_KMT,
        VirtualMemoryHandleType.FABRIC: _h.CU_MEM_HANDLE_TYPE_FABRIC,
    }
    _g = driver.CUmemAllocationGranularity_flags
    _granularity = {  # noqa: RUF012
        VirtualMemoryGranularityType.RECOMMENDED: _g.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        VirtualMemoryGranularityType.MINIMUM: _g.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
    }
    _l = driver.CUmemLocationType
    _location_type = {  # noqa: RUF012
        VirtualMemoryLocationType.DEVICE: _l.CU_MEM_LOCATION_TYPE_DEVICE,
        VirtualMemoryLocationType.HOST: _l.CU_MEM_LOCATION_TYPE_HOST,
        VirtualMemoryLocationType.HOST_NUMA: _l.CU_MEM_LOCATION_TYPE_HOST_NUMA,
        VirtualMemoryLocationType.HOST_NUMA_CURRENT: _l.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
    }
    _t = driver.CUmemAllocationType
    # CUDA 13+ exposes MANAGED in CUmemAllocationType; older 12.x does not
    _allocation_type = {VirtualMemoryAllocationType.PINNED: _t.CU_MEM_ALLOCATION_TYPE_PINNED}  # noqa: RUF012
    if binding_version() >= (13, 0, 0):
        _allocation_type[VirtualMemoryAllocationType.MANAGED] = _t.CU_MEM_ALLOCATION_TYPE_MANAGED

    @staticmethod
    def _access_to_flags(spec: VirtualMemoryAccessType | None) -> int:
        flags = VirtualMemoryResourceOptions._access_flags.get(spec)
        if flags is None:
            raise ValueError(f"Unknown access spec: {spec!r}")
        return flags  # type: ignore[no-any-return]

    @staticmethod
    def _allocation_type_to_driver(spec: VirtualMemoryAllocationType) -> int:
        alloc_type = VirtualMemoryResourceOptions._allocation_type.get(spec)
        if alloc_type is None:
            raise ValueError(f"Unsupported allocation_type: {spec!r}")
        return alloc_type  # type: ignore[no-any-return]

    @staticmethod
    def _location_type_to_driver(spec: VirtualMemoryLocationType) -> int:
        loc_type = VirtualMemoryResourceOptions._location_type.get(spec)
        if loc_type is None:
            raise ValueError(f"Unsupported location_type: {spec!r}")
        return loc_type  # type: ignore[no-any-return]

    @staticmethod
    def _handle_type_to_driver(spec: VirtualMemoryHandleType | None) -> int:
        if spec == "win32":
            raise NotImplementedError("win32 is currently not supported, please reach out to the CUDA Python team")
        handle_type = VirtualMemoryResourceOptions._handle_types.get(spec)
        if handle_type is None:
            raise ValueError(f"Unsupported handle_type: {spec!r}")
        return handle_type  # type: ignore[no-any-return]

    @staticmethod
    def _granularity_to_driver(spec: VirtualMemoryGranularityType) -> int:
        granularity = VirtualMemoryResourceOptions._granularity.get(spec)
        if granularity is None:
            raise ValueError(f"Unsupported granularity: {spec!r}")
        return granularity  # type: ignore[no-any-return]


cdef inline size_t _align_up(size_t size, size_t gran) noexcept nogil:
    return (size + gran - 1) // gran * gran


cdef inline bint _is_default_token(cydriver.CUstream s) noexcept nogil:
    cdef uintptr_t h = <uintptr_t>s
    return h == 0 or h == <uintptr_t>cydriver.CU_STREAM_LEGACY or h == <uintptr_t>cydriver.CU_STREAM_PER_THREAD


cdef bint _stream_is_capturing(cydriver.CUstream s) except -1:
    cdef cydriver.CUstreamCaptureStatus cap_status
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        HANDLE_RETURN(cydriver.cuStreamGetCaptureInfo(s, &cap_status, NULL, NULL, NULL, NULL, NULL))
    ELSE:
        HANDLE_RETURN(cydriver.cuStreamGetCaptureInfo(s, &cap_status, NULL, NULL, NULL, NULL))
    return cap_status == cydriver.CU_STREAM_CAPTURE_STATUS_ACTIVE


cdef int _raise_last_error() except -1:
    """Raise the status a handle factory recorded when it returned empty."""
    HANDLE_RETURN(get_last_error())
    raise RuntimeError(
        "internal cuda.core error, please report: a virtual memory handle factory "
        "returned an empty handle without recording a CUDA error"
    )


cdef class VirtualMemoryBuffer(Buffer):
    """A :class:`Buffer` returned by :class:`VirtualMemoryResource`.

    The buffer owns its address reservations, physical allocations and
    mappings through its device pointer handle; closing it is the only way
    to release them. A buffer returned by
    :meth:`VirtualMemoryResource.modify_allocation` aliases the buffer it
    was grown from: the two share their physical memory, and that memory is
    freed when the last buffer that maps it closes.
    """

    def close(self, stream: Stream | GraphBuilder | None = None) -> None:
        """Release this buffer's share of its address range.

        The mappings, reservations and physical allocations go away when the
        last buffer that maps them closes. Before it unmaps, the resource
        synchronizes every deallocation stream the buffers of the range
        recorded. Virtual memory deallocation is synchronous and cannot be
        captured, so closing on a capturing stream raises and leaves the
        buffer open.

        Parameters
        ----------
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`, optional
            If given, replaces the recorded deallocation stream, as for
            :meth:`Buffer.close`.
        """
        cdef Stream s
        cdef StreamHandle h
        cdef cydriver.CUstream raw
        if not self._h_ptr:
            return
        if stream is not None:
            s = Stream_accept(stream)
            raw = as_cu(s._h_stream)
        else:
            h = deallocation_stream(self._h_ptr)
            raw = as_cu(h)
        # Default-stream tokens are checked by the range deleter under their
        # bound context; a real stream can be checked here and refused.
        if not _is_default_token(raw) and _stream_is_capturing(raw):
            raise RuntimeError(
                "cannot close a VirtualMemoryResource buffer on a capturing stream: "
                "virtual memory deallocation is synchronous and cannot be captured"
            )
        Buffer.close(self, stream)


cdef class VirtualMemoryResource(MemoryResource):
    """Create a device memory resource that uses the CUDA VMM APIs to allocate memory.

    Parameters
    ----------
    device_id : Device | int
        Device for which a memory resource is constructed.

    config : VirtualMemoryResourceOptions, optional
        A configuration object for the VirtualMemoryResource


    Warning
    -------
        This is a low-level API that is provided only for convenience. Make sure you fully understand
        how CUDA Virtual Memory Management works before using this. Other MemoryResource subclasses
        in cuda.core should already meet the common needs.

    Notes
    -----
    Every buffer this resource returns is a :class:`VirtualMemoryBuffer` that
    owns its address reservations, physical allocations and mappings; closing
    the buffer releases them. :meth:`deallocate` is not involved in that path.
    """

    cdef:
        public object device
        public object config

    def __init__(self, device_id: Device | int, config: VirtualMemoryResourceOptions | None = None) -> None:
        self.device = Device(device_id)
        self.config = check_or_create_options(
            VirtualMemoryResourceOptions, config, "VirtualMemoryResource options", keep_none=False
        )
        if self.config.location_type in _HOST_LOCATION_TYPES:
            self.device = None
            # The driver rejects an exportable handle type for host memory.
            if self.config.handle_type is not None:
                raise ValueError(
                    "host-located virtual memory cannot have an exportable handle type; "
                    "pass handle_type=None"
                )

        if self.device is not None and not self.device.properties.virtual_memory_management_supported:
            raise RuntimeError("VirtualMemoryResource requires CUDA VMM API support")

        # Validate RDMA support if requested
        if (
            self.config.gpu_direct_rdma
            and self.device is not None
            and not self.device.properties.gpu_direct_rdma_supported
        ):
            raise RuntimeError("GPU Direct RDMA is not supported on this device")

    cdef int _fill_prop(self, object cfg, cydriver.CUmemAllocationProp* prop) except -1:
        # The location comes from the resource; the rest may come from a
        # per-call configuration.
        cdef int alloc_type = int(VirtualMemoryResourceOptions._allocation_type_to_driver(cfg.allocation_type))
        cdef int loc_type = int(VirtualMemoryResourceOptions._location_type_to_driver(self.config.location_type))
        cdef int handle_type = int(VirtualMemoryResourceOptions._handle_type_to_driver(cfg.handle_type))
        memset(prop, 0, sizeof(cydriver.CUmemAllocationProp))
        prop.type = <cydriver.CUmemAllocationType>alloc_type
        prop.location.type = <cydriver.CUmemLocationType>loc_type
        prop.location.id = self.device.device_id if self.device is not None else -1
        prop.allocFlags.gpuDirectRDMACapable = 1 if cfg.gpu_direct_rdma else 0
        prop.requestedHandleTypes = <cydriver.CUmemAllocationHandleType>handle_type
        prop.win32HandleMetaData = NULL
        return 0

    cdef int _fill_access(
        self, object cfg, const cydriver.CUmemAllocationProp* prop,
        vector[cydriver.CUmemAccessDesc]& descs,
    ) except -1:
        cdef cydriver.CUmemAccessDesc d
        cdef int owner_flags = int(VirtualMemoryResourceOptions._access_to_flags(cfg.self_access))
        cdef int peer_flags = int(VirtualMemoryResourceOptions._access_to_flags(cfg.peer_access))
        if owner_flags:
            memset(&d, 0, sizeof(d))
            d.location.type = prop.location.type
            d.location.id = prop.location.id
            d.flags = <cydriver.CUmemAccess_flags>owner_flags
            descs.push_back(d)
        if peer_flags:
            for peer_dev in cfg.peers:
                memset(&d, 0, sizeof(d))
                d.location.type = cydriver.CU_MEM_LOCATION_TYPE_DEVICE
                d.location.id = int(peer_dev)
                d.flags = <cydriver.CUmemAccess_flags>peer_flags
                descs.push_back(d)
        return 0

    cdef size_t _granularity(self, object cfg, const cydriver.CUmemAllocationProp* prop) except? 0:
        cdef size_t gran = 0
        cdef int flag = int(VirtualMemoryResourceOptions._granularity_to_driver(cfg.granularity))
        with nogil:
            HANDLE_RETURN(cydriver.cuMemGetAllocationGranularity(
                &gran, prop, <cydriver.CUmemAllocationGranularity_flags>flag))
        return gran

    cdef int _record_deallocation_stream(self, const DevicePtrHandle& h_ptr, Stream s) except -1:
        cdef StreamHandle h
        cdef ContextHandle h_ctx
        if s is not None and not Stream_is_default_token(s):
            h = s._h_stream
        elif self.device is None:
            # Host-located memory needs no context to free, and a default-stream
            # token has no context to bind to: record nothing.
            return 0
        else:
            # Bind the legacy default-stream token to the device's primary
            # context so the free is ordered correctly no matter what is current
            # then, and so allocate() never depends on a current context.
            # get_primary_context keeps its own cache; nothing is cached here.
            h_ctx = get_primary_context(self.device.device_id)
            if not h_ctx:
                _raise_last_error()
            h = create_context_bound_legacy_stream(h_ctx)
            if not h:
                _raise_last_error()
        HANDLE_RETURN(set_deallocation_stream(h_ptr, h))
        return 0

    cdef int _copy_deallocation_stream(self, const DevicePtrHandle& dst, const DevicePtrHandle& src) except -1:
        cdef StreamHandle h = deallocation_stream(src)
        if h:
            HANDLE_RETURN(set_deallocation_stream(dst, h))
        return 0

    def allocate(self, size_t size, *, stream: Stream | GraphBuilder | None = None) -> VirtualMemoryBuffer:
        """
        Allocate a buffer of the given size using CUDA virtual memory.

        Parameters
        ----------
        size : int
            The size in bytes of the buffer to allocate. It is rounded up to the
            allocation granularity; the returned buffer reports the rounded size.
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`, optional
            Keyword-only. The allocation itself is synchronous. A real stream is
            recorded as the buffer's deallocation stream and synchronized when
            the buffer closes; with `None` or a default-stream token the legacy
            default stream of the resource's device is recorded instead.

        Returns
        -------
        VirtualMemoryBuffer
            A buffer that owns its reservation, physical allocation and mapping.

        Raises
        ------
        CUDAError
            If any CUDA driver API call fails during allocation. Nothing is
            left allocated when this method raises.
        """
        cdef Stream s = None
        if stream is not None:
            s = Stream_accept(stream)
        return self._allocate(self.config, size, s)

    cdef Buffer _allocate(self, object cfg, size_t size, Stream s):
        """Allocate ``size`` bytes with ``cfg``; ``s`` is the accepted stream or None."""
        cdef cydriver.CUmemAllocationProp prop
        cdef vector[cydriver.CUmemAccessDesc] descs
        cdef size_t gran, aligned, addr_align
        cdef cydriver.CUdeviceptr hint
        cdef MemAllocationHandle h_alloc
        cdef VaReservationHandle h_res
        cdef VaMappingHandle h_map
        cdef VmmRangeHandle rng
        cdef DevicePtrHandle h_ptr

        if size == 0:
            # Nothing to reserve or map; an empty buffer with a non-owning handle.
            return Buffer_from_deviceptr_handle(deviceptr_create_ref(0), 0, self, None, VirtualMemoryBuffer)

        self._fill_prop(cfg, &prop)
        self._fill_access(cfg, &prop, descs)
        gran = self._granularity(cfg, &prop)
        aligned = _align_up(size, gran)
        addr_align = cfg.addr_align or gran
        hint = cfg.addr_hint or 0

        # Every handle below is a local: if a later step fails, the locals die
        # in reverse order and undo everything created so far.
        with nogil:
            h_alloc = create_mem_allocation_handle(aligned, prop, descs.data(), descs.size())
        if not h_alloc:
            _raise_last_error()
        with nogil:
            h_res = create_va_reservation_handle(aligned, addr_align, hint)
        if not h_res:
            _raise_last_error()
        with nogil:
            h_map = create_va_mapping_handle(as_cu(h_res), h_alloc, h_res)
        if not h_map:
            _raise_last_error()

        rng = create_vmm_range(as_cu(h_res))
        vmm_range_append(rng, h_map)
        h_ptr = deviceptr_create_vmm(as_cu(h_res), rng)
        if not h_ptr:
            _raise_last_error()
        self._record_deallocation_stream(h_ptr, s)
        return Buffer_from_deviceptr_handle(h_ptr, aligned, self, None, VirtualMemoryBuffer)

    def modify_allocation(
        self, buf: Buffer, size_t new_size, config: VirtualMemoryResourceOptions | None = None
    ) -> VirtualMemoryBuffer:
        """
        Grow a buffer of this resource to at least ``new_size`` bytes.

        The buffer passed in stays open and usable. The returned buffer aliases
        it: both map the same physical memory, which is freed when the last of
        the two closes. When the driver can extend the address range in place,
        the returned buffer has the same pointer; otherwise it has a new one and
        the existing contents are reachable through both.

        This method is not thread-safe with respect to two buffers that share an
        address range.

        Parameters
        ----------
        buf : VirtualMemoryBuffer
            A buffer returned by :meth:`allocate` or by this method.
        new_size : int
            The requested total size in bytes; rounded up to the granularity.
        config : VirtualMemoryResourceOptions, optional
            Configuration for the new physical memory chunk only. Existing
            chunks keep the access they were created with, and the resource's
            own configuration is unchanged.

        Returns
        -------
        VirtualMemoryBuffer
            ``buf`` itself when it already covers ``new_size``; otherwise a new
            buffer of the rounded size.

        Raises
        ------
        TypeError
            If ``buf`` did not come from this resource.
        CUDAError
            If a driver call fails. ``buf`` is untouched when this method raises.
        """
        cdef Buffer b
        cdef VmmRangeHandle rng, rng_new
        cdef object cfg
        cdef cydriver.CUmemAllocationProp prop
        cdef vector[cydriver.CUmemAccessDesc] descs
        cdef size_t gran, req, total, add, count, addr_align, offset, i, chunk
        cdef cydriver.CUdeviceptr base, base_new
        cdef MemAllocationHandle h_alloc, a
        cdef VaReservationHandle h_res, h_res_new
        cdef VaMappingHandle h_map, m, m2
        cdef DevicePtrHandle h_ptr2, h_ptr_new
        cdef object new_buf

        if not isinstance(buf, Buffer):
            raise TypeError(f"buf must be a Buffer, got {type(buf).__name__}")
        b = <Buffer>buf
        Buffer_check_open(b)
        if b.memory_resource is not self:
            raise TypeError("buf was not allocated by this VirtualMemoryResource")
        cfg = self.config if config is None else check_or_create_options(
            VirtualMemoryResourceOptions, config, "VirtualMemoryResource options", keep_none=False
        )
        if b._size == 0:
            # An empty buffer maps nothing; the request is a fresh allocation.
            return self._allocate(cfg, new_size, None)
        rng = vmm_range(b._h_ptr)
        if not rng:
            raise TypeError("buf was not allocated by VirtualMemoryResource.allocate")

        self._fill_prop(cfg, &prop)
        self._fill_access(cfg, &prop, descs)
        gran = self._granularity(cfg, &prop)
        req = _align_up(new_size, gran)
        total = vmm_range_total(rng)
        base = as_cu(b._h_ptr)

        if req <= b._size:
            return buf
        if req <= total:
            # A shorter alias asking for what the range already maps.
            h_ptr2 = deviceptr_create_vmm(base, rng)
            if not h_ptr2:
                _raise_last_error()
            self._copy_deallocation_stream(h_ptr2, b._h_ptr)
            return Buffer_from_deviceptr_handle(h_ptr2, req, self, None, VirtualMemoryBuffer)

        # The new chunk is a whole number of granules; the result covers it.
        add = _align_up(req - total, gran)
        req = total + add
        count = vmm_range_count(rng)

        # Grow in place: reserve the range right after the current one. The
        # driver raises alignment 0 to its default, so the hint is well formed.
        with nogil:
            h_res = create_va_reservation_handle(add, 0, base + total)
        if not h_res:
            get_last_error()  # the probe may fail; that is not an error here
        elif as_cu(h_res) != base + total:
            h_res.reset()  # granted elsewhere: free it and move instead
        else:
            with nogil:
                h_alloc = create_mem_allocation_handle(add, prop, descs.data(), descs.size())
            if not h_alloc:
                _raise_last_error()
            with nogil:
                h_map = create_va_mapping_handle(base + total, h_alloc, h_res)
            if not h_map:
                _raise_last_error()
            vmm_range_reserve(rng, count + 1)
            h_ptr2 = deviceptr_create_vmm(base, rng)
            if not h_ptr2:
                _raise_last_error()
            self._copy_deallocation_stream(h_ptr2, b._h_ptr)
            new_buf = Buffer_from_deviceptr_handle(h_ptr2, req, self, None, VirtualMemoryBuffer)
            # Last step, and it cannot throw after the reserve above: the input
            # buffer is untouched if anything before this raised.
            vmm_range_append(rng, h_map)
            return new_buf

        # Move: a new range that maps every existing allocation, then the new one.
        # The allocations are shared with the input buffer's range.
        addr_align = cfg.addr_align or gran
        with nogil:
            h_res_new = create_va_reservation_handle(req, addr_align, 0)
        if not h_res_new:
            _raise_last_error()
        base_new = as_cu(h_res_new)
        rng_new = create_vmm_range(base_new)
        vmm_range_reserve(rng_new, count + 1)
        offset = 0
        for i in range(count):
            m = vmm_range_mapping(rng, i)
            a = va_mapping_allocation(m)
            chunk = mem_allocation_size(a)
            with nogil:
                m2 = create_va_mapping_handle(base_new + offset, a, h_res_new)
            if not m2:
                _raise_last_error()
            vmm_range_append(rng_new, m2)
            offset += chunk
        with nogil:
            h_alloc = create_mem_allocation_handle(add, prop, descs.data(), descs.size())
        if not h_alloc:
            _raise_last_error()
        with nogil:
            m2 = create_va_mapping_handle(base_new + offset, h_alloc, h_res_new)
        if not m2:
            _raise_last_error()
        vmm_range_append(rng_new, m2)
        h_ptr_new = deviceptr_create_vmm(base_new, rng_new)
        if not h_ptr_new:
            _raise_last_error()
        self._copy_deallocation_stream(h_ptr_new, b._h_ptr)
        return Buffer_from_deviceptr_handle(h_ptr_new, req, self, None, VirtualMemoryBuffer)

    def deallocate(self, ptr: DevicePointerType, size: int, *, stream: Stream | GraphBuilder | None = None) -> None:
        """
        Unmap and free one address range that was reserved and mapped outside this resource.

        Buffers returned by :meth:`allocate` and :meth:`modify_allocation` free
        themselves when they close and never call this method. It exists for
        raw pointers wrapped with :meth:`Buffer.from_handle` with ``mr`` set to
        this resource: the range must be exactly one reservation, and the caller
        must already have released its own ``cuMemCreate`` handle, so the
        physical memory is freed by the unmap.

        Parameters
        ----------
        ptr : DevicePointerType
            The start of the reservation.
        size : int
            The size of the reservation in bytes.
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`, optional
            Keyword-only. If given, ``stream.sync()`` is called before the
            range is unmapped, except for a default-stream token on a
            host-located resource, which has no context to synchronize in.
        """
        cdef cydriver.CUdeviceptr devptr = 0 if ptr is None else <cydriver.CUdeviceptr>int(ptr)
        cdef size_t nbytes = size
        cdef Stream s
        if stream is not None:
            s = Stream_accept(stream)
            # A host-located resource records no stream, so Buffer teardown
            # passes an unbound default-stream token, which has no context to
            # synchronize in. There is nothing queued on it to wait for.
            if self.device is not None or not Stream_is_default_token(s):
                s.sync()
        if devptr == 0 or nbytes == 0:
            return
        with nogil:
            HANDLE_RETURN(cydriver.cuMemUnmap(devptr, nbytes))
            HANDLE_RETURN(cydriver.cuMemAddressFree(devptr, nbytes))

    @property
    def is_device_accessible(self) -> bool:
        """
        Indicates whether the allocated memory is accessible from the device.
        """
        return self.config.location_type == "device"

    @property
    def is_host_accessible(self) -> bool:
        """
        Indicates whether the allocated memory is accessible from the host.
        """
        return self.config.location_type in _HOST_LOCATION_TYPES

    @property
    def is_ipc_enabled(self) -> bool:
        """Return False. Buffers of this resource cannot be shared through IPC descriptors."""
        return False

    @property
    def device_id(self) -> int:
        """
        Get the device ID associated with this memory resource.

        Returns:
            int: CUDA device ID. -1 if the memory resource allocates host memory
        """
        return self.device.device_id if self.device is not None else -1

    def __repr__(self) -> str:
        """
        Return a string representation of the VirtualMemoryResource.

        Returns:
            str: A string describing the object
        """
        return f"<VirtualMemoryResource device={self.device}>"
