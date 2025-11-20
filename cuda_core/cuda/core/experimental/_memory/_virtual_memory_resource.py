# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Literal, Union

from cuda.core.experimental._device import Device
from cuda.core.experimental._memory._buffer import Buffer, MemoryResource
from cuda.core.experimental._utils.cuda_utils import (
    Transaction,
    check_or_create_options,
    driver,
    get_binding_version,
)
from cuda.core.experimental._utils.cuda_utils import (
    _check_driver_error as raise_if_driver_error,
)

if TYPE_CHECKING:
    from cuda.core.experimental._graph import GraphBuilder
    from cuda.core.experimental._stream import Stream

__all__ = ["VirtualMemoryResourceOptions", "VirtualMemoryResource"]

VirtualMemoryHandleTypeT = Union[Literal["posix_fd", "generic", "win32", "win32_kmt", "fabric"], None]
VirtualMemoryLocationTypeT = Literal["device", "host", "host_numa", "host_numa_current"]
VirtualMemoryGranularityT = Literal["minimum", "recommended"]
VirtualMemoryAccessTypeT = Union[Literal["rw", "r"], None]
VirtualMemoryAllocationTypeT = Literal["pinned", "managed"]


@dataclass
class VirtualMemoryResourceOptions:
    """A configuration object for the VirtualMemoryResource
       Stores configuration information which tells the resource how to use the CUDA VMM APIs

    Attributes
    ----------
    allocation_type: :obj:`~_memory.VirtualMemoryAllocationTypeT`
        Controls the type of allocation.
    location_type: :obj:`~_memory.VirtualMemoryLocationTypeT`
        Controls the location of the allocation.
    handle_type: :obj:`~_memory.VirtualMemoryHandleTypeT`
        Export handle type for the physical allocation. Use
        ``"posix_fd"`` on Linux if you plan to
        import/export the allocation (required for cuMemRetainAllocationHandle).
        Use `None` if you don't need an exportable handle.
    gpu_direct_rdma: bool
        Hint that the allocation should be GDR-capable (if supported).
    granularity: :obj:`~_memory.VirtualMemoryGranularityT`
        Controls granularity query and size rounding.
    addr_hint: int
        A (optional) virtual address hint to try to reserve at. Setting it to 0 lets the CUDA driver decide.
    addr_align: int
        Alignment for the VA reservation. If `None`, use the queried granularity.
    peers: Iterable[int]
        Extra device IDs that should be granted access in addition to ``device``.
    self_access: :obj:`~_memory.VirtualMemoryAccessTypeT`
        Access flags for the owning device.
    peer_access: :obj:`~_memory.VirtualMemoryAccessTypeT`
        Access flags for peers.
    """

    # Human-friendly strings; normalized in __post_init__
    allocation_type: VirtualMemoryAllocationTypeT = "pinned"
    location_type: VirtualMemoryLocationTypeT = "device"
    handle_type: VirtualMemoryHandleTypeT = "posix_fd"
    granularity: VirtualMemoryGranularityT = "recommended"
    gpu_direct_rdma: bool = False
    addr_hint: int | None = 0
    addr_align: int | None = None
    peers: Iterable[int] = field(default_factory=tuple)
    self_access: VirtualMemoryAccessTypeT = "rw"
    peer_access: VirtualMemoryAccessTypeT = "rw"
    win32_handle_metadata: int | None = 0

    _a = driver.CUmemAccess_flags
    _access_flags = {"rw": _a.CU_MEM_ACCESS_FLAGS_PROT_READWRITE, "r": _a.CU_MEM_ACCESS_FLAGS_PROT_READ, None: 0}
    _h = driver.CUmemAllocationHandleType
    _handle_types = {
        None: _h.CU_MEM_HANDLE_TYPE_NONE,
        "posix_fd": _h.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        "win32": _h.CU_MEM_HANDLE_TYPE_WIN32,
        "win32_kmt": _h.CU_MEM_HANDLE_TYPE_WIN32_KMT,
        "fabric": _h.CU_MEM_HANDLE_TYPE_FABRIC,
    }
    _g = driver.CUmemAllocationGranularity_flags
    _granularity = {
        "recommended": _g.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        "minimum": _g.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
    }
    _l = driver.CUmemLocationType
    _location_type = {
        "device": _l.CU_MEM_LOCATION_TYPE_DEVICE,
        "host": _l.CU_MEM_LOCATION_TYPE_HOST,
        "host_numa": _l.CU_MEM_LOCATION_TYPE_HOST_NUMA,
        "host_numa_current": _l.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
    }
    # CUDA 13+ exposes MANAGED in CUmemAllocationType; older 12.x does not
    _a = driver.CUmemAllocationType
    _allocation_type = {"pinned": _a.CU_MEM_ALLOCATION_TYPE_PINNED}
    ver_major, ver_minor = get_binding_version()
    if ver_major >= 13:
        _allocation_type["managed"] = _a.CU_MEM_ALLOCATION_TYPE_MANAGED

    @staticmethod
    def _access_to_flags(spec: str):
        flags = VirtualMemoryResourceOptions._access_flags.get(spec)
        if flags is None:
            raise ValueError(f"Unknown access spec: {spec!r}")
        return flags

    @staticmethod
    def _allocation_type_to_driver(spec: str):
        alloc_type = VirtualMemoryResourceOptions._allocation_type.get(spec)
        if alloc_type is None:
            raise ValueError(f"Unsupported allocation_type: {spec!r}")
        return alloc_type

    @staticmethod
    def _location_type_to_driver(spec: str):
        loc_type = VirtualMemoryResourceOptions._location_type.get(spec)
        if loc_type is None:
            raise ValueError(f"Unsupported location_type: {spec!r}")
        return loc_type

    @staticmethod
    def _handle_type_to_driver(spec: str):
        handle_type = VirtualMemoryResourceOptions._handle_types.get(spec)
        if handle_type is None:
            raise ValueError(f"Unsupported handle_type: {spec!r}")
        return handle_type

    @staticmethod
    def _granularity_to_driver(spec: str):
        granularity = VirtualMemoryResourceOptions._granularity.get(spec)
        if granularity is None:
            raise ValueError(f"Unsupported granularity: {spec!r}")
        return granularity


class VirtualMemoryResource(MemoryResource):
    """Create a device memory resource that uses the CUDA VMM APIs to allocate memory.

    Parameters
    ----------
    device_id : Device | int
        Device for which a memory resource is constructed.

    config : VirtualMemoryResourceOptions
        A configuration object for the VirtualMemoryResource
    """

    def __init__(self, device_id: Device | int, config: VirtualMemoryResourceOptions = None):
        self.device = Device(device_id)
        self.config = check_or_create_options(
            VirtualMemoryResourceOptions, config, "VirtualMemoryResource options", keep_none=False
        )
        # Matches ("host", "host_numa", "host_numa_current")
        if "host" in self.config.location_type:
            self.device = None

        if not self.device and self.config.location_type == "device":
            raise RuntimeError("VirtualMemoryResource requires a device for device memory allocations")

        if self.device and not self.device.properties.virtual_memory_management_supported:
            raise RuntimeError("VirtualMemoryResource requires CUDA VMM API support")

        # Validate RDMA support if requested
        if (
            self.config.gpu_direct_rdma
            and self.device is not None
            and not self.device.properties.gpu_direct_rdma_supported
        ):
            raise RuntimeError("GPU Direct RDMA is not supported on this device")

    @staticmethod
    def _align_up(size: int, gran: int) -> int:
        """
        Align a size up to the nearest multiple of a granularity.
        """
        return (size + gran - 1) & ~(gran - 1)

    def modify_allocation(self, buf: Buffer, new_size: int, config: VirtualMemoryResourceOptions = None) -> Buffer:
        """
        Grow an existing allocation using CUDA VMM, with a configurable policy.

        This implements true growing allocations that preserve the base pointer
        by extending the virtual address range and mapping additional physical memory.

        This function uses transactional allocation: if any step fails, the original buffer is not modified and
        all steps the function took are rolled back so a new allocation is not created.

        Parameters
        ----------
        buf : Buffer
            The existing buffer to grow
        new_size : int
            The new total size for the allocation
        config : VirtualMemoryResourceOptions, optional
            Configuration for the new physical memory chunks. If None, uses current config.

        Returns
        -------
        Buffer
            The same buffer with updated size and properties, preserving the original pointer
        """
        if config is not None:
            self.config = config

        # Build allocation properties for new chunks
        prop = driver.CUmemAllocationProp()
        prop.type = VirtualMemoryResourceOptions._allocation_type_to_driver(self.config.allocation_type)
        prop.location.type = VirtualMemoryResourceOptions._location_type_to_driver(self.config.location_type)
        prop.location.id = self.device.device_id
        prop.allocFlags.gpuDirectRDMACapable = 1 if self.config.gpu_direct_rdma else 0
        prop.requestedHandleTypes = VirtualMemoryResourceOptions._handle_type_to_driver(self.config.handle_type)
        prop.win32HandleMetaData = self.config.win32_handle_metadata if self.config.win32_handle_metadata else 0

        # Query granularity
        gran_flag = VirtualMemoryResourceOptions._granularity_to_driver(self.config.granularity)
        res, gran = driver.cuMemGetAllocationGranularity(prop, gran_flag)
        raise_if_driver_error(res)

        # Calculate sizes
        additional_size = new_size - buf.size
        if additional_size <= 0:
            # Same size: only update access policy if needed; avoid zero-sized driver calls
            descs = self._build_access_descriptors(prop)
            if descs:
                (res,) = driver.cuMemSetAccess(int(buf.handle), buf.size, descs, len(descs))
                raise_if_driver_error(res)
            return buf

        aligned_additional_size = VirtualMemoryResource._align_up(additional_size, gran)
        total_aligned_size = VirtualMemoryResource._align_up(new_size, gran)
        aligned_prev_size = total_aligned_size - aligned_additional_size
        addr_align = self.config.addr_align or gran

        # Try to extend the existing VA range first
        res, new_ptr = driver.cuMemAddressReserve(
            aligned_additional_size,
            addr_align,
            int(buf.handle) + aligned_prev_size,  # fixedAddr hint - aligned end of current range
            0,
        )

        if res != driver.CUresult.CUDA_SUCCESS or new_ptr != (int(buf.handle) + aligned_prev_size):
            # Check for specific errors that are not recoverable with the slow path
            if res in (
                driver.CUresult.CUDA_ERROR_INVALID_VALUE,
                driver.CUresult.CUDA_ERROR_NOT_PERMITTED,
                driver.CUresult.CUDA_ERROR_NOT_INITIALIZED,
                driver.CUresult.CUDA_ERROR_NOT_SUPPORTED,
            ):
                raise_if_driver_error(res)
            (res2,) = driver.cuMemAddressFree(new_ptr, aligned_additional_size)
            raise_if_driver_error(res2)
            # Fallback: couldn't extend contiguously, need full remapping
            return self._grow_allocation_slow_path(
                buf, new_size, prop, aligned_additional_size, total_aligned_size, addr_align
            )
        else:
            # Success! We can extend the VA range contiguously
            return self._grow_allocation_fast_path(buf, new_size, prop, aligned_additional_size, new_ptr)

    def _grow_allocation_fast_path(
        self, buf: Buffer, new_size: int, prop: driver.CUmemAllocationProp, aligned_additional_size: int, new_ptr: int
    ) -> Buffer:
        """
        Fast path for growing a virtual memory allocation when the new region can be
        reserved contiguously after the existing buffer.

        This function creates and maps new physical memory for the additional size,
        sets access permissions, and updates the buffer size in place (the pointer
        remains unchanged).

        Args:
            buf (Buffer):
                The buffer to grow.

            new_size (int):
                The new total size in bytes.

            prop (driver.CUmemAllocationProp):
                Allocation properties for the new memory.

            aligned_additional_size (int):
                The size of the new region to allocate, aligned to granularity.

            new_ptr (int):
                The address of the newly reserved contiguous VA region (should
                be at the end of the current buffer).

        Returns:
            Buffer: The same buffer object with its size updated to `new_size`.
        """
        with Transaction() as trans:
            # Create new physical memory for the additional size
            trans.append(
                lambda np=new_ptr, s=aligned_additional_size: raise_if_driver_error(driver.cuMemAddressFree(np, s)[0])
            )
            res, new_handle = driver.cuMemCreate(aligned_additional_size, prop, 0)
            raise_if_driver_error(res)
            # Register undo for creation
            trans.append(lambda h=new_handle: raise_if_driver_error(driver.cuMemRelease(h)[0]))

            # Map the new physical memory to the extended VA range
            (res,) = driver.cuMemMap(new_ptr, aligned_additional_size, 0, new_handle, 0)
            raise_if_driver_error(res)
            # Register undo for mapping
            trans.append(
                lambda np=new_ptr, s=aligned_additional_size: raise_if_driver_error(driver.cuMemUnmap(np, s)[0])
            )

            # Set access permissions for the new portion
            descs = self._build_access_descriptors(prop)
            if descs:
                (res,) = driver.cuMemSetAccess(new_ptr, aligned_additional_size, descs, len(descs))
                raise_if_driver_error(res)

            # All succeeded, cancel undo actions
            trans.commit()

        # Update the buffer size (pointer stays the same)
        buf._size = new_size
        return buf

    def _grow_allocation_slow_path(
        self,
        buf: Buffer,
        new_size: int,
        prop: driver.CUmemAllocationProp,
        aligned_additional_size: int,
        total_aligned_size: int,
        addr_align: int,
    ) -> Buffer:
        """
        Slow path for growing a virtual memory allocation when the new region cannot be
        reserved contiguously after the existing buffer.

        This function reserves a new, larger virtual address (VA) range, remaps the old
        physical memory to the beginning of the new VA range, creates and maps new physical
        memory for the additional size, sets access permissions, and updates the buffer's
        pointer and size.

        Args:
            buf (Buffer): The buffer to grow.
            new_size (int): The new total size in bytes.
            prop (driver.CUmemAllocationProp): Allocation properties for the new memory.
            aligned_additional_size (int): The size of the new region to allocate, aligned to granularity.
            total_aligned_size (int): The total new size to reserve, aligned to granularity.
            addr_align (int): The required address alignment for the new VA range.

        Returns:
            Buffer: The buffer object updated with the new pointer and size.
        """
        with Transaction() as trans:
            # Reserve a completely new, larger VA range
            res, new_ptr = driver.cuMemAddressReserve(total_aligned_size, addr_align, 0, 0)
            raise_if_driver_error(res)
            # Register undo for VA reservation
            trans.append(
                lambda np=new_ptr, s=total_aligned_size: raise_if_driver_error(driver.cuMemAddressFree(np, s)[0])
            )

            # Get the old allocation handle for remapping
            result, old_handle = driver.cuMemRetainAllocationHandle(buf.handle)
            raise_if_driver_error(result)
            # Register undo for old_handle
            trans.append(lambda h=old_handle: raise_if_driver_error(driver.cuMemRelease(h)[0]))

            # Unmap the old VA range (aligned previous size)
            aligned_prev_size = total_aligned_size - aligned_additional_size
            (result,) = driver.cuMemUnmap(int(buf.handle), aligned_prev_size)
            raise_if_driver_error(result)

            def _remap_old():
                # Try to remap the old physical memory back to the original VA range
                try:
                    (res,) = driver.cuMemMap(int(buf.handle), aligned_prev_size, 0, old_handle, 0)
                    raise_if_driver_error(res)
                except Exception:  # noqa: S110
                    # TODO: consider logging this exception
                    pass

            trans.append(_remap_old)

            # Remap the old physical memory to the new VA range (aligned previous size)
            (res,) = driver.cuMemMap(int(new_ptr), aligned_prev_size, 0, old_handle, 0)
            raise_if_driver_error(res)

            # Register undo for mapping
            trans.append(lambda np=new_ptr, s=aligned_prev_size: raise_if_driver_error(driver.cuMemUnmap(np, s)[0]))

            # Create new physical memory for the additional size
            res, new_handle = driver.cuMemCreate(aligned_additional_size, prop, 0)
            raise_if_driver_error(res)

            # Register undo for new physical memory
            trans.append(lambda h=new_handle: raise_if_driver_error(driver.cuMemRelease(h)[0]))

            # Map the new physical memory to the extended portion (aligned offset)
            (res,) = driver.cuMemMap(int(new_ptr) + aligned_prev_size, aligned_additional_size, 0, new_handle, 0)
            raise_if_driver_error(res)

            # Register undo for mapping
            trans.append(
                lambda base=int(new_ptr), offs=aligned_prev_size, s=aligned_additional_size: raise_if_driver_error(
                    driver.cuMemUnmap(base + offs, s)[0]
                )
            )

            # Set access permissions for the entire new range
            descs = self._build_access_descriptors(prop)
            if descs:
                (res,) = driver.cuMemSetAccess(new_ptr, total_aligned_size, descs, len(descs))
                raise_if_driver_error(res)

            # All succeeded, cancel undo actions
            trans.commit()

        # Free the old VA range (aligned previous size)
        (res2,) = driver.cuMemAddressFree(int(buf.handle), aligned_prev_size)
        raise_if_driver_error(res2)

        # Invalidate the old buffer so its destructor won't try to free again
        buf._clear()

        # Return a new Buffer for the new mapping
        return Buffer.from_handle(ptr=new_ptr, size=new_size, mr=self)

    def _build_access_descriptors(self, prop: driver.CUmemAllocationProp) -> list:
        """
        Build access descriptors for memory access permissions.

        Returns
        -------
        list
            List of CUmemAccessDesc objects for setting memory access
        """
        descs = []

        # Owner access
        owner_flags = VirtualMemoryResourceOptions._access_to_flags(self.config.self_access)
        if owner_flags:
            d = driver.CUmemAccessDesc()
            d.location.type = prop.location.type
            d.location.id = prop.location.id
            d.flags = owner_flags
            descs.append(d)

        # Peer device access
        peer_flags = VirtualMemoryResourceOptions._access_to_flags(self.config.peer_access)
        if peer_flags:
            for peer_dev in self.config.peers:
                d = driver.CUmemAccessDesc()
                d.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                d.location.id = int(peer_dev)
                d.flags = peer_flags
                descs.append(d)

        return descs

    def allocate(self, size: int, stream: Stream | GraphBuilder | None = None) -> Buffer:
        """
        Allocate a buffer of the given size using CUDA virtual memory.

        Parameters
        ----------
        size : int
            The size in bytes of the buffer to allocate.
        stream : Stream, optional
            CUDA stream to associate with the allocation (not currently supported).

        Returns
        -------
        Buffer
            A Buffer object representing the allocated virtual memory.

        Raises
        ------
        NotImplementedError
            If a stream is provided or if the location type is not device memory.
        CUDAError
            If any CUDA driver API call fails during allocation.

        Notes
        -----
        This method uses transactional allocation: if any step fails, all resources
        allocated so far are automatically cleaned up. The allocation is performed
        with the configured granularity, access permissions, and peer access as
        specified in the resource's configuration.
        """
        if stream is not None:
            raise NotImplementedError("Stream is not supported with VirtualMemoryResource")

        config = self.config
        # ---- Build allocation properties ----
        prop = driver.CUmemAllocationProp()
        prop.type = VirtualMemoryResourceOptions._allocation_type_to_driver(config.allocation_type)
        prop.location.type = VirtualMemoryResourceOptions._location_type_to_driver(config.location_type)
        prop.location.id = self.device.device_id if config.location_type == "device" else -1
        prop.allocFlags.gpuDirectRDMACapable = 1 if config.gpu_direct_rdma else 0
        prop.requestedHandleTypes = VirtualMemoryResourceOptions._handle_type_to_driver(config.handle_type)
        prop.win32HandleMetaData = self.config.win32_handle_metadata if self.config.win32_handle_metadata else 0

        # ---- Query and apply granularity ----
        # Choose min vs recommended granularity per config
        gran_flag = VirtualMemoryResourceOptions._granularity_to_driver(config.granularity)
        res, gran = driver.cuMemGetAllocationGranularity(prop, gran_flag)
        raise_if_driver_error(res)

        aligned_size = VirtualMemoryResource._align_up(size, gran)
        addr_align = config.addr_align or gran

        # ---- Transactional allocation ----
        with Transaction() as trans:
            # ---- Create physical memory ----
            res, handle = driver.cuMemCreate(aligned_size, prop, 0)
            raise_if_driver_error(res)
            # Register undo for physical memory
            trans.append(lambda h=handle: raise_if_driver_error(driver.cuMemRelease(h)[0]))

            # ---- Reserve VA space ----
            # Potentially, use a separate size for the VA reservation from the physical allocation size
            res, ptr = driver.cuMemAddressReserve(aligned_size, addr_align, config.addr_hint, 0)
            raise_if_driver_error(res)
            # Register undo for VA reservation
            trans.append(lambda p=ptr, s=aligned_size: raise_if_driver_error(driver.cuMemAddressFree(p, s)[0]))

            # ---- Map physical memory into VA ----
            (res,) = driver.cuMemMap(ptr, aligned_size, 0, handle, 0)
            trans.append(lambda p=ptr, s=aligned_size: raise_if_driver_error(driver.cuMemUnmap(p, s)[0]))
            raise_if_driver_error(res)

            # ---- Set access for owner + peers ----
            descs = self._build_access_descriptors(prop)
            if descs:
                (res,) = driver.cuMemSetAccess(ptr, aligned_size, descs, len(descs))
                raise_if_driver_error(res)

            trans.commit()

        # Done — return a Buffer that tracks this VA range
        buf = Buffer.from_handle(ptr=ptr, size=aligned_size, mr=self)
        return buf

    def deallocate(self, ptr: int, size: int, stream: Stream | GraphBuilder | None = None) -> None:
        """
        Deallocate memory on the device using CUDA VMM APIs.
        """
        result, handle = driver.cuMemRetainAllocationHandle(ptr)
        raise_if_driver_error(result)
        (result,) = driver.cuMemUnmap(ptr, size)
        raise_if_driver_error(result)
        (result,) = driver.cuMemAddressFree(ptr, size)
        raise_if_driver_error(result)
        (result,) = driver.cuMemRelease(handle)
        raise_if_driver_error(result)

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
        return self.config.location_type == "host"

    @property
    def device_id(self) -> int:
        """
        Get the device ID associated with this memory resource.

        Returns:
            int: CUDA device ID. -1 if the memory resource allocates host memory
        """
        return self.device.device_id if self.config.location_type == "device" else -1

    def __repr__(self) -> str:
        """
        Return a string representation of the VirtualMemoryResource.

        Returns:
            str: A string describing the object
        """
        return f"<VirtualMemoryResource device={self.device}>"
