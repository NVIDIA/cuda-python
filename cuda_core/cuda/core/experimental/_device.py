# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import threading
from typing import Union

from cuda.core.experimental._context import Context, ContextOptions
from cuda.core.experimental._memory import Buffer, MemoryResource, _DefaultAsyncMempool, _SynchronousMemoryResource
from cuda.core.experimental._stream import Stream, StreamOptions, default_stream
from cuda.core.experimental._utils import ComputeCapability, CUDAError, driver, handle_return, precondition, runtime

_tls = threading.local()
_tls_lock = threading.Lock()


class DeviceProperties:
    """
    Represents the properties of a CUDA device.

    Attributes
    ----------
    name : str
        ASCII string identifying the device.
    uuid : cudaUUID_t
        16-byte unique identifier.
    total_global_mem : int
        Total amount of global memory available on the device in bytes.
    shared_mem_per_block : int
        Maximum amount of shared memory available to a thread block in bytes.
    regs_per_block : int
        Maximum number of 32-bit registers available to a thread block.
    warp_size : int
        Warp size in threads.
    mem_pitch : int
        Maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated through
        cudaMallocPitch().
    max_threads_per_block : int
        Maximum number of threads per block.
    max_threads_dim : tuple
        Maximum size of each dimension of a block.
    max_grid_size : tuple
        Maximum size of each dimension of a grid.
    clock_rate : int
        Clock frequency in kilohertz.
    cluster_launch : bool
        Indicates whether or not the device supports cluster launch.
    total_const_mem : int
        Total amount of constant memory available on the device in bytes.
    major : int
        Major revision number defining the device's compute capability.
    minor : int
        Minor revision number defining the device's compute capability.
    texture_alignment : int
        Alignment requirement; texture base addresses that are aligned to textureAlignment bytes do not need an
        offset applied to texture fetches.
    texture_pitch_alignment : int
        Pitch alignment requirement for 2D texture references that are bound to pitched memory.
    device_overlap : bool
        Indicates whether or not the device can concurrently copy memory between host and device while executing
        a kernel.
    multi_processor_count : int
        Number of multiprocessors on the device.
    kernel_exec_timeout_enabled : bool
        Indicates whether or not there is a run time limit for kernels executed on the device.
    integrated : bool
        Indicates whether or not the device is an integrated (motherboard) GPU.
    can_map_host_memory : bool
        Indicates whether or not the device can map host memory into the CUDA address space for use with
        cudaHostAlloc()/cudaHostGetDevicePointer().
    compute_mode : int
        Compute mode that the device is currently in.
    max_texture_1d : int
        Maximum 1D texture size.
    max_texture_1d_mipmap : int
        Maximum 1D mipmapped texture size.
    max_texture_1d_linear : int
        Maximum 1D texture size for textures bound to linear memory.
    max_texture_2d : tuple
        Maximum 2D texture dimensions.
    max_texture_2d_mipmap : tuple
        Maximum 2D mipmapped texture dimensions.
    max_texture_2d_linear : tuple
        Maximum 2D texture dimensions for 2D textures bound to pitch linear memory.
    max_texture_2d_gather : tuple
        Maximum 2D texture dimensions if texture gather operations have to be performed.
    max_texture_3d : tuple
        Maximum 3D texture dimensions.
    max_texture_3d_alt : tuple
        Maximum alternate 3D texture dimensions.
    max_texture_cubemap : int
        Maximum cubemap texture width or height.
    max_texture_1d_layered : tuple
        Maximum 1D layered texture dimensions.
    max_texture_2d_layered : tuple
        Maximum 2D layered texture dimensions.
    max_texture_cubemap_layered : tuple
        Maximum cubemap layered texture dimensions.
    max_surface_1d : int
        Maximum 1D surface size.
    max_surface_2d : tuple
        Maximum 2D surface dimensions.
    max_surface_3d : tuple
        Maximum 3D surface dimensions.
    max_surface_1d_layered : tuple
        Maximum 1D layered surface dimensions.
    max_surface_2d_layered : tuple
        Maximum 2D layered surface dimensions.
    max_surface_cubemap : int
        Maximum cubemap surface width or height.
    max_surface_cubemap_layered : tuple
        Maximum cubemap layered surface dimensions.
    surface_alignment : int
        Alignment requirements for surfaces.
    concurrent_kernels : bool
        Indicates whether or not the device supports executing multiple kernels within the same context
        simultaneously.
    ecc_enabled : bool
        Indicates whether or not the device has ECC support turned on.
    pci_bus_id : int
        PCI bus identifier of the device.
    pci_device_id : int
        PCI device (sometimes called slot) identifier of the device.
    pci_domain_id : int
        PCI domain identifier of the device.
    tcc_driver : bool
        Indicates whether or not the device is using a TCC driver.
    async_engine_count : int
        1 when the device can concurrently copy memory between host and device while executing a kernel.
        It is 2 when the device can concurrently copy memory between host and device in both directions
        and execute a kernel at the same time. It is 0 if neither of these is supported.
    unified_addressing : bool
        Indicates whether or not the device shares a unified address space with the host.
    memory_clock_rate : int
        Peak memory clock frequency in kilohertz.
    memory_bus_width : int
        Memory bus width in bits.
    l2_cache_size : int
        L2 cache size in bytes.
    persisting_l2_cache_max_size : int
        L2 cache's maximum persisting lines size in bytes.
    max_threads_per_multi_processor : int
        Number of maximum resident threads per multiprocessor.
    stream_priorities_supported : bool
        Indicates whether or not the device supports stream priorities.
    global_l1_cache_supported : bool
        Indicates whether or not the device supports caching of globals in L1 cache.
    local_l1_cache_supported : bool
        Indicates whether or not the device supports caching of locals in L1 cache.
    shared_mem_per_multiprocessor : int
        Maximum amount of shared memory available to a multiprocessor in bytes; this amount is shared by all
        thread blocks simultaneously resident on a multiprocessor.
    regs_per_multiprocessor : int
        Maximum number of 32-bit registers available to a multiprocessor; this number is shared by all thread
        blocks simultaneously resident on a multiprocessor.
    managed_memory : bool
        Indicates whether or not the device supports allocating managed memory on this system.
    is_multi_gpu_board : bool
        Indicates whether or not the device is on a multi-GPU board (e.g. Gemini cards).
    multi_gpu_board_group_id : int
        Unique identifier for a group of devices associated with the same board. Devices on the same
        multi-GPU board will share the same identifier.
    single_to_double_precision_perf_ratio : int
        Ratio of single precision performance (in floating-point operations per second) to double precision
        performance.
    pageable_memory_access : bool
        Indicates whether or not the device supports coherently accessing pageable memory without calling
        cudaHostRegister on it.
    concurrent_managed_access : bool
        Indicates whether or not the device can coherently access managed memory concurrently with the CPU.
    compute_preemption_supported : bool
        Indicates whether or not the device supports Compute Preemption.
    can_use_host_pointer_for_registered_mem : bool
        Indicates whether or not the device can access host registered memory at the same virtual address as
        the CPU.
    cooperative_launch : bool
        Indicates whether or not the device supports launching cooperative kernels via
        cudaLaunchCooperativeKernel.
    cooperative_multi_device_launch : bool
        Indicates whether or not the device supports launching cooperative kernels via
        cudaLaunchCooperativeKernelMultiDevice.
    pageable_memory_access_uses_host_page_tables : bool
        Indicates whether or not the device accesses pageable memory via the host's page tables.
    direct_managed_mem_access_from_host : bool
        Indicates whether or not the host can directly access managed memory on the device without migration.
    access_policy_max_window_size : int
        Maximum value of cudaAccessPolicyWindow::num_bytes.
    reserved_shared_mem_per_block : int
        Shared memory reserved by CUDA driver per block in bytes.
    host_register_supported : bool
        Indicates whether or not the device supports host memory registration via cudaHostRegister.
    sparse_cuda_array_supported : bool
        Indicates whether or not the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays.
    host_register_read_only_supported : bool
        Indicates whether or not the device supports using the cudaHostRegister flag cudaHostRegisterReadOnly
        to register memory
        that must be mapped as read-only to the GPU.
    timeline_semaphore_interop_supported : bool
        Indicates whether or not external timeline semaphore interop is supported on the device.
    memory_pools_supported : bool
        Indicates whether or not the device supports using the cudaMallocAsync and cudaMemPool family of APIs.
    gpu_direct_rdma_supported : bool
        Indicates whether or not the device supports GPUDirect RDMA APIs.
    gpu_direct_rdma_flush_writes_options : int
        Bitmask to be interpreted according to the cudaFlushGPUDirectRDMAWritesOptions enum.
    gpu_direct_rdma_writes_ordering : int
        See the cudaGPUDirectRDMAWritesOrdering enum for numerical values.
    memory_pool_supported_handle_types : int
        Bitmask of handle types supported with mempool-based IPC.
    deferred_mapping_cuda_array_supported : bool
        Indicates whether or not the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays.
    ipc_event_supported : bool
        Indicates whether or not the device supports IPC Events.
    unified_function_pointers : bool
        Indicates whether or not the device supports unified pointers.
    host_native_atomic_supported : bool
        Indicates whether or not the link between the device and the host supports native atomic operations.
    luid : bytes
        8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms.
    luid_device_node_mask : int
        LUID device node mask. Value is undefined on TCC and non-Windows platforms.
    max_blocks_per_multi_processor : int
        Maximum number of resident blocks per multiprocessor.
    """

    def _init(device_id):
        self = DeviceProperties.__new__(DeviceProperties)

        prop = handle_return(runtime.cudaGetDeviceProperties(device_id))

        self.name = prop.name.decode("utf-8")
        self.uuid = prop.uuid.bytes
        self.total_global_mem = prop.totalGlobalMem
        self.shared_mem_per_block = prop.sharedMemPerBlock
        self.regs_per_block = prop.regsPerBlock
        self.warp_size = prop.warpSize
        self.mem_pitch = prop.memPitch
        self.max_threads_per_block = prop.maxThreadsPerBlock
        self.max_threads_dim = tuple(prop.maxThreadsDim)
        self.max_grid_size = tuple(prop.maxGridSize)
        self.clock_rate = prop.clockRate
        self.cluster_launch = bool(prop.clusterLaunch)
        self.total_const_mem = prop.totalConstMem
        self.major = prop.major
        self.minor = prop.minor
        self.texture_alignment = prop.textureAlignment
        self.texture_pitch_alignment = prop.texturePitchAlignment
        self.device_overlap = bool(prop.deviceOverlap)
        self.multi_processor_count = prop.multiProcessorCount
        self.kernel_exec_timeout_enabled = bool(prop.kernelExecTimeoutEnabled)
        self.integrated = bool(prop.integrated)
        self.can_map_host_memory = bool(prop.canMapHostMemory)
        self.compute_mode = prop.computeMode
        self.max_texture_1d = prop.maxTexture1D
        self.max_texture_1d_mipmap = prop.maxTexture1DMipmap
        self.max_texture_2d = tuple(prop.maxTexture2D)
        self.max_texture_2d_mipmap = tuple(prop.maxTexture2DMipmap)
        self.max_texture_2d_linear = tuple(prop.maxTexture2DLinear)
        self.max_texture_2d_gather = tuple(prop.maxTexture2DGather)
        self.max_texture_3d = tuple(prop.maxTexture3D)
        self.max_texture_3d_alt = tuple(prop.maxTexture3DAlt)
        self.max_texture_cubemap = prop.maxTextureCubemap
        self.max_texture_1d_layered = tuple(prop.maxTexture1DLayered)
        self.max_texture_2d_layered = tuple(prop.maxTexture2DLayered)
        self.max_texture_cubemap_layered = tuple(prop.maxTextureCubemapLayered)
        self.max_surface_1d = prop.maxSurface1D
        self.max_surface_2d = tuple(prop.maxSurface2D)
        self.max_surface_3d = tuple(prop.maxSurface3D)
        self.max_surface_1d_layered = tuple(prop.maxSurface1DLayered)
        self.max_surface_2d_layered = tuple(prop.maxSurface2DLayered)
        self.max_surface_cubemap = prop.maxSurfaceCubemap
        self.max_surface_cubemap_layered = tuple(prop.maxSurfaceCubemapLayered)
        self.surface_alignment = prop.surfaceAlignment
        self.concurrent_kernels = bool(prop.concurrentKernels)
        self.ecc_enabled = bool(prop.ECCEnabled)
        self.pci_bus_id = prop.pciBusID
        self.pci_device_id = prop.pciDeviceID
        self.pci_domain_id = prop.pciDomainID
        self.tcc_driver = bool(prop.tccDriver)
        self.async_engine_count = prop.asyncEngineCount
        self.unified_addressing = bool(prop.unifiedAddressing)
        self.memory_clock_rate = prop.memoryClockRate
        self.memory_bus_width = prop.memoryBusWidth
        self.l2_cache_size = prop.l2CacheSize
        self.persisting_l2_cache_max_size = prop.persistingL2CacheMaxSize
        self.max_threads_per_multi_processor = prop.maxThreadsPerMultiProcessor
        self.stream_priorities_supported = bool(prop.streamPrioritiesSupported)
        self.global_l1_cache_supported = bool(prop.globalL1CacheSupported)
        self.local_l1_cache_supported = bool(prop.localL1CacheSupported)
        self.shared_mem_per_multiprocessor = prop.sharedMemPerMultiprocessor
        self.regs_per_multiprocessor = prop.regsPerMultiprocessor
        self.managed_memory = bool(prop.managedMemory)
        self.is_multi_gpu_board = bool(prop.isMultiGpuBoard)
        self.multi_gpu_board_group_id = prop.multiGpuBoardGroupID
        self.pageable_memory_access = bool(prop.pageableMemoryAccess)
        self.concurrent_managed_access = bool(prop.concurrentManagedAccess)
        self.compute_preemption_supported = bool(prop.computePreemptionSupported)
        self.can_use_host_pointer_for_registered_mem = bool(prop.canUseHostPointerForRegisteredMem)
        self.cooperative_launch = bool(prop.cooperativeLaunch)
        self.cooperative_multi_device_launch = bool(prop.cooperativeMultiDeviceLaunch)
        self.pageable_memory_access_uses_host_page_tables = bool(prop.pageableMemoryAccessUsesHostPageTables)
        self.direct_managed_mem_access_from_host = bool(prop.directManagedMemAccessFromHost)
        self.access_policy_max_window_size = prop.accessPolicyMaxWindowSize
        self.reserved_shared_mem_per_block = prop.reservedSharedMemPerBlock
        self.host_register_supported = bool(prop.hostRegisterSupported)
        self.sparse_cuda_array_supported = bool(prop.sparseCudaArraySupported)
        self.host_register_read_only_supported = bool(prop.hostRegisterReadOnlySupported)
        self.timeline_semaphore_interop_supported = bool(prop.timelineSemaphoreInteropSupported)
        self.memory_pools_supported = bool(prop.memoryPoolsSupported)
        self.gpu_direct_rdma_supported = bool(prop.gpuDirectRDMASupported)
        self.gpu_direct_rdma_flush_writes_options = prop.gpuDirectRDMAFlushWritesOptions
        self.gpu_direct_rdma_writes_ordering = prop.gpuDirectRDMAWritesOrdering
        self.memory_pool_supported_handle_types = prop.memoryPoolSupportedHandleTypes
        self.deferred_mapping_cuda_array_supported = bool(prop.deferredMappingCudaArraySupported)
        self.ipc_event_supported = bool(prop.ipcEventSupported)
        self.unified_function_pointers = bool(prop.unifiedFunctionPointers)
        self.host_native_atomic_supported = bool(prop.hostNativeAtomicSupported)
        self.luid = prop.luid
        self.luid_device_node_mask = prop.luidDeviceNodeMask
        self.max_blocks_per_multi_processor = prop.maxBlocksPerMultiProcessor
        return self

    def __init__(self, device_id):
        raise RuntimeError("DeviceProperties should not be instantiated directly")


class Device:
    """Represent a GPU and act as an entry point for cuda.core features.

    This is a singleton object that helps ensure interoperability
    across multiple libraries imported in the process to both see
    and use the same GPU device.

    While acting as the entry point, many other CUDA resources can be
    allocated such as streams and buffers. Any :obj:`~_context.Context` dependent
    resource created through this device, will continue to refer to
    this device's context.

    Newly returned :obj:`~_device.Device` objects are thread-local singletons
    for a specified device.

    Note
    ----
    Will not initialize the GPU.

    Parameters
    ----------
    device_id : int, optional
        Device ordinal to return a :obj:`~_device.Device` object for.
        Default value of `None` return the currently used device.

    """

    __slots__ = ("_id", "_mr", "_has_inited", "_properties")

    def __new__(cls, device_id=None):
        # important: creating a Device instance does not initialize the GPU!
        if device_id is None:
            device_id = handle_return(runtime.cudaGetDevice())
            assert isinstance(device_id, int), f"{device_id=}"
        else:
            total = handle_return(runtime.cudaGetDeviceCount())
            if not isinstance(device_id, int) or not (0 <= device_id < total):
                raise ValueError(f"device_id must be within [0, {total}), got {device_id}")

        # ensure Device is singleton
        with _tls_lock:
            if not hasattr(_tls, "devices"):
                total = handle_return(runtime.cudaGetDeviceCount())
                _tls.devices = []
                for dev_id in range(total):
                    dev = super().__new__(cls)
                    dev._id = dev_id
                    # If the device is in TCC mode, or does not support memory pools for some other reason,
                    # use the SynchronousMemoryResource which does not use memory pools.
                    if (
                        handle_return(
                            runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported, 0)
                        )
                    ) == 1:
                        dev._mr = _DefaultAsyncMempool(dev_id)
                    else:
                        dev._mr = _SynchronousMemoryResource(dev_id)

                    dev._has_inited = False
                    dev._properties = None
                    _tls.devices.append(dev)

        return _tls.devices[device_id]

    def _check_context_initialized(self, *args, **kwargs):
        if not self._has_inited:
            raise CUDAError("the device is not yet initialized, perhaps you forgot to call .set_current() first?")

    @property
    def device_id(self) -> int:
        """Return device ordinal."""
        return self._id

    @property
    def pci_bus_id(self) -> str:
        """Return a PCI Bus Id string for this device."""
        bus_id = handle_return(runtime.cudaDeviceGetPCIBusId(13, self._id))
        return bus_id[:12].decode()

    @property
    def uuid(self) -> str:
        """Return a UUID for the device.

        Returns 16-octets identifying the device. If the device is in
        MIG mode, returns its MIG UUID which uniquely identifies the
        subscribed MIG compute instance.

        Note
        ----
        MIG UUID is only returned when device is in MIG mode and the
        driver is older than CUDA 11.4.

        """
        driver_ver = handle_return(driver.cuDriverGetVersion())
        if driver_ver >= 11040:
            uuid = handle_return(driver.cuDeviceGetUuid_v2(self._id))
        else:
            uuid = handle_return(driver.cuDeviceGetUuid(self._id))
        uuid = uuid.bytes.hex()
        # 8-4-4-4-12
        return f"{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:]}"

    @property
    def name(self) -> str:
        """Return the device name."""
        # Use 256 characters to be consistent with CUDA Runtime
        name = handle_return(driver.cuDeviceGetName(256, self._id))
        name = name.split(b"\0")[0]
        return name.decode()

    @property
    def properties(self) -> DeviceProperties:
        """Return information about the compute-device."""
        if self._properties is None:
            self._properties = DeviceProperties._init(self._id)

        return self._properties

    @property
    def compute_capability(self) -> ComputeCapability:
        """Return a named tuple with 2 fields: major and minor."""
        major = handle_return(
            runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, self._id)
        )
        minor = handle_return(
            runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, self._id)
        )
        return ComputeCapability(major, minor)

    @property
    @precondition(_check_context_initialized)
    def context(self) -> Context:
        """Return the current :obj:`~_context.Context` associated with this device.

        Note
        ----
        Device must be initialized.

        """
        ctx = handle_return(driver.cuCtxGetCurrent())
        assert int(ctx) != 0
        return Context._from_ctx(ctx, self._id)

    @property
    def memory_resource(self) -> MemoryResource:
        """Return :obj:`~_memory.MemoryResource` associated with this device."""
        return self._mr

    @memory_resource.setter
    def memory_resource(self, mr):
        if not isinstance(mr, MemoryResource):
            raise TypeError
        self._mr = mr

    @property
    def default_stream(self) -> Stream:
        """Return default CUDA :obj:`~_stream.Stream` associated with this device.

        The type of default stream returned depends on if the environment
        variable CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM is set.

        If set, returns a per-thread default stream. Otherwise returns
        the legacy stream.

        """
        return default_stream()

    def __int__(self):
        """Return device_id."""
        return self._id

    def __repr__(self):
        return f"<Device {self._id} ({self.name})>"

    def set_current(self, ctx: Context = None) -> Union[Context, None]:
        """Set device to be used for GPU executions.

        Initializes CUDA and sets the calling thread to a valid CUDA
        context. By default the primary context is used, but optional `ctx`
        parameter can be used to explicitly supply a :obj:`~_context.Context` object.

        Providing a `ctx` causes the previous set context to be popped and returned.

        Parameters
        ----------
        ctx : :obj:`~_context.Context`, optional
            Optional context to push onto this device's current thread stack.

        Returns
        -------
        Union[:obj:`~_context.Context`, None], optional
            Popped context.

        Examples
        --------
        Acts as an entry point of this object. Users always start a code by
        calling this method, e.g.

        >>> from cuda.core.experimental import Device
        >>> dev0 = Device(0)
        >>> dev0.set_current()
        >>> # ... do work on device 0 ...

        """
        if ctx is not None:
            if not isinstance(ctx, Context):
                raise TypeError("a Context object is required")
            if ctx._id != self._id:
                raise RuntimeError(
                    "the provided context was created on a different "
                    f"device {ctx._id} other than the target {self._id}"
                )
            prev_ctx = handle_return(driver.cuCtxPopCurrent())
            handle_return(driver.cuCtxPushCurrent(ctx._handle))
            self._has_inited = True
            if int(prev_ctx) != 0:
                return Context._from_ctx(prev_ctx, self._id)
        else:
            ctx = handle_return(driver.cuCtxGetCurrent())
            if int(ctx) == 0:
                # use primary ctx
                ctx = handle_return(driver.cuDevicePrimaryCtxRetain(self._id))
                handle_return(driver.cuCtxPushCurrent(ctx))
            else:
                ctx_id = handle_return(driver.cuCtxGetDevice())
                if ctx_id != self._id:
                    # use primary ctx
                    ctx = handle_return(driver.cuDevicePrimaryCtxRetain(self._id))
                    handle_return(driver.cuCtxPushCurrent(ctx))
                else:
                    # no-op, a valid context already exists and is set current
                    pass
            self._has_inited = True

    def create_context(self, options: ContextOptions = None) -> Context:
        """Create a new :obj:`~_context.Context` object.

        Note
        ----
        The newly context will not be set as current.

        Parameters
        ----------
        options : :obj:`~_context.ContextOptions`, optional
            Customizable dataclass for context creation options.

        Returns
        -------
        :obj:`~_context.Context`
            Newly created context object.

        """
        raise NotImplementedError("TODO")

    @precondition(_check_context_initialized)
    def create_stream(self, obj=None, options: StreamOptions = None) -> Stream:
        """Create a Stream object.

        New stream objects can be created in two different ways:

        1) Create a new CUDA stream with customizable `options`.
        2) Wrap an existing foreign `obj` supporting the __cuda_stream__ protocol.

        Option (2) internally holds a reference to the foreign object
        such that the lifetime is managed.

        Note
        ----
        Device must be initialized.

        Parameters
        ----------
        obj : Any, optional
            Any object supporting the __cuda_stream__ protocol.
        options : :obj:`~_stream.StreamOptions`, optional
            Customizable dataclass for stream creation options.

        Returns
        -------
        :obj:`~_stream.Stream`
            Newly created stream object.

        """
        return Stream._init(obj=obj, options=options)

    @precondition(_check_context_initialized)
    def allocate(self, size, stream=None) -> Buffer:
        """Allocate device memory from a specified stream.

        Allocates device memory of `size` bytes on the specified `stream`
        using the memory resource currently associated with this Device.

        Parameter `stream` is optional, using a default stream by default.

        Note
        ----
        Device must be initialized.

        Parameters
        ----------
        size : int
            Number of bytes to allocate.
        stream : :obj:`~_stream.Stream`, optional
            The stream establishing the stream ordering semantic.
            Default value of `None` uses default stream.

        Returns
        -------
        :obj:`~_memory.Buffer`
            Newly created buffer object.

        """
        if stream is None:
            stream = default_stream()
        return self._mr.allocate(size, stream)

    @precondition(_check_context_initialized)
    def sync(self):
        """Synchronize the device.

        Note
        ----
        Device must be initialized.

        """
        handle_return(runtime.cudaDeviceSynchronize())
