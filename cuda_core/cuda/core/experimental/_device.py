# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import threading
from typing import Optional, Union
import warnings

from cuda import cuda, cudart
from cuda.core.experimental._utils import handle_return, ComputeCapability, CUDAError, \
                             precondition
from cuda.core.experimental._context import Context, ContextOptions
from cuda.core.experimental._memory import _DefaultAsyncMempool, Buffer, MemoryResource
from cuda.core.experimental._stream import default_stream, Stream, StreamOptions


_tls = threading.local()
_tls_lock = threading.Lock()


class Device:
    """Represents a GPU and acts as an entry point for cuda.core features.

    This is a singleton object that helps ensure interoperability
    across multiple libraries imported in the process to both see
    and use the same GPU device.

    While acting as the entry point, many other CUDA resources can be
    allocated such as streams and buffers. Any :obj:`Context` dependent
    resource created through this device, will continue to refer to
    this device's context.

    """
    __slots__ = ("_id", "_mr", "_has_inited")

    def __new__(cls, device_id=None):
        """Create and return a singleton :obj:`Device` object.

        Creates and returns a thread-local singleton :obj:`Device` object
        corresponding to a specific device.

        Note
        ----
        Will not initialize the GPU.

        Parameters
        ----------
        device_id : int, optional
            Device ordinal to return a :obj:`Device` object for.
            Default value of `None` return the currently used device.

        """
        # important: creating a Device instance does not initialize the GPU!
        if device_id is None:
            device_id = handle_return(cudart.cudaGetDevice())
            assert isinstance(device_id, int), f"{device_id=}"
        else:
            total = handle_return(cudart.cudaGetDeviceCount())
            if not isinstance(device_id, int) or not (0 <= device_id < total):
                raise ValueError(
                    f"device_id must be within [0, {total}), got {device_id}")

        # ensure Device is singleton
        with _tls_lock:
            if not hasattr(_tls, "devices"):
                total = handle_return(cudart.cudaGetDeviceCount())
                _tls.devices = []
                for dev_id in range(total):
                    dev = super().__new__(cls)
                    dev._id = dev_id
                    dev._mr = _DefaultAsyncMempool(dev_id)
                    dev._has_inited = False
                    _tls.devices.append(dev)

        return _tls.devices[device_id]

    def _check_context_initialized(self, *args, **kwargs):
        if not self._has_inited:
            raise CUDAError("the device is not yet initialized, "
                            "perhaps you forgot to call .set_current() first?")

    @property
    def device_id(self) -> int:
        """Return device ordinal."""
        return self._id

    @property
    def pci_bus_id(self) -> str:
        """Return a PCI Bus Id string for this device."""
        bus_id = handle_return(cudart.cudaDeviceGetPCIBusId(13, self._id))
        return bus_id[:12].decode()

    @property
    def uuid(self) -> str:
        """Return a UUID for the device.

        Returns 16-octets identifying the device. If the device is in
        MIG mode, returns its MIG UUID which uniquely identifies the
        subscribed MIG compute instance.

        """
        driver_ver = handle_return(cuda.cuDriverGetVersion())
        if driver_ver >= 11040:
            uuid = handle_return(cuda.cuDeviceGetUuid_v2(self._id))
        else:
            uuid = handle_return(cuda.cuDeviceGetUuid(self._id))
        uuid = uuid.bytes.hex()
        # 8-4-4-4-12
        return f"{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:]}"

    @property
    def name(self) -> str:
        """Return the device name."""
        # CUDA Runtime uses up to 256 characters, use the same for consistency
        name = handle_return(cuda.cuDeviceGetName(256, self._id))
        name = name.split(b'\0')[0]
        return name.decode()

    @property
    def properties(self) -> dict:
        """Return information about the compute-device."""
        # TODO: pythonize the key names
        return handle_return(cudart.cudaGetDeviceProperties(self._id))

    @property
    def compute_capability(self) -> ComputeCapability:
        """Return a named tuple with 2 fields: major and minor."""
        major = handle_return(cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, self._id))
        minor = handle_return(cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, self._id))
        return ComputeCapability(major, minor)

    @property
    @precondition(_check_context_initialized)
    def context(self) -> Context:
        """Return the current :obj:`Context` associated with this device.

        Note
        ----
        Device must be initialized.

        """
        ctx = handle_return(cuda.cuCtxGetCurrent())
        assert int(ctx) != 0
        return Context._from_ctx(ctx, self._id)

    @property
    def memory_resource(self) -> MemoryResource:
        """Return :obj:`MemoryResource` associated with this device."""
        return self._mr

    @memory_resource.setter
    def memory_resource(self, mr):
        if not isinstance(mr, MemoryResource):
            raise TypeError
        self._mr = mr

    @property
    def default_stream(self) -> Stream:
        """Return default CUDA :obj:`Stream` associated with this device.

        Returns per-thread default stream if environment is set with
        CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM, otherwise return a legacy stream.

        """
        return default_stream()

    def __int__(self):
        """Return device_id."""
        return self._id

    def __repr__(self):
        return f"<Device {self._id} ({self.name})>"

    def set_current(self, ctx: Context=None) -> Union[Context, None]:
        """Set device to be used for GPU executions.

        Initializes CUDA and sets the calling thread to a valid CUDA
        context. By default the primary context is used, but optional `ctx`
        parameter can be used to explicitly supply a :obj:`Context` object.

        Providing a `ctx` causes the previous set context to be popped and returned.

        Parameters
        ----------
        ctx : Context, optional
            Optional context to push onto this device's current thread stack

        Returns
        -------
        Union[Context, None], optional
            Popped context

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
                raise RuntimeError("the provided context was created on a different "
                                  f"device {ctx._id} other than the target {self._id}")
            prev_ctx = handle_return(cuda.cuCtxPopCurrent())
            handle_return(cuda.cuCtxPushCurrent(ctx._handle))
            self._has_inited = True
            if int(prev_ctx) != 0:
                return Context._from_ctx(prev_ctx, self._id)
        else:
            ctx = handle_return(cuda.cuCtxGetCurrent())
            if int(ctx) == 0:
                # use primary ctx
                ctx = handle_return(cuda.cuDevicePrimaryCtxRetain(self._id))
                handle_return(cuda.cuCtxPushCurrent(ctx))
            else:
                ctx_id = handle_return(cuda.cuCtxGetDevice())
                if ctx_id != self._id:
                    # use primary ctx
                    ctx = handle_return(cuda.cuDevicePrimaryCtxRetain(self._id))
                    handle_return(cuda.cuCtxPushCurrent(ctx))
                else:
                    # no-op, a valid context already exists and is set current
                    pass
            self._has_inited = True

    def create_context(self, options: ContextOptions = None) -> Context:
        """Create a new :obj:`Context` object.

        Note
        ----
        The newly context will not be set as current.

        Parameters
        ----------
        options : ContextOptions, optional
            Customizable dataclass for context creation options

        Returns
        -------
        :obj:`Context`
            Newly created Context object

        """
        raise NotImplementedError("TODO")

    @precondition(_check_context_initialized)
    def create_stream(self, obj=None, options: StreamOptions=None) -> Stream:
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
        options : :obj:`StreamOptions`, optional
            Customizable dataclass for stream creation options

        Returns
        -------
        :obj:`Stream`
            Newly created Stream object

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
        stream : :obj:`Stream`, optional
            The stream establishing the stream ordering semantic.
            Default value of `None` uses default stream.

        Returns
        -------
        :obj:`Buffer`
            Newly created Buffer object

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
        handle_return(cudart.cudaDeviceSynchronize())
