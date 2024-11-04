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

    __slots__ = ("_id", "_mr", "_has_inited")

    def __new__(cls, device_id=None):
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
        return self._id

    @property
    def pci_bus_id(self) -> str:
        bus_id = handle_return(cudart.cudaDeviceGetPCIBusId(13, self._id))
        return bus_id[:12].decode()

    @property
    def uuid(self) -> str:
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
        # assuming a GPU name is less than 128 characters...
        name = handle_return(cuda.cuDeviceGetName(128, self._id))
        name = name.split(b'\0')[0]
        return name.decode()

    @property
    def properties(self) -> dict:
        # TODO: pythonize the key names
        return handle_return(cudart.cudaGetDeviceProperties(self._id))

    @property
    def compute_capability(self) -> ComputeCapability:
        """Returns a named tuple with 2 fields: major and minor. """
        major = handle_return(cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, self._id))
        minor = handle_return(cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, self._id))
        return ComputeCapability(major, minor)

    @property
    @precondition(_check_context_initialized)
    def context(self) -> Context:
        ctx = handle_return(cuda.cuCtxGetCurrent())
        assert int(ctx) != 0
        return Context._from_ctx(ctx, self._id)

    @property
    def memory_resource(self) -> MemoryResource:
        return self._mr

    @memory_resource.setter
    def memory_resource(self, mr):
        if not isinstance(mr, MemoryResource):
            raise TypeError
        self._mr = mr

    @property
    def default_stream(self) -> Stream:
        return default_stream()

    def __int__(self):
        return self._id

    def __repr__(self):
        return f"<Device {self._id} ({self.name})>"

    def set_current(self, ctx: Context=None) -> Union[Context, None]:
        """
        Entry point of this object. Users always start a code by
        calling this method, e.g.
        
        >>> from cuda.core.experimental import Device
        >>> dev0 = Device(0)
        >>> dev0.set_current()
        >>> # ... do work on device 0 ...
        
        The optional ctx argument is for advanced users to bind a
        CUDA context with the device. In this case, the previously
        set context is popped and returned to the user.
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
        # Create a Context object (but do NOT set it current yet!).
        # ContextOptions is a dataclass for setting e.g. affinity or CIG
        # options. 
        raise NotImplementedError("TODO")

    @precondition(_check_context_initialized)
    def create_stream(self, obj=None, options: StreamOptions=None) -> Stream:
        # Create a Stream object by either holding a newly created
        # CUDA stream or wrapping an existing foreign object supporting
        # the __cuda_stream__ protocol. In the latter case, a reference
        # to obj is held internally so that its lifetime is managed.
        return Stream._init(obj=obj, options=options)

    @precondition(_check_context_initialized)
    def allocate(self, size, stream=None) -> Buffer:
        if stream is None:
            stream = default_stream()
        return self._mr.allocate(size, stream)

    @precondition(_check_context_initialized)
    def sync(self):
        handle_return(cudart.cudaDeviceSynchronize())
