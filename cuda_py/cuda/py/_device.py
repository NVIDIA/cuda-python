import threading
import warnings

from cuda import cuda, cudart
from cuda.py._utils import handle_return, CUDAError
from cuda.py._context import Context
from cuda.py._memory import _DefaultAsyncMempool, MemoryResource
from cuda.py._stream import default_stream, Stream


_tls = threading.local()
_tls_lock = threading.Lock()


class Device:

    __slots__ = ("_id", "_mr")
    Stream = Stream

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
                    _tls.devices.append(dev)

        return _tls.devices[device_id]

    @property
    def device_id(self):
        return self._id

    @property
    def pci_bus_id(self):
        bus_id = handle_return(cudart.cudaDeviceGetPCIBusId(13, self._id))
        return bus_id[:12].decode()

    @property
    def uuid(self):
        driver_ver = handle_return(cuda.cuDriverGetVersion())
        if driver_ver >= 11040:
            uuid = handle_return(cuda.cuDeviceGetUuid_v2(self._id))
        else:
            uuid = handle_return(cuda.cuDeviceGetUuid(self._id))
        uuid = uuid.bytes.hex()
        # 8-4-4-4-12
        return f"{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:]}"

    @property
    def name(self):
        # assuming a GPU name is less than 128 characters...
        name = handle_return(cuda.cuDeviceGetName(128, self._id))
        name = name.split(b'\0')[0]
        return name.decode()

    @property
    def properties(self):
        return handle_return(cudart.cudaGetDeviceProperties(self._id))

    @property
    def compute_capability(self):
        major = handle_return(cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, self._id))
        minor = handle_return(cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, self._id))
        return (major, minor)

    @property
    def context(self):
        ctx = handle_return(cuda.cuCtxGetCurrent())
        if int(ctx) == 0:
            raise CUDAError("the device is not yet initialized, "
                            "perhaps you forgot to call .use() first?")
        return Context._from_ctx(ctx, self._id)

    @property
    def memory_resource(self):
        return self._mr

    @memory_resource.setter
    def memory_resource(self, mr):
        if not isinstance(mr, MemoryResource):
            raise TypeError
        self._mr = mr

    @property
    def default_stream(self):
        return default_stream()

    def __int__(self):
        return self._id

    def __repr__(self):
        return f"<Device {self._id} ({self.name})>"

    def use(self, ctx=None):
        if ctx is not None:
            if not isinstance(ctx, Context):
                raise TypeError("a Context object is required")
            if ctx._id != self._id:
                raise RuntimeError("the provided context was created on a different "
                                  f"device {ctx._id} other than the target {self._id}")
            prev_ctx = handle_return(cuda.cuCtxPopCurrent())
            handle_return(cuda.cuCtxPushCurrent(ctx._handle))
            if int(prev_ctx) == 0:
                return None
            else:
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

    def allocate(self, size, stream=None):
        if stream is None:
            stream = default_stream()
        return self._mr.allocate(size, stream)

    def sync(self):
        handle_return(cudart.cudaDeviceSynchronize())
