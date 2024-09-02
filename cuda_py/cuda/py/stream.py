import os

from cuda import cuda, cudart
from cuda.py._utils import handle_return


class Stream:

    __slots__ = ("_handle", "_nonblocking", "_priority", "_owner", "_builtin")

    def __init__(self, obj=None, *, nonblocking=True, priority=None):
        # minimal requirements for the destructor
        self._handle = None
        self._owner = None
        self._builtin = False

        if obj is not None:
            if not hasattr(obj, "__cuda_stream__"):
                raise ValueError
            self._handle = cuda.CUstream(obj.__cuda_stream__())
            # TODO: check if obj is created under the current context/device
            self._owner = obj
            self._nonblocking = None  # delayed
            self._priority = None  # delayed
            return

        if nonblocking:
            flags = cuda.CUstream_flags.CU_STREAM_NON_BLOCKING
        else:
            flags = cuda.CUstream_flags.CU_STREAM_DEFAULT

        if priority is not None:
            high, low = handle_return(
                cudart.cudaDeviceGetStreamPriorityRange())
            if not (low <= priority <= high):
                raise ValueError(f"{priority=} is out of range {[low, high]}")
        else:
            priority = 0

        self._handle = handle_return(
            cuda.cuStreamCreateWithPriority(flags, priority))
        self._owner = None  # TODO: hold the Context object?
        self._nonblocking = nonblocking
        self._priority = priority

    def __del__(self):
        if self._owner is None and self._handle and not self._builtin:
            handle_return(cuda.cuStreamDestroy(self._handle))

    def __cuda_stream__(self):
        return int(self._handle)

    @property
    def nonblocking(self):
        if self._nonblocking is None:
            flag = handle_return(cuda.cuStreamGetFlags(self._handle))
            if flag == cuda.CUstream_flags.CU_STREAM_NON_BLOCKING:
                self._nonblocking = True
            else:
                self._nonblocking = False
        return self._nonblocking

    @property
    def priority(self):
        if self._priority is None:
            prio = handle_return(cuda.cuStreamGetPriority(self._handle))
            self._priority = prio
        return self._priority

    def sync(self):
        handle_return(cuda.cuStreamSynchronize(self._handle))


class _LegacyDefaultStream(Stream):

    def __init__(self):
        self._handle = cuda.CUstream(cuda.CU_STREAM_LEGACY)
        self._owner = None
        self._nonblocking = None  # delayed
        self._priority = None  # delayed
        self._builtin = True


class _PerThreadDefaultStream(Stream):

    def __init__(self):
        self._handle = cuda.CUstream(cuda.CU_STREAM_PER_THREAD)
        self._owner = None
        self._nonblocking = None  # delayed
        self._priority = None  # delayed
        self._builtin = True


LEGACY_DEFAULT_STREAM = _LegacyDefaultStream()
PER_THREAD_DEFAULT_STREAM = _PerThreadDefaultStream()


def default_stream():
    # TODO: flip the default
    use_ptds = int(os.environ.get('CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM', 0))
    if use_ptds:
        return PER_THREAD_DEFAULT_STREAM
    else:
        return LEGACY_DEFAULT_STREAM
