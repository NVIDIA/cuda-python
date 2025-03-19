# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from dataclasses import dataclass
from typing import Optional, Any

from cuda.core.experimental._utils.clear_error_support import assert_type
from cuda.core.experimental._utils.cuda_utils import driver


@dataclass
class ContextOptions:
    pass  # TODO


class Context:
    __slots__ = ("_handle", "_id")

    def __new__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Context objects cannot be instantiated directly. Please use Device or Stream APIs.")

    @classmethod
    def _from_ctx(cls, obj: driver.CUcontext, dev_id: int) -> "Context":
        assert_type(obj, driver.CUcontext)
        ctx = super().__new__(cls)
        ctx._handle = obj
        ctx._id = dev_id
        return ctx

    @classmethod
    def _init(cls, device_id: int, options: Optional[ContextOptions] = None) -> "Context":
        """Initialize a new context."""
        handle = driver.CUcontext()
        handle_return(driver.cuCtxCreate(handle, options, device_id))
        return cls._from_ctx(handle, device_id)

    @classmethod
    def current(cls) -> Optional["Context"]:
        """Get the current context."""
        handle = driver.CUcontext()
        handle_return(driver.cuCtxGetCurrent(handle))
        if int(handle) == 0:
            return None
        device_id = driver.CUdevice()
        handle_return(driver.cuCtxGetDevice(device_id))
        return cls._from_ctx(handle, device_id)

    def set_current(self) -> None:
        """Set this context as the current context."""
        handle_return(driver.cuCtxSetCurrent(self._handle))

    def pop_current(self) -> None:
        """Pop this context from the current thread's context stack."""
        handle_return(driver.cuCtxPopCurrent(self._handle))

    def push_current(self) -> None:
        """Push this context onto the current thread's context stack."""
        handle_return(driver.cuCtxPushCurrent(self._handle))

    @property
    def handle(self) -> driver.CUcontext:
        """Get the CUDA context handle."""
        return self._handle

    @property
    def device_id(self) -> int:
        """Get the device ID associated with this context."""
        return self._id

    def __repr__(self) -> str:
        """Return a string representation of the context."""
        return f"Context(device_id={self._id})"
