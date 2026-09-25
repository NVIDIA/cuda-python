# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bridge the CUDA driver's ``cuLogs*`` error-log-management callback into
the standard :mod:`logging` module.

The CUDA driver (CUDA 12.9+) maintains an internal, human-readable error/
warning log stream (see the *Error Log Management* section of the CUDA
driver API docs) that is otherwise only observable via the ``CUDA_LOG_FILE``
environment variable or stderr. This module lets a caller route that stream
into any :class:`logging.Logger`, so it can be filtered, captured (e.g. via
pytest's ``caplog``), and forwarded using ordinary Python logging tooling.

xref: https://github.com/NVIDIA/cuda-python/issues/671
"""

from __future__ import annotations

import ctypes
import logging
import threading

from cuda.bindings import driver as _driver
from cuda.core._utils.cuda_utils import handle_return

__all__ = ["register_cuda_log_bridge", "unregister_cuda_log_bridge"]

# C callback signature expected by cuLogsRegisterCallback:
#     void callback(void *userData, CUlogLevel logLevel, char *message, size_t length)
#
# Note: cuLogsRegisterCallback requires an actual C function pointer. The
# cuda.bindings wrapper does *not* accept a plain Python callable directly
# (it only ever converts the `callbackFunc` argument to an integer address),
# so we build a real function pointer via ctypes and pass its address.
_CUlogsCallback_functype = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t)

# CUlogLevel currently only defines two levels (CUDA 12.9 cuLogs* API).
# Anything unrecognized is conservatively mapped to ERROR.
_CUDA_LEVEL_TO_PY_LEVEL = {
    int(_driver.CUlogLevel.CU_LOG_LEVEL_ERROR): logging.ERROR,
    int(_driver.CUlogLevel.CU_LOG_LEVEL_WARNING): logging.WARNING,
}

_lock = threading.Lock()
# Must hold a reference to the ctypes callback for as long as it is
# registered: once it is garbage collected, the driver would be left
# holding a dangling function pointer and crash the process on the next
# log message.
_c_callback: ctypes._FuncPointer | None = None
_callback_handle = None  # CUlogsCallbackHandle, needed to unregister


def _make_c_callback(logger: logging.Logger) -> ctypes._FuncPointer:
    """Build a ctypes C function pointer that forwards each driver log
    record to ``logger``.
    """

    def _on_log(_user_data, log_level, message_ptr, length):
        # This runs inside a ctypes callback trampoline, potentially from a
        # CUDA-driver-internal thread. It must never raise or block.
        try:
            raw = ctypes.string_at(message_ptr, length) if message_ptr else b""
            msg = raw.decode("utf-8", errors="replace").rstrip("\n")
            py_level = _CUDA_LEVEL_TO_PY_LEVEL.get(log_level, logging.ERROR)
            logger.log(py_level, "[CUDA Driver] %s", msg)
        except Exception:
            logger.exception("cuda_log_bridge: error while forwarding driver log")

    return _CUlogsCallback_functype(_on_log)


def register_cuda_log_bridge(logger: logging.Logger | None = None) -> None:
    """Register a bridge that forwards CUDA driver ``cuLogs*`` messages to
    a Python :class:`logging.Logger`.

    This is a process-wide, idempotent operation: the driver only supports
    a global log stream (not scoped to a context/stream/thread), so calling
    this again while a bridge is already registered is a no-op -- the
    logger passed to the first call remains in effect.

    Parameters
    ----------
    logger : logging.Logger, optional
        The logger that driver messages should be forwarded to. Defaults
        to ``logging.getLogger("cuda.driver")``.

    Raises
    ------
    cuda.core._utils.cuda_utils.CUDAError
        If the underlying ``cuLogsRegisterCallback`` call fails, e.g.
        because the loaded CUDA driver predates the 12.9 cuLogs* API.
    """
    global _c_callback, _callback_handle

    if logger is None:
        logger = logging.getLogger("cuda.driver")

    with _lock:
        if _callback_handle is not None:
            return  # already registered; idempotent

        c_callback = _make_c_callback(logger)
        addr = ctypes.cast(c_callback, ctypes.c_void_p).value

        handle = handle_return(_driver.cuLogsRegisterCallback(addr, None))

        # Keep the ctypes callback object alive for as long as it is
        # registered (see the module-level comment on _c_callback above).
        _c_callback = c_callback
        _callback_handle = handle


def unregister_cuda_log_bridge() -> None:
    """Unregister the bridge previously installed by
    :func:`register_cuda_log_bridge`.

    Idempotent: safe to call even if no bridge is currently registered.

    Raises
    ------
    cuda.core._utils.cuda_utils.CUDAError
        If the underlying ``cuLogsUnregisterCallback`` call fails.
    """
    global _c_callback, _callback_handle

    with _lock:
        if _callback_handle is None:
            return  # nothing registered; idempotent

        handle_return(_driver.cuLogsUnregisterCallback(_callback_handle))

        _c_callback = None
        _callback_handle = None
