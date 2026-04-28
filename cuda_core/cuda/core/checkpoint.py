# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass
from enum import IntEnum as _IntEnum
from typing import Any as _Any

from cuda.core._utils.cuda_utils import handle_return as _handle_cuda_return

try:
    from cuda.bindings import driver as _driver
except ImportError:
    from cuda import cuda as _driver


class ProcessState(_IntEnum):
    """
    CUDA checkpoint state for a process.
    """

    RUNNING = 0
    LOCKED = 1
    CHECKPOINTED = 2
    FAILED = 3


@_dataclass(frozen=True)
class Process:
    """
    CUDA process that can be locked, checkpointed, restored, and unlocked.

    Parameters
    ----------
    pid : int
        Process ID of the CUDA process.
    """

    pid: int

    def __post_init__(self):
        _check_pid(self.pid)

    @property
    def state(self) -> ProcessState:
        """
        CUDA checkpoint state for this process.
        """
        driver = _get_driver()
        state = _handle_return(driver, driver.cuCheckpointProcessGetState(self.pid))
        return ProcessState(int(state))

    @property
    def restore_thread_id(self) -> int:
        """
        CUDA restore thread ID for this process.
        """
        driver = _get_driver()
        return _handle_return(driver, driver.cuCheckpointProcessGetRestoreThreadId(self.pid))

    def lock(self, timeout_ms: int = 0) -> None:
        """
        Lock this process, blocking further CUDA API calls.

        Parameters
        ----------
        timeout_ms : int, optional
            Timeout in milliseconds. A value of 0 indicates no timeout.
        """
        driver = _get_driver()
        args = driver.CUcheckpointLockArgs()
        args.timeoutMs = _check_timeout_ms(timeout_ms)
        _handle_return(driver, driver.cuCheckpointProcessLock(self.pid, args))

    def checkpoint(self) -> None:
        """
        Checkpoint the GPU memory contents of this locked process.
        """
        driver = _get_driver()
        _handle_return(driver, driver.cuCheckpointProcessCheckpoint(self.pid, None))

    def restore(self, gpu_mapping: _Mapping[_Any, _Any] | None = None) -> None:
        """
        Restore this checkpointed process.

        Parameters
        ----------
        gpu_mapping : mapping, optional
            GPU UUID remapping from each checkpointed GPU UUID to the GPU UUID
            to restore onto. If provided, the mapping must contain every
            checkpointed GPU UUID.
        """
        driver = _get_driver()
        args = _make_restore_args(driver, gpu_mapping)
        _handle_return(driver, driver.cuCheckpointProcessRestore(self.pid, args))

    def unlock(self) -> None:
        """
        Unlock this locked process so it can resume CUDA API calls.
        """
        driver = _get_driver()
        _handle_return(driver, driver.cuCheckpointProcessUnlock(self.pid, None))


def _get_driver():
    required = (
        "cuCheckpointProcessCheckpoint",
        "cuCheckpointProcessGetRestoreThreadId",
        "cuCheckpointProcessGetState",
        "cuCheckpointProcessLock",
        "cuCheckpointProcessRestore",
        "cuCheckpointProcessUnlock",
        "CUcheckpointGpuPair",
        "CUcheckpointLockArgs",
        "CUcheckpointRestoreArgs",
    )
    missing = [name for name in required if not hasattr(_driver, name)]
    if missing:
        raise RuntimeError(
            f"CUDA checkpointing requires cuda.bindings with CUDA checkpoint API support. Missing: {', '.join(missing)}"
        )
    return _driver


def _handle_return(driver, result):
    err = result[0]
    not_supported_errors = (
        getattr(driver.CUresult, "CUDA_ERROR_NOT_FOUND", None),
        getattr(driver.CUresult, "CUDA_ERROR_NOT_SUPPORTED", None),
    )
    if err in not_supported_errors:
        raise RuntimeError(
            "CUDA checkpointing is not supported by the installed NVIDIA driver. "
            "Upgrade to a driver version with CUDA checkpoint API support."
        )

    return _handle_cuda_return(result)


def _check_pid(pid: int) -> int:
    if isinstance(pid, bool) or not isinstance(pid, int):
        raise TypeError("pid must be an int")
    if pid <= 0:
        raise ValueError("pid must be a positive int")
    return pid


def _check_timeout_ms(timeout_ms: int) -> int:
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
        raise TypeError("timeout_ms must be an int")
    if timeout_ms < 0:
        raise ValueError("timeout_ms must be >= 0")
    return timeout_ms


def _make_restore_args(driver, gpu_mapping: _Mapping[_Any, _Any] | None):
    if gpu_mapping is None:
        return None
    if not isinstance(gpu_mapping, _Mapping):
        raise TypeError("gpu_mapping must be a mapping from checkpointed GPU UUID to restore GPU UUID")

    pairs = []
    for old_uuid, new_uuid in gpu_mapping.items():
        pair = driver.CUcheckpointGpuPair()
        pair.oldUuid = old_uuid
        pair.newUuid = new_uuid
        pairs.append(pair)

    if not pairs:
        return None

    args = driver.CUcheckpointRestoreArgs()
    args.gpuPairs = pairs
    args.gpuPairsCount = len(pairs)
    return args


__all__ = [
    "Process",
    "ProcessState",
]
