# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping as _Mapping
from typing import Any as _Any
from typing import Literal as _Literal

from cuda.core._utils.cuda_utils import handle_return as _handle_cuda_return
from cuda.core._utils.version import binding_version as _binding_version
from cuda.core._utils.version import driver_version as _driver_version

try:
    from cuda.bindings import driver as _driver
except ImportError:
    from cuda import cuda as _driver


ProcessStateT = _Literal["running", "locked", "checkpointed", "failed"]

_PROCESS_STATE_NAMES: dict[int, ProcessStateT] = {
    0: "running",
    1: "locked",
    2: "checkpointed",
    3: "failed",
}

_REQUIRED_BINDING_ATTRS = (
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
_REQUIRED_DRIVER_VERSION = (12, 8, 0)
_driver_capability_checked = False


class Process:
    """
    CUDA process that can be locked, checkpointed, restored, and unlocked.

    Parameters
    ----------
    pid : int
        Process ID of the CUDA process.
    """

    __slots__ = ("pid",)

    def __init__(self, pid: int):
        self.pid = _check_pid(pid)

    @property
    def state(self) -> ProcessStateT:
        """
        CUDA checkpoint state for this process.
        """
        driver = _get_driver()
        state = _call_driver(driver, driver.cuCheckpointProcessGetState, self.pid)
        state_value = int(state)
        try:
            return _PROCESS_STATE_NAMES[state_value]
        except KeyError as e:
            raise RuntimeError(f"Unknown CUDA checkpoint process state: {state_value}") from e

    @property
    def restore_thread_id(self) -> int:
        """
        CUDA restore thread ID for this process.
        """
        driver = _get_driver()
        return _call_driver(driver, driver.cuCheckpointProcessGetRestoreThreadId, self.pid)

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
        _call_driver(driver, driver.cuCheckpointProcessLock, self.pid, args)

    def checkpoint(self) -> None:
        """
        Checkpoint the GPU memory contents of this locked process.
        """
        driver = _get_driver()
        _call_driver(driver, driver.cuCheckpointProcessCheckpoint, self.pid, None)

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
        _call_driver(driver, driver.cuCheckpointProcessRestore, self.pid, args)

    def unlock(self) -> None:
        """
        Unlock this locked process so it can resume CUDA API calls.
        """
        driver = _get_driver()
        _call_driver(driver, driver.cuCheckpointProcessUnlock, self.pid, None)


def _get_driver():
    global _driver_capability_checked
    if _driver_capability_checked:
        return _driver

    binding_ver = _binding_version()
    if not _binding_version_supports_checkpoint(binding_ver):
        raise RuntimeError(
            "CUDA checkpointing requires cuda.bindings with CUDA checkpoint API support. "
            f"Found cuda.bindings {'.'.join(str(part) for part in binding_ver[:3])}."
        )

    missing = [name for name in _REQUIRED_BINDING_ATTRS if not hasattr(_driver, name)]
    if missing:
        raise RuntimeError(
            f"CUDA checkpointing requires cuda.bindings with CUDA checkpoint API support. Missing: {', '.join(missing)}"
        )

    driver_ver = _driver_version()
    if driver_ver < _REQUIRED_DRIVER_VERSION:
        raise RuntimeError(
            "CUDA checkpointing is not supported by the installed NVIDIA driver. "
            "Upgrade to a driver version with CUDA checkpoint API support."
        )

    _driver_capability_checked = True
    return _driver


def _binding_version_supports_checkpoint(version) -> bool:
    major, minor, patch = version[:3]
    return (major == 12 and (minor, patch) >= (8, 0)) or (major == 13 and (minor, patch) >= (0, 2)) or major > 13


def _call_driver(driver, func, *args):
    try:
        result = func(*args)
    except RuntimeError as e:
        if "cuCheckpointProcess" in str(e) and "not found" in str(e):
            raise RuntimeError(
                "CUDA checkpointing is not supported by the installed NVIDIA driver. "
                "Upgrade to a driver version with CUDA checkpoint API support."
            ) from e
        raise
    return _handle_return(driver, result)


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
]
