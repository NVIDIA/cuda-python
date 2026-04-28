# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

import pytest

from cuda.core import checkpoint


class _DriverProcessState(IntEnum):
    CU_PROCESS_STATE_RUNNING = 0
    CU_PROCESS_STATE_LOCKED = 1
    CU_PROCESS_STATE_CHECKPOINTED = 2
    CU_PROCESS_STATE_FAILED = 3


class _DriverResult(IntEnum):
    CUDA_SUCCESS = 0
    CUDA_ERROR_NOT_FOUND = 500
    CUDA_ERROR_NOT_SUPPORTED = 801


class _Uuid:
    pass


class _CheckpointGpuPair:
    def __init__(self):
        self.oldUuid = None
        self.newUuid = None


class _CheckpointLockArgs:
    def __init__(self):
        self.timeoutMs = None


class _CheckpointRestoreArgs:
    def __init__(self):
        self.gpuPairs = None
        self.gpuPairsCount = None


class _Driver:
    CUresult = _DriverResult
    CUprocessState = _DriverProcessState
    CUcheckpointGpuPair = _CheckpointGpuPair
    CUcheckpointLockArgs = _CheckpointLockArgs
    CUcheckpointRestoreArgs = _CheckpointRestoreArgs

    def __init__(self):
        self.calls = []

    def cuCheckpointProcessGetState(self, pid):
        self.calls.append(("get_state", pid))
        return (0, self.CUprocessState.CU_PROCESS_STATE_CHECKPOINTED)

    def cuCheckpointProcessGetRestoreThreadId(self, pid):
        self.calls.append(("get_restore_thread_id", pid))
        return (0, 123)

    def cuCheckpointProcessLock(self, pid, args):
        self.calls.append(("lock", pid, args))
        return (0,)

    def cuCheckpointProcessCheckpoint(self, pid, args):
        self.calls.append(("checkpoint", pid, args))
        return (0,)

    def cuCheckpointProcessRestore(self, pid, args):
        self.calls.append(("restore", pid, args))
        return (0,)

    def cuCheckpointProcessUnlock(self, pid, args):
        self.calls.append(("unlock", pid, args))
        return (0,)


@pytest.fixture
def checkpoint_driver(monkeypatch):
    driver = _Driver()
    monkeypatch.setattr(checkpoint, "_get_driver", lambda: driver)

    def handle_return(driver, result):
        if len(result) == 1:
            return None
        return result[1]

    monkeypatch.setattr(checkpoint, "_handle_return", handle_return)
    return driver


def test_public_checkpoint_symbols():
    assert checkpoint.ProcessState.CHECKPOINTED == 2
    assert "Process" in checkpoint.__all__
    assert "ProcessState" in checkpoint.__all__
    for name in ("Any", "Mapping", "IntEnum", "dataclass", "handle_return"):
        assert not hasattr(checkpoint, name)


def test_process_state(checkpoint_driver):
    state = checkpoint.Process(42).state

    assert state is checkpoint.ProcessState.CHECKPOINTED
    assert checkpoint_driver.calls == [("get_state", 42)]


def test_process_restore_thread_id(checkpoint_driver):
    tid = checkpoint.Process(42).restore_thread_id

    assert tid == 123
    assert checkpoint_driver.calls == [("get_restore_thread_id", 42)]


def test_process_lock_sets_timeout_ms(checkpoint_driver):
    checkpoint.Process(42).lock(timeout_ms=500)

    opname, pid, args = checkpoint_driver.calls[0]
    assert opname == "lock"
    assert pid == 42
    assert isinstance(args, _CheckpointLockArgs)
    assert args.timeoutMs == 500


def test_process_checkpoint_and_unlock_pass_null_args(checkpoint_driver):
    process = checkpoint.Process(42)
    process.checkpoint()
    process.unlock()

    assert checkpoint_driver.calls == [
        ("checkpoint", 42, None),
        ("unlock", 42, None),
    ]


def test_process_restore_accepts_gpu_uuid_mapping(checkpoint_driver):
    old_uuid = _Uuid()
    new_uuid = _Uuid()

    checkpoint.Process(42).restore(gpu_mapping={old_uuid: new_uuid})

    opname, pid, args = checkpoint_driver.calls[0]
    assert opname == "restore"
    assert pid == 42
    assert isinstance(args, _CheckpointRestoreArgs)
    assert args.gpuPairsCount == 1
    assert len(args.gpuPairs) == 1
    assert args.gpuPairs[0].oldUuid is old_uuid
    assert args.gpuPairs[0].newUuid is new_uuid


def test_process_restore_empty_gpu_mapping_uses_null_args(checkpoint_driver):
    checkpoint.Process(42).restore(gpu_mapping={})

    assert checkpoint_driver.calls == [("restore", 42, None)]


@pytest.mark.parametrize(
    ("args", "error_type", "match"),
    [
        (("123",), TypeError, "pid must be an int"),
        ((True,), TypeError, "pid must be an int"),
        ((0,), ValueError, "pid must be a positive int"),
    ],
)
def test_process_rejects_invalid_pid(checkpoint_driver, args, error_type, match):
    with pytest.raises(error_type, match=match):
        checkpoint.Process(*args)


@pytest.mark.parametrize(
    ("timeout_ms", "error_type", "match"),
    [
        (-1, ValueError, "timeout_ms must be >= 0"),
        (1.5, TypeError, "timeout_ms must be an int"),
        (True, TypeError, "timeout_ms must be an int"),
    ],
)
def test_process_lock_rejects_invalid_timeout(checkpoint_driver, timeout_ms, error_type, match):
    with pytest.raises(error_type, match=match):
        checkpoint.Process(42).lock(timeout_ms=timeout_ms)


def test_process_restore_rejects_invalid_gpu_mapping(checkpoint_driver):
    with pytest.raises(TypeError, match="gpu_mapping must be a mapping"):
        checkpoint.Process(42).restore(gpu_mapping=[object()])


@pytest.mark.parametrize(
    "error_name",
    [
        "CUDA_ERROR_NOT_FOUND",
        "CUDA_ERROR_NOT_SUPPORTED",
    ],
)
def test_checkpoint_apis_reject_unsupported_driver(error_name):
    driver = _Driver()
    result = (getattr(driver.CUresult, error_name),)

    with pytest.raises(RuntimeError, match="CUDA checkpointing is not supported"):
        checkpoint._handle_return(driver, result)
