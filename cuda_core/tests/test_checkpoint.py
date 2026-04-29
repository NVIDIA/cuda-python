# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

import pytest

from cuda.core import checkpoint


class _MockDriverProcessState(IntEnum):
    CU_PROCESS_STATE_RUNNING = 0
    CU_PROCESS_STATE_LOCKED = 1
    CU_PROCESS_STATE_CHECKPOINTED = 2
    CU_PROCESS_STATE_FAILED = 3


class _MockDriverResult(IntEnum):
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


class _MockDriver:
    CUresult = _MockDriverResult
    CUprocessState = _MockDriverProcessState
    CUcheckpointGpuPair = _CheckpointGpuPair
    CUcheckpointLockArgs = _CheckpointLockArgs
    CUcheckpointRestoreArgs = _CheckpointRestoreArgs

    def __init__(self, process_state=_MockDriverProcessState.CU_PROCESS_STATE_CHECKPOINTED):
        self.calls = []
        self.process_state = process_state

    def cuCheckpointProcessGetState(self, pid):
        self.calls.append(("get_state", pid))
        return (0, self.process_state)

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
    driver = _MockDriver()
    monkeypatch.setattr(checkpoint, "_get_driver", lambda: driver)

    def handle_return(driver, result):
        if len(result) == 1:
            return None
        return result[1]

    monkeypatch.setattr(checkpoint, "_handle_return", handle_return)
    return driver


def test_public_checkpoint_symbols():
    assert set(checkpoint.ProcessStateT.__args__) == {"running", "locked", "checkpointed", "failed"}
    assert checkpoint.__all__ == ["Process"]
    for name in ("Any", "Mapping", "Literal", "IntEnum", "dataclass", "handle_return", "ProcessState"):
        assert not hasattr(checkpoint, name)


@pytest.mark.parametrize(
    ("process_state", "expected"),
    [
        (_MockDriverProcessState.CU_PROCESS_STATE_RUNNING, "running"),
        (_MockDriverProcessState.CU_PROCESS_STATE_LOCKED, "locked"),
        (_MockDriverProcessState.CU_PROCESS_STATE_CHECKPOINTED, "checkpointed"),
        (_MockDriverProcessState.CU_PROCESS_STATE_FAILED, "failed"),
    ],
)
def test_process_state(checkpoint_driver, process_state, expected):
    checkpoint_driver.process_state = process_state

    state = checkpoint.Process(42).state

    assert state == expected
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
    driver = _MockDriver()
    result = (getattr(driver.CUresult, error_name),)

    with pytest.raises(RuntimeError, match="CUDA checkpointing is not supported"):
        checkpoint._handle_return(driver, result)


def test_get_driver_caches_capability_check(monkeypatch):
    calls = {"binding_version": 0, "driver_version": 0}

    def binding_version():
        calls["binding_version"] += 1
        return (13, 0, 2)

    def driver_version():
        calls["driver_version"] += 1
        return (12, 8, 0)

    driver = _MockDriver()
    monkeypatch.setattr(checkpoint, "_driver", driver)
    monkeypatch.setattr(checkpoint, "_driver_capability_checked", False)
    monkeypatch.setattr(checkpoint, "_binding_version", binding_version)
    monkeypatch.setattr(checkpoint, "_driver_version", driver_version)

    assert checkpoint._get_driver() is driver
    assert checkpoint._get_driver() is driver
    assert calls == {"binding_version": 1, "driver_version": 1}


@pytest.mark.parametrize("binding_version", [(12, 7, 0), (13, 0, 1)])
def test_get_driver_rejects_unsupported_binding_version(monkeypatch, binding_version):
    monkeypatch.setattr(checkpoint, "_driver", _MockDriver())
    monkeypatch.setattr(checkpoint, "_driver_capability_checked", False)
    monkeypatch.setattr(checkpoint, "_binding_version", lambda: binding_version)

    with pytest.raises(RuntimeError, match="CUDA checkpointing requires cuda.bindings"):
        checkpoint._get_driver()


def test_get_driver_rejects_missing_binding_symbols(monkeypatch):
    monkeypatch.setattr(checkpoint, "_driver", object())
    monkeypatch.setattr(checkpoint, "_driver_capability_checked", False)
    monkeypatch.setattr(checkpoint, "_binding_version", lambda: (13, 0, 2))

    with pytest.raises(RuntimeError, match="Missing: cuCheckpointProcessCheckpoint"):
        checkpoint._get_driver()


def test_get_driver_rejects_unsupported_driver_version(monkeypatch):
    monkeypatch.setattr(checkpoint, "_driver", _MockDriver())
    monkeypatch.setattr(checkpoint, "_driver_capability_checked", False)
    monkeypatch.setattr(checkpoint, "_binding_version", lambda: (13, 0, 2))
    monkeypatch.setattr(checkpoint, "_driver_version", lambda: (12, 7, 0))

    with pytest.raises(RuntimeError, match="CUDA checkpointing is not supported"):
        checkpoint._get_driver()


def test_checkpoint_apis_translate_missing_runtime_symbol():
    driver = _MockDriver()

    def missing_checkpoint_symbol():
        raise RuntimeError('Function "cuCheckpointProcessLock" not found')

    with pytest.raises(RuntimeError, match="CUDA checkpointing is not supported"):
        checkpoint._call_driver(driver, missing_checkpoint_symbol)
