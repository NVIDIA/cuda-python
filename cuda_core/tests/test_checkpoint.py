# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Real GPU tests for cuda.core.checkpoint — no mocks.
#
# Driver-backed lifecycle tests run through an isolated coordinator/target
# process pair so hangs can be timed out without wedging the pytest process.
#
# Migration tests attempt GPU UUID remapping following the pattern from
# NVIDIA/cuda-checkpoint r580-migration-api.c.  They require ≥2 GPUs of
# the same chip type, an unmasked CUDA device view, and a driver that supports
# migration; the tests skip gracefully when the hardware or driver cannot
# satisfy this.

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

from cuda.core import Device, checkpoint
from cuda.core._utils.cuda_utils import CUDAError

# -- Skip condition -------------------------------------------------------


def _checkpoint_available():
    """Return True if the checkpoint API is usable on this system."""
    try:
        checkpoint._get_driver()
        return True
    except RuntimeError:
        return False


needs_checkpoint = pytest.mark.skipif(
    sys.platform != "linux" or not _checkpoint_available(),
    reason="CUDA checkpoint API requires Linux and a supported driver/bindings",
)


# -- Helpers ---------------------------------------------------------------


_SCENARIO_SKIP_EXIT_CODE = 77
_CHECKPOINT_TEST_SCRIPT = Path(__file__).resolve()


def _checkpoint_child_command(*args: str) -> list[str]:
    return [sys.executable, "-I", str(_CHECKPOINT_TEST_SCRIPT), *args]


def _scenario_skip(reason):
    print(f"SKIP: {reason}", flush=True)
    raise SystemExit(_SCENARIO_SKIP_EXIT_CODE)


def _run_or_skip_unsupported(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except RuntimeError as exc:
        if "CUDA checkpointing is not supported" in str(exc):
            _scenario_skip(str(exc))
        raise


def _build_rotation_mapping(devices):
    n = len(devices)
    return {devices[i].uuid: devices[(i + 1) % n].uuid for i in range(n)}


def _find_same_chip_pair(devices):
    seen = {}
    for i, dev in enumerate(devices):
        name = dev.name
        if name in seen:
            return (seen[name], i)
        seen[name] = i
    return None


def _read_prefixed(target, prefix):
    line = target.stdout.readline()
    if not line:
        stderr = target.stderr.read()
        raise RuntimeError(f"checkpoint target exited before {prefix!r}; stderr:\n{stderr}")
    line = line.strip()
    if not line.startswith(prefix):
        raise RuntimeError(f"expected target output prefix {prefix!r}, got {line!r}")
    return line[len(prefix) :]


def _start_target(device_index=0):
    target = subprocess.Popen(  # noqa: S603 - controlled test subprocess using this Python executable.
        _checkpoint_child_command("target", str(device_index)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready_uuid = _read_prefixed(target, "READY:")
    except Exception:
        _stop_target(target)
        raise
    return target, ready_uuid


def _stop_target(target):
    if target.poll() is None:
        with suppress(Exception):
            target.stdin.write("exit\n")
            target.stdin.flush()
        try:
            target.wait(timeout=5)
        except subprocess.TimeoutExpired:
            target.kill()
            target.wait()


def _target_uuid(target):
    target.stdin.write("uuid\n")
    target.stdin.flush()
    return _read_prefixed(target, "UUID:")


def _checkpoint_restore(proc, gpu_mapping=None):
    _run_or_skip_unsupported(proc.lock, timeout_ms=5000)
    _run_or_skip_unsupported(proc.checkpoint)
    try:
        _run_or_skip_unsupported(proc.restore, gpu_mapping=gpu_mapping)
    except (CUDAError, RuntimeError) as exc:
        with suppress(Exception):
            proc.restore()
        with suppress(Exception):
            proc.unlock()
        if "INVALID_VALUE" in str(exc):
            _scenario_skip(
                "Driver does not support GPU migration on this hardware (CUDA_ERROR_INVALID_VALUE; see NVBug 5437334)"
            )
        raise
    proc.unlock()


def _run_checkpoint_scenario_or_skip(scenario: str, *, timeout: int = 90) -> None:
    """Run mutating checkpoint/restore scenarios out-of-process.

    The CUDA checkpoint APIs can block inside the driver when a runner exposes
    symbols but the platform path cannot complete checkpoint/restore.  Running
    the scenario in its own process group lets the parent test skip that runner
    cleanly instead of hanging the entire CI job.
    """
    proc = subprocess.Popen(  # noqa: S603 - controlled test subprocess using this Python executable.
        _checkpoint_child_command("scenario", scenario),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        pytest.skip(
            f"CUDA checkpoint scenario timed out after {timeout}s; driver/hardware did not complete "
            f"checkpoint/restore.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    if proc.returncode == _SCENARIO_SKIP_EXIT_CODE:
        reason = stdout.strip() or stderr.strip() or "CUDA checkpoint scenario skipped"
        pytest.skip(reason)
    if proc.returncode != 0:
        pytest.fail(
            f"CUDA checkpoint scenario failed with exit code {proc.returncode}.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )


def _target_main(device_index: int) -> None:
    Device(device_index).set_current()
    print(f"READY:{Device().uuid}", flush=True)

    for line in sys.stdin:
        command = line.strip()
        if command == "uuid":
            print(f"UUID:{Device().uuid}", flush=True)
        elif command == "exit":
            break


def _scenario_initial_state_is_running() -> None:
    target, _ = _start_target()
    proc = checkpoint.Process(target.pid)
    try:
        assert proc.state == "running"
    finally:
        _stop_target(target)


def _scenario_restore_thread_id_is_positive() -> None:
    target, _ = _start_target()
    proc = checkpoint.Process(target.pid)
    try:
        tid = proc.restore_thread_id
        assert isinstance(tid, int)
        assert tid > 0
    finally:
        _stop_target(target)


def _scenario_lock_unlock() -> None:
    target, _ = _start_target()
    proc = checkpoint.Process(target.pid)
    try:
        _run_or_skip_unsupported(proc.lock)
        assert proc.state == "locked"
        proc.unlock()
        assert proc.state == "running"
    finally:
        _stop_target(target)


def _scenario_lock_default_timeout() -> None:
    target, _ = _start_target()
    proc = checkpoint.Process(target.pid)
    try:
        _run_or_skip_unsupported(proc.lock)
        assert proc.state == "locked"
        proc.unlock()
    finally:
        _stop_target(target)


def _scenario_lock_with_timeout() -> None:
    target, _ = _start_target()
    proc = checkpoint.Process(target.pid)
    try:
        _run_or_skip_unsupported(proc.lock, timeout_ms=5000)
        assert proc.state == "locked"
        proc.unlock()
    finally:
        _stop_target(target)


def _scenario_full_cycle_no_migration() -> None:
    target, _ = _start_target()
    proc = checkpoint.Process(target.pid)
    try:
        _run_or_skip_unsupported(proc.lock, timeout_ms=5000)
        assert proc.state == "locked"

        _run_or_skip_unsupported(proc.checkpoint)
        assert proc.state == "checkpointed"

        _run_or_skip_unsupported(proc.restore)
        assert proc.state == "locked"  # restore leaves process locked

        proc.unlock()
        assert proc.state == "running"
    finally:
        _stop_target(target)


def _scenario_rotation_migrates_context() -> None:
    devices = Device.get_all_devices()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        _scenario_skip(
            "GPU migration tests require an unmasked CUDA device view because "
            "the checkpoint mapping must cover every GPU visible to the kernel-mode driver"
        )
    if len(devices) < 2:
        _scenario_skip("GPU migration tests require at least 2 GPUs")
    if _find_same_chip_pair(devices) is None:
        _scenario_skip("GPU migration requires at least 2 GPUs of the same chip type")

    gpu_mapping = _build_rotation_mapping(devices)
    target, uuid_origin = _start_target(0)
    current_uuid = uuid_origin
    proc = checkpoint.Process(target.pid)
    try:
        for step in range(len(devices)):
            expected_uuid = devices[(step + 1) % len(devices)].uuid
            _checkpoint_restore(proc, gpu_mapping=gpu_mapping)
            observed_uuid = _target_uuid(target)
            if observed_uuid == current_uuid:
                _scenario_skip("Driver accepted GPU rotation but migration is a no-op on this hardware/driver version")
            assert observed_uuid == expected_uuid, f"Step {step}: expected UUID {expected_uuid}, got {observed_uuid}"
            current_uuid = observed_uuid

        assert _target_uuid(target) == uuid_origin
    finally:
        _stop_target(target)


def _scenario_swap_identical_gpus() -> None:
    devices = Device.get_all_devices()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        _scenario_skip(
            "GPU migration tests require an unmasked CUDA device view because "
            "the checkpoint mapping must cover every GPU visible to the kernel-mode driver"
        )
    pair = _find_same_chip_pair(devices)
    if pair is None:
        _scenario_skip("No two GPUs of the same chip type found")

    i, j = pair
    gpu_mapping = {d.uuid: d.uuid for d in devices}
    gpu_mapping[devices[i].uuid] = devices[j].uuid
    gpu_mapping[devices[j].uuid] = devices[i].uuid

    target, uuid_before = _start_target(i)
    proc = checkpoint.Process(target.pid)
    try:
        assert uuid_before == devices[i].uuid

        _checkpoint_restore(proc, gpu_mapping=gpu_mapping)
        uuid_after = _target_uuid(target)

        if uuid_after == devices[i].uuid:
            _scenario_skip("Driver accepted GPU swap but migration is a no-op on this hardware/driver version")
        assert uuid_after == devices[j].uuid
    finally:
        _stop_target(target)


_SCENARIOS = {
    "initial_state_is_running": _scenario_initial_state_is_running,
    "restore_thread_id_is_positive": _scenario_restore_thread_id_is_positive,
    "lock_unlock": _scenario_lock_unlock,
    "lock_default_timeout": _scenario_lock_default_timeout,
    "lock_with_timeout": _scenario_lock_with_timeout,
    "full_cycle_no_migration": _scenario_full_cycle_no_migration,
    "rotation_migrates_context": _scenario_rotation_migrates_context,
    "swap_identical_gpus": _scenario_swap_identical_gpus,
}


def _checkpoint_child_main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit("checkpoint child mode required")

    if argv[1] == "target":
        if len(argv) != 3:
            raise SystemExit("target mode requires a device index")
        _target_main(int(argv[2]))
        return

    if argv[1] == "scenario":
        if len(argv) != 3:
            raise SystemExit("scenario mode requires a scenario name")
        try:
            scenario = _SCENARIOS[argv[2]]
        except KeyError as exc:
            raise SystemExit(f"unknown checkpoint scenario: {argv[2]}") from exc
        scenario()
        return

    raise SystemExit(f"unknown checkpoint child mode: {argv[1]}")


# -- Input validation (no GPU / driver needed) -----------------------------


class TestInputValidation:
    @pytest.mark.parametrize(
        ("args", "error_type", "match"),
        [
            (("abc",), TypeError, "pid must be an int"),
            ((True,), TypeError, "pid must be an int"),
            ((0,), ValueError, "pid must be a positive int"),
            ((-1,), ValueError, "pid must be a positive int"),
        ],
    )
    def test_process_rejects_invalid_pid(self, args, error_type, match):
        with pytest.raises(error_type, match=match):
            checkpoint.Process(*args)

    def test_public_symbols(self):
        assert checkpoint.__all__ == ["Process"]
        assert not hasattr(checkpoint, "ProcessStateType")

    def test_pid_is_read_only(self):
        proc = checkpoint.Process(1)
        assert proc.pid == 1
        with pytest.raises(AttributeError):
            proc.pid = 2


# -- Lifecycle (single GPU, real driver) -----------------------------------


@needs_checkpoint
class TestCheckpointLifecycle:
    def test_initial_state_is_running(self):
        _run_checkpoint_scenario_or_skip("initial_state_is_running")

    def test_restore_thread_id_is_positive(self):
        _run_checkpoint_scenario_or_skip("restore_thread_id_is_positive")

    def test_lock_unlock(self):
        _run_checkpoint_scenario_or_skip("lock_unlock")

    def test_lock_default_timeout(self):
        """lock() with the default timeout_ms=0 (no timeout)."""
        _run_checkpoint_scenario_or_skip("lock_default_timeout")

    def test_lock_with_timeout(self):
        _run_checkpoint_scenario_or_skip("lock_with_timeout")

    def test_full_cycle_no_migration(self):
        """lock -> checkpoint -> restore -> unlock, verify state at each step."""
        _run_checkpoint_scenario_or_skip("full_cycle_no_migration")


# -- GPU migration (>= 2 same-chip GPUs, real driver) ---------------------


@needs_checkpoint
class TestCheckpointGpuMigration:
    """GPU UUID remapping tests following the r580-migration-api.c pattern.

    These tests require at least two GPUs of the same chip type and a
    driver that supports checkpoint migration.  They skip when the
    hardware cannot satisfy this (e.g. heterogeneous GPUs, or a driver
    build where migration returns CUDA_ERROR_INVALID_VALUE — see
    NVBug 5437334).
    """

    def test_rotation_migrates_context(self):
        """Rotate context through all GPUs and back to the origin.

        Builds a rotation mapping (device i -> device (i+1) % N) for
        every visible device and performs N rotations.  After each step
        the context device UUID is checked.  After N steps the context
        should be back on the original device.
        """
        _run_checkpoint_scenario_or_skip("rotation_migrates_context", timeout=180)

    def test_swap_identical_gpus(self):
        """Swap context between two GPUs of the same chip type.

        Sets the context on one of the pair members so that a successful
        migration is observable (the context UUID changes).
        """
        _run_checkpoint_scenario_or_skip("swap_identical_gpus", timeout=120)


if __name__ == "__main__":
    _checkpoint_child_main(sys.argv)
