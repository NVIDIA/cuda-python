# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Real GPU tests for cuda.core.checkpoint — no mocks.
#
# Lifecycle tests self-checkpoint the current process (os.getpid()) and
# exercise lock / checkpoint / restore / unlock through the real driver.
#
# Migration tests attempt GPU UUID remapping following the pattern from
# NVIDIA/cuda-checkpoint r580-migration-api.c.  They require ≥2 GPUs of
# the same chip type and a driver that supports migration; the tests skip
# gracefully when the hardware or driver cannot satisfy this.

import os
import sys
from contextlib import suppress

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


def _build_rotation_mapping(devices):
    """GPU i UUID -> GPU (i+1) % N UUID for every visible device.

    Returns a ``{str: str}`` dict of UUID strings suitable for
    :meth:`~checkpoint.Process.restore`.
    """
    n = len(devices)
    return {devices[i].uuid: devices[(i + 1) % n].uuid for i in range(n)}


def _find_same_chip_pair(devices):
    """Return (i, j) indices of two devices with the same name, or None."""
    seen = {}
    for i, dev in enumerate(devices):
        name = dev.name
        if name in seen:
            return (seen[name], i)
        seen[name] = i
    return None


def _run_or_skip_unsupported(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except RuntimeError as exc:
        if "CUDA checkpointing is not supported" in str(exc):
            pytest.skip(str(exc))
        raise


# -- Fixtures --------------------------------------------------------------


@pytest.fixture
def self_process(init_cuda):
    """checkpoint.Process wrapping os.getpid(), with safety unlock on teardown.

    Records the initial device so tests that call ``set_current()`` on a
    different device (e.g. migration tests) are side-effect free.
    """
    original_device = init_cuda
    proc = checkpoint.Process(os.getpid())
    yield proc
    # Ensure the process is not left locked if the test fails mid-lifecycle.
    try:
        st = proc.state
    except Exception:
        st = None
    if st == "checkpointed":
        with suppress(Exception):
            proc.restore()
        with suppress(Exception):
            proc.unlock()
    elif st == "locked":
        with suppress(Exception):
            proc.unlock()
    # Restore the original device so init_cuda's teardown pops the right context.
    original_device.set_current()


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
        assert not hasattr(checkpoint, "ProcessStateT")


# -- Lifecycle (single GPU, real driver) -----------------------------------


@needs_checkpoint
class TestCheckpointLifecycle:
    def test_initial_state_is_running(self, self_process):
        assert self_process.state == "running"

    def test_restore_thread_id_is_positive(self, self_process):
        tid = self_process.restore_thread_id
        assert isinstance(tid, int)
        assert tid > 0

    def test_lock_unlock(self, self_process):
        _run_or_skip_unsupported(self_process.lock)
        assert self_process.state == "locked"
        self_process.unlock()
        assert self_process.state == "running"

    def test_lock_default_timeout(self, self_process):
        """lock() with the default timeout_ms=0 (no timeout)."""
        _run_or_skip_unsupported(self_process.lock)
        assert self_process.state == "locked"
        self_process.unlock()

    def test_lock_with_timeout(self, self_process):
        _run_or_skip_unsupported(self_process.lock, timeout_ms=5000)
        assert self_process.state == "locked"
        self_process.unlock()

    def test_full_cycle_no_migration(self, self_process):
        """lock -> checkpoint -> restore -> unlock, verify state at each step."""
        _run_or_skip_unsupported(self_process.lock)
        assert self_process.state == "locked"

        _run_or_skip_unsupported(self_process.checkpoint)
        assert self_process.state == "checkpointed"

        _run_or_skip_unsupported(self_process.restore)
        assert self_process.state == "locked"  # restore leaves process locked

        self_process.unlock()
        assert self_process.state == "running"


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

    @staticmethod
    def _try_migration(proc, gpu_mapping):
        """Attempt a single checkpoint-restore with migration.

        Returns True on success.  Skips the test if the driver rejects
        the migration with CUDA_ERROR_INVALID_VALUE (known limitation
        on some architectures / driver versions).
        """
        _run_or_skip_unsupported(proc.lock)
        _run_or_skip_unsupported(proc.checkpoint)
        try:
            _run_or_skip_unsupported(proc.restore, gpu_mapping=gpu_mapping)
        except (CUDAError, RuntimeError) as exc:
            # Recover: restore without migration, then unlock.
            proc.restore()
            proc.unlock()
            if "INVALID_VALUE" in str(exc):
                pytest.skip(
                    "Driver does not support GPU migration on this hardware "
                    "(CUDA_ERROR_INVALID_VALUE — see NVBug 5437334)"
                )
            raise
        proc.unlock()
        return True

    def test_rotation_migrates_context(self, self_process):
        """Rotate context through all GPUs and back to the origin.

        Builds a rotation mapping (device i -> device (i+1) % N) for
        every visible device and performs N rotations.  After each step
        the context device UUID is checked.  After N steps the context
        should be back on the original device.
        """
        devices = Device.get_all_devices()
        if len(devices) < 2:
            pytest.skip("GPU migration tests require at least 2 GPUs")
        if _find_same_chip_pair(devices) is None:
            pytest.skip("GPU migration requires at least 2 GPUs of the same chip type")

        gpu_mapping = _build_rotation_mapping(devices)
        uuid_origin = Device().uuid

        for step in range(len(devices)):
            expected_uuid = devices[(step + 1) % len(devices)].uuid

            self._try_migration(self_process, gpu_mapping)

            assert Device().uuid == expected_uuid, f"Step {step}: expected UUID {expected_uuid}, got {Device().uuid}"

        # After N rotations, back at the origin.
        assert Device().uuid == uuid_origin

    def test_swap_identical_gpus(self, self_process):
        """Swap context between two GPUs of the same chip type.

        Sets the context on one of the pair members so that a successful
        migration is observable (the context UUID changes).
        """
        devices = Device.get_all_devices()
        pair = _find_same_chip_pair(devices)
        if pair is None:
            pytest.skip("No two GPUs of the same chip type found")

        i, j = pair
        # Place context on device i so the swap is observable.
        devices[i].set_current()

        # Build an identity mapping, then swap the pair (using UUID strings).
        gpu_mapping = {d.uuid: d.uuid for d in devices}
        gpu_mapping[devices[i].uuid] = devices[j].uuid
        gpu_mapping[devices[j].uuid] = devices[i].uuid

        assert Device().uuid == devices[i].uuid

        self._try_migration(self_process, gpu_mapping)
        uuid_after = Device().uuid

        if uuid_after == devices[i].uuid:
            pytest.skip("Driver accepted GPU swap but migration is a no-op on this hardware/driver version")
        assert uuid_after == devices[j].uuid
