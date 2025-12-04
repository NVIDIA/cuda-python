# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test that warnings are emitted when multiprocessing start method is 'fork'
and IPC objects are serialized.

These tests use mocking to simulate the 'fork' start method without actually
using fork, avoiding the need for subprocess isolation.
"""

import warnings
from unittest.mock import patch

from cuda.core.experimental import DeviceMemoryResource, DeviceMemoryResourceOptions, EventOptions
from cuda.core.experimental._event import _reduce_event
from cuda.core.experimental._memory._ipc import (
    _deep_reduce_device_memory_resource,
    _reduce_allocation_handle,
)
from cuda.core.experimental._utils.cuda_utils import reset_fork_warning


def test_warn_on_fork_method_device_memory_resource(ipc_device):
    """Test that warning is emitted when DeviceMemoryResource is pickled with fork method."""
    device = ipc_device
    device.set_current()
    options = DeviceMemoryResourceOptions(max_size=2097152, ipc_enabled=True)
    mr = DeviceMemoryResource(device, options=options)

    with patch("multiprocessing.get_start_method", return_value="fork"), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Reset the warning flag to allow testing
        reset_fork_warning()

        # Trigger the reduction function directly
        _deep_reduce_device_memory_resource(mr)

        # Check that warning was emitted
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}: {[str(warning.message) for warning in w]}"
        warning = w[0]
        assert warning.category is UserWarning
        assert "fork" in str(warning.message).lower()
        assert "spawn" in str(warning.message).lower()
        assert "undefined behavior" in str(warning.message).lower()

    mr.close()


def test_warn_on_fork_method_allocation_handle(ipc_device):
    """Test that warning is emitted when IPCAllocationHandle is pickled with fork method."""
    device = ipc_device
    device.set_current()
    options = DeviceMemoryResourceOptions(max_size=2097152, ipc_enabled=True)
    mr = DeviceMemoryResource(device, options=options)
    alloc_handle = mr.get_allocation_handle()

    with patch("multiprocessing.get_start_method", return_value="fork"), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Reset the warning flag to allow testing
        reset_fork_warning()

        # Trigger the reduction function directly
        _reduce_allocation_handle(alloc_handle)

        # Check that warning was emitted
        assert len(w) == 1
        warning = w[0]
        assert warning.category is UserWarning
        assert "fork" in str(warning.message).lower()

    mr.close()


def test_warn_on_fork_method_event(mempool_device):
    """Test that warning is emitted when Event is pickled with fork method."""
    device = mempool_device
    device.set_current()
    stream = device.create_stream()
    ipc_event_options = EventOptions(ipc_enabled=True)
    event = stream.record(options=ipc_event_options)

    with patch("multiprocessing.get_start_method", return_value="fork"), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Reset the warning flag to allow testing
        reset_fork_warning()

        # Trigger the reduction function directly
        _reduce_event(event)

        # Check that warning was emitted
        assert len(w) == 1
        warning = w[0]
        assert warning.category is UserWarning
        assert "fork" in str(warning.message).lower()

    event.close()


def test_no_warning_with_spawn_method(ipc_device):
    """Test that no warning is emitted when start method is 'spawn'."""
    device = ipc_device
    device.set_current()
    options = DeviceMemoryResourceOptions(max_size=2097152, ipc_enabled=True)
    mr = DeviceMemoryResource(device, options=options)

    with patch("multiprocessing.get_start_method", return_value="spawn"), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Reset the warning flag to allow testing
        reset_fork_warning()

        # Trigger the reduction function directly
        _deep_reduce_device_memory_resource(mr)

        # Check that no fork-related warning was emitted
        fork_warnings = [warning for warning in w if "fork" in str(warning.message).lower()]
        assert len(fork_warnings) == 0, f"Unexpected warning: {fork_warnings[0].message if fork_warnings else None}"

    mr.close()


def test_warning_emitted_only_once(ipc_device):
    """Test that warning is only emitted once even when multiple objects are pickled."""
    device = ipc_device
    device.set_current()
    options = DeviceMemoryResourceOptions(max_size=2097152, ipc_enabled=True)
    mr1 = DeviceMemoryResource(device, options=options)
    mr2 = DeviceMemoryResource(device, options=options)

    with patch("multiprocessing.get_start_method", return_value="fork"), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Reset the warning flag to allow testing
        reset_fork_warning()

        # Trigger reduction multiple times
        _deep_reduce_device_memory_resource(mr1)
        _deep_reduce_device_memory_resource(mr2)

        # Check that warning was emitted only once
        fork_warnings = [warning for warning in w if "fork" in str(warning.message).lower()]
        assert len(fork_warnings) == 1, f"Expected 1 warning, got {len(fork_warnings)}"

    mr1.close()
    mr2.close()
