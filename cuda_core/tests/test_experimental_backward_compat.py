# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for backward compatibility of cuda.core.experimental namespace.

These tests verify that the experimental namespace forwarding stubs work
correctly and emit appropriate deprecation warnings.

Note: This test function is assumed to be the only function importing
cuda.core.experimental in the test suite to avoid race conditions when
tests run in parallel.
"""

import sys

import pytest


def test_experimental_backward_compatibility():
    """Test backward compatibility of cuda.core.experimental namespace.

    This single test function combines all experimental namespace tests to
    avoid race conditions when tests run in parallel. All tests that need to
    verify deprecation warnings or module state should be in this function.
    """
    # Defensive: ensure module is not cached (handles case where it might
    # already be imported by other tests or conftest)
    if "cuda.core.experimental" in sys.modules:
        del sys.modules["cuda.core.experimental"]

    # Test 1: Main module import - should emit deprecation warning
    with pytest.deprecated_call():
        import cuda.core.experimental

    # Test that symbols are accessible
    assert hasattr(cuda.core.experimental, "Device")
    assert hasattr(cuda.core.experimental, "Stream")
    assert hasattr(cuda.core.experimental, "Buffer")
    assert hasattr(cuda.core.experimental, "system")

    # Test 2: Direct imports - should emit deprecation warning
    # Clear cached module again to ensure warning is emitted
    del sys.modules["cuda.core.experimental"]

    with pytest.deprecated_call():
        from cuda.core.experimental import (
            Buffer,
            Device,
            Stream,
        )

    # Verify objects are usable
    assert Device is not None
    assert Stream is not None
    assert Buffer is not None

    # Test 3: Symbols are the same objects as core
    import cuda.core

    # Compare classes/types
    assert cuda.core.experimental.Device is cuda.core.Device
    assert cuda.core.experimental.Stream is cuda.core.Stream
    assert cuda.core.experimental.Buffer is cuda.core.Buffer
    assert cuda.core.experimental.MemoryResource is cuda.core.MemoryResource
    assert cuda.core.experimental.Program is cuda.core.Program
    assert cuda.core.experimental.Kernel is cuda.core.Kernel
    assert cuda.core.experimental.ObjectCode is cuda.core.ObjectCode
    assert cuda.core.experimental.Graph is cuda.core.Graph
    assert cuda.core.experimental.GraphBuilder is cuda.core.GraphBuilder
    assert cuda.core.experimental.Event is cuda.core.Event
    assert cuda.core.experimental.Linker is cuda.core.Linker

    # Compare singletons
    assert cuda.core.experimental.system is cuda.core.system

    # Test 4: Utils module works
    # Note: The deprecation warning is only emitted once at import time when
    # cuda.core.experimental is first imported. Accessing utils or importing
    # from utils does not trigger additional warnings since utils is already
    # set as an attribute in the module namespace.
    assert hasattr(cuda.core.experimental, "utils")
    assert cuda.core.experimental.utils is not None

    # Should have expected utilities (no warning on import from utils submodule)
    from cuda.core.experimental.utils import StridedMemoryView, args_viewable_as_strided_memory

    assert StridedMemoryView is not None
    assert args_viewable_as_strided_memory is not None

    # Test 5: Options classes are accessible
    assert hasattr(cuda.core.experimental, "EventOptions")
    assert hasattr(cuda.core.experimental, "StreamOptions")
    assert hasattr(cuda.core.experimental, "LaunchConfig")
    assert hasattr(cuda.core.experimental, "ProgramOptions")
    assert hasattr(cuda.core.experimental, "LinkerOptions")
    assert hasattr(cuda.core.experimental, "GraphCompleteOptions")
    assert hasattr(cuda.core.experimental, "GraphDebugPrintOptions")
    assert hasattr(cuda.core.experimental, "DeviceMemoryResourceOptions")
    assert hasattr(cuda.core.experimental, "VirtualMemoryResourceOptions")

    # Verify they're the same objects
    assert cuda.core.experimental.EventOptions is cuda.core.EventOptions
    assert cuda.core.experimental.StreamOptions is cuda.core.StreamOptions
    assert cuda.core.experimental.LaunchConfig is cuda.core.LaunchConfig

    # Test 6: Memory-related classes are accessible
    assert hasattr(cuda.core.experimental, "MemoryResource")
    assert hasattr(cuda.core.experimental, "DeviceMemoryResource")
    assert hasattr(cuda.core.experimental, "LegacyPinnedMemoryResource")
    assert hasattr(cuda.core.experimental, "VirtualMemoryResource")
    assert hasattr(cuda.core.experimental, "GraphMemoryResource")

    # Verify they're the same objects
    assert cuda.core.experimental.MemoryResource is cuda.core.MemoryResource
    assert cuda.core.experimental.DeviceMemoryResource is cuda.core.DeviceMemoryResource

    # Test 7: Objects can be instantiated through experimental namespace
    # (No deprecation warning expected since module is already imported)
    device = cuda.core.experimental.Device()

    assert device is not None

    # Verify it's the same type
    from cuda.core import Device as CoreDevice

    assert isinstance(device, CoreDevice)
