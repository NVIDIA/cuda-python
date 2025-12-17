# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for backward compatibility of cuda.core.experimental namespace.

These tests verify that the experimental namespace forwarding stubs work
correctly and emit appropriate deprecation warnings.
"""

import pytest


# Test that experimental imports still work
def test_experimental_imports_work():
    """Test that imports from experimental namespace still work."""
    # Clear cached module to ensure warning is emitted
    import sys

    if "cuda.core.experimental" in sys.modules:
        del sys.modules["cuda.core.experimental"]

    # Test main module import - should emit deprecation warning
    with pytest.deprecated_call():
        import cuda.core.experimental

    # Test that symbols are accessible
    assert hasattr(cuda.core.experimental, "Device")
    assert hasattr(cuda.core.experimental, "Stream")
    assert hasattr(cuda.core.experimental, "Buffer")
    assert hasattr(cuda.core.experimental, "system")


def test_experimental_symbols_are_same_objects():
    """Test that experimental namespace symbols are the same objects as core."""
    import cuda.core
    import cuda.core.experimental

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


def test_experimental_direct_imports():
    """Test that direct imports from experimental submodules work."""
    # Clear any cached imports to ensure warnings are emitted
    import sys

    if "cuda.core.experimental" in sys.modules:
        del sys.modules["cuda.core.experimental"]

    # Test various import patterns - warning is emitted once at module import time
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


def test_experimental_submodule_access():
    """Test that accessing experimental underscored submodules raises AttributeError.

    Underscored modules are not public APIs and should not be accessible through
    the experimental namespace.
    """
    import cuda.core.experimental

    # Underscored modules should not be accessible (__getattr__ removed per reviewer feedback)
    with pytest.raises(AttributeError):
        _ = cuda.core.experimental._device
    with pytest.raises(AttributeError):
        _ = cuda.core.experimental._stream
    with pytest.raises(AttributeError):
        _ = cuda.core.experimental._memory


def test_experimental_utils_module():
    """Test that experimental.utils module works.

    Note: The deprecation warning is only emitted once at import time when
    cuda.core.experimental is first imported. Accessing utils or importing
    from utils does not trigger additional warnings since utils is already
    set as an attribute in the module namespace.
    """
    import cuda.core.experimental

    # Should be able to access utils (no warning on access, only on initial import)
    assert hasattr(cuda.core.experimental, "utils")
    assert cuda.core.experimental.utils is not None

    # Should have expected utilities (no warning on import from utils submodule)
    from cuda.core.experimental.utils import StridedMemoryView, args_viewable_as_strided_memory

    assert StridedMemoryView is not None
    assert args_viewable_as_strided_memory is not None


def test_experimental_options_classes():
    """Test that options classes are accessible."""
    import cuda.core.experimental

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


def test_experimental_memory_classes():
    """Test that memory-related classes are accessible."""
    import cuda.core.experimental

    assert hasattr(cuda.core.experimental, "MemoryResource")
    assert hasattr(cuda.core.experimental, "DeviceMemoryResource")
    assert hasattr(cuda.core.experimental, "LegacyPinnedMemoryResource")
    assert hasattr(cuda.core.experimental, "VirtualMemoryResource")
    assert hasattr(cuda.core.experimental, "GraphMemoryResource")

    # Verify they're the same objects
    assert cuda.core.experimental.MemoryResource is cuda.core.MemoryResource
    assert cuda.core.experimental.DeviceMemoryResource is cuda.core.DeviceMemoryResource


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_experimental_instantiations():
    """Test that objects can be instantiated through experimental namespace."""
    from cuda.core.experimental import Device

    # Should be able to create objects
    device = Device()
    assert device is not None

    # Verify it's the same type
    from cuda.core import Device as CoreDevice

    assert isinstance(device, CoreDevice)
