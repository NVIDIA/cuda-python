# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for cuda.core.typing public type aliases and protocols."""


def test_typing_module_imports():
    """All type aliases and protocols are importable from cuda.core.typing."""
    from cuda.core.typing import (
        DevicePointerT,
        IsStreamT,
        VirtualMemoryAccessTypeT,
        VirtualMemoryAllocationTypeT,
        VirtualMemoryGranularityT,
        VirtualMemoryHandleTypeT,
        VirtualMemoryLocationTypeT,
    )

    # Verify they are not None (sanity check)
    for name, obj in (
        ("DevicePointerT", DevicePointerT),
        ("IsStreamT", IsStreamT),
        ("VirtualMemoryAccessTypeT", VirtualMemoryAccessTypeT),
        ("VirtualMemoryAllocationTypeT", VirtualMemoryAllocationTypeT),
        ("VirtualMemoryGranularityT", VirtualMemoryGranularityT),
        ("VirtualMemoryHandleTypeT", VirtualMemoryHandleTypeT),
        ("VirtualMemoryLocationTypeT", VirtualMemoryLocationTypeT),
    ):
        assert obj is not None, f"{name} should not be None"


def test_typing_matches_private_definitions():
    """cuda.core.typing re-exports match the original private definitions."""
    from cuda.core._memory._buffer import DevicePointerT as _DevicePointerT
    from cuda.core._memory._virtual_memory_resource import (
        VirtualMemoryAccessTypeT as _VirtualMemoryAccessTypeT,
        VirtualMemoryAllocationTypeT as _VirtualMemoryAllocationTypeT,
        VirtualMemoryGranularityT as _VirtualMemoryGranularityT,
        VirtualMemoryHandleTypeT as _VirtualMemoryHandleTypeT,
        VirtualMemoryLocationTypeT as _VirtualMemoryLocationTypeT,
    )
    from cuda.core._stream import IsStreamT as _IsStreamT
    from cuda.core.typing import (
        DevicePointerT,
        IsStreamT,
        VirtualMemoryAccessTypeT,
        VirtualMemoryAllocationTypeT,
        VirtualMemoryGranularityT,
        VirtualMemoryHandleTypeT,
        VirtualMemoryLocationTypeT,
    )

    assert DevicePointerT is _DevicePointerT
    assert IsStreamT is _IsStreamT
    assert VirtualMemoryAccessTypeT is _VirtualMemoryAccessTypeT
    assert VirtualMemoryAllocationTypeT is _VirtualMemoryAllocationTypeT
    assert VirtualMemoryGranularityT is _VirtualMemoryGranularityT
    assert VirtualMemoryHandleTypeT is _VirtualMemoryHandleTypeT
    assert VirtualMemoryLocationTypeT is _VirtualMemoryLocationTypeT
