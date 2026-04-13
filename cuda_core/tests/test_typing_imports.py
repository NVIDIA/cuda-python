# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for cuda.core.typing public type aliases and protocols."""


def test_typing_module_imports():
    """All type aliases and protocols are importable from cuda.core.typing."""
    from cuda.core.typing import (
        DevicePointerT,
        IsStreamT,
    )

    assert DevicePointerT is not None
    assert IsStreamT is not None


def test_typing_matches_private_definitions():
    """cuda.core.typing re-exports match the original private definitions."""
    from cuda.core._memory._buffer import DevicePointerT as _DevicePointerT
    from cuda.core._stream import IsStreamT as _IsStreamT
    from cuda.core.typing import (
        DevicePointerT,
        IsStreamT,
    )

    assert DevicePointerT is _DevicePointerT
    assert IsStreamT is _IsStreamT
