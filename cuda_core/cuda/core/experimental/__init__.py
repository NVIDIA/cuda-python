# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Backward compatibility stubs for cuda.core.experimental namespace.

This module provides forwarding stubs that import from the new cuda.core.*
locations and emit deprecation warnings. Users should migrate to importing
directly from cuda.core instead of cuda.core.experimental.

The experimental namespace will be removed in v1.0.0.

"""


def _warn_deprecated():
    """Emit a deprecation warning for using the experimental namespace.

    Note: This warning is only when the experimental module is first imported.
    Subsequent accesses to attributes (like utils, Device, etc.) do not trigger
    additional warnings since they are already set in the module namespace.
    """
    import warnings

    warnings.warn(
        "The cuda.core.experimental namespace is deprecated. "
        "Please import directly from cuda.core instead. "
        "For example, use 'from cuda.core import Device' instead of "
        "'from cuda.core.experimental import Device'. "
        "The experimental namespace will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=3,
    )


# Import from new locations and re-export
_warn_deprecated()


from cuda.core import managed_memory, system, utils

# Make utils accessible as a submodule for backward compatibility
__import__("sys").modules[__spec__.name + ".managed_memory"] = managed_memory
__import__("sys").modules[__spec__.name + ".utils"] = utils


from cuda.core._device import Device
from cuda.core._event import Event, EventOptions
from cuda.core._graph import (
    Graph,
    GraphBuilder,
    GraphCompleteOptions,
    GraphDebugPrintOptions,
)
from cuda.core._launch_config import LaunchConfig
from cuda.core._launcher import launch
from cuda.core._layout import _StridedLayout
from cuda.core._linker import Linker, LinkerOptions
from cuda.core._memory import (
    Buffer,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    GraphMemoryResource,
    LegacyPinnedMemoryResource,
    ManagedMemoryResource,
    ManagedMemoryResourceOptions,
    MemoryResource,
    PinnedMemoryResource,
    PinnedMemoryResourceOptions,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
)
from cuda.core._module import Kernel, ObjectCode
from cuda.core._program import Program, ProgramOptions
from cuda.core._stream import Stream, StreamOptions
