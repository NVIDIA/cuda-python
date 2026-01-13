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


from cuda.core import system, utils  # noqa: E402

# Make utils accessible as a submodule for backward compatibility
__import__("sys").modules[__spec__.name + ".utils"] = utils


from cuda.core._device import Device  # noqa: E402
from cuda.core._event import Event, EventOptions  # noqa: E402
from cuda.core._graph import (  # noqa: E402
    Graph,
    GraphBuilder,
    GraphCompleteOptions,
    GraphDebugPrintOptions,
)
from cuda.core._launch_config import LaunchConfig  # noqa: E402
from cuda.core._launcher import launch  # noqa: E402
from cuda.core._layout import _StridedLayout  # noqa: E402
from cuda.core._linker import Linker, LinkerOptions  # noqa: E402
from cuda.core._memory import (  # noqa: E402
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
from cuda.core._module import Kernel, ObjectCode  # noqa: E402
from cuda.core._program import Program, ProgramOptions  # noqa: E402
from cuda.core._stream import Stream, StreamOptions  # noqa: E402
