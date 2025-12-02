# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Backward compatibility stubs for cuda.core.experimental namespace.

This module provides forwarding stubs that import from the new cuda.core.*
locations and emit deprecation warnings. Users should migrate to importing
directly from cuda.core instead of cuda.core.experimental.

The experimental namespace will be removed in a future release.
"""

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For type checkers, import from the new location
    from cuda.core import (
        Buffer,
        Device,
        DeviceMemoryResource,
        DeviceMemoryResourceOptions,
        Event,
        EventOptions,
        Graph,
        GraphBuilder,
        GraphCompleteOptions,
        GraphDebugPrintOptions,
        Kernel,
        LaunchConfig,
        LegacyPinnedMemoryResource,
        Linker,
        LinkerOptions,
        MemoryResource,
        ObjectCode,
        Program,
        ProgramOptions,
        Stream,
        StreamOptions,
        VirtualMemoryResource,
        VirtualMemoryResourceOptions,
        launch,
        system,
        utils,
    )


def _warn_deprecated():
    """Emit a deprecation warning for using the experimental namespace."""
    warnings.warn(
        "The cuda.core.experimental namespace is deprecated. "
        "Please import directly from cuda.core instead. "
        "For example, use 'from cuda.core import Device' instead of "
        "'from cuda.core.experimental import Device'. "
        "The experimental namespace will be removed in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


# Import from new locations and re-export
_warn_deprecated()

from cuda.core import utils  # noqa: E402
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
from cuda.core._linker import Linker, LinkerOptions  # noqa: E402
from cuda.core._memory import (  # noqa: E402
    Buffer,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    GraphMemoryResource,
    LegacyPinnedMemoryResource,
    MemoryResource,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
)
from cuda.core._module import Kernel, ObjectCode  # noqa: E402
from cuda.core._program import Program, ProgramOptions  # noqa: E402
from cuda.core._stream import Stream, StreamOptions  # noqa: E402
from cuda.core._system import System  # noqa: E402

system = System()
__import__("sys").modules[__spec__.name + ".system"] = system
del System

# Also create forwarding stubs for submodules
# These will be imported lazily when accessed
def __getattr__(name):
    """Forward attribute access to the new location with deprecation warning."""
    if name in ("_device", "_event", "_graph", "_launch_config", "_launcher", 
                "_linker", "_memory", "_module", "_program", "_stream", "_system", 
                "_utils", "_context", "_dlpack", "_kernel_arg_handler", 
                "_launch_config", "_memoryview"):
        _warn_deprecated()
        # Import the submodule from the new location
        import importlib
        new_name = name.lstrip("_")
        try:
            return importlib.import_module(f"cuda.core.{new_name}")
        except ImportError:
            # Fallback to underscore-prefixed name
            return importlib.import_module(f"cuda.core.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
