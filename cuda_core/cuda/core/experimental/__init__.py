# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

try:
    from cuda import bindings
except ImportError:
    raise ImportError("cuda.bindings 12.x or 13.x must be installed") from None
else:
    cuda_major, cuda_minor = bindings.__version__.split(".")[:2]
    if cuda_major not in ("12", "13"):
        raise ImportError("cuda.bindings 12.x or 13.x must be installed")

import importlib
import sys

# Load _resource_handles with RTLD_GLOBAL so its C++ symbols are available
# to other extension modules that depend on them (_context, _device, etc.)
# This must happen before importing any dependent modules.
if sys.platform != "win32":
    import os

    _old_dlopen_flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
    try:
        from cuda.core.experimental import _resource_handles  # noqa: F401
    finally:
        sys.setdlopenflags(_old_dlopen_flags)
    del _old_dlopen_flags, os
else:
    from cuda.core.experimental import _resource_handles  # noqa: F401

subdir = f"cu{cuda_major}"
try:
    versioned_mod = importlib.import_module(f".{subdir}", __package__)
    # Import all symbols from the module
    globals().update(versioned_mod.__dict__)
except ImportError:
    # This is not a wheel build, but a conda or local build, do nothing
    pass
else:
    del versioned_mod
finally:
    del bindings, importlib, subdir, cuda_major, cuda_minor

from cuda.core.experimental import utils  # noqa: E402
from cuda.core.experimental._device import Device  # noqa: E402
from cuda.core.experimental._event import Event, EventOptions  # noqa: E402
from cuda.core.experimental._graph import (  # noqa: E402
    Graph,
    GraphBuilder,
    GraphCompleteOptions,
    GraphDebugPrintOptions,
)
from cuda.core.experimental._launch_config import LaunchConfig  # noqa: E402
from cuda.core.experimental._launcher import launch  # noqa: E402
from cuda.core.experimental._linker import Linker, LinkerOptions  # noqa: E402
from cuda.core.experimental._memory import (  # noqa: E402
    Buffer,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    GraphMemoryResource,
    LegacyPinnedMemoryResource,
    MemoryResource,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
)
from cuda.core.experimental._module import Kernel, ObjectCode  # noqa: E402
from cuda.core.experimental._program import Program, ProgramOptions  # noqa: E402
from cuda.core.experimental._stream import Stream, StreamOptions  # noqa: E402
from cuda.core.experimental._system import System  # noqa: E402

system = System()
__import__("sys").modules[__spec__.name + ".system"] = system
del System
