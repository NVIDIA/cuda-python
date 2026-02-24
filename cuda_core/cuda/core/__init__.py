# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._version import __version__

try:
    from cuda import bindings
except ImportError:
    raise ImportError("cuda.bindings 12.x or 13.x must be installed") from None
else:
    cuda_major, cuda_minor = bindings.__version__.split(".")[:2]
    if cuda_major not in ("12", "13"):
        raise ImportError("cuda.bindings 12.x or 13.x must be installed")

import importlib

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

from cuda.core import system, utils  # noqa: E402
from cuda.core._device import Device  # noqa: E402
from cuda.core._event import Event, EventOptions  # noqa: E402
from cuda.core._graph import (  # noqa: E402
    Graph,
    GraphBuilder,
    GraphCompleteOptions,
    GraphDebugPrintOptions,
)
from cuda.core._graphics import GraphicsResource  # noqa: E402
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
from cuda.core._memoryview import (  # noqa: E402
    StridedMemoryView,  # noqa: E402
    args_viewable_as_strided_memory,  # noqa: E402
)
from cuda.core._module import Kernel, ObjectCode  # noqa: E402
from cuda.core._program import Program, ProgramOptions  # noqa: E402
from cuda.core._stream import (  # noqa: E402
    LEGACY_DEFAULT_STREAM,
    PER_THREAD_DEFAULT_STREAM,
    Stream,
    StreamOptions,
)
