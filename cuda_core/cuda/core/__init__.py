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

from cuda.core import managed_memory, system, utils
from cuda.core._device import Device
from cuda.core._event import Event, EventOptions
from cuda.core._graph import (
    Graph,
    GraphBuilder,
    GraphCompleteOptions,
    GraphDebugPrintOptions,
)
from cuda.core._graphics import GraphicsResource
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
from cuda.core._memoryview import (
    StridedMemoryView,
    args_viewable_as_strided_memory,
)
from cuda.core._module import Kernel, ObjectCode
from cuda.core._program import Program, ProgramOptions
from cuda.core._stream import (
    LEGACY_DEFAULT_STREAM,
    PER_THREAD_DEFAULT_STREAM,
    Stream,
    StreamOptions,
)
from cuda.core._tensor_map import TensorMapDescriptor, TensorMapDescriptorOptions
