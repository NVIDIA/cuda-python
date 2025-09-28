# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

try:
    import cuda.bindings
except ImportError as e:
    raise ImportError("cuda.bindings 12.x or 13.x must be installed")
else:
    cuda_major, cuda_minor = cuda.bindings.__version__.split(".")[:2]
    if cuda_major not in ("12", "13"):
        raise ImportError("cuda.bindings 12.x or 13.x must be installed")

import importlib
subdir = f"cu{cuda_major}"
try:
    verioned_mod = importlib.import_module(f".{subdir}", __package__)
    # Import all symbols from the module
    globals().update(verioned_mod.__dict__)
except ImportError:
    # This is not a wheel build, but a conda or local build, do nothing
    pass
else:
    del verioned_mod
finally:
    del cuda.bindings, importlib, subdir, cuda_major, cuda_minor

from cuda.core.experimental import utils
from cuda.core.experimental._device import Device
from cuda.core.experimental._event import Event, EventOptions
from cuda.core.experimental._graph import (
    Graph,
    GraphBuilder,
    GraphCompleteOptions,
    GraphDebugPrintOptions,
)
from cuda.core.experimental._launch_config import LaunchConfig
from cuda.core.experimental._launcher import launch
from cuda.core.experimental._linker import Linker, LinkerOptions
from cuda.core.experimental._memory import (
    Buffer,
    DeviceMemoryResource,
    IPCChannel,
    LegacyPinnedMemoryResource,
    MemoryResource,
)
from cuda.core.experimental._module import Kernel, ObjectCode
from cuda.core.experimental._program import Program, ProgramOptions
from cuda.core.experimental._stream import Stream, StreamOptions
from cuda.core.experimental._system import System

system = System()
__import__("sys").modules[__spec__.name + ".system"] = system
del System
