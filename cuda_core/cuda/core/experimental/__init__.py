# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

try:
    import cuda.bindings
except ImportError:
    raise ImportError("cuda.bindings 12.x or 13.x must be installed") from None
else:
    cuda_major, cuda_minor = cuda.bindings.__version__.split(".")[:2]
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
    del cuda.bindings, importlib, subdir, cuda_major, cuda_minor

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
