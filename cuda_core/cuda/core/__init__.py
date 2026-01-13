# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

from cuda.core._version import __version__

# Version validation: detect setuptools-scm fallback versions (e.g., 0.1.dev...)
# This check must be kept in sync with similar checks in cuda.bindings and cuda.pathfinder
if not os.environ.get("CUDA_PYTHON_ALLOW_FALLBACK_VERSIONING"):
    version_parts = __version__.split(".")
    if len(version_parts) < 2:
        raise RuntimeError(
            f"Invalid version format: '{__version__}'. "
            f"The version detection system failed. "
            f"This usually means git tags are not available (e.g., shallow clone or zip archive). "
            f"To fix: ensure you have a full git checkout with tags, or set "
            f"CUDA_PYTHON_ALLOW_FALLBACK_VERSIONING=1 to disable this check."
        )
    try:
        major, minor = int(version_parts[0]), int(version_parts[1])
    except ValueError:
        raise RuntimeError(
            f"Invalid version format: '{__version__}'. "
            f"The version detection system failed. "
            f"This usually means git tags are not available (e.g., shallow clone or zip archive). "
            f"To fix: ensure you have a full git checkout with tags, or set "
            f"CUDA_PYTHON_ALLOW_FALLBACK_VERSIONING=1 to disable this check."
        ) from None
    if major == 0 and minor <= 1:
        raise RuntimeError(
            f"Invalid version detected: '{__version__}'. "
            f"The version detection system failed silently and produced a fallback version. "
            f"This usually means git tags are not available (e.g., shallow clone or zip archive). "
            f"To fix: ensure you have a full git checkout with tags, or set "
            f"CUDA_PYTHON_ALLOW_FALLBACK_VERSIONING=1 to disable this check."
        )

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
from cuda.core._stream import Stream, StreamOptions  # noqa: E402
