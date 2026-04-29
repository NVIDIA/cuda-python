# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._memoryview import (
    StridedMemoryView,
    args_viewable_as_strided_memory,
)

__all__ = [
    "FileStreamProgramCache",
    "InMemoryProgramCache",
    "ProgramCacheResource",
    "StridedMemoryView",
    "args_viewable_as_strided_memory",
    "make_program_cache_key",
]

# Lazily expose the program-cache APIs so ``from cuda.core.utils import
# StridedMemoryView`` stays lightweight -- the cache backend pulls in driver,
# NVRTC, and module-load machinery that memoryview-only consumers do not need.
_LAZY_CACHE_ATTRS = frozenset(
    {
        "FileStreamProgramCache",
        "InMemoryProgramCache",
        "ProgramCacheResource",
        "make_program_cache_key",
    }
)


def __getattr__(name):
    if name in _LAZY_CACHE_ATTRS:
        from cuda.core.utils import _program_cache

        value = getattr(_program_cache, name)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module 'cuda.core.utils' has no attribute {name!r}")


def __dir__():
    # Merge the lazy public API with the real module namespace so REPL and
    # introspection tools still surface ``__file__``, ``__spec__``, etc.
    return sorted(set(globals()) | set(__all__))
