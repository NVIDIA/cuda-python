# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent program cache for cuda.core.

Public surface:

* :class:`ProgramCacheResource` -- bytes-in / bytes-out ABC.
* :class:`InMemoryProgramCache` -- thread-safe LRU dict-backed cache.
* :class:`FileStreamProgramCache` -- atomic, multi-process directory cache.
* :func:`make_program_cache_key` -- key derivation for arbitrary
  ``Program`` configurations.

The package is split into submodules by concern. Tests that need to
monkeypatch internals (Windows flag, version probes, helpers, ...)
should reach into the owning submodule (e.g.
``_program_cache._file_stream._IS_WINDOWS``,
``_program_cache._keys._linker_backend_and_version``) rather than the
package object: the symbols re-exported here are only convenience
aliases and don't intercept calls within the submodules.
"""

from __future__ import annotations

from ._abc import ProgramCacheResource
from ._file_stream import FileStreamProgramCache
from ._in_memory import InMemoryProgramCache
from ._keys import make_program_cache_key

__all__ = [
    "FileStreamProgramCache",
    "InMemoryProgramCache",
    "ProgramCacheResource",
    "make_program_cache_key",
]
