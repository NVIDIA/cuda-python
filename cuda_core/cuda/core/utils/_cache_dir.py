# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared user-cache-root resolution for cuda.core's on-disk caches."""

from __future__ import annotations

import os
from pathlib import Path

# Exposed as a module-level flag so tests can toggle it without monkeypatching
# ``os.name`` itself (pathlib reads ``os.name`` at instantiation time).
_IS_WINDOWS = os.name == "nt"


def _default_cache_dir() -> Path:
    """OS-conventional root for cuda.core's on-disk caches.

    Resolves to the user-cache root for the calling user, with a
    ``cuda-python`` vendor leaf so callers can each place their own cache
    under a stable, shared root:

    * Linux: ``$XDG_CACHE_HOME/cuda-python``
      (default ``~/.cache/cuda-python`` per the XDG Base Directory spec).
    * Windows: ``%LOCALAPPDATA%\\cuda-python``
      (Windows uses local AppData -- caches don't roam; falls back to
      ``~/AppData/Local`` if the env var is unset).

    CUDA does not support macOS, so no macOS branch is provided.

    Callers append their own leaf directory, e.g. ``program-cache`` or
    ``nvrtc-headers``.
    """
    if _IS_WINDOWS:
        local_app_data = os.environ.get("LOCALAPPDATA")
        root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "cuda-python"
