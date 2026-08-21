# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import threading
from pathlib import Path

_DIGEST_CHARS = 32


_lock = threading.Lock()
_source_dir: tempfile.TemporaryDirectory[str] | None = None


def _ensure_source_dir() -> Path:
    """Return the store directory, creating it if needed. Caller holds ``_lock``."""
    global _source_dir
    if _source_dir is None:
        _source_dir = tempfile.TemporaryDirectory(prefix="cuda-core-jit-")
    root = Path(_source_dir.name)
    # Re-created rather than assumed: a /tmp reaper can delete the tree out from
    # under a long-running process.
    root.mkdir(parents=True, exist_ok=True)
    return root


def source_dir() -> Path:
    """The process-scoped directory holding materialized JIT source."""
    with _lock:
        return _ensure_source_dir()


def materialize(code: bytes, suffix: str = ".cu") -> str | None:

    digest = hashlib.sha256(code).hexdigest()[:_DIGEST_CHARS]
    try:
        with _lock:
            target = _ensure_source_dir() / f"{digest}{suffix}"
            # Writers are serialized and the directory is ours alone, so an
            # entry that exists is one a previous caller finished writing.
            if not target.exists():
                try:
                    target.write_bytes(code)
                except BaseException:
                    # Otherwise a half-written entry survives to be mistaken
                    # for a complete one.
                    with contextlib.suppress(OSError):
                        target.unlink()
                    raise
            return os.fspath(target)
    except OSError:
        # A read only or full filesystem, or a sandbox that forbids the temp
        # dir should pass
        return None
