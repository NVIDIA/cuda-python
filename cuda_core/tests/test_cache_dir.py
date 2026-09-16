# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.agent_authored(model="claude-sonnet-5")
def test_user_cache_dir_lives_under_platform_root(monkeypatch, tmp_path):
    """The shared user-cache root (``cuda.core.utils._cache_dir``) is platform-specific:

    * Linux: ``$XDG_CACHE_HOME`` or ``~/.cache``.
    * Windows: ``%LOCALAPPDATA%`` or ``~/AppData/Local``.

    Both branches must end in ``cuda-python``; that suffix is what guarantees a
    stable on-disk layout across releases, and callers (e.g. the file-stream
    cache, NVRTC's bundled-headers cache) each append their own leaf under it.
    """
    from pathlib import Path

    from cuda.core.utils import _cache_dir
    from cuda.core.utils._cache_dir import _default_cache_dir

    # Path must end with cuda-python regardless of platform.
    assert _default_cache_dir().parts[-1] == "cuda-python"

    # Linux branch: XDG_CACHE_HOME wins when set.
    monkeypatch.setattr(_cache_dir, "_IS_WINDOWS", False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert _default_cache_dir() == tmp_path / "xdg" / "cuda-python"

    # Linux branch: falls back to ``~/.cache`` when XDG_CACHE_HOME is unset.
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    assert _default_cache_dir() == tmp_path / "home" / ".cache" / "cuda-python"

    # Windows branch: LOCALAPPDATA wins when set.
    monkeypatch.setattr(_cache_dir, "_IS_WINDOWS", True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    assert _default_cache_dir() == tmp_path / "appdata" / "cuda-python"

    # Windows branch: falls back to ``~/AppData/Local`` when LOCALAPPDATA is unset.
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert _default_cache_dir() == tmp_path / "home" / "AppData" / "Local" / "cuda-python"
