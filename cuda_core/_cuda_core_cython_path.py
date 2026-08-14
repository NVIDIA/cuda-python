# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Expose editable cuda-bindings declarations to filesystem-based Cython lookup."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

_CUDA_BINDINGS_DECLARATION = Path("cuda", "bindings", "cydriver.pxd")


def _validated_root(root: str | os.PathLike[str]) -> Path | None:
    try:
        root_path = Path(root).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    return root_path if (root_path / _CUDA_BINDINGS_DECLARATION).is_file() else None


def _find_on_sys_path() -> Path | None:
    for entry in sys.path:
        if isinstance(entry, str) and (root := _validated_root(entry or os.getcwd())):
            return root
    return None


def _root_from_package_path(package_path: str | os.PathLike[str]) -> Path | None:
    bindings_path = Path(package_path)
    if bindings_path.name != "bindings" or bindings_path.parent.name != "cuda":
        return None
    return _validated_root(bindings_path.parent.parent)


def _find_from_spec() -> Path | None:
    """Resolve a physical root without importing ``cuda.bindings`` itself."""
    try:
        spec = importlib.util.find_spec("cuda.bindings")
    except ModuleNotFoundError:
        return None
    if spec is None:
        return None

    for location in spec.submodule_search_locations or ():
        if root := _root_from_package_path(location):
            return root

    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        return _root_from_package_path(Path(spec.origin).parent)
    return None


def _distribution() -> importlib.metadata.Distribution | None:
    try:
        return importlib.metadata.distribution("cuda-bindings")
    except importlib.metadata.PackageNotFoundError:
        return None


def _path_from_file_url(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None

    path = url2pathname(parsed.path)
    if parsed.netloc and parsed.netloc != "localhost":
        path = f"//{parsed.netloc}{path}"
    return Path(path)


def _report_invalid_editable(message: str, *, strict: bool) -> None:
    if strict:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _find_editable_root(*, strict: bool) -> Path | None:
    distribution = _distribution()
    if distribution is None:
        return None

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        return None
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        _report_invalid_editable("cuda-bindings has an invalid direct_url.json", strict=strict)
        return None
    if not isinstance(direct_url, dict):
        _report_invalid_editable("cuda-bindings has an invalid direct_url.json", strict=strict)
        return None

    dir_info = direct_url.get("dir_info")
    if not isinstance(dir_info, dict) or dir_info.get("editable") is not True:
        return None

    url = direct_url.get("url")
    if isinstance(url, str):
        source_path = _path_from_file_url(url)
        if source_path is not None and (root := _validated_root(source_path)):
            return root

    # PEP 660 meta finders can expose a source layout that differs from the
    # project root recorded in direct_url.json.
    if root := _find_from_spec():
        return root

    _report_invalid_editable(
        "cuda-bindings is marked as editable, but its physical "
        f"{_CUDA_BINDINGS_DECLARATION} declaration could not be found",
        strict=strict,
    )
    return None


def find_cuda_bindings_cython_root() -> str | None:
    """Return a transient Cython root only when normal ``sys.path`` is insufficient."""
    if _find_on_sys_path() is not None:
        return None

    root = _find_editable_root(strict=True) or _find_from_spec()
    if root is None:
        distribution = _distribution()
        if distribution is not None:
            declaration = Path(distribution.locate_file(_CUDA_BINDINGS_DECLARATION))
            if declaration.is_file():
                root = _validated_root(declaration.parents[2])
    if root is None:
        raise RuntimeError(f"Could not find the physical cuda-bindings {_CUDA_BINDINGS_DECLARATION} declaration")
    return os.fspath(root)


def add_editable_cuda_bindings_path() -> None:
    """Expose editable bindings declarations in the final runtime environment."""
    if _find_on_sys_path() is not None:
        return

    root = _find_editable_root(strict=False) or _find_from_spec()
    if root is not None:
        root_string = os.fspath(root)
        if root_string not in sys.path:
            sys.path.append(root_string)
