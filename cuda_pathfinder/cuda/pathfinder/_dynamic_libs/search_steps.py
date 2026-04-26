# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable search steps for locating NVIDIA libraries.

Each find step is a callable with signature::

    (SearchContext) -> FindResult | None

Find steps locate a library file on disk without loading it.  The
orchestrator in :mod:`load_nvidia_dynamic_lib` handles loading, the
already-loaded check, and dependency resolution.

Step sequences are defined per search strategy so that adding a new
step or strategy only requires adding a function and a tuple entry.

This module is intentionally platform-agnostic: it does not branch on the
current operating system. Platform differences are routed through the
:data:`~cuda.pathfinder._dynamic_libs.search_platform.PLATFORM` instance.
"""

import glob
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NoReturn, cast

from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.search_platform import PLATFORM, SearchPlatform
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FindResult:
    """A library file located on disk (not yet loaded)."""

    abs_path: str
    found_via: str


@dataclass
class SearchContext:
    """Mutable state accumulated during the search cascade."""

    desc: LibDescriptor
    platform: SearchPlatform = PLATFORM
    error_messages: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)

    @property
    def libname(self) -> str:
        return self.desc.name  # type: ignore[no-any-return]  # mypy can't resolve new sibling module

    @property
    def lib_searched_for(self) -> str:
        return cast(str, self.platform.lib_searched_for(self.libname))

    def raise_not_found(self) -> NoReturn:
        err = ", ".join(self.error_messages)
        att = "\n".join(self.attachments)
        raise DynamicLibNotFoundError(f'Failure finding "{self.lib_searched_for}": {err}\n{att}')


#: Type alias for a find step callable.
FindStep = Callable[[SearchContext], FindResult | None]


def _find_lib_dir_using_anchor(desc: LibDescriptor, platform: SearchPlatform, anchor_point: str) -> str | None:
    """Find the library directory under *anchor_point* using the descriptor's relative paths."""
    rel_dirs = platform.anchor_rel_dirs(desc)
    for rel_path in rel_dirs:
        for dirname in sorted(glob.glob(os.path.join(anchor_point, rel_path))):
            if os.path.isdir(dirname):
                return os.path.normpath(dirname)
    return None


def _find_using_lib_dir(ctx: SearchContext, lib_dir: str | None) -> str | None:
    """Find a library file in a resolved lib directory."""
    if lib_dir is None:
        return None
    return cast(
        str | None,
        ctx.platform.find_in_lib_dir(
            lib_dir,
            ctx.libname,
            ctx.lib_searched_for,
            ctx.error_messages,
            ctx.attachments,
        ),
    )


def _derive_ctk_root_linux(resolved_lib_path: str) -> str | None:
    """Derive CTK root from Linux canary path.

    Supports:
    - ``$CTK_ROOT/lib64/libfoo.so.*``
    - ``$CTK_ROOT/lib/libfoo.so.*``
    - ``$CTK_ROOT/targets/<triple>/lib64/libfoo.so.*``
    - ``$CTK_ROOT/targets/<triple>/lib/libfoo.so.*``
    """
    lib_dir = os.path.dirname(resolved_lib_path)
    basename = os.path.basename(lib_dir)
    if basename in ("lib64", "lib"):
        parent = os.path.dirname(lib_dir)
        grandparent = os.path.dirname(parent)
        if os.path.basename(grandparent) == "targets":
            return os.path.dirname(grandparent)
        return parent
    return None


def _derive_ctk_root_windows(resolved_lib_path: str) -> str | None:
    """Derive CTK root from Windows canary path.

    Supports:
    - ``$CTK_ROOT/bin/x64/foo.dll`` (CTK 13 style)
    - ``$CTK_ROOT/bin/foo.dll`` (CTK 12 style)
    """
    import ntpath

    lib_dir = ntpath.dirname(resolved_lib_path)
    basename = ntpath.basename(lib_dir).lower()
    if basename == "x64":
        parent = ntpath.dirname(lib_dir)
        if ntpath.basename(parent).lower() == "bin":
            return ntpath.dirname(parent)
    elif basename == "bin":
        return ntpath.dirname(lib_dir)
    return None


def derive_ctk_root(resolved_lib_path: str) -> str | None:
    """Derive CTK root from a resolved canary library path."""
    ctk_root = _derive_ctk_root_linux(resolved_lib_path)
    if ctk_root is not None:
        return ctk_root
    return _derive_ctk_root_windows(resolved_lib_path)


def find_via_ctk_root(ctx: SearchContext, ctk_root: str) -> FindResult | None:
    """Find a library under a previously derived CTK root."""
    lib_dir = _find_lib_dir_using_anchor(ctx.desc, ctx.platform, ctk_root)
    abs_path = _find_using_lib_dir(ctx, lib_dir)
    if abs_path is None:
        return None
    return FindResult(abs_path, "system-ctk-root")


# ---------------------------------------------------------------------------
# Find steps
# ---------------------------------------------------------------------------


_PATH_OVERRIDE_ENV_PREFIX = "CUDA_PATHFINDER_"
_PATH_OVERRIDE_ENV_SUFFIX = "_PATH_OVERRIDE"


def path_override_env_var(libname: str) -> str:
    """Return the per-library override environment variable name."""
    return f"{_PATH_OVERRIDE_ENV_PREFIX}{libname.upper()}{_PATH_OVERRIDE_ENV_SUFFIX}"


def find_via_path_override(ctx: SearchContext) -> FindResult | None:
    """Resolve a library via the per-library override environment variable.

    The variable name is ``CUDA_PATHFINDER_<LIBNAME_UPPER>_PATH_OVERRIDE``.

    Value semantics:
    - Unset or empty: this step is a no-op and returns ``None``.
    - Path to an existing regular file: used as the resolved library file.
    - Path to an existing directory: searched for the library file using the
      same platform logic as other anchor-based steps.
    - Anything else (path does not exist, directory has no matching library):
      raises :class:`DynamicLibNotFoundError`. An explicit override that fails
      to resolve must not silently fall through to other search steps.
    """
    env_var = path_override_env_var(ctx.libname)
    override = os.environ.get(env_var)
    if not override:
        return None

    found_via = f"override({env_var})"

    if os.path.isfile(override):
        return FindResult(os.path.normpath(override), found_via)

    if os.path.isdir(override):
        abs_path = _find_using_lib_dir(ctx, override)
        if abs_path is not None:
            return FindResult(abs_path, found_via)
        err = ", ".join(ctx.error_messages) or f"no matching file under {override!r}"
        att = "\n".join(ctx.attachments)
        raise DynamicLibNotFoundError(
            f"{env_var}={override!r} is set but {ctx.lib_searched_for!r} was not found there: {err}\n{att}"
        )

    raise DynamicLibNotFoundError(f"{env_var}={override!r} is set but the path does not exist as a file or directory.")


def find_in_site_packages(ctx: SearchContext) -> FindResult | None:
    """Search pip wheel install locations."""
    rel_dirs = ctx.platform.site_packages_rel_dirs(ctx.desc)
    if not rel_dirs:
        return None
    abs_path = ctx.platform.find_in_site_packages(rel_dirs, ctx.lib_searched_for, ctx.error_messages, ctx.attachments)
    if abs_path is not None:
        return FindResult(abs_path, "site-packages")
    return None


def find_in_conda(ctx: SearchContext) -> FindResult | None:
    """Search ``$CONDA_PREFIX``."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None
    anchor = ctx.platform.conda_anchor_point(conda_prefix)
    lib_dir = _find_lib_dir_using_anchor(ctx.desc, ctx.platform, anchor)
    abs_path = _find_using_lib_dir(ctx, lib_dir)
    if abs_path is not None:
        return FindResult(abs_path, "conda")
    return None


def find_in_cuda_path(ctx: SearchContext) -> FindResult | None:
    """Search ``$CUDA_PATH`` / ``$CUDA_HOME``.

    On Windows, this is the normal fallback for system-installed CTK DLLs when
    they are not already discoverable via the native ``LoadLibraryExW(..., 0)``
    path used by :func:`cuda.pathfinder._dynamic_libs.load_dl_windows.load_with_system_search`.
    Python 3.8+ does not include ``PATH`` in that native DLL search.

    The returned ``found_via`` is always ``"CUDA_PATH"`` regardless of which
    environment variable actually provided the value.
    """
    cuda_home = get_cuda_path_or_home()
    if cuda_home is None:
        return None
    lib_dir = _find_lib_dir_using_anchor(ctx.desc, ctx.platform, cuda_home)
    abs_path = _find_using_lib_dir(ctx, lib_dir)
    if abs_path is not None:
        return FindResult(abs_path, "CUDA_PATH")
    return None


# ---------------------------------------------------------------------------
# Step sequences per strategy
# ---------------------------------------------------------------------------

#: Find steps that run before the already-loaded check and system search.
#: The path-override step has the highest priority and fails loudly if the
#: override env var is set but the library cannot be resolved from it.
EARLY_FIND_STEPS: tuple[FindStep, ...] = (find_via_path_override, find_in_site_packages, find_in_conda)

#: Find steps that run after system search fails.
LATE_FIND_STEPS: tuple[FindStep, ...] = (find_in_cuda_path,)


# ---------------------------------------------------------------------------
# Cascade runner
# ---------------------------------------------------------------------------


def run_find_steps(ctx: SearchContext, steps: tuple[FindStep, ...]) -> FindResult | None:
    """Run find steps in order, returning the first hit."""
    for step in steps:
        result = step(ctx)
        if result is not None:
            return result
    return None
