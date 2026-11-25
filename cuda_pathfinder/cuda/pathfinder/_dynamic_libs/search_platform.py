# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-specific rules for filesystem search steps.

The goal is to keep :mod:`search_steps` platform-agnostic: it should not branch
on OS flags like ``IS_WINDOWS``. Instead, it works on a :class:`SearchContext`
that this module resolves for the target platform.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import NoReturn

from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import is_suppressed_dll_file
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.windows_arch import windows_pe_matches_arch, windows_python_arch


@dataclass(frozen=True, slots=True)
class NotFound:
    """Why a search missed: the text for the final not-found error."""

    error_messages: tuple[str, ...] = ()
    attachments: tuple[str, ...] = ()

    def __add__(self, other: NotFound) -> NotFound:
        return NotFound(self.error_messages + other.error_messages, self.attachments + other.attachments)


@dataclass(frozen=True, slots=True)
class SearchContext:
    """One library's search inputs, resolved for one target platform.

    Build one with :func:`linux_search_context`, :func:`windows_search_context`,
    or :func:`search_context` (the platform this process runs on).
    """

    desc: LibDescriptor
    #: ``None`` on Linux; ``"x64"`` or ``"arm64"`` on Windows.
    windows_arch: str | None
    #: Wording that names the library in error messages.
    lib_searched_for: str
    site_packages_rel_dirs: tuple[str, ...]
    #: Path parts between ``$CONDA_PREFIX`` and the directory the anchors hang off.
    conda_anchor_parts: tuple[str, ...]
    anchor_rel_dirs: tuple[str, ...]
    install_root_env_vars: tuple[str, ...]
    install_root_env_rel_dirs: tuple[str, ...]
    #: Globs relative to the Windows Program Files directory.
    program_files_rel_globs: tuple[str, ...]
    #: Windows architecture that a candidate DLL must match; ``None`` means no check.
    pe_arch_to_check: str | None

    def conda_anchor_point(self, conda_prefix: str) -> str:
        return os.path.join(conda_prefix, *self.conda_anchor_parts)

    def program_files_root_globs(self) -> tuple[str, ...]:
        program_files = os.environ.get("PROGRAMW6432") or os.environ.get("PROGRAMFILES")
        if not program_files:
            return ()
        return tuple(os.path.join(program_files, rel_glob) for rel_glob in self.program_files_rel_globs)

    def raise_not_found(self, missed: NotFound) -> NoReturn:
        err = ", ".join(missed.error_messages)
        att = "\n".join(missed.attachments)
        raise DynamicLibNotFoundError(f'Failure finding "{self.lib_searched_for}": {err}\n{att}')


def linux_search_context(desc: LibDescriptor) -> SearchContext:
    return SearchContext(
        desc=desc,
        windows_arch=None,
        lib_searched_for=" or ".join(desc.linux_sonames),
        site_packages_rel_dirs=desc.site_packages_linux,
        conda_anchor_parts=(),
        anchor_rel_dirs=desc.anchor_rel_dirs_linux,
        install_root_env_vars=desc.install_root_env_vars_linux,
        install_root_env_rel_dirs=desc.install_root_env_rel_dirs_linux,
        program_files_rel_globs=(),
        pe_arch_to_check=None,
    )


def windows_search_context(desc: LibDescriptor, target_arch: str) -> SearchContext:
    arch_is_supported = target_arch in desc.supported_windows_arch
    return SearchContext(
        desc=desc,
        windows_arch=target_arch,
        lib_searched_for=f"known {desc.name} DLL",
        site_packages_rel_dirs=desc.site_packages_windows.for_arch(target_arch),
        conda_anchor_parts=("Library",),
        anchor_rel_dirs=desc.anchor_rel_dirs_windows.for_arch(target_arch),
        install_root_env_vars=desc.install_root_env_vars_windows if arch_is_supported else (),
        install_root_env_rel_dirs=(
            desc.install_root_env_rel_dirs_windows.for_arch(target_arch) if arch_is_supported else ()
        ),
        program_files_rel_globs=desc.program_files_root_globs_windows.for_arch(target_arch),
        pe_arch_to_check=target_arch if desc.requires_windows_binary_arch_check else None,
    )


def search_context(desc: LibDescriptor) -> SearchContext:
    """Resolve *desc* for the platform this process runs on."""
    if IS_WINDOWS:
        return windows_search_context(desc, windows_python_arch())
    return linux_search_context(desc)


def _listdir_attachment(dirpath: str) -> tuple[str, ...]:
    return (f'  listdir("{dirpath}"):', *(f"    {node}" for node in sorted(os.listdir(dirpath))))


def _no_such_file_in_sub_dirs(sub_dirs: Sequence[str], file_description: str) -> NotFound:
    attachments: tuple[str, ...] = ()
    for sub_dir in find_sub_dirs_all_sitepackages(sub_dirs):
        attachments += _listdir_attachment(sub_dir)
    return NotFound((f"No such file: {file_description}",), attachments)


def _find_descriptor_so_under_dir(dirpath: str, desc: LibDescriptor) -> str | None:
    for soname in desc.linux_sonames:
        path = os.path.join(dirpath, soname)
        if os.path.isfile(path):
            return path
    return None


def _find_descriptor_dll_under_dir(
    dirpath: str,
    desc: LibDescriptor,
    target_arch: str | None = None,
) -> str | None:
    def candidate_is_usable(path: str) -> bool:
        if not os.path.isfile(path):
            return False
        if is_suppressed_dll_file(os.path.basename(path)):
            return False
        return target_arch is None or windows_pe_matches_arch(path, target_arch)

    for dll_basename in desc.windows_dlls:
        path = os.path.join(dirpath, dll_basename)
        if candidate_is_usable(path):
            return path
    return None


def _find_lib_file(ctx: SearchContext, dirpath: str) -> str | None:
    """Return the first library file under *dirpath* that the target platform accepts."""
    if ctx.windows_arch is None:
        return _find_descriptor_so_under_dir(dirpath, ctx.desc)
    return _find_descriptor_dll_under_dir(dirpath, ctx.desc, ctx.pe_arch_to_check)


def find_lib_in_site_packages(ctx: SearchContext) -> str | NotFound:
    sub_dirs_searched: list[tuple[str, ...]] = []
    for rel_dir in ctx.site_packages_rel_dirs:
        sub_dir = PurePath(rel_dir).parts
        for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
            lib_path = _find_lib_file(ctx, abs_dir)
            if lib_path is not None:
                return lib_path
        sub_dirs_searched.append(sub_dir)
    missed = NotFound()
    for sub_dir in sub_dirs_searched:
        missed += _no_such_file_in_sub_dirs(sub_dir, ctx.lib_searched_for)
    return missed


def find_lib_in_dir(ctx: SearchContext, lib_dir: str) -> str | NotFound:
    lib_path = _find_lib_file(ctx, lib_dir)
    if lib_path is not None:
        return lib_path
    if ctx.pe_arch_to_check is None:
        error_message = f"No such file: {ctx.lib_searched_for}"
    else:
        error_message = f"No {ctx.pe_arch_to_check}-compatible PE file: {ctx.lib_searched_for}"
    if not os.path.isdir(lib_dir):
        attachments: tuple[str, ...] = (f'  listdir("{lib_dir}"):', "    DIRECTORY DOES NOT EXIST")
    else:
        attachments = _listdir_attachment(lib_dir)
    return NotFound((error_message,), attachments)
