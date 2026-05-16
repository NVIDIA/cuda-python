# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform abstraction for filesystem search steps.

The goal is to keep :mod:`search_steps` platform-agnostic: it should not branch
on OS flags like ``IS_WINDOWS``. Instead, it calls through the single
``PLATFORM`` instance exported here.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import is_suppressed_dll_file
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


def _no_such_file_in_sub_dirs(
    sub_dirs: Sequence[str], file_wild: str, error_messages: list[str], attachments: list[str]
) -> None:
    error_messages.append(f"No such file: {file_wild}")
    for sub_dir in find_sub_dirs_all_sitepackages(sub_dirs):
        attachments.append(f'  listdir("{sub_dir}"):')
        for node in sorted(os.listdir(sub_dir)):
            attachments.append(f"    {node}")


def _find_so_in_rel_dirs(
    rel_dirs: tuple[str, ...],
    so_basename: str,
    error_messages: list[str],
    attachments: list[str],
) -> str | None:
    sub_dirs_searched: list[tuple[str, ...]] = []
    file_wild = so_basename + "*"
    for rel_dir in rel_dirs:
        sub_dir = tuple(rel_dir.split(os.path.sep))
        for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
            # Exact unversioned match first; fall back to versioned names because some
            # distros only ship lib<name>.so.<major> (e.g. conda libcupti). Only one match
            # is expected in practice. Sort in reverse so the newest-sorting name wins if
            # multiple coexist, matching the newest-first bias elsewhere in pathfinder
            # (see LinuxSearchPlatform.find_in_lib_dir and load_dl_linux._candidate_sonames).
            # Issue #1732 tracks the deferred question of raising on true ambiguity.
            so_name = os.path.join(abs_dir, so_basename)
            if os.path.isfile(so_name):
                return so_name
            for so_name in sorted(glob.glob(os.path.join(abs_dir, file_wild)), reverse=True):
                if os.path.isfile(so_name):
                    return so_name
        sub_dirs_searched.append(sub_dir)
    for sub_dir in sub_dirs_searched:
        _no_such_file_in_sub_dirs(sub_dir, file_wild, error_messages, attachments)
    return None


def _find_dll_under_dir(dirpath: str, file_wild: str) -> str | None:
    for path in sorted(glob.glob(os.path.join(dirpath, file_wild))):
        if not os.path.isfile(path):
            continue
        if not is_suppressed_dll_file(os.path.basename(path)):
            return path
    return None


def _find_dll_in_rel_dirs(
    rel_dirs: tuple[str, ...],
    lib_searched_for: str,
    error_messages: list[str],
    attachments: list[str],
) -> str | None:
    sub_dirs_searched: list[tuple[str, ...]] = []
    for rel_dir in rel_dirs:
        sub_dir = tuple(rel_dir.split(os.path.sep))
        for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
            dll_name = _find_dll_under_dir(abs_dir, lib_searched_for)
            if dll_name is not None:
                return dll_name
        sub_dirs_searched.append(sub_dir)
    for sub_dir in sub_dirs_searched:
        _no_such_file_in_sub_dirs(sub_dir, lib_searched_for, error_messages, attachments)
    return None


class SearchPlatform(Protocol):
    def lib_searched_for(self, libname: str) -> str: ...

    def site_packages_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def conda_anchor_point(self, conda_prefix: str) -> str: ...

    def anchor_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None: ...

    def find_in_lib_dir(
        self,
        lib_dir: str,
        libname: str,
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None: ...


@dataclass(frozen=True, slots=True)
class LinuxSearchPlatform:
    def lib_searched_for(self, libname: str) -> str:
        return f"lib{libname}.so"

    def site_packages_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.site_packages_linux)

    def conda_anchor_point(self, conda_prefix: str) -> str:
        return conda_prefix

    def anchor_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.anchor_rel_dirs_linux)

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        return _find_so_in_rel_dirs(rel_dirs, lib_searched_for, error_messages, attachments)

    def find_in_lib_dir(
        self,
        lib_dir: str,
        _libname: str,
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        # Most libraries have both unversioned and versioned files/symlinks (exact match first)
        so_name = os.path.join(lib_dir, lib_searched_for)
        if os.path.isfile(so_name):
            return so_name
        # Some libraries only exist as versioned files (e.g., libcupti.so.13 in conda),
        # so the glob fallback is needed
        file_wild = lib_searched_for + "*"
        # Only one match is expected, but to ensure deterministic behavior in unexpected
        # situations, and to be internally consistent, we sort in reverse order with the
        # intent to return the newest version first. Issue #1732 tracks the deferred
        # question of raising on true ambiguity.
        for so_name in sorted(glob.glob(os.path.join(lib_dir, file_wild)), reverse=True):
            if os.path.isfile(so_name):
                return so_name
        error_messages.append(f"No such file: {file_wild}")
        attachments.append(f'  listdir("{lib_dir}"):')
        if not os.path.isdir(lib_dir):
            attachments.append("    DIRECTORY DOES NOT EXIST")
        else:
            for node in sorted(os.listdir(lib_dir)):
                attachments.append(f"    {node}")
        return None


@dataclass(frozen=True, slots=True)
class WindowsSearchPlatform:
    def lib_searched_for(self, libname: str) -> str:
        return f"{libname}*.dll"

    def site_packages_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.site_packages_windows)

    def conda_anchor_point(self, conda_prefix: str) -> str:
        return os.path.join(conda_prefix, "Library")

    def anchor_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.anchor_rel_dirs_windows)

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        return _find_dll_in_rel_dirs(rel_dirs, lib_searched_for, error_messages, attachments)

    def find_in_lib_dir(
        self,
        lib_dir: str,
        libname: str,
        _lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        file_wild = libname + "*.dll"
        dll_name = _find_dll_under_dir(lib_dir, file_wild)
        if dll_name is not None:
            return dll_name
        error_messages.append(f"No such file: {file_wild}")
        attachments.append(f'  listdir("{lib_dir}"):')
        if not os.path.isdir(lib_dir):
            attachments.append("    DIRECTORY DOES NOT EXIST")
        else:
            for node in sorted(os.listdir(lib_dir)):
                attachments.append(f"    {node}")
        return None


PLATFORM: SearchPlatform = WindowsSearchPlatform() if IS_WINDOWS else LinuxSearchPlatform()
