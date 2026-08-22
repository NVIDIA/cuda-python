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
from pathlib import PurePath
from typing import Protocol, cast

from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import is_suppressed_dll_file
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.windows_arch import windows_pe_matches_arch, windows_python_arch


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
        sub_dir = PurePath(rel_dir).parts
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

    # Prefer the descriptor's known DLL names in its established search order.
    # Prefix matching remains the forward-compatible fallback for libraries
    # whose version is encoded in the filename (for example CUPTI).
    for dll_basename in reversed(cast(tuple[str, ...], desc.windows_dlls)):
        path = os.path.join(dirpath, dll_basename)
        if candidate_is_usable(path):
            return path

    if desc.windows_dll_match_mode == "prefix":
        file_wild = os.path.join(dirpath, f"{desc.name}*.dll")
        for path in sorted(glob.glob(file_wild)):
            if candidate_is_usable(path):
                return path
    return None


def _find_dll_in_rel_dirs(
    rel_dirs: tuple[str, ...],
    desc: LibDescriptor,
    target_arch: str,
    lib_searched_for: str,
    error_messages: list[str],
    attachments: list[str],
) -> str | None:
    sub_dirs_searched: list[tuple[str, ...]] = []
    checked_arch = target_arch if desc.requires_windows_binary_arch_check else None
    for rel_dir in rel_dirs:
        sub_dir = PurePath(rel_dir).parts
        for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
            dll_name = _find_descriptor_dll_under_dir(abs_dir, desc, checked_arch)
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

    def install_root_env_vars(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def install_root_env_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def program_files_root_globs(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        desc: LibDescriptor,
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None: ...

    def find_in_lib_dir(
        self,
        lib_dir: str,
        desc: LibDescriptor,
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

    def install_root_env_vars(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.install_root_env_vars_linux)

    def install_root_env_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.install_root_env_rel_dirs_linux)

    def program_files_root_globs(self, _desc: LibDescriptor) -> tuple[str, ...]:
        return ()

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        _desc: LibDescriptor,
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        return _find_so_in_rel_dirs(rel_dirs, lib_searched_for, error_messages, attachments)

    def find_in_lib_dir(
        self,
        lib_dir: str,
        _desc: LibDescriptor,
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
    target_arch: str

    def lib_searched_for(self, libname: str) -> str:
        return f"{libname}*.dll"

    def site_packages_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.site_packages_windows.for_arch(self.target_arch))

    def conda_anchor_point(self, conda_prefix: str) -> str:
        return os.path.join(conda_prefix, "Library")

    def anchor_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.anchor_rel_dirs_windows.for_arch(self.target_arch))

    def install_root_env_vars(self, desc: LibDescriptor) -> tuple[str, ...]:
        if self.target_arch not in desc.supported_windows_arch:
            return ()
        return cast(tuple[str, ...], desc.install_root_env_vars_windows)

    def install_root_env_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        if self.target_arch not in desc.supported_windows_arch:
            return ()
        return cast(tuple[str, ...], desc.install_root_env_rel_dirs_windows.for_arch(self.target_arch))

    def program_files_root_globs(self, desc: LibDescriptor) -> tuple[str, ...]:
        program_files = os.environ.get("PROGRAMW6432") or os.environ.get("PROGRAMFILES")
        if not program_files:
            return ()
        rel_globs = desc.program_files_root_globs_windows.for_arch(self.target_arch)
        return tuple(os.path.join(program_files, rel_glob) for rel_glob in rel_globs)

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        desc: LibDescriptor,
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        return _find_dll_in_rel_dirs(
            rel_dirs,
            desc,
            self.target_arch,
            lib_searched_for,
            error_messages,
            attachments,
        )

    def find_in_lib_dir(
        self,
        lib_dir: str,
        desc: LibDescriptor,
        _lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        file_wild = desc.name + "*.dll"
        target_arch = self.target_arch if desc.requires_windows_binary_arch_check else None
        dll_name = _find_descriptor_dll_under_dir(lib_dir, desc, target_arch)
        if dll_name is not None:
            return dll_name
        if target_arch is None:
            error_messages.append(f"No such file: {file_wild}")
        else:
            error_messages.append(f"No {target_arch}-compatible PE file: {file_wild}")
        attachments.append(f'  listdir("{lib_dir}"):')
        if not os.path.isdir(lib_dir):
            attachments.append("    DIRECTORY DOES NOT EXIST")
        else:
            for node in sorted(os.listdir(lib_dir)):
                attachments.append(f"    {node}")
        return None


def _platform_for_current_system() -> SearchPlatform:
    if IS_WINDOWS:
        return WindowsSearchPlatform(target_arch=windows_python_arch())
    return LinuxSearchPlatform()


PLATFORM = _platform_for_current_system()
