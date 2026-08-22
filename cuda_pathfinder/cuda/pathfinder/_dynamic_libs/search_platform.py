# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform abstraction for filesystem search steps.

The goal is to keep :mod:`search_steps` platform-agnostic: it should not branch
on OS flags like ``IS_WINDOWS``. Instead, it calls through the single
``PLATFORM`` instance exported here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
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
        for node in sorted(entry.name for entry in Path(sub_dir).iterdir()):
            attachments.append(f"    {node}")


def _find_so_in_rel_dirs(
    rel_dirs: tuple[str, ...],
    so_basename: str,
    error_messages: list[str],
    attachments: list[str],
) -> Path | None:
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
            abs_dir_path = Path(abs_dir)
            so_path = abs_dir_path / so_basename
            if so_path.is_file():
                return so_path
            for so_path in sorted(abs_dir_path.glob(file_wild), reverse=True):
                if so_path.is_file():
                    return so_path
        sub_dirs_searched.append(sub_dir)
    for sub_dir in sub_dirs_searched:
        _no_such_file_in_sub_dirs(sub_dir, file_wild, error_messages, attachments)
    return None


def _find_dll_under_dir(dirpath: Path, file_wild: str, target_arch: str | None = None) -> Path | None:
    for path in sorted(dirpath.glob(file_wild)):
        if not path.is_file():
            continue
        if is_suppressed_dll_file(path.name):
            continue
        # windows_pe_matches_arch() lives in _utils and is still str-typed.
        if target_arch is not None and not windows_pe_matches_arch(str(path), target_arch):
            continue
        return path
    return None


def _find_dll_in_rel_dirs(
    rel_dirs: tuple[str, ...],
    lib_searched_for: str,
    error_messages: list[str],
    attachments: list[str],
) -> Path | None:
    sub_dirs_searched: list[tuple[str, ...]] = []
    for rel_dir in rel_dirs:
        sub_dir = PurePath(rel_dir).parts
        for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
            dll_path = _find_dll_under_dir(Path(abs_dir), lib_searched_for)
            if dll_path is not None:
                return dll_path
        sub_dirs_searched.append(sub_dir)
    for sub_dir in sub_dirs_searched:
        _no_such_file_in_sub_dirs(sub_dir, lib_searched_for, error_messages, attachments)
    return None


class SearchPlatform(Protocol):
    def lib_searched_for(self, libname: str) -> str: ...

    def site_packages_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def conda_anchor_point(self, conda_prefix: str) -> Path: ...

    def anchor_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> Path | None: ...

    def find_in_lib_dir(
        self,
        lib_dir: Path,
        desc: LibDescriptor,
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> Path | None: ...


@dataclass(frozen=True, slots=True)
class LinuxSearchPlatform:
    def lib_searched_for(self, libname: str) -> str:
        return f"lib{libname}.so"

    def site_packages_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.site_packages_linux)

    def conda_anchor_point(self, conda_prefix: str) -> Path:
        return Path(conda_prefix)

    def anchor_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.anchor_rel_dirs_linux)

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> Path | None:
        return _find_so_in_rel_dirs(rel_dirs, lib_searched_for, error_messages, attachments)

    def find_in_lib_dir(
        self,
        lib_dir: Path,
        _desc: LibDescriptor,
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> Path | None:
        # Most libraries have both unversioned and versioned files/symlinks (exact match first)
        so_path = lib_dir / lib_searched_for
        if so_path.is_file():
            return so_path
        # Some libraries only exist as versioned files (e.g., libcupti.so.13 in conda),
        # so the glob fallback is needed
        file_wild = lib_searched_for + "*"
        # Only one match is expected, but to ensure deterministic behavior in unexpected
        # situations, and to be internally consistent, we sort in reverse order with the
        # intent to return the newest version first. Issue #1732 tracks the deferred
        # question of raising on true ambiguity.
        for so_path in sorted(lib_dir.glob(file_wild), reverse=True):
            if so_path.is_file():
                return so_path
        error_messages.append(f"No such file: {file_wild}")
        attachments.append(f'  listdir("{lib_dir}"):')
        if not lib_dir.is_dir():
            attachments.append("    DIRECTORY DOES NOT EXIST")
        else:
            for node in sorted(entry.name for entry in lib_dir.iterdir()):
                attachments.append(f"    {node}")
        return None


@dataclass(frozen=True, slots=True)
class WindowsSearchPlatform:
    target_arch: str

    def lib_searched_for(self, libname: str) -> str:
        return f"{libname}*.dll"

    def site_packages_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.site_packages_windows.for_arch(self.target_arch))

    def conda_anchor_point(self, conda_prefix: str) -> Path:
        return Path(conda_prefix, "Library")

    def anchor_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.anchor_rel_dirs_windows.for_arch(self.target_arch))

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> Path | None:
        return _find_dll_in_rel_dirs(rel_dirs, lib_searched_for, error_messages, attachments)

    def find_in_lib_dir(
        self,
        lib_dir: Path,
        desc: LibDescriptor,
        _lib_searched_for: str,
        error_messages: list[str],
        attachments: list[str],
    ) -> Path | None:
        file_wild = desc.name + "*.dll"
        target_arch = self.target_arch if desc.requires_windows_binary_arch_check else None
        dll_path = _find_dll_under_dir(lib_dir, file_wild, target_arch)
        if dll_path is not None:
            return dll_path
        if target_arch is None:
            error_messages.append(f"No such file: {file_wild}")
        else:
            error_messages.append(f"No {target_arch}-compatible PE file: {file_wild}")
        attachments.append(f'  listdir("{lib_dir}"):')
        if not lib_dir.is_dir():
            attachments.append("    DIRECTORY DOES NOT EXIST")
        else:
            for node in sorted(entry.name for entry in lib_dir.iterdir()):
                attachments.append(f"    {node}")
        return None


def _platform_for_current_system() -> SearchPlatform:
    if IS_WINDOWS:
        return WindowsSearchPlatform(target_arch=windows_python_arch())
    return LinuxSearchPlatform()


PLATFORM = _platform_for_current_system()
