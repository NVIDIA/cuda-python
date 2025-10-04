# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
from collections.abc import Sequence

from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
    SITE_PACKAGES_LIBDIRS_LINUX,
    SITE_PACKAGES_LIBDIRS_WINDOWS,
    is_suppressed_dll_file,
)
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
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


def _find_so_using_nvidia_lib_dirs(
    libname: str, so_basename: str, error_messages: list[str], attachments: list[str]
) -> str | None:
    rel_dirs = SITE_PACKAGES_LIBDIRS_LINUX.get(libname)
    if rel_dirs is not None:
        sub_dirs_searched = []
        file_wild = so_basename + "*"
        for rel_dir in rel_dirs:
            sub_dir = tuple(rel_dir.split(os.path.sep))
            for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
                # First look for an exact match
                so_name = os.path.join(abs_dir, so_basename)
                if os.path.isfile(so_name):
                    return so_name
                # Look for a versioned library
                # Using sort here mainly to make the result deterministic.
                for so_name in sorted(glob.glob(os.path.join(abs_dir, file_wild))):
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


def _find_dll_using_nvidia_bin_dirs(
    libname: str, lib_searched_for: str, error_messages: list[str], attachments: list[str]
) -> str | None:
    rel_dirs = SITE_PACKAGES_LIBDIRS_WINDOWS.get(libname)
    if rel_dirs is not None:
        sub_dirs_searched = []
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


def _find_lib_dir_using_anchor_point(libname: str, anchor_point: str, linux_lib_dir: str) -> str | None:
    # Resolve paths for the four cases:
    #    Windows/Linux x nvvm yes/no
    if IS_WINDOWS:
        if libname == "nvvm":  # noqa: SIM108
            rel_paths = [
                "nvvm/bin/*",  # CTK 13
                "nvvm/bin",  # CTK 12
            ]
        else:
            rel_paths = [
                "bin/x64",  # CTK 13
                "bin",  # CTK 12
            ]
    else:
        if libname == "nvvm":  # noqa: SIM108
            rel_paths = ["nvvm/lib64"]
        else:
            rel_paths = [linux_lib_dir]

    for rel_path in rel_paths:
        for dirname in sorted(glob.glob(os.path.join(anchor_point, rel_path))):
            if os.path.isdir(dirname):
                return dirname

    return None


def _find_lib_dir_using_cuda_home(libname: str) -> str | None:
    cuda_home = get_cuda_home_or_path()
    if cuda_home is None:
        return None
    return _find_lib_dir_using_anchor_point(libname, anchor_point=cuda_home, linux_lib_dir="lib64")


def _find_lib_dir_using_conda_prefix(libname: str) -> str | None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None
    return _find_lib_dir_using_anchor_point(
        libname, anchor_point=os.path.join(conda_prefix, "Library") if IS_WINDOWS else conda_prefix, linux_lib_dir="lib"
    )


def _find_so_using_lib_dir(
    lib_dir: str, so_basename: str, error_messages: list[str], attachments: list[str]
) -> str | None:
    so_name = os.path.join(lib_dir, so_basename)
    if os.path.isfile(so_name):
        return so_name
    error_messages.append(f"No such file: {so_name}")
    attachments.append(f'  listdir("{lib_dir}"):')
    if not os.path.isdir(lib_dir):
        attachments.append("    DIRECTORY DOES NOT EXIST")
    else:
        for node in sorted(os.listdir(lib_dir)):
            attachments.append(f"    {node}")
    return None


def _find_dll_using_lib_dir(
    lib_dir: str, libname: str, error_messages: list[str], attachments: list[str]
) -> str | None:
    file_wild = libname + "*.dll"
    dll_name = _find_dll_under_dir(lib_dir, file_wild)
    if dll_name is not None:
        return dll_name
    error_messages.append(f"No such file: {file_wild}")
    attachments.append(f'  listdir("{lib_dir}"):')
    for node in sorted(os.listdir(lib_dir)):
        attachments.append(f"    {node}")
    return None


class _FindNvidiaDynamicLib:
    def __init__(self, libname: str):
        self.libname = libname
        if IS_WINDOWS:
            self.lib_searched_for = f"{libname}*.dll"
        else:
            self.lib_searched_for = f"lib{libname}.so"
        self.error_messages: list[str] = []
        self.attachments: list[str] = []
        self.abs_path: str | None = None

    def try_site_packages(self) -> str | None:
        if IS_WINDOWS:
            return _find_dll_using_nvidia_bin_dirs(
                self.libname,
                self.lib_searched_for,
                self.error_messages,
                self.attachments,
            )
        else:
            return _find_so_using_nvidia_lib_dirs(
                self.libname,
                self.lib_searched_for,
                self.error_messages,
                self.attachments,
            )

    def try_with_conda_prefix(self) -> str | None:
        return self._find_using_lib_dir(_find_lib_dir_using_conda_prefix(self.libname))

    def try_with_cuda_home(self) -> str | None:
        return self._find_using_lib_dir(_find_lib_dir_using_cuda_home(self.libname))

    def _find_using_lib_dir(self, lib_dir: str | None) -> str | None:
        if lib_dir is None:
            return None
        if IS_WINDOWS:
            return _find_dll_using_lib_dir(
                lib_dir,
                self.libname,
                self.error_messages,
                self.attachments,
            )
        else:
            return _find_so_using_lib_dir(
                lib_dir,
                self.lib_searched_for,
                self.error_messages,
                self.attachments,
            )

    def raise_not_found_error(self) -> None:
        err = ", ".join(self.error_messages)
        att = "\n".join(self.attachments)
        raise DynamicLibNotFoundError(f'Failure finding "{self.lib_searched_for}": {err}\n{att}')
