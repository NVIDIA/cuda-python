# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import os
from collections.abc import Sequence
from typing import Optional

from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
    IS_WINDOWS,
    is_suppressed_dll_file,
)
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs, find_sub_dirs_all_sitepackages


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
) -> Optional[str]:
    file_wild = so_basename + "*"
    nvidia_sub_dirs_list: list[tuple[str, ...]] = [("nvidia", "*", "lib")]  # works also for CTK 13 nvvm
    if libname == "nvvm":
        nvidia_sub_dirs_list.append(("nvidia", "*", "nvvm", "lib64"))  # CTK 12
    for nvidia_sub_dirs in nvidia_sub_dirs_list:
        for lib_dir in find_sub_dirs_all_sitepackages(nvidia_sub_dirs):
            # First look for an exact match
            so_name = os.path.join(lib_dir, so_basename)
            if os.path.isfile(so_name):
                return so_name
            # Look for a versioned library
            # Using sort here mainly to make the result deterministic.
            for so_name in sorted(glob.glob(os.path.join(lib_dir, file_wild))):
                if os.path.isfile(so_name):
                    return so_name
    _no_such_file_in_sub_dirs(nvidia_sub_dirs, file_wild, error_messages, attachments)
    return None


def _find_dll_under_dir(dirpath: str, file_wild: str) -> Optional[str]:
    for path in sorted(glob.glob(os.path.join(dirpath, file_wild))):
        if not os.path.isfile(path):
            continue
        if not is_suppressed_dll_file(os.path.basename(path)):
            return path
    return None


def _find_dll_using_nvidia_bin_dirs(
    libname: str, lib_searched_for: str, error_messages: list[str], attachments: list[str]
) -> Optional[str]:
    nvidia_sub_dirs_list: list[tuple[str, ...]] = [
        ("nvidia", "*", "bin"),  # CTK 12
        ("nvidia", "*", "bin", "*"),  # CTK 13, e.g. site-packages\nvidia\cu13\bin\x86_64\
    ]
    if libname == "nvvm":
        nvidia_sub_dirs_list.append(("nvidia", "*", "nvvm", "bin"))  # Only for CTK 12
    for nvidia_sub_dirs in nvidia_sub_dirs_list:
        for bin_dir in find_sub_dirs_all_sitepackages(nvidia_sub_dirs):
            dll_name = _find_dll_under_dir(bin_dir, lib_searched_for)
            if dll_name is not None:
                return dll_name
    _no_such_file_in_sub_dirs(nvidia_sub_dirs, lib_searched_for, error_messages, attachments)
    return None


def _get_cuda_home() -> Optional[str]:
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home is None:
        cuda_home = os.environ.get("CUDA_PATH")
    return cuda_home


def _find_lib_dir_using_cuda_home(libname: str) -> Optional[str]:
    cuda_home = _get_cuda_home()
    if cuda_home is None:
        return None
    subdirs_list: tuple[tuple[str, ...], ...]
    if IS_WINDOWS:
        if libname == "nvvm":  # noqa: SIM108
            subdirs_list = (
                ("nvvm", "bin", "*"),  # CTK 13
                ("nvvm", "bin"),  # CTK 12
            )
        else:
            subdirs_list = (
                ("bin", "x64"),  # CTK 13
                ("bin",),  # CTK 12
            )
    else:
        if libname == "nvvm":  # noqa: SIM108
            subdirs_list = (("nvvm", "lib64"),)
        else:
            subdirs_list = (
                ("lib64",),  # CTK
                ("lib",),  # Conda
            )
    for sub_dirs in subdirs_list:
        dirname: str  # work around bug in mypy
        for dirname in find_sub_dirs((cuda_home,), sub_dirs):
            return dirname
    return None


def _find_so_using_lib_dir(
    lib_dir: str, so_basename: str, error_messages: list[str], attachments: list[str]
) -> Optional[str]:
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
) -> Optional[str]:
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
        self.error_messages: list[str] = []
        self.attachments: list[str] = []
        self.abs_path = None

        if IS_WINDOWS:
            self.lib_searched_for = f"{libname}*.dll"
            if self.abs_path is None:
                self.abs_path = _find_dll_using_nvidia_bin_dirs(
                    libname,
                    self.lib_searched_for,
                    self.error_messages,
                    self.attachments,
                )
        else:
            self.lib_searched_for = f"lib{libname}.so"
            if self.abs_path is None:
                self.abs_path = _find_so_using_nvidia_lib_dirs(
                    libname,
                    self.lib_searched_for,
                    self.error_messages,
                    self.attachments,
                )

    def retry_with_cuda_home_priority_last(self) -> None:
        cuda_home_lib_dir = _find_lib_dir_using_cuda_home(self.libname)
        if cuda_home_lib_dir is not None:
            if IS_WINDOWS:
                self.abs_path = _find_dll_using_lib_dir(
                    cuda_home_lib_dir,
                    self.libname,
                    self.error_messages,
                    self.attachments,
                )
            else:
                self.abs_path = _find_so_using_lib_dir(
                    cuda_home_lib_dir,
                    self.lib_searched_for,
                    self.error_messages,
                    self.attachments,
                )

    def raise_if_abs_path_is_None(self) -> str:  # noqa: N802
        if self.abs_path:
            return self.abs_path
        err = ", ".join(self.error_messages)
        att = "\n".join(self.attachments)
        raise DynamicLibNotFoundError(f'Failure finding "{self.lib_searched_for}": {err}\n{att}')


@functools.cache
def find_nvidia_dynamic_lib(libname: str) -> str:
    return _FindNvidiaDynamicLib(libname).raise_if_abs_path_is_None()
