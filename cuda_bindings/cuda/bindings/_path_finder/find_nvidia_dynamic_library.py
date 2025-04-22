# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import glob
import os

from .cuda_paths import IS_WIN32, get_cuda_paths
from .sys_path_find_sub_dirs import sys_path_find_sub_dirs


def _no_such_file_in_sub_dirs(sub_dirs, file_wild, error_messages, attachments):
    error_messages.append(f"No such file: {file_wild}")
    for sub_dir in sys_path_find_sub_dirs(sub_dirs):
        attachments.append(f'  listdir("{sub_dir}"):')
        for node in sorted(os.listdir(sub_dir)):
            attachments.append(f"    {node}")


def _find_so_using_nvidia_lib_dirs(libname, so_basename, error_messages, attachments):
    if libname == "nvvm":  # noqa: SIM108
        nvidia_sub_dirs = ("nvidia", "*", "nvvm", "lib64")
    else:
        nvidia_sub_dirs = ("nvidia", "*", "lib")
    file_wild = so_basename + "*"
    for lib_dir in sys_path_find_sub_dirs(nvidia_sub_dirs):
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


def _append_to_os_environ_path(dirpath):
    curr_path = os.environ.get("PATH")
    os.environ["PATH"] = dirpath if curr_path is None else os.pathsep.join((curr_path, dirpath))


def _find_dll_using_nvidia_bin_dirs(libname, error_messages, attachments):
    if libname == "nvvm":  # noqa: SIM108
        nvidia_sub_dirs = ("nvidia", "*", "nvvm", "bin")
    else:
        nvidia_sub_dirs = ("nvidia", "*", "bin")
    file_wild = libname + "*.dll"
    for bin_dir in sys_path_find_sub_dirs(nvidia_sub_dirs):
        dll_name = None
        have_builtins = False
        for path in sorted(glob.glob(os.path.join(bin_dir, file_wild))):
            # nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-win_amd64.whl:
            #     nvidia\cuda_nvrtc\bin\
            #         nvrtc-builtins64_128.dll
            #         nvrtc64_120_0.alt.dll
            #         nvrtc64_120_0.dll
            # See also:
            # https://github.com/NVIDIA/cuda-python/pull/563#discussion_r2054427641
            node = os.path.basename(path)
            if node.endswith(".alt.dll"):
                continue
            if "-builtins" in node:
                have_builtins = True
                continue
            if dll_name is not None:
                continue
            if os.path.isfile(path):
                dll_name = path
        if dll_name is not None:
            if have_builtins:
                # Add the DLL directory to the search path
                os.add_dll_directory(bin_dir)
                # Update PATH as a fallback for dependent DLL resolution
                _append_to_os_environ_path(bin_dir)
            return dll_name
    _no_such_file_in_sub_dirs(nvidia_sub_dirs, file_wild, error_messages, attachments)
    return None


def _get_cuda_paths_info(key, error_messages):
    env_path_tuple = get_cuda_paths()[key]
    if not env_path_tuple:
        error_messages.append(f'Failure obtaining get_cuda_paths()["{key}"]')
        return None
    if not env_path_tuple.info:
        error_messages.append(f'Failure obtaining get_cuda_paths()["{key}"].info')
        return None
    return env_path_tuple.info


def _find_so_using_cudalib_dir(so_basename, error_messages, attachments):
    cudalib_dir = _get_cuda_paths_info("cudalib_dir", error_messages)
    if cudalib_dir is None:
        return None
    primary_so_dir = cudalib_dir + "/"
    candidate_so_dirs = [primary_so_dir]
    libs = ["/lib/", "/lib64/"]
    for _ in range(2):
        alt_dir = libs[0].join(primary_so_dir.rsplit(libs[1], 1))
        if alt_dir not in candidate_so_dirs:
            candidate_so_dirs.append(alt_dir)
        libs.reverse()
    candidate_so_names = [so_dirname + so_basename for so_dirname in candidate_so_dirs]
    for so_name in candidate_so_names:
        if os.path.isfile(so_name):
            return so_name
        error_messages.append(f"No such file: {so_name}")
    for so_dirname in candidate_so_dirs:
        attachments.append(f'  listdir("{so_dirname}"):')
        if not os.path.isdir(so_dirname):
            attachments.append("    DIRECTORY DOES NOT EXIST")
        else:
            for node in sorted(os.listdir(so_dirname)):
                attachments.append(f"    {node}")
    return None


def _find_dll_using_cudalib_dir(libname, error_messages, attachments):
    cudalib_dir = _get_cuda_paths_info("cudalib_dir", error_messages)
    if cudalib_dir is None:
        return None
    file_wild = libname + "*.dll"
    for dll_name in sorted(glob.glob(os.path.join(cudalib_dir, file_wild))):
        if os.path.isfile(dll_name):
            return dll_name
    error_messages.append(f"No such file: {file_wild}")
    attachments.append(f'  listdir("{cudalib_dir}"):')
    for node in sorted(os.listdir(cudalib_dir)):
        attachments.append(f"    {node}")
    return None


class _find_nvidia_dynamic_library:
    def __init__(self, libname: str):
        self.libname = libname
        self.error_messages = []
        self.attachments = []
        self.abs_path = None

        if IS_WIN32:
            self.abs_path = _find_dll_using_nvidia_bin_dirs(libname, self.error_messages, self.attachments)
            if self.abs_path is None:
                if libname == "nvvm":
                    self.abs_path = _get_cuda_paths_info("nvvm", self.error_messages)
                else:
                    self.abs_path = _find_dll_using_cudalib_dir(libname, self.error_messages, self.attachments)
            self.lib_searched_for = f"{libname}*.dll"
        else:
            self.lib_searched_for = f"lib{libname}.so"
            self.abs_path = _find_so_using_nvidia_lib_dirs(
                libname, self.lib_searched_for, self.error_messages, self.attachments
            )
            if self.abs_path is None:
                if libname == "nvvm":
                    self.abs_path = _get_cuda_paths_info("nvvm", self.error_messages)
                else:
                    self.abs_path = _find_so_using_cudalib_dir(
                        self.lib_searched_for, self.error_messages, self.attachments
                    )

    def raise_if_abs_path_is_None(self):
        if self.abs_path:
            return self.abs_path
        err = ", ".join(self.error_messages)
        att = "\n".join(self.attachments)
        raise RuntimeError(f'Failure finding "{self.lib_searched_for}": {err}\n{att}')


@functools.cache
def find_nvidia_dynamic_library(libname: str) -> str:
    return _find_nvidia_dynamic_library(libname).raise_if_abs_path_is_None()
