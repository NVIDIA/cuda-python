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
        for node in sorted(glob.glob(os.path.join(lib_dir, file_wild))):
            so_name = os.path.join(lib_dir, node)
            if os.path.isfile(so_name):
                return so_name
    _no_such_file_in_sub_dirs(nvidia_sub_dirs, file_wild, error_messages, attachments)
    return None


def _find_dll_using_nvidia_bin_dirs(libname, error_messages, attachments):
    if libname == "nvvm":  # noqa: SIM108
        nvidia_sub_dirs = ("nvidia", "*", "nvvm", "bin")
    else:
        nvidia_sub_dirs = ("nvidia", "*", "bin")
    file_wild = libname + "*.dll"
    for bin_dir in sys_path_find_sub_dirs(nvidia_sub_dirs):
        for node in sorted(glob.glob(os.path.join(bin_dir, file_wild))):
            dll_name = os.path.join(bin_dir, node)
            if os.path.isfile(dll_name):
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
    error_messages = []
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
    for node in sorted(glob.glob(os.path.join(cudalib_dir, file_wild))):
        dll_name = os.path.join(cudalib_dir, node)
        if os.path.isfile(dll_name):
            return dll_name
    error_messages.append(f"No such file: {file_wild}")
    attachments.append(f'  listdir("{cudalib_dir}"):')
    for node in sorted(os.listdir(cudalib_dir)):
        attachments.append(f"    {node}")
    return None


@functools.cache
def find_nvidia_dynamic_library(name: str) -> str:
    error_messages = []
    attachments = []

    if IS_WIN32:
        dll_name = _find_dll_using_nvidia_bin_dirs(name, error_messages, attachments)
        if dll_name is None:
            if name == "nvvm":
                dll_name = _get_cuda_paths_info("nvvm", error_messages)
            else:
                dll_name = _find_dll_using_cudalib_dir(name, error_messages, attachments)
        if dll_name is None:
            attachments = "\n".join(attachments)
            raise RuntimeError(f"Failure finding {name}*.dll: {', '.join(error_messages)}\n{attachments}")
        return dll_name

    so_basename = f"lib{name}.so"
    so_name = _find_so_using_nvidia_lib_dirs(name, so_basename, error_messages, attachments)
    if so_name is None:
        if name == "nvvm":
            so_name = _get_cuda_paths_info("nvvm", error_messages)
        else:
            so_name = _find_so_using_cudalib_dir(so_basename, error_messages, attachments)
    if so_name is None:
        attachments = "\n".join(attachments)
        raise RuntimeError(f"Failure finding {so_basename}: {', '.join(error_messages)}\n{attachments}")
    return so_name