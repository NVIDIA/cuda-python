# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import os

from .cuda_paths import get_cuda_paths
from .find_nvidia_lib_dirs import find_nvidia_lib_dirs


def _find_using_nvidia_lib_dirs(so_basename, error_messages, attachments):
    for lib_dir in find_nvidia_lib_dirs():
        so_name = os.path.join(lib_dir, so_basename)
        if os.path.isfile(so_name):
            return so_name
        error_messages.append(f"No such file: {so_name}")
    for lib_dir in find_nvidia_lib_dirs():
        attachments.append(f"  listdir({repr(lib_dir)}):")
        for node in sorted(os.listdir(lib_dir)):
            attachments.append(f"    {node}")
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


def _find_using_lib_dir(so_basename, error_messages, attachments):
    lib_dir = _get_cuda_paths_info("cudalib_dir", error_messages)
    primary_so_dir = lib_dir + "/"
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
        attachments.append(f"  listdir({repr(so_dirname)}):")
        if not os.path.isdir(so_dirname):
            attachments.append("    DIRECTORY DOES NOT EXIST")
        else:
            for node in sorted(os.listdir(so_dirname)):
                attachments.append(f"    {node}")
    return None


@functools.cache
def find_nvidia_dynamic_library(libbasename):
    so_basename = f"lib{libbasename}.so"
    error_messages = []
    attachments = []
    so_name = _find_using_nvidia_lib_dirs(so_basename, error_messages, attachments)
    if so_name is None:
        if libbasename == "nvvm":
            so_name = _get_cuda_paths_info("nvvm", error_messages)
        else:
            so_name = _find_using_lib_dir(so_basename, error_messages, attachments)
    if so_name is None:
        attachments = "\n".join(attachments)
        raise RuntimeError(f"Unable to load {so_basename} from: {', '.join(error_messages)}\n{attachments}")
    return so_name
