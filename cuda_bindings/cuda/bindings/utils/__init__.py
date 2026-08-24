# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable

from ._nvvm_utils import check_nvvm_compiler_options
from ._ptx_utils import get_minimal_required_cuda_ver_from_ptx_ver, get_ptx_ver
from ._version_check import warn_if_cuda_major_version_mismatch

_handle_getters: dict[type, Callable[[Any], int]] = {}


def _add_cuda_native_handle_getter(t: type, getter: Callable[[Any], int]) -> None:
    _handle_getters[t] = getter


def get_cuda_native_handle(obj: Any) -> int:
    """Returns the address of the provided CUDA Python object as a Python int.

    Parameters
    ----------
    obj : Any
        CUDA Python object

    Returns
    -------
    int : The object address.
    """
    obj_type = type(obj)
    try:
        getter = _handle_getters[obj_type]
    except KeyError:
        raise TypeError("Unknown type: " + str(obj_type)) from None
    # Deliberately outside the try: a KeyError raised by the getter itself is a
    # bug in that getter, not an unregistered type.
    return getter(obj)
