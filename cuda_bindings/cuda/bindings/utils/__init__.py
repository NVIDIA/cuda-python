# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from typing import Any, Callable

from ._ptx_utils import get_minimal_required_cuda_ver_from_ptx_ver, get_ptx_ver

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
        return _handle_getters[obj_type](obj)
    except KeyError:
        raise TypeError("Unknown type: " + str(obj_type)) from None
