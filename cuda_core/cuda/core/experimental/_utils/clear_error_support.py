# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from typing import Any, Type, Union


def assert_type(obj: Any, expected_type: Type[Any]) -> None:
    """Ensure obj is of expected_type, else raise AssertionError with a clear message."""
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected type {expected_type.__name__}, but got {type(obj).__name__}")


def assert_type_str_or_bytes(obj: Any) -> None:
    """Ensure obj is of type str or bytes, else raise AssertionError with a clear message."""
    if not isinstance(obj, (str, bytes)):
        raise TypeError(f"Expected type str or bytes, but got {type(obj).__name__}")


def raise_code_path_meant_to_be_unreachable() -> None:
    raise RuntimeError("This code path is meant to be unreachable.")
