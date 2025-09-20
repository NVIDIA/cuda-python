# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


def assert_type(obj, expected_type):
    """Ensure obj is of expected_type, else raise AssertionError with a clear message."""
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected type {expected_type.__name__}, but got {type(obj).__name__}")


def assert_type_str_or_bytes_like(obj):
    """Ensure obj is of type str or bytes, else raise AssertionError with a clear message."""
    if not isinstance(obj, (str, bytes, bytearray)):
        raise TypeError(f"Expected type str or bytes or bytearray, but got {type(obj).__name__}")


def raise_code_path_meant_to_be_unreachable():
    raise RuntimeError("This code path is meant to be unreachable.")
