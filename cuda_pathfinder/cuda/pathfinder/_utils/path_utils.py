# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def _abs_norm(path: str | None) -> str | None:
    """Normalize and convert a path to an absolute path.

    Args:
        path (str or None): The path to normalize and make absolute.

    Returns:
        str or None: The normalized absolute path, or None if the input is None
        or empty.
    """
    if path:
        return os.path.normpath(os.path.abspath(path))
    return None
