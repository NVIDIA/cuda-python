# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path


def resolve_path(*parts: str | os.PathLike[str]) -> str:
    """Build a canonical absolute path, resolving symlinks and parent components."""
    return str(Path(*parts).resolve())
