# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

_CUDA_VERSION_RE = re.compile(r"^\s*#\s*define\s+CUDA_VERSION\s+(?P<encoded>\d+)\b", re.MULTILINE)


class ReadCudaHeaderVersionError(RuntimeError):
    """Raised when ``read_cuda_header_version()`` cannot determine the CTK version from ``cuda.h``."""


@dataclass(frozen=True, slots=True)
class CudaToolkitVersion:
    """CUDA Toolkit version encoded by the ``CUDA_VERSION`` macro in ``cuda.h``."""

    encoded: int
    major: int
    minor: int


def parse_cuda_header_version(header_text: str) -> CudaToolkitVersion | None:
    """Parse the CUDA Toolkit major/minor version from ``cuda.h`` text."""
    match = _CUDA_VERSION_RE.search(header_text)
    if match is None:
        return None
    encoded = int(match.group("encoded"))
    return CudaToolkitVersion(
        encoded=encoded,
        major=encoded // 1000,
        minor=(encoded % 1000) // 10,
    )


@functools.cache
def read_cuda_header_version(cuda_header_path: str) -> CudaToolkitVersion:
    """Read and parse the CUDA Toolkit major/minor version from ``cuda.h``."""
    try:
        header_text = Path(cuda_header_path).read_text(encoding="utf-8", errors="replace")
        version = parse_cuda_header_version(header_text)
        if version is None:
            raise RuntimeError(f"{cuda_header_path!r} does not define CUDA_VERSION.")
        return version
    except Exception as exc:
        raise ReadCudaHeaderVersionError(
            f"Failed to read the CUDA Toolkit version from cuda.h at {cuda_header_path!r}."
        ) from exc
