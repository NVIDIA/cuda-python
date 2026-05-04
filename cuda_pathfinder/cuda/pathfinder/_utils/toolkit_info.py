# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

_CUDA_VERSION_RE = re.compile(r"^\s*#\s*define\s+CUDA_VERSION\s+(?P<encoded>\d+)\b", re.MULTILINE)
EncodedCudaVersionT = TypeVar("EncodedCudaVersionT", bound="EncodedCudaVersion")


@dataclass(frozen=True, slots=True)
class EncodedCudaVersion:
    """CUDA major/minor version represented in CUDA's integer ``encoded`` form."""

    encoded: int
    major: int
    minor: int

    @classmethod
    def from_encoded(cls: type[EncodedCudaVersionT], encoded: int | str) -> EncodedCudaVersionT:
        if isinstance(encoded, str):
            try:
                encoded_int = int(encoded)
            except ValueError as exc:
                raise ValueError(
                    f"{cls.__name__}.from_encoded() expected an integer or decimal string, got {encoded!r}."
                ) from exc
        elif isinstance(encoded, int):
            encoded_int = encoded
        else:
            raise TypeError(
                f"{cls.__name__}.from_encoded() expected an integer or decimal string, got {type(encoded).__name__}."
            )
        if encoded_int < 0:
            raise ValueError(
                f"{cls.__name__}.from_encoded() expected a non-negative encoded CUDA version, got {encoded_int}."
            )
        # CUDA encodes versions as major * 1000 + minor * 10. The least-significant
        # decimal is ignored here: it is 0 in all CUDA releases and is not a patch version.
        return cls(
            encoded=encoded_int,
            major=encoded_int // 1000,
            minor=(encoded_int % 1000) // 10,
        )


class ReadCudaHeaderVersionError(RuntimeError):
    """Raised when ``read_cuda_header_version()`` cannot determine the CTK version from ``cuda.h``."""


@dataclass(frozen=True, slots=True)
class CudaToolkitVersion(EncodedCudaVersion):
    """CUDA Toolkit version encoded by the ``CUDA_VERSION`` macro in ``cuda.h``."""


def parse_cuda_header_version(header_text: str) -> CudaToolkitVersion | None:
    """Parse the CUDA Toolkit major/minor version from ``cuda.h`` text."""
    match = _CUDA_VERSION_RE.search(header_text)
    if match is None:
        return None
    return CudaToolkitVersion.from_encoded(match.group("encoded"))


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
