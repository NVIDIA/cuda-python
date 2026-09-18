# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The cuda-bindings version floor: the single source of truth.

cuda.core supports two CUDA major series at a time and requires, for each, a
minimum cuda-bindings version (the *floor*) at build time and at run time. The
floor is the newest cuda-bindings release of that series that the CI source
root can build, normally the release cuda.core's own wheels are built against.
See https://github.com/NVIDIA/cuda-python/issues/2783 and the support policy.

This module is imported by the build backend (``build_hooks.py``, by file
path, because the package is not importable during its own build), by
``cuda/core/__init__.py`` at import time, and by tests that keep the static
pins in ``pyproject.toml`` and ``ci/versions.yml`` in step. It therefore uses
the standard library only and must not import anything from ``cuda``.

Bumping a floor is a release-note item under "Breaking Changes". Bump it in the
same PR that first uses a cuda-bindings API newer than the old floor; the CI
rows that install the floor bindings fail otherwise.
"""

from __future__ import annotations

import re

__all__ = [
    "CUDA_BINDINGS_FLOOR",
    "SUPPORTED_CUDA_MAJORS",
    "check_installed_bindings",
    "cuda_version_of",
    "format_version",
    "pip_requirement",
    "release_triple",
    "required_minimum",
]

# Minimum cuda-bindings release per CUDA major series, as a (major, minor, patch)
# triple. Keep in step with the `cu12`/`cu13` extras in pyproject.toml (tested).
CUDA_BINDINGS_FLOOR: dict[int, tuple[int, int, int]] = {
    12: (12, 9, 8),
    13: (13, 4, 1),
}

SUPPORTED_CUDA_MAJORS = tuple(sorted(CUDA_BINDINGS_FLOOR))

_RELEASE_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def release_triple(version: str) -> tuple[int, int, int] | None:
    """The leading ``major.minor.patch`` of a version string, or None.

    Pre-release and development suffixes are ignored, so ``13.4.1``,
    ``13.4.1a0`` and ``13.4.2.dev249+gabcdef`` yield (13, 4, 1), (13, 4, 1)
    and (13, 4, 2). A string without three leading integers (for example the
    ``0.1.dev1`` that setuptools-scm reports for a shallow clone) yields None.
    """
    m = _RELEASE_RE.match(version.strip())
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def format_version(triple: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in triple)


def cuda_version_of(triple: tuple[int, int, int]) -> int:
    """The ``CUDA_VERSION`` macro value (e.g. 13040) for a version triple's major.minor."""
    return triple[0] * 1000 + triple[1] * 10


def pip_requirement(major: int) -> str:
    """The pip requirement that pins cuda-bindings to the floor and the major."""
    return f"cuda-bindings>={format_version(CUDA_BINDINGS_FLOOR[major])},=={major}.*"


def required_minimum(cuda_major: int, header_cuda_version: int) -> tuple[int, int, int]:
    """The minimum cuda-bindings a build accepts at run time.

    A build accepts the floor of its major series, and never a cuda-bindings
    whose minor is older than the ``cuda.h`` the build compiled against: the
    driver function-pointer keys the C++ layer looks up in cuda-bindings are
    derived from that header's macros, so an older minor may lack them.
    """
    floor = CUDA_BINDINGS_FLOOR[cuda_major]
    header_minor = (header_cuda_version // 1000, header_cuda_version // 10 % 100, 0)
    return max(floor, header_minor)


def check_installed_bindings(
    installed_version: str,
    build_cuda_major: int,
    build_cuda_version: int,
    core_version: str,
) -> tuple[int, int, int]:
    """Validate the installed cuda-bindings against this build; return its triple.

    Raises ImportError with an actionable message when the installed
    cuda-bindings is not a release of a supported major, is not the major
    this build was compiled for, or is older than the build's minimum.
    """
    installed = release_triple(installed_version)
    if installed is None or installed[0] not in CUDA_BINDINGS_FLOOR:
        majors = " or ".join(f"{m}.x" for m in SUPPORTED_CUDA_MAJORS)
        raise ImportError(f"cuda-bindings {majors} must be installed (found {installed_version})")
    major = installed[0]
    if major != build_cuda_major:
        raise ImportError(
            f"this cuda.core {core_version} build is for CUDA {build_cuda_major}, but the installed "
            f"cuda-bindings is {installed_version}. Install cuda-bindings {build_cuda_major}.x, "
            f"or a cuda.core build for CUDA {major}."
        )
    minimum = required_minimum(major, build_cuda_version)
    if installed < minimum:
        floor = format_version(minimum)
        raise ImportError(
            f"cuda.core {core_version} requires cuda-bindings >= {floor} for CUDA {major} "
            f"(found {installed_version}). Upgrade with: pip install -U 'cuda-bindings>={floor},=={major}.*'"
        )
    return installed
