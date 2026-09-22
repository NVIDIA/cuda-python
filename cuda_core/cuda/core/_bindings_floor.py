# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The cuda-bindings version floor: how it is read and how it is enforced.

cuda.core supports two CUDA major series at a time and requires, for each, a
minimum cuda-bindings version (the *floor*) at build time and at run time. The
floor is the newest cuda-bindings release of that series at the time of the
cuda.core release, normally the release cuda.core's own wheels are built
against. See https://github.com/NVIDIA/cuda-python/issues/2783 and the support
policy in the documentation.

The floors are declared in exactly one place: the ``cu12`` and ``cu13`` extras
in ``pyproject.toml``, each of which pins ``cuda-bindings>=<floor>,==<major>.*``.
Everything else derives from them through :func:`floors_from_extras`:

- the build backend (``build_hooks.py``) checks the installed cuda-bindings
  and header against the floor and records the floor of the build in the
  generated ``_build_info.py``;
- ``cuda/core/__init__.py`` checks the installed cuda-bindings against that
  record with :func:`check_installed_bindings`;
- the documentation reads the floors into substitutions (``docs/source/conf.py``);
- ``toolshed/check_cuda_core_bindings_floor.py`` (a pre-commit hook) checks
  that the CI toolkit pins in ``ci/versions.yml`` sit in the floors' minors.

This module is loaded by file path during the build and by the tools above,
so it uses the standard library only and must not import anything from
``cuda``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

__all__ = [
    "bindings_requirement",
    "check_installed_bindings",
    "cuda_version_of",
    "floors_from_extras",
    "format_version",
    "release_triple",
    "required_minimum",
]

_RELEASE_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")
_EXTRA_RE = re.compile(r"^cu(\d+)$")
# A cuda-bindings requirement, with or without extras such as `[all]`, up to
# an optional environment marker. The name must be followed by `[`, an
# operator, a marker, whitespace or the end so that other names do not match.
_REQUIREMENT_RE = re.compile(
    r"^\s*cuda-bindings(?=[\[<>=!~;\s]|$)\s*(?:\[[^\]]*\])?\s*(?P<specifiers>[^;]*?)\s*(?:;.*)?$"
)
_FLOOR_SPEC_RE = re.compile(r"^>=(\d+)\.(\d+)\.(\d+)$")
_MAJOR_SPEC_RE = re.compile(r"^==(\d+)\.\*$")


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


def floors_from_extras(extras: Mapping[str, Sequence[str]]) -> dict[int, tuple[int, int, int]]:
    """The floor per CUDA major, from the ``cu<major>`` extras of ``pyproject.toml``.

    ``extras`` is the parsed ``[project.optional-dependencies]`` table. Each
    ``cu<N>`` extra must list exactly one cuda-bindings requirement of the
    form ``cuda-bindings[...]>=<major>.<minor>.<patch>,==<N>.*`` (the order
    of the two specifiers does not matter). Anything else raises ValueError
    naming the extra, so a typo fails the build and the pre-commit hook
    instead of shifting the floor silently.
    """
    floors: dict[int, tuple[int, int, int]] = {}
    for extra, requirements in extras.items():
        m = _EXTRA_RE.match(extra)
        if m is None:
            continue
        major = int(m.group(1))
        matches = [(r, rm) for r in requirements if (rm := _REQUIREMENT_RE.match(r)) is not None]
        if len(matches) != 1:
            raise ValueError(
                f"the {extra!r} extra must list exactly one cuda-bindings requirement, found {len(matches)}"
            )
        requirement, requirement_match = matches[0]
        specifiers = [s.strip() for s in requirement_match.group("specifiers").split(",") if s.strip()]
        floor_matches = [fm for s in specifiers if (fm := _FLOOR_SPEC_RE.match(s)) is not None]
        major_matches = [mm for s in specifiers if (mm := _MAJOR_SPEC_RE.match(s)) is not None]
        if len(specifiers) != 2 or len(floor_matches) != 1 or len(major_matches) != 1:
            raise ValueError(
                f"the {extra!r} extra must pin cuda-bindings as '>=<major>.<minor>.<patch>,==<major>.*', "
                f"found {requirement!r}"
            )
        floor = (int(floor_matches[0].group(1)), int(floor_matches[0].group(2)), int(floor_matches[0].group(3)))
        pinned_major = int(major_matches[0].group(1))
        if floor[0] != major or pinned_major != major:
            raise ValueError(
                f"the {extra!r} extra pins cuda-bindings {requirement!r}, whose majors do not match CUDA {major}"
            )
        floors[major] = floor
    if not floors:
        raise ValueError("pyproject.toml declares no cu<major> extra")
    return dict(sorted(floors.items()))


def bindings_requirement(floor: tuple[int, int, int]) -> str:
    """The pip requirement that pins cuda-bindings to ``floor`` and its major, without extras."""
    return f"cuda-bindings>={format_version(floor)},=={floor[0]}.*"


def required_minimum(floor: tuple[int, int, int], header_cuda_version: int) -> tuple[int, int, int]:
    """The minimum cuda-bindings a build accepts at run time.

    A build accepts the floor of its major series, and never a cuda-bindings
    whose minor is older than the ``cuda.h`` the build compiled against: the
    driver function-pointer keys the C++ layer looks up in cuda-bindings are
    derived from that header's macros, so an older minor may lack them.
    """
    header_minor = (header_cuda_version // 1000, header_cuda_version // 10 % 100, 0)
    return max(floor, header_minor)


def check_installed_bindings(
    installed_version: str,
    build_cuda_major: int,
    build_cuda_version: int,
    build_floor: tuple[int, int, int],
    core_version: str,
) -> tuple[int, int, int]:
    """Validate the installed cuda-bindings against a build; return its triple.

    ``build_cuda_major``, ``build_cuda_version`` and ``build_floor`` are the
    build's record in ``_build_info.py``. Raises ImportError with an actionable
    message when the installed cuda-bindings is not a release, is not of the
    major this build was compiled for, or is older than the build's minimum.
    """
    installed = release_triple(installed_version)
    if installed is None:
        raise ImportError(f"a cuda-bindings {build_cuda_major}.x release is required (found {installed_version})")
    major = installed[0]
    if major != build_cuda_major:
        raise ImportError(
            f"this cuda.core {core_version} build is for CUDA {build_cuda_major}, but the installed "
            f"cuda-bindings is {installed_version}. Install cuda-bindings {build_cuda_major}.x "
            f"(pip install 'cuda-bindings=={build_cuda_major}.*'), or a cuda.core build for CUDA {major} if one exists."
        )
    minimum = required_minimum(build_floor, build_cuda_version)
    if installed < minimum:
        floor = format_version(minimum)
        raise ImportError(
            f"cuda.core {core_version} requires cuda-bindings >= {floor} for CUDA {major} "
            f"(found {installed_version}). Upgrade with: pip install -U 'cuda-bindings>={floor},=={major}.*'"
        )
    return installed
