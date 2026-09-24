# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The cuda-bindings version floor: how it is read and how it is enforced.

cuda.core supports two CUDA major series at a time. For each series, it
requires a minimum cuda-bindings version, the *floor*, at build time and at
run time. The floor is the newest cuda-bindings release of that series at the
time of the cuda.core release. Normally that is the release that the cuda.core
wheels are built against. See https://github.com/NVIDIA/cuda-python/issues/2783
and the support policy in the documentation.

Exactly one place declares the floors: the ``cu12`` and ``cu13`` extras in
``pyproject.toml``. Each extra pins ``cuda-bindings>=<floor>,<<next major>``.
Everything else derives from them through :func:`floors_from_extras`:

- The build backend, ``build_hooks.py``, checks the installed cuda-bindings
  against the floor. It checks that the ``cuda.h`` that it compiles against is
  the one that cuda-bindings was generated from. It records the floor and the
  header in the generated ``_build_info.py``.
- ``cuda/core/__init__.py`` checks the installed cuda-bindings against that
  record with :func:`check_installed_bindings`.
- The documentation reads the floors into substitutions in ``docs/source/conf.py``.
- ``toolshed/check_cuda_core_bindings_floor.py``, a pre-commit hook, checks
  that the CI toolkit pins in ``ci/versions.yml`` sit in the minors of the floors.

The build and the tools above load this module by file path. It therefore uses
the standard library only and must not import anything from ``cuda``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

__all__ = [
    "SUPPORT_URL",
    "bindings_requirement",
    "check_installed_bindings",
    "cuda_version_of",
    "floors_from_extras",
    "format_version",
    "release_triple",
]

_RELEASE_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")
_EXTRA_RE = re.compile(r"^cu(\d+)$")
# A cuda-bindings requirement, with or without extras such as `[all]`, up to
# an optional environment marker. After the name, the lookahead requires `[`, an
# operator, a marker, whitespace or the end, so that other names do not match.
_REQUIREMENT_RE = re.compile(
    r"^\s*cuda-bindings(?=[\[<>=!~;\s]|$)\s*(?:\[[^\]]*\])?\s*(?P<specifiers>[^;]*?)\s*(?:;.*)?$"
)
_FLOOR_SPEC_RE = re.compile(r"^>=(\d+)\.(\d+)\.(\d+)$")
# The upper bound: `<14`, `<14.0` or `<14.0.0`, or the equivalent `==13.*`.
_UPPER_SPEC_RE = re.compile(r"^<(\d+)(?:\.0)*$")
_MAJOR_SPEC_RE = re.compile(r"^==(\d+)\.\*$")

# The support policy section the error messages refer to.
SUPPORT_URL = "https://nvidia.github.io/cuda-python/cuda-core/latest/support.html#cuda-core-bindings-floor"


def release_triple(version: str) -> tuple[int, int, int] | None:
    """The leading ``major.minor.patch`` of a version string, or None.

    The function ignores pre-release and development suffixes, so ``13.4.1``,
    ``13.4.1a0`` and ``13.4.2.dev249+gabcdef`` yield (13, 4, 1), (13, 4, 1)
    and (13, 4, 2). A string without three leading integers yields None, for
    example the ``0.1.dev1`` that setuptools-scm reports for a shallow clone.
    """
    m = _RELEASE_RE.match(version.strip())
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def format_version(triple: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in triple)


def cuda_version_of(triple: tuple[int, int, int]) -> int:
    """The ``CUDA_VERSION`` macro value for the major.minor of a version triple, for example 13040."""
    return triple[0] * 1000 + triple[1] * 10


def floors_from_extras(extras: Mapping[str, Sequence[str]]) -> dict[int, tuple[int, int, int]]:
    """The floor per CUDA major, from the ``cu<major>`` extras of ``pyproject.toml``.

    ``extras`` is the parsed ``[project.optional-dependencies]`` table. Each
    ``cu<N>`` extra must list exactly one cuda-bindings requirement with two
    specifiers, in either order. One is the floor, ``>=<N>.<minor>.<patch>``.
    The other is an upper bound that excludes the next major, ``<<N+1>`` or
    ``==<N>.*``. Anything else raises ValueError with the name of the extra, so
    a typo fails the build and the pre-commit hook instead of a silent shift of
    the floor.
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
        confined = [cm for s in specifiers if (cm := _confined_major(s)) is not None]
        if len(specifiers) != 2 or len(floor_matches) != 1 or len(confined) != 1:
            raise ValueError(
                f"the {extra!r} extra must pin cuda-bindings as '>=<floor>,<<next major>', "
                f"for example '>=13.4.1,<14', but it lists {requirement!r}"
            )
        floor = (int(floor_matches[0].group(1)), int(floor_matches[0].group(2)), int(floor_matches[0].group(3)))
        if floor[0] != major or confined[0] != major:
            raise ValueError(
                f"the {extra!r} extra pins cuda-bindings {requirement!r}, which does not confine it to CUDA {major}"
            )
        floors[major] = floor
    if not floors:
        raise ValueError("pyproject.toml declares no cu<major> extra")
    return dict(sorted(floors.items()))


def _confined_major(specifier: str) -> int | None:
    """The one major a specifier confines cuda-bindings to: ``<14`` -> 13, ``==13.*`` -> 13, else None."""
    if (m := _UPPER_SPEC_RE.match(specifier)) is not None:
        return int(m.group(1)) - 1
    if (m := _MAJOR_SPEC_RE.match(specifier)) is not None:
        return int(m.group(1))
    return None


def bindings_requirement(floor: tuple[int, int, int], below: tuple[int, ...] | None = None) -> str:
    """The pip requirement for cuda-bindings at or above ``floor`` and below ``below``, without extras.

    ``below`` defaults to the next major: ``(13, 4, 1)`` gives ``cuda-bindings>=13.4.1,<14``.
    With ``below=(13, 5)`` it gives ``cuda-bindings>=13.4.1,<13.5``.
    """
    upper = format_version(below) if below is not None else str(floor[0] + 1)
    return f"cuda-bindings>={format_version(floor)},<{upper}"


def header_minor(cuda_version: int) -> tuple[int, int]:
    """The (major, minor) of a ``CUDA_VERSION`` macro value: 13040 -> (13, 4)."""
    return cuda_version // 1000, cuda_version // 10 % 100


def check_installed_bindings(
    installed_version: str,
    installed_cuda_version: int,
    build_cuda_major: int,
    build_cuda_version: int,
    build_floor: tuple[int, int, int],
    core_version: str,
) -> tuple[int, int, int]:
    """Validate the installed cuda-bindings against a build and return its triple.

    ``installed_version`` and ``installed_cuda_version`` are the ``__version__``
    and ``driver.CUDA_VERSION`` of the installed cuda-bindings. The latter is
    the ``cuda.h`` that it was generated from, for example 13040.
    ``build_cuda_major``, ``build_cuda_version`` and ``build_floor`` are the
    build's record in ``_build_info.py``.

    Raises ImportError with an actionable message unless the installed
    cuda-bindings passes all of these checks:

    - It is a release.
    - It is of the major that this build was compiled for.
    - It is at least the floor.
    - It was generated from a ``cuda.h`` minor at least as new as the one that
      the build compiled against. The driver function table that the C++ layer
      looks up in cuda-bindings is keyed by the macros of that header, so an
      older minor may lack entries.

    The check compares headers as ``CUDA_VERSION`` values, not version strings,
    because a development build of cuda-bindings carries the previous release's
    version string.
    """
    installed = release_triple(installed_version)
    if installed is None:
        raise ImportError(
            f"cuda.core requires a cuda-bindings {build_cuda_major}.x release, "
            f"but the installed cuda-bindings version is {installed_version}"
        )
    major = installed[0]
    if major != build_cuda_major:
        raise ImportError(
            f"This cuda.core {core_version} build is for CUDA {build_cuda_major}, but the installed "
            f"cuda-bindings is {installed_version}. Install cuda-bindings {build_cuda_major}.x with: "
            f'pip install "cuda-bindings=={build_cuda_major}.*". If a cuda.core build for CUDA {major} exists, '
            "install it instead."
        )
    if installed < tuple(build_floor):
        floor = format_version(build_floor)
        raise ImportError(
            f"cuda.core {core_version} requires cuda-bindings >= {floor} for CUDA {major}, but cuda-bindings "
            f'{installed_version} is installed. Upgrade with: pip install -U "cuda-bindings>={floor},<{major + 1}"'
        )
    built_against, generated_from = header_minor(build_cuda_version), header_minor(installed_cuda_version)
    if generated_from < built_against:
        needed = f"{built_against[0]}.{built_against[1]}"
        raise ImportError(
            f"cuda.core {core_version} was compiled against CUDA {needed} headers and needs cuda-bindings "
            f"{needed} or newer, but cuda-bindings {installed_version} is installed. This does not require a "
            f"newer CUDA driver or toolkit: cuda.core supports older CUDA {major}.x versions at run time, "
            f'see {SUPPORT_URL}. Upgrade with: pip install -U "cuda-bindings>={needed}.0,<{major + 1}"'
        )
    return installed
