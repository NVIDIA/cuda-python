# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Internal support for error-enum explanations.

``cuda_core`` keeps frozen 13.1.1 fallback tables for older ``cuda-bindings``
releases. Driver/runtime error enums carry usable ``__doc__`` text starting in
the 12.x backport line at ``cuda-bindings`` 12.9.6, and in the mainline 13.x
series at ``cuda-bindings`` 13.2.0. This module decides which source to use
and normalizes generated docstrings so user-facing ``CUDAError`` messages stay
presentable.

The cleanup rules here were derived while validating generated enum docstrings
in PR #1805. Keep them narrow and remove them when codegen quirks or fallback
support are no longer needed.
"""

from __future__ import annotations

import importlib.metadata
import re
from typing import Any

_MIN_12X_BINDING_VERSION_FOR_ENUM_DOCSTRINGS = (12, 9, 6)
_MIN_13X_BINDING_VERSION_FOR_ENUM_DOCSTRINGS = (13, 2, 0)


# ``version.pyx`` cannot be reused here (circular import via ``cuda_utils``).
def _binding_version() -> tuple[int, int, int]:
    """Return the installed ``cuda-bindings`` version, or a conservative old value."""
    try:
        parts = importlib.metadata.version("cuda-bindings").split(".")[:3]
    except importlib.metadata.PackageNotFoundError:
        return (0, 0, 0)  # For very old versions of cuda-python
    return tuple(int(v) for v in parts)


def _binding_version_has_usable_enum_docstrings(version: tuple[int, int, int]) -> bool:
    """Whether released bindings are known to carry usable error-enum ``__doc__`` text."""
    return (
        _MIN_12X_BINDING_VERSION_FOR_ENUM_DOCSTRINGS <= version < (13, 0, 0)
        or version >= _MIN_13X_BINDING_VERSION_FOR_ENUM_DOCSTRINGS
    )


def _fix_hyphenation_wordwrap_spacing(s: str) -> str:
    """Remove spaces around hyphens introduced by line wrapping in generated ``__doc__`` text.

    This is a narrow workaround for wrapped forms such as ``non- linear`` that
    would otherwise look awkward in user-facing messages.
    """
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"([a-z])- ([a-z])", r"\1-\2", s)
        s = re.sub(r"([a-z]) -([a-z])", r"\1-\2", s)
    return s


def clean_enum_member_docstring(doc: str | None) -> str | None:
    """Turn an enum member ``__doc__`` into plain text.

    The generated enum docstrings are already close to user-facing prose, but
    they may contain Sphinx inline roles, line wrapping, or a small known
    codegen defect. Normalize only those differences so the text is suitable
    for error messages.
    """
    if doc is None:
        return None
    s = doc
    # Known codegen bug on cudaErrorIncompatibleDriverContext. Remove once fixed
    # in cuda-bindings code generation. Do not use a raw string for the needle:
    # r"\n..." would not match the real newline present in __doc__.
    s = s.replace("\n:py:obj:`~.Interactions`", ' "Interactions ')
    s = re.sub(
        r":(?:py:)?(?:obj|func|meth|class|mod|data|const|exc):`([^`]+)`",
        lambda m: re.sub(r"^~?\.", "", m.group(1)),
        s,
    )
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _fix_hyphenation_wordwrap_spacing(s)
    return s


class DocstringBackedExplanations:
    """``dict.get``-like lookup over enum-member ``__doc__`` strings.

    Once the bindings-version gate says docstrings are available, use them
    exclusively. Missing docstrings should surface as ``None`` / ``default``
    rather than silently mixing in frozen fallback prose.
    """

    __slots__ = ("_enum_type",)

    def __init__(self, enum_type: Any) -> None:
        self._enum_type = enum_type

    def get(self, code: int, default: str | None = None) -> str | None:
        try:
            member = self._enum_type(code)
        except ValueError:
            return default

        raw_doc = member.__doc__
        if raw_doc is None:
            return default

        return clean_enum_member_docstring(raw_doc)


def get_best_available_explanations(
    enum_type: Any, fallback: dict[int, str | tuple[str, ...]]
) -> DocstringBackedExplanations | dict[int, str | tuple[str, ...]]:
    """Pick one explanation source per bindings version.

    Use enum-member ``__doc__`` only for bindings versions known to expose
    usable per-member text (12.9.6+ in the 12.x backport line, 13.2.0+ in the
    13.x mainline). Otherwise keep using the frozen 13.1.1 fallback tables.
    """
    if not _binding_version_has_usable_enum_docstrings(_binding_version()):
        return fallback
    return DocstringBackedExplanations(enum_type)
