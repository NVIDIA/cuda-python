# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Internal support for error-enum explanations.

``cuda_core`` keeps frozen 13.1.1 fallback tables for older ``cuda-bindings``
releases. Starting with ``cuda-bindings`` 13.2.0, driver/runtime error enums
carry usable ``__doc__`` text. This module decides which source to use and
normalizes generated docstrings so user-facing ``CUDAError`` messages stay
close to the long-form explanation prose.

The cleanup rules here were derived while validating docstring-vs-dict parity
in PR #1805. Keep them narrow and remove them when codegen / fallback support is
no longer needed.
"""

from __future__ import annotations

import importlib.metadata
import re
from typing import Any

_MIN_BINDING_VERSION_FOR_ENUM_DOCSTRINGS = (13, 2, 0)


# ``version.pyx`` cannot be reused here (circular import via ``cuda_utils``).
def _binding_version() -> tuple[int, int, int]:
    """Return the installed ``cuda-bindings`` version, or a conservative old value."""
    try:
        parts = importlib.metadata.version("cuda-bindings").split(".")[:3]
    except importlib.metadata.PackageNotFoundError:
        return (0, 0, 0)  # For very old versions of cuda-python
    return tuple(int(v) for v in parts)


def _strip_doxygen_double_colon_prefixes(s: str) -> str:
    """Remove Doxygen-style ``::`` before CUDA identifiers (not C++ ``Foo::Bar`` scope).

    The frozen fallback tables come from CUDA header comments and therefore use
    Doxygen ``::name`` references. Generated enum ``__doc__`` text uses Sphinx
    roles instead, so parity checks need a small amount of normalization.
    """
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(?<![A-Za-z0-9_])::+([A-Za-z_][A-Za-z0-9_]*)", r"\1", s)
    return s


def _fix_hyphenation_wordwrap_spacing(s: str) -> str:
    """Remove spaces around hyphens introduced by line wrapping in generated ``__doc__`` text.

    This is a narrow workaround for wrapped forms such as ``non- linear`` that
    otherwise differ from the single-line fallback prose.
    """
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"([a-z])- ([a-z])", r"\1-\2", s)
        s = re.sub(r"([a-z]) -([a-z])", r"\1-\2", s)
    return s


def clean_enum_member_docstring(doc: str | None) -> str | None:
    """Turn an enum member ``__doc__`` into plain text.

    The generated enum docstrings are already close to the fallback explanation
    prose, but not byte-identical: they may contain Sphinx inline roles, line
    wrapping, or a small known codegen defect. Normalize only those differences
    so the text is suitable for user-facing error messages.
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

    ``cuda-bindings`` < 13.2.0: use the frozen 13.1.1 fallback tables.
    ``cuda-bindings`` >= 13.2.0: use enum-member ``__doc__`` exclusively.
    """
    if _binding_version() < _MIN_BINDING_VERSION_FOR_ENUM_DOCSTRINGS:
        return fallback
    return DocstringBackedExplanations(enum_type)
