# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal support for error-enum explanations.

Driver and runtime error enums in ``cuda-bindings`` carry per-member
``__doc__`` text (since 12.9.6 in the 12.x line and 13.2.0 in the 13.x line;
every ``cuda-bindings`` that ``cuda.core`` accepts has it). This module
normalizes those generated docstrings so user-facing ``CUDAError`` messages
stay presentable.

The cleanup rules here were derived while validating generated enum docstrings
in PR #1805. Keep them narrow and remove them when the codegen quirks are gone.
"""

from __future__ import annotations

import re
from typing import Any

_RST_INLINE_ROLE_RE = re.compile(r":(?:[a-z]+:)?[a-z]+:`([^`]+)`")
_WORDWRAP_HYPHEN_AFTER_RE = re.compile(r"(?<=[0-9A-Za-z_])- (?=[0-9A-Za-z_])")
_WORDWRAP_HYPHEN_BEFORE_RE = re.compile(r"(?<=[0-9A-Za-z_]) -(?=[0-9A-Za-z_])")


def _fix_hyphenation_wordwrap_spacing(s: str) -> str:
    """Remove spaces around hyphens introduced by line wrapping in generated ``__doc__`` text.

    This targets asymmetric wrap artifacts such as ``non- linear`` or
    ``GPU- Direct`` while leaving intentional ``a - b`` separators alone.
    """
    prev = None
    while prev != s:
        prev = s
        s = _WORDWRAP_HYPHEN_AFTER_RE.sub("-", s)
        s = _WORDWRAP_HYPHEN_BEFORE_RE.sub("-", s)
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
    # in cuda-bindings code generation.
    s = s.replace("\n:py:obj:`~.Interactions`", ' "Interactions ')
    # Drop a leading "~." or "." after removing the surrounding RST inline role.
    s = _RST_INLINE_ROLE_RE.sub(lambda m: re.sub(r"^~?\.", "", m.group(1)), s)
    # Strip simple bold emphasis markers.
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    # Strip simple italic emphasis markers.
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    # Collapse wrapped lines and repeated spaces.
    s = re.sub(r"\s+", " ", s).strip()
    s = _fix_hyphenation_wordwrap_spacing(s)
    return s


class DocstringBackedExplanations:
    """Expose enum-member ``__doc__`` text via ``dict.get``.

    Keeps the ``.get(int(error))`` lookup shape used by ``cuda_utils.pyx``.
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
