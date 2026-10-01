# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.core._utils.enum_explanations_helpers import (
    DocstringBackedExplanations,
    clean_enum_member_docstring,
)


class _FakeEnumMember:
    def __init__(self, doc):
        self.__doc__ = doc


class _FakeEnumType:
    def __init__(self, members):
        self._members = members

    def __call__(self, code):
        try:
            return self._members[code]
        except KeyError as e:
            raise ValueError(code) from e


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("a\nb  c", "a b c", id="collapse_whitespace"),
        pytest.param("  x  \n ", "x", id="strip_padding"),
        pytest.param(
            "see\n:py:obj:`~.cuInit()` or :py:obj:`cuCtxDestroy()`",
            "see cuInit() or cuCtxDestroy()",
            id="rst_py_domain_role",
        ),
        pytest.param(
            "x :py:func:`~.cudaMalloc()` y",
            "x cudaMalloc() y",
            id="rst_py_role",
        ),
        pytest.param(
            "x :c:func:`cuLaunchKernel` y",
            "x cuLaunchKernel y",
            id="rst_non_py_domain_role",
        ),
        pytest.param("x :term:`device` y", "x device y", id="rst_role_without_domain"),
        pytest.param("**Note:** text", "Note: text", id="strip_bold"),
        pytest.param("*Note* text", "Note text", id="strip_italic"),
        pytest.param("[Deprecated]\n", "[Deprecated]", id="deprecated_line"),
        pytest.param("non- linear", "non-linear", id="hyphen_space_after"),
        pytest.param("word -word", "word-word", id="hyphen_space_before"),
        pytest.param("GPU- Direct", "GPU-Direct", id="hyphen_space_after_uppercase"),
        pytest.param("peer -GPU", "peer-GPU", id="hyphen_space_before_uppercase"),
        pytest.param("L2- cache", "L2-cache", id="hyphen_space_after_digit"),
        pytest.param(
            "Common causes are - a. bad access",
            "Common causes are - a. bad access",
            id="preserve_dash_separator",
        ),
        pytest.param(
            'Please see\n:py:obj:`~.Interactions`with the CUDA Driver API" for more information.',
            'Please see "Interactions with the CUDA Driver API" for more information.',
            id="codegen_broken_interactions_role",
        ),
    ],
)
def test_clean_enum_member_docstring_examples(raw, expected):
    assert clean_enum_member_docstring(raw) == expected


def test_clean_enum_member_docstring_none_input():
    assert clean_enum_member_docstring(None) is None


def test_docstring_backed_get_returns_default_for_non_enum_code():
    lut = DocstringBackedExplanations(_FakeEnumType({}))
    assert lut.get(-1) is None
    assert lut.get(-1, default="sentinel") == "sentinel"


def test_docstring_backed_get_returns_default_for_missing_docstring():
    lut = DocstringBackedExplanations(_FakeEnumType({7: _FakeEnumMember(None)}))
    assert lut.get(7) is None
    assert lut.get(7, default="sentinel") == "sentinel"
