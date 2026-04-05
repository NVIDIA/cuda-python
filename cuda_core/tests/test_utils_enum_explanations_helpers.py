# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

from cuda.core._utils.enum_explanations_helpers import (
    DocstringBackedExplanations,
    _strip_doxygen_double_colon_prefixes,
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
            id="sphinx_py_obj_roles",
        ),
        pytest.param(
            "x :py:func:`~.cudaMalloc()` y",
            "x cudaMalloc() y",
            id="sphinx_py_func_role",
        ),
        pytest.param("**Note:** text", "Note: text", id="strip_bold"),
        pytest.param("*Note* text", "Note text", id="strip_italic"),
        pytest.param("[Deprecated]\n", "[Deprecated]", id="deprecated_line"),
        pytest.param("non- linear", "non-linear", id="hyphen_space_after"),
        pytest.param("word -word", "word-word", id="hyphen_space_before"),
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("see ::CUDA_SUCCESS", "see CUDA_SUCCESS", id="type_ref"),
        pytest.param("Foo::Bar unchanged", "Foo::Bar unchanged", id="cpp_scope_preserved"),
        pytest.param("::cuInit() and ::CUstream", "cuInit() and CUstream", id="multiple_prefixes"),
    ],
)
def test_strip_doxygen_double_colon_prefixes(raw, expected):
    assert _strip_doxygen_double_colon_prefixes(raw) == expected


def test_docstring_backed_get_returns_default_for_non_enum_code():
    lut = DocstringBackedExplanations(_FakeEnumType({}))
    assert lut.get(-1) is None
    assert lut.get(-1, default="sentinel") == "sentinel"


def test_docstring_backed_get_returns_default_for_missing_docstring():
    lut = DocstringBackedExplanations(_FakeEnumType({7: _FakeEnumMember(None)}))
    assert lut.get(7) is None
    assert lut.get(7, default="sentinel") == "sentinel"


def test_docstring_backed_get_returns_default_for_unknown_code():
    lut = DocstringBackedExplanations(_FakeEnumType({}))
    assert lut.get(99, default="sentinel") == "sentinel"


def test_docstring_backed_get_returns_default_for_missing_docstring_without_fallback():
    lut = DocstringBackedExplanations(_FakeEnumType({7: _FakeEnumMember(None)}))
    assert lut.get(7, default="sentinel") == "sentinel"


def test_get_best_available_explanations_uses_fallback_before_13_2(monkeypatch):
    import cuda.core._utils.enum_explanations_helpers as cleanup

    fallback = {7: "fallback text"}
    monkeypatch.setattr(cleanup, "_binding_version", lambda: (13, 1, 1))
    assert cleanup.get_best_available_explanations(_FakeEnumType({7: _FakeEnumMember("doc")}), fallback) is fallback


def test_get_best_available_explanations_prefers_docstrings_from_13_2(monkeypatch):
    import cuda.core._utils.enum_explanations_helpers as cleanup

    fallback = {7: "fallback text"}
    monkeypatch.setattr(cleanup, "_binding_version", lambda: (13, 2, 0))
    expl = cleanup.get_best_available_explanations(
        _FakeEnumType({7: _FakeEnumMember("clean me")}),
        fallback,
    )
    assert isinstance(expl, DocstringBackedExplanations)
    assert expl.get(7) == "clean me"


def test_driver_cu_result_explanations_get_matches_clean_docstring():
    pytest.importorskip("cuda.bindings")
    from cuda.bindings import driver
    from cuda.core._utils.driver_cu_result_explanations import DRIVER_CU_RESULT_EXPLANATIONS

    e = driver.CUresult.CUDA_SUCCESS
    code = int(e)
    assert DRIVER_CU_RESULT_EXPLANATIONS.get(code) == clean_enum_member_docstring(e.__doc__)
