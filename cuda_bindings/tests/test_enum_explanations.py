# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import importlib
import importlib.metadata
import re
import textwrap

import pytest

from cuda.bindings import driver, runtime

_EXPLANATION_MODULES = [
    ("driver_cu_result_explanations", "DRIVER_CU_RESULT_EXPLANATIONS", driver.CUresult),
    ("runtime_cuda_error_explanations", "RUNTIME_CUDA_ERROR_EXPLANATIONS", runtime.cudaError_t),
]

# Explanation dicts are maintained for the same toolkit as cuda-bindings; enum members
# carry docstrings from code generation (reportedly aligned since cuda-bindings 13.2.0).
_MIN_BINDING_VERSION_FOR_DOCSTRING_COMPARE = (13, 2)


def _get_binding_version():
    try:
        major_minor = importlib.metadata.version("cuda-bindings").split(".")[:2]
    except importlib.metadata.PackageNotFoundError:
        major_minor = importlib.metadata.version("cuda-python").split(".")[:2]
    return tuple(int(v) for v in major_minor)


def _explanation_text_from_dict_value(value):
    """Flatten a dict entry to a single str (entries are str or tuple of str fragments)."""
    if isinstance(value, tuple):
        return "".join(value)
    return value


def clean_enum_member_docstring(doc: str | None) -> str | None:
    """Turn a FastEnum member ``__doc__`` into plain text for display or fallback logic.

    Always: collapse all whitespace (including newlines) to single spaces and strip ends.

    Best-effort: remove common Sphinx/reST inline markup seen in generated CUDA docs,
    e.g. ``:py:obj:`~.cudaGetLastError()` `` -> ``cudaGetLastError()`` (relative ``~.`` is
    dropped). Does not aim for perfect reST parsing—only patterns that appear on these
    enums in practice.

    Returns ``None`` if ``doc`` is ``None``; otherwise returns a non-empty or empty str.
    """
    if doc is None:
        return None
    s = doc
    # Sphinx roles with a single backtick-delimited target (most common on these enums).
    # Strip the role and keep the inner text; drop leading ~. used for same-module refs.
    s = re.sub(
        r":(?:py:)?(?:obj|func|meth|class|mod|data|const|exc):`([^`]+)`",
        lambda m: re.sub(r"^~?\.", "", m.group(1)),
        s,
    )
    # Inline emphasis / strong (rare in error blurbs)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    # Collapse whitespace (newlines -> spaces) and trim
    s = re.sub(r"\s+", " ", s).strip()
    return s


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\nb  c", "a b c"),
        ("  x  \n ", "x"),
        (
            "see\n:py:obj:`~.cuInit()` or :py:obj:`cuCtxDestroy()`",
            "see cuInit() or cuCtxDestroy()",
        ),
        (
            "x :py:func:`~.cudaMalloc()` y",
            "x cudaMalloc() y",
        ),
        ("**Note:** text", "Note: text"),
        ("[Deprecated]\n", "[Deprecated]"),
    ],
)
def test_clean_enum_member_docstring_examples(raw, expected):
    assert clean_enum_member_docstring(raw) == expected


def test_clean_enum_member_docstring_none_input():
    assert clean_enum_member_docstring(None) is None


@pytest.mark.xfail(
    reason=(
        "Enum member __doc__ is not byte-identical to explanation dicts in current "
        "releases (Sphinx/RST and line breaks in __doc__ vs ::-style refs in dicts; "
        "some deprecated codes use a short [Deprecated] docstring). Remove xfail when "
        "dicts and generated docstrings share one source of truth."
    ),
    strict=False,
)
@pytest.mark.parametrize("module_name,dict_name,enum_type", _EXPLANATION_MODULES)
def test_explanations_dict_matches_enum_member_docstrings(module_name, dict_name, enum_type):
    """Each explanation dict value should match the corresponding enum member's __doc__.

    cuda-bindings 13.2+ attaches per-member documentation on driver ``CUresult`` and
    runtime ``cudaError_t``; this test checks it against the hand-maintained dicts.

    If this fails, differences may include whitespace, line breaks, Sphinx/RST markup
    in ``__doc__`` vs raw ``::symbol()`` text in the dicts—normalizing whitespace is
    a possible follow-up.

    Marked xfail while dict text and generated ``__doc__`` differ; run
    ``pytest --runxfail`` on this test to print the full mismatch report.
    """
    if _get_binding_version() < _MIN_BINDING_VERSION_FOR_DOCSTRING_COMPARE:
        pytest.skip(
            "Enum __doc__ vs explanation dict compare is only run for "
            f"cuda-bindings >= {_MIN_BINDING_VERSION_FOR_DOCSTRING_COMPARE[0]}.{_MIN_BINDING_VERSION_FOR_DOCSTRING_COMPARE[1]}"
        )

    mod = importlib.import_module(f"cuda.bindings._utils.{module_name}")
    expl_dict = getattr(mod, dict_name)

    mismatches = []
    for error in enum_type:
        code = int(error)
        assert code in expl_dict
        expected = _explanation_text_from_dict_value(expl_dict[code])
        actual = error.__doc__
        if actual is None:
            continue
        if expected != actual:
            mismatches.append((error, expected, actual))

    if not mismatches:
        return

    lines = [
        f"{len(mismatches)} enum member(s) where dict text != __doc__ (strict equality):",
    ]
    for error, expected, actual in mismatches[:15]:
        lines.append(f"  {error!r}")
        lines.append("    dict:")
        lines.extend("    | " + ln for ln in textwrap.wrap(repr(expected), width=100) or [""])
        lines.append("    __doc__:")
        lines.extend("    | " + ln for ln in textwrap.wrap(repr(actual), width=100) or [""])
    if len(mismatches) > 15:
        lines.append(f"  ... and {len(mismatches) - 15} more")
    pytest.fail("\n".join(lines))


@pytest.mark.parametrize("module_name,dict_name,enum_type", _EXPLANATION_MODULES)
def test_explanations_health(module_name, dict_name, enum_type):
    mod = importlib.import_module(f"cuda.bindings._utils.{module_name}")
    expl_dict = getattr(mod, dict_name)

    known_codes = set()
    for error in enum_type:
        code = int(error)
        assert code in expl_dict
        known_codes.add(code)

    if _get_binding_version() >= (13, 0):
        extra_expl = sorted(set(expl_dict.keys()) - known_codes)
        assert not extra_expl
