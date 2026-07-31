# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guards for cuda.bindings Cython warning cleanliness (#2450)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_BINDINGS_ROOT = Path(__file__).resolve().parents[1]
_CUDA_BINDINGS = _BINDINGS_ROOT / "cuda" / "bindings"

# cpdef returning a Python object must not carry an exception clause; Cython
# warns (and with warning_errors, fails) that the clause is ignored.
_CPDEF_OBJECT_EXCEPT_RE = re.compile(
    r"^\s*cpdef\s+object\s+\w+\s*\([^)]*\)\s+except\b",
    re.MULTILINE,
)
_MODULE_GET_ATTRIBUTES_EXCEPT_RE = re.compile(
    r"^\s*cpdef\s+module_get_attributes\s*\([^)]*\)\s+except\b",
    re.MULTILINE,
)


@pytest.mark.agent_authored(model="grok-4.5")
def test_build_hooks_enable_cython_warning_errors():
    """Source builds must treat Cython warnings as errors."""
    source = (_BINDINGS_ROOT / "build_hooks.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = {
        node.targets[0].attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
    }
    assert "warning_errors" in assigned, (
        "cuda_bindings/build_hooks.py must set Cython Options.warning_errors = True"
    )


@pytest.mark.agent_authored(model="grok-4.5")
def test_windll_pxd_declares_load_library_flag_as_const():
    """Assignment in a ``.pxd`` is not executed; declare the Win32 flag as const."""
    text = (_CUDA_BINDINGS / "_lib" / "windll.pxd").read_text(encoding="utf-8")
    assert "LOAD_LIBRARY_SEARCH_SYSTEM32" in text
    assert re.search(
        r"^\s*const\s+DWORD\s+LOAD_LIBRARY_SEARCH_SYSTEM32\s*$",
        text,
        re.MULTILINE,
    ), "LOAD_LIBRARY_SEARCH_SYSTEM32 must be declared as const DWORD (no assignment)"
    assert re.search(
        r"LOAD_LIBRARY_SEARCH_SYSTEM32\s*=",
        text,
    ) is None


@pytest.mark.agent_authored(model="grok-4.5")
@pytest.mark.parametrize("filename", ["cudla.pxd", "cudla.pyx"])
def test_cudla_cpdef_python_returns_have_no_except_clause(filename):
    """Regression for cybind emitting ``except *`` on Python-returning cpdefs."""
    text = (_CUDA_BINDINGS / filename).read_text(encoding="utf-8")
    assert _CPDEF_OBJECT_EXCEPT_RE.search(text) is None, (
        f"{filename} has cpdef object ... except ..., which Cython warns about"
    )
    # Declared without an explicit object return type, but returns Python values.
    assert _MODULE_GET_ATTRIBUTES_EXCEPT_RE.search(text) is None, (
        f"{filename}: cpdef module_get_attributes(...) except ... is invalid for Python returns"
    )
