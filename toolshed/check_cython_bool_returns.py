# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject Cython declarations that return ``bint`` for a C++ function returning ``bool``.

A consumer module calls a cimported function through a pointer of the type in
the ``.pxd``, not the type in the C++ header. ``bint`` is ``int``, but MSVC
sets only the low byte of the return register for a ``bool``, so on Windows
the caller can read ``false`` as ``true``. Declare such returns as::

    from libcpp cimport bool as cpp_bool

    cdef cpp_bool f(...) noexcept nogil
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "cuda_core" / "cuda" / "core"
CXX_SUFFIXES = (".hpp", ".h", ".cpp")
CYTHON_SUFFIXES = (".pyx", ".pxd", ".pxi")

# [inline|static|...] bool name(     -- the return type may sit on its own line
_CXX_BOOL_FUNCTION = re.compile(
    r"""(?:^|[;{}]|\]\])\s*
    (?:(?:inline|static|constexpr|virtual|extern)\s+)*
    bool\s+(?P<name>[A-Za-z_]\w*)\s*\(""",
    re.MULTILINE | re.VERBOSE,
)
# [cdef ][inline ]bint name ["c::name"] (
_CYTHON_BINT_FUNCTION = re.compile(
    r"""^[ \t]*(?:cdef\s+)?(?:inline\s+)?
    bint\s+(?P<name>[A-Za-z_]\w*)\s*
    (?:"(?P<cname>[^"]*)"\s*)?\(""",
    re.MULTILINE | re.VERBOSE,
)


def _blank(match):
    return re.sub(r"[^\n]", " ", match.group(0))


def _strip_cxx_comments(text):
    text = re.sub(r"(?s)/\*.*?\*/", _blank, text)
    return re.sub(r"//[^\n]*", _blank, text)


def _strip_cython_comments(text):
    # Keep the quoted C names in extern declarations; only drop ``#`` comments
    # and docstrings, whose examples could otherwise look like declarations.
    text = re.sub(r'(?s)(?:"""|\'\'\').*?(?:"""|\'\'\')', _blank, text)
    return re.sub(r"#[^\n]*", _blank, text)


def collect_cxx_bool_functions(root=SOURCE_ROOT):
    names = set()
    for path in root.rglob("*"):
        if path.suffix in CXX_SUFFIXES:
            text = _strip_cxx_comments(path.read_text(encoding="utf-8"))
            names.update(m.group("name") for m in _CXX_BOOL_FUNCTION.finditer(text))
    return names


def find_violations(text, bool_functions):
    """Yield ``(offset, message)`` for each ``bint`` declaration of a C++ ``bool`` function."""
    for match in _CYTHON_BINT_FUNCTION.finditer(_strip_cython_comments(text)):
        name = match.group("name")
        cname = match.group("cname")
        target = cname.rsplit("::", 1)[-1] if cname else name
        if target in bool_functions:
            yield (
                match.start("name"),
                f"{name!r} returns C++ bool; declare its return type as cpp_bool, not bint",
            )


def main(root=SOURCE_ROOT):
    bool_functions = collect_cxx_bool_functions(root)
    failed = False
    for path in sorted(root.rglob("*")):
        if path.suffix not in CYTHON_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        for offset, message in find_violations(text, bool_functions):
            lineno = text.count("\n", 0, offset) + 1
            print(f"{path.relative_to(REPO_ROOT)}:{lineno}: {message}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
