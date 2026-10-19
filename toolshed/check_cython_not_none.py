# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject ``def`` parameters annotated with a bare cdef class but lacking ``not None``.

Cython lets ``None`` through for a parameter typed as an extension type unless
the parameter says ``not None``, and an annotation Cython cannot resolve to a
type constrains nothing at all. Either way ``None`` can reach C code and crash
instead of raising ``TypeError``. Write one of::

    def f(Buffer buf not None: Buffer)    # Cython knows Buffer
    def f(buf not None: Buffer)           # Cython does not know Buffer
    def f(buf: Buffer | None)             # None is a valid value
"""

import re
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_SEARCH_ROOTS = ("cuda_core/cuda", "cuda_bindings/cuda")

_CDEF_CLASS = re.compile(r"^\s*cdef\s+(?:public\s+|final\s+)*class\s+(\w+)", re.MULTILINE)
_DEF_START = re.compile(r"^[ \t]*(?:cp)?def\b[^\n(]*\(", re.MULTILINE)
_IDENT = r"[A-Za-z_]\w*"
# [Type ]name[ not None][: annotation][ = default]
_PARAM = re.compile(
    rf"""\A\s*
    (?P<typed>(?:{_IDENT}(?:\.{_IDENT})*\s+)?)
    (?P<name>\*{{0,2}}{_IDENT})
    (?P<notnone>\s+not\s+None)?
    (?:\s*:\s*(?P<annotation>.+?))?
    (?:\s*=\s*(?P<default>.+?))?
    \s*\Z""",
    re.VERBOSE | re.DOTALL,
)
_SINGLE_NAME = re.compile(rf"\A{_IDENT}(?:\.{_IDENT})*\Z")


class Violation(NamedTuple):
    offset: int  # start of the parameter text
    name_end: int  # where ``not None`` belongs
    name: str
    typed: bool  # already has a leading C-level type, e.g. ``Buffer buf``
    annotation: str
    message: str


def collect_cdef_classes(root=REPO_ROOT):
    names = set()
    for search_root in CLASS_SEARCH_ROOTS:
        for path in (root / search_root).rglob("*"):
            if path.suffix in (".pyx", ".pxd"):
                names.update(_CDEF_CLASS.findall(path.read_text(encoding="utf-8")))
    return names


def _blank(match):
    return re.sub(r"[^\n]", " ", match.group(0))


def _strip_comments(text):
    # Blank out docstrings and ``#`` comments (keeping offsets stable) so that
    # example code inside them is not mistaken for a signature.
    text = re.sub(r'(?s)(?:"""|\'\'\').*?(?:"""|\'\'\')', _blank, text)
    return re.sub(r"#[^\n]*", _blank, text)


def _split_signature(text, start):
    """Yield ``(offset, text)`` for each top-level parameter after ``start``."""
    depth = 0
    param_start = start
    pos = start
    while pos < len(text):
        char = text[pos]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                yield param_start, text[param_start:pos]
                return
            depth -= 1
        elif char == "," and depth == 0:
            yield param_start, text[param_start:pos]
            param_start = pos + 1
        pos += 1


def _includes_none(annotation):
    return re.search(r"\bNone\b", annotation) is not None


def _is_overload_stub(clean, def_start):
    preceding = clean[:def_start].rstrip().rsplit("\n", 1)[-1].strip()
    return preceding == "@overload"


def find_violations(text, cdef_classes):
    """Return the ``Violation`` records for one source."""
    clean = _strip_comments(text)
    violations = []
    for match in _DEF_START.finditer(clean):
        if _is_overload_stub(clean, match.start()):
            continue
        for offset, param in _split_signature(clean, match.end()):
            parsed = _PARAM.match(param)
            if parsed is None or parsed.group("annotation") is None:
                continue
            annotation = parsed.group("annotation").strip()
            if not _SINGLE_NAME.match(annotation):
                continue
            if annotation.rsplit(".", 1)[-1] not in cdef_classes:
                continue
            name = parsed.group("name")
            # ``self``/``cls`` are never None, and ``not None`` is not allowed
            # on ``*args`` / ``**kwargs``.
            if name in ("self", "cls") or name.startswith("*"):
                continue
            if parsed.group("default") is not None and parsed.group("default").strip() == "None":
                message = f"parameter {name!r} defaults to None; annotate it as '{annotation} | None'"
            elif parsed.group("notnone") is None:
                message = f"parameter {name!r} is annotated with cdef class {annotation!r} but lacks 'not None'"
            else:
                continue
            violations.append(
                Violation(
                    offset=offset + (len(param) - len(param.lstrip())),
                    name_end=offset + parsed.end("name"),
                    name=name,
                    typed=bool(parsed.group("typed").strip()),
                    annotation=annotation,
                    message=message,
                )
            )
    return violations


def main(argv):
    cdef_classes = collect_cdef_classes()
    failed = False
    for arg in argv:
        path = Path(arg)
        if path.suffix != ".pyx":
            continue
        text = path.read_text(encoding="utf-8")
        for violation in find_violations(text, cdef_classes):
            lineno = text.count("\n", 0, violation.offset) + 1
            print(f"{arg}:{lineno}: {violation.message}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
