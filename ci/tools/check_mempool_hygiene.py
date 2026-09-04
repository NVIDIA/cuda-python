# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that tests do not create uncapped CUDA memory pools.

A pool created without ``max_size`` reserves an address-space window sized from
installed device memory rather than from what the test allocates, and the whole
cuda_core suite shares one process. Enough of those reservations exhaust the
address space, after which the rest of the session fails with
CUDA_ERROR_OUT_OF_MEMORY on a device with free physical memory.

See cuda_core/tests/AGENTS.md for the rule this enforces.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TREE = ROOT / "cuda_core" / "tests"

# Managed pools cannot be right-sized: cuMemPoolCreate requires maxSize == 0 for
# managed pools, so ManagedMemoryResourceOptions has no max_size to set.
CAPPABLE_OPTIONS = frozenset({"DeviceMemoryResourceOptions", "PinnedMemoryResourceOptions"})
CAPPABLE_RESOURCES = frozenset({"DeviceMemoryResource", "PinnedMemoryResource"})

OPT_OUT_MARKER = "uncapped-pool-ok"


def _callee_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_capped(node: ast.Call) -> bool:
    # ``**kwargs`` (arg is None) may carry max_size; do not guess.
    return any(kw.arg is None or kw.arg == "max_size" for kw in node.keywords)


def _dict_is_capped(node: ast.Dict) -> bool:
    for key in node.keys:
        if key is None:  # ``**other`` inside the literal
            return True
        if isinstance(key, ast.Constant) and key.value == "max_size":
            return True
    return False


def _iter_calls(tree: ast.AST):
    """Yield ``(call, statements)`` for every call in ``tree``.

    ``statements`` is the chain of ``ast.stmt`` ancestors, outermost first, so
    ``statements[-1]`` is the statement the call belongs to. The chain is what
    bounds an opt-out marker: a marker annotates the statement it sits in (or a
    statement containing it), not whatever the next line happens to construct.
    """
    stack: list[tuple[ast.AST, tuple[ast.stmt, ...]]] = [(tree, ())]
    while stack:
        node, stmts = stack.pop()
        if isinstance(node, ast.stmt):
            stmts = (*stmts, node)
        if isinstance(node, ast.Call):
            yield node, stmts
        stack.extend((child, stmts) for child in ast.iter_child_nodes(node))


def _opted_out(lines: list[str], node: ast.Call, stmts: tuple[ast.stmt, ...]) -> bool:
    """True if an opt-out marker annotates ``node``.

    A marker counts when it is on a line belonging to the call's own statement
    (including continuation lines of a multi-line construction), on the header
    line of a compound statement containing the call, or on a dedicated comment
    line immediately above the call's statement.

    A marker anywhere else does not count. Previously the whole line above was
    accepted unconditionally, so a trailing marker annotating the *previous*
    statement silently exempted the next one, and a marker appearing inside an
    unrelated string literal exempted whatever followed it.
    """
    nearest = stmts[-1] if stmts else node
    first = nearest.lineno - 1  # 0-based index of the statement's first line
    end = getattr(node, "end_lineno", node.lineno)
    if any(OPT_OUT_MARKER in line for line in lines[first:end]):
        return True
    # Headers of the compound statements containing the call ("with", "for",
    # "def", ...): a marker there annotates a block this call is part of.
    if any(OPT_OUT_MARKER in lines[s.lineno - 1] for s in stmts[:-1]):
        return True
    if first == 0:
        return False
    above = lines[first - 1].strip()
    return above.startswith("#") and OPT_OUT_MARKER in above


def violations_in(path: Path) -> list[str]:
    """Return one message per uncapped pool construction in ``path``."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    found = []
    for node, stmts in _iter_calls(ast.parse(source, filename=str(path))):
        name = _callee_name(node)
        if name in CAPPABLE_OPTIONS:
            uncapped = not _is_capped(node)
        elif name in CAPPABLE_RESOURCES:
            # The options may also be given as a dict literal.
            dicts = [arg for arg in [*node.args, *(kw.value for kw in node.keywords)] if isinstance(arg, ast.Dict)]
            uncapped = any(not _dict_is_capped(d) for d in dicts)
        else:
            continue
        if uncapped and not _opted_out(lines, node, stmts):
            found.append(f"{path.as_posix()}:{node.lineno}: {name} without max_size")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"Files to check. Defaults to every .py under {DEFAULT_TREE.relative_to(ROOT).as_posix()}.",
    )
    args = parser.parse_args(argv)

    paths = args.paths or sorted(DEFAULT_TREE.rglob("*.py"))
    violations = sorted(v for path in paths if path.suffix == ".py" for v in violations_in(path))
    if not violations:
        return 0

    print("error: memory pools created by tests must set max_size:", file=sys.stderr)
    for violation in violations:
        print(f"  - {violation}", file=sys.stderr)
    print(
        f"Use the suite-wide POOL_SIZE from cuda_core/tests/helpers/constants.py, or annotate a\n"
        f"deliberate exception with a '# {OPT_OUT_MARKER}: <reason>' comment.\n"
        f"See cuda_core/tests/AGENTS.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
