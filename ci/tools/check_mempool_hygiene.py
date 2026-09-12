# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that tests do not create uncapped CUDA memory pools.

A pool created without ``max_size`` -- or with ``max_size=0``, which is the
same request expressed explicitly -- reserves an address-space window sized from
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


MISSING_CAP = "without max_size"
ZERO_CAP = "with max_size=0, which is the uncapped default"


def _is_zero_literal(node: ast.expr) -> bool:
    """True for a literal ``0``.

    ``CUmemPoolProps.maxSize == 0`` asks the driver for its system-dependent
    default, which is the uncapped pool this check exists to prevent -- see
    "When set to 0, defaults to a system-dependent value" in the
    ``DeviceMemoryResourceOptions`` / ``PinnedMemoryResourceOptions``
    docstrings. Only literals are inspected: a named constant may well be the
    suite-wide POOL_SIZE, and this checker does not guess.
    """
    return isinstance(node, ast.Constant) and node.value == 0


def _cap_problem(node: ast.Call) -> str | None:
    """Describe why ``node``'s options leave the pool uncapped, else ``None``."""
    # ``**kwargs`` (arg is None) may carry max_size; do not guess.
    if any(kw.arg is None for kw in node.keywords):
        return None
    for kw in node.keywords:
        if kw.arg == "max_size":
            return ZERO_CAP if _is_zero_literal(kw.value) else None
    return MISSING_CAP


def _dict_cap_problem(node: ast.Dict) -> str | None:
    """``_cap_problem`` for options given as a dict literal."""
    if any(key is None for key in node.keys):  # ``**other`` inside the literal
        return None
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == "max_size":
            return ZERO_CAP if _is_zero_literal(value) else None
    return MISSING_CAP


def _opted_out(lines: list[str], node: ast.AST) -> bool:
    """True if the call, or the line above it, carries the opt-out marker."""
    start = max(node.lineno - 2, 0)  # -1 for 0-based, -1 more for a preceding comment
    end = getattr(node, "end_lineno", node.lineno)
    return any(OPT_OUT_MARKER in line for line in lines[start:end])


def violations_in(path: Path) -> list[str]:
    """Return one message per uncapped pool construction in ``path``."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    found = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node)
        if name in CAPPABLE_OPTIONS:
            problem = _cap_problem(node)
        elif name in CAPPABLE_RESOURCES:
            # The options may also be given as a dict literal.
            dicts = [arg for arg in [*node.args, *(kw.value for kw in node.keywords)] if isinstance(arg, ast.Dict)]
            problems = [p for p in (_dict_cap_problem(d) for d in dicts) if p is not None]
            problem = problems[0] if problems else None
        else:
            continue
        if problem is not None and not _opted_out(lines, node):
            found.append(f"{path.as_posix()}:{node.lineno}: {name} {problem}")
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

    print("error: memory pools created by tests must set max_size to a non-zero value:", file=sys.stderr)
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
