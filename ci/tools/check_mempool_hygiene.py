# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that tests do not create uncapped CUDA memory pools.

A pool created without ``max_size`` reserves an address-space window sized from
installed device memory rather than from what the test allocates, and the whole
cuda_core suite shares one process. Enough of those reservations exhaust the
address space, after which the rest of the session fails with
CUDA_ERROR_OUT_OF_MEMORY on a device with free physical memory.

Two rules are enforced:

* cuda_core: ``DeviceMemoryResource`` / ``PinnedMemoryResource`` and their
  options classes must be given ``max_size``.
* cuda.bindings: a pool properties object (``MemPoolProps_v1``,
  ``CUmemPoolProps``, ``cudaMemPoolProps``) assigned to a name must have
  ``max_size`` / ``maxSize`` set in the same function (or at module level, for
  module-level code). Graph alloc-node ``params.poolProps`` is a field, not a
  constructor call, so it is not checked; that pool cannot be capped anyway.

Only ``.py`` files are checked. ``ast`` cannot parse Cython, so ``.pyx`` tests
(e.g. ``cuda_bindings_*/tests/cython``) are out of scope and must be capped by
hand.

See cuda_core/tests/AGENTS.md and cuda_bindings_13/AGENTS.md for the rules this
enforces.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TREES = (
    ROOT / "cuda_core" / "tests",
    ROOT / "cuda_bindings_12" / "tests",
    ROOT / "cuda_bindings_13" / "tests",
)

# Managed pools cannot be right-sized: cuMemPoolCreate requires maxSize == 0 for
# managed pools, so ManagedMemoryResourceOptions has no max_size to set.
CAPPABLE_OPTIONS = frozenset({"DeviceMemoryResourceOptions", "PinnedMemoryResourceOptions"})
CAPPABLE_RESOURCES = frozenset({"DeviceMemoryResource", "PinnedMemoryResource"})

# Pool properties types of the cuda.bindings APIs; the size cap is a plain
# attribute, spelled ``max_size`` on the v2 types and ``maxSize`` on the others.
POOL_PROPS_TYPES = frozenset(
    {"MemPoolProps_v1", "CUmemPoolProps", "CUmemPoolProps_v1", "CUmemPoolProps_st", "cudaMemPoolProps"}
)
POOL_PROPS_CAP_ATTRIBUTES = frozenset({"max_size", "maxSize"})

OPT_OUT_MARKER = "uncapped-pool-ok"

RULE_RESOURCE = "resource"
RULE_PROPS = "props"


class Violation(NamedTuple):
    rule: str
    message: str


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


def _opted_out(lines: list[str], node: ast.AST) -> bool:
    """True if the call, or the line above it, carries the opt-out marker."""
    start = max(node.lineno - 2, 0)  # -1 for 0-based, -1 more for a preceding comment
    end = getattr(node, "end_lineno", node.lineno)
    return any(OPT_OUT_MARKER in line for line in lines[start:end])


def _walk_scope(scope: ast.AST) -> Iterator[ast.AST]:
    """Yield ``scope``'s descendants, yielding but not entering nested function/lambda/class bodies."""
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            pending.extend(ast.iter_child_nodes(node))


def _assign_targets(node: ast.AST) -> list[ast.expr]:
    """The assignment targets of ``node``, or ``[]`` if it is not an assignment."""
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return [node.target]
    return []


def _pool_props_violations(path: Path, scope: ast.AST, lines: list[str]) -> list[Violation]:
    """Report each ``var = PoolPropsType(...)`` with no ``var.max_size`` set in the same scope.

    Recurses so a cap only counts in the scope that built the props; a cap on a
    same-named variable elsewhere does not excuse it.
    """
    found: list[Violation] = []
    constructed: list[tuple[ast.Assign, str, str]] = []  # (node, variable, type name)
    capped: set[str] = set()
    for node in _walk_scope(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found += _pool_props_violations(path, node, lines)
            continue
        for target in _assign_targets(node):
            if (
                isinstance(target, ast.Attribute)
                and target.attr in POOL_PROPS_CAP_ATTRIBUTES
                and isinstance(target.value, ast.Name)
            ):
                capped.add(target.value.id)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.value, ast.Call):
            (target,) = node.targets
            type_name = _callee_name(node.value)
            if isinstance(target, ast.Name) and type_name in POOL_PROPS_TYPES:
                constructed.append((node, target.id, type_name))
    for node, variable, type_name in constructed:
        if variable not in capped and not _opted_out(lines, node):
            found.append(
                Violation(
                    RULE_PROPS,
                    f"{path.as_posix()}:{node.lineno}: {type_name} assigned to '{variable}' "
                    f"without {variable}.max_size / {variable}.maxSize",
                )
            )
    return found


def _resource_violations(path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node)
        if name in CAPPABLE_OPTIONS:
            uncapped = not _is_capped(node)
        elif name in CAPPABLE_RESOURCES:
            # The options may also be given as a dict literal.
            dicts = [arg for arg in [*node.args, *(kw.value for kw in node.keywords)] if isinstance(arg, ast.Dict)]
            uncapped = any(not _dict_is_capped(d) for d in dicts)
        else:
            continue
        if uncapped and not _opted_out(lines, node):
            found.append(Violation(RULE_RESOURCE, f"{path.as_posix()}:{node.lineno}: {name} without max_size"))
    return found


def _violations(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    return [*_resource_violations(path, tree, lines), *_pool_props_violations(path, tree, lines)]


def violations_in(path: Path) -> list[str]:
    """Return one message per uncapped pool construction in ``path``."""
    return [violation.message for violation in _violations(path)]


def _report(header: str, messages: list[str], hint: str) -> None:
    """Print ``header``, the bulleted ``messages``, and ``hint`` to stderr; a no-op if empty."""
    if not messages:
        return
    print(header, file=sys.stderr)
    for message in messages:
        print(f"  - {message}", file=sys.stderr)
    print(hint, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files to check. Defaults to every .py under "
        + ", ".join(tree.relative_to(ROOT).as_posix() for tree in DEFAULT_TREES)
        + ".",
    )
    args = parser.parse_args(argv)

    paths = args.paths or sorted(path for tree in DEFAULT_TREES for path in tree.rglob("*.py"))
    violations = sorted(v for path in paths if path.suffix == ".py" for v in _violations(path))
    if not violations:
        return 0

    _report(
        "error: memory pools created by tests must set max_size:",
        [v.message for v in violations if v.rule == RULE_RESOURCE],
        f"Use the suite-wide POOL_SIZE from cuda_core/tests/helpers/constants.py, or annotate a\n"
        f"deliberate exception with a '# {OPT_OUT_MARKER}: <reason>' comment.\n"
        f"See cuda_core/tests/AGENTS.md.",
    )
    _report(
        "error: pool properties created by tests must set max_size / maxSize:",
        [v.message for v in violations if v.rule == RULE_PROPS],
        f"Set max_size (v2 bindings) or maxSize (other bindings) on the props object you constructed,\n"
        f"using POOL_SIZE from cuda_python_test_helpers.memory, or annotate a deliberate exception\n"
        f"with a '# {OPT_OUT_MARKER}: <reason>' comment. See cuda_bindings_13/AGENTS.md.",
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
