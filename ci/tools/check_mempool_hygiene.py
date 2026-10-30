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
    """Yield the nodes of ``scope`` without entering nested functions, lambdas, or classes."""
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            pending.extend(ast.iter_child_nodes(node))


def _assigned_pool_props(node: ast.AST) -> tuple[str, str] | None:
    """Return ``(variable, type name)`` for ``variable = <PoolPropsType>(...)``."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.value, ast.Call)):
        return None
    target = node.targets[0]
    callee = _callee_name(node.value)
    if isinstance(target, ast.Name) and callee in POOL_PROPS_TYPES:
        return target.id, callee
    return None


def _capped_variables(scope: ast.AST) -> set[str]:
    """Names with ``name.max_size`` or ``name.maxSize`` assigned directly in ``scope``."""
    capped = set()
    for node in _walk_scope(scope):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        capped.update(
            target.value.id
            for target in targets
            if isinstance(target, ast.Attribute)
            and target.attr in POOL_PROPS_CAP_ATTRIBUTES
            and isinstance(target.value, ast.Name)
        )
    return capped


def _pool_props_violations(path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
    found = []
    scopes = [tree, *(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))]
    for scope in scopes:
        capped = _capped_variables(scope)
        for node in _walk_scope(scope):
            assigned = _assigned_pool_props(node)
            if assigned is None:
                continue
            variable, type_name = assigned
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

    resource_messages = [v.message for v in violations if v.rule == RULE_RESOURCE]
    props_messages = [v.message for v in violations if v.rule == RULE_PROPS]
    if resource_messages:
        print("error: memory pools created by tests must set max_size:", file=sys.stderr)
        for message in resource_messages:
            print(f"  - {message}", file=sys.stderr)
        print(
            f"Use the suite-wide POOL_SIZE from cuda_core/tests/helpers/constants.py, or annotate a\n"
            f"deliberate exception with a '# {OPT_OUT_MARKER}: <reason>' comment.\n"
            f"See cuda_core/tests/AGENTS.md.",
            file=sys.stderr,
        )
    if props_messages:
        print("error: pool properties created by tests must set max_size / maxSize:", file=sys.stderr)
        for message in props_messages:
            print(f"  - {message}", file=sys.stderr)
        print(
            f"Set max_size (v2 bindings) or maxSize (other bindings) on the props object you constructed,\n"
            f"using POOL_SIZE from cuda_python_test_helpers.memory, or annotate a deliberate exception\n"
            f"with a '# {OPT_OUT_MARKER}: <reason>' comment. See cuda_bindings_13/AGENTS.md.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
