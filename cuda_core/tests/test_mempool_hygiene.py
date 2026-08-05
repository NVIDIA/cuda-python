# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static guard against uncapped memory pools in the test suite.

A pool created without ``max_size`` reserves an address-space window sized from
device memory rather than from what the test allocates, and the whole suite
shares one process. Enough of those reservations exhaust the address space and
the rest of the session fails with ``CUDA_ERROR_OUT_OF_MEMORY`` on a device with
free physical memory (issue #2381). See AGENTS.md in this directory.

This check is static rather than runtime so that it also covers pools created
by tests that are skipped on the current platform.
"""

import ast
import pathlib

import pytest

TESTS_ROOT = pathlib.Path(__file__).parent

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


def _opted_out(lines: list[str], node: ast.AST) -> bool:
    """True if the call, or the line above it, carries the opt-out marker."""
    start = max(node.lineno - 2, 0)  # -1 for 0-based, -1 more for a preceding comment
    end = getattr(node, "end_lineno", node.lineno)
    return any(OPT_OUT_MARKER in line for line in lines[start:end])


def _violations_in(path: pathlib.Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    found = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
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
            found.append(f"{path.relative_to(TESTS_ROOT).as_posix()}:{node.lineno}: {name} without max_size")
    return found


@pytest.mark.agent_authored(model="claude-opus-5")
def test_no_uncapped_memory_pools():
    violations = sorted(v for path in TESTS_ROOT.rglob("*.py") for v in _violations_in(path))
    assert not violations, (
        "Memory pools created by tests must set max_size (use POOL_SIZE = 2097152).\n"
        f"Annotate a deliberate exception with a '# {OPT_OUT_MARKER}: <reason>' comment.\n"
        "See cuda_core/tests/AGENTS.md.\n" + "\n".join(violations)
    )
