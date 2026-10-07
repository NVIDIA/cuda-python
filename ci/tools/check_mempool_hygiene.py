# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the memory-pool hygiene of the cuda_core tests.

A memory resource constructed with an options object creates a pool that the
test owns; one constructed without options wraps the device's current pool and
owns nothing. The whole cuda_core suite shares one process, and an owned pool
keeps its address-space reservation until the pool is destroyed and the
stream-ordered frees of its allocations retire. Three rules keep that bounded:

1. max_size: a Device or Pinned pool sets ``max_size``. A pool without it
   reserves an address-space window sized from installed device memory rather
   than from what the test allocates, and enough of those reservations exhaust
   the address space, after which the rest of the session fails with
   CUDA_ERROR_OUT_OF_MEMORY on a device with free physical memory. Opt out with
   ``# uncapped-pool-ok: <reason>``.
2. close: a Device or Pinned pool is bound to a name and closed in the function
   that creates it, by ``.close()`` on that name or by a ``for`` loop that
   closes every element of the list the name holds. Memory resources are not
   context managers. Opt out with ``# unclosed-pool-ok: <reason>`` when the
   close happens elsewhere.
3. owns_pool: a test that creates a pool of any kind, directly or through a
   function in the same module, carries ``@pytest.mark.owns_pool`` on the
   function, on its class, or in the module's ``pytestmark``; a fixture that
   creates a pool calls ``request.node.add_marker("owns_pool")``. The marker
   makes the ``init_cuda`` teardown run ``gc.collect()`` before it drains the
   context. Calls are followed by bare name and through ``self``; a pool created
   through a function passed in as a parameter is not seen.

See cuda_core/tests/AGENTS.md for the rules this enforces.
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

# Every options class. Passing one to a resource creates an owned pool.
POOL_OPTIONS = CAPPABLE_OPTIONS | {"ManagedMemoryResourceOptions", "VirtualMemoryResourceOptions"}
# Resource constructors and the number of leading positional arguments that are
# not options (the device). More positional arguments, or any keyword, pass options.
POOL_RESOURCES = {
    "DeviceMemoryResource": 1,
    "PinnedMemoryResource": 0,
    "ManagedMemoryResource": 0,
    "VirtualMemoryResource": 1,
}
# Factories in cuda_core/tests/helpers/memory.py; they create a pool when given options.
POOL_HELPERS = frozenset({"create_managed_memory_resource_or_skip", "create_pinned_memory_resource_or_xfail"})
HELPER_NON_OPTION_KEYWORDS = frozenset({"xfail_device"})
CAPPABLE_HELPERS = frozenset({"create_pinned_memory_resource_or_xfail"})

UNCAPPED_MARKER = "uncapped-pool-ok"
UNCLOSED_MARKER = "unclosed-pool-ok"
OWNS_POOL = "owns_pool"


def _callee_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _decorator_name(node: ast.AST) -> str:
    """The last name of a decorator or mark expression: ``pytest.mark.owns_pool`` -> ``owns_pool``."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
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


def _passes_options(node: ast.Call) -> bool:
    """True if a resource constructor or pool helper call passes options, so it creates a pool."""
    name = _callee_name(node)
    if name in POOL_RESOURCES:
        return len(node.args) > POOL_RESOURCES[name] or bool(node.keywords)
    if name in POOL_HELPERS:
        return bool(node.args) or any(kw.arg not in HELPER_NON_OPTION_KEYWORDS for kw in node.keywords)
    return False


def _creates_pool(node: ast.Call) -> bool:
    """True if the call creates an owned pool of any kind (rule 3)."""
    return _callee_name(node) in POOL_OPTIONS or _passes_options(node)


def _creates_cappable_pool(node: ast.Call) -> bool:
    """True if the call creates a Device or Pinned pool (rule 2)."""
    return _callee_name(node) in CAPPABLE_RESOURCES | CAPPABLE_HELPERS and _passes_options(node)


def _opted_out(lines: list[str], node: ast.AST, markers: tuple[str, ...]) -> bool:
    """True if the call, or the line above it, carries one of the opt-out markers."""
    start = max(node.lineno - 2, 0)  # -1 for 0-based, -1 more for a preceding comment
    end = getattr(node, "end_lineno", node.lineno)
    return any(marker in line for marker in markers for line in lines[start:end])


def _calls(node: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _uncapped_pools(tree: ast.Module, lines: list[str], path: Path) -> list[str]:
    found = []
    for node in _calls(tree):
        name = _callee_name(node)
        if name in CAPPABLE_OPTIONS:
            uncapped = not _is_capped(node)
        elif name in CAPPABLE_RESOURCES:
            # The options may also be given as a dict literal.
            dicts = [arg for arg in [*node.args, *(kw.value for kw in node.keywords)] if isinstance(arg, ast.Dict)]
            uncapped = any(not _dict_is_capped(d) for d in dicts)
        else:
            continue
        if uncapped and not _opted_out(lines, node, (UNCAPPED_MARKER,)):
            found.append(f"{path.as_posix()}:{node.lineno}: {name} without max_size")
    return found


def _names_closed_in(node: ast.AST) -> set[str]:
    """Names ``x`` with an ``x.close(...)`` call inside ``node``."""
    return {
        call.func.value.id
        for call in _calls(node)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "close" and isinstance(call.func.value, ast.Name)
    }


def _unclosed_pools(func: ast.FunctionDef, lines: list[str], path: Path) -> list[str]:
    closed_names = _names_closed_in(func)
    for node in ast.walk(func):
        # ``for mr in mrs: mr.close()`` closes every pool bound to ``mrs``.
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Name)
            and node.target.id in _names_closed_in(node)
        ):
            closed_names.add(node.iter.id)
    bound_to = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            # Every call in the value, so a pool built inside a comprehension is bound to the list's name.
            for call in _calls(node.value):
                bound_to[id(call)] = node.targets[0].id
    found = []
    for call in _calls(func):
        if not _creates_cappable_pool(call) or bound_to.get(id(call)) in closed_names:
            continue
        # A call marked uncapped-pool-ok raises before any pool exists, so there is nothing to close.
        if _opted_out(lines, call, (UNCAPPED_MARKER, UNCLOSED_MARKER)):
            continue
        found.append(f"{path.as_posix()}:{call.lineno}: {_callee_name(call)} in {func.name} is not closed")
    return found


def _functions(tree: ast.Module):
    """Yield ``(function, enclosing classes)`` for module-level functions and methods."""

    def visit(body, classes):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield node, classes
            elif isinstance(node, ast.ClassDef):
                yield from visit(node.body, [*classes, node])

    yield from visit(tree.body, [])


def _callee_names(func: ast.FunctionDef) -> set[str]:
    """Names of the functions ``func`` calls by bare name or through ``self``/``cls``."""
    names = set()
    for call in _calls(func):
        target = call.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id in ("self", "cls")
        ):
            names.add(target.attr)
    return names


def _pool_creators(functions) -> set[int]:
    """Ids of the functions that create a pool, directly or through another function in the module."""
    creators = {id(func) for func, _ in functions if any(_creates_pool(call) for call in _calls(func))}
    callees = {id(func): _callee_names(func) for func, _ in functions}
    changed = True
    while changed:
        creator_names = {func.name for func, _ in functions if id(func) in creators}
        changed = False
        for func, _ in functions:
            if id(func) not in creators and callees[id(func)] & creator_names:
                creators.add(id(func))
                changed = True
    return creators


def _module_marks(tree: ast.Module) -> set[str]:
    marks = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
            marks.update(_decorator_name(value) for value in values)
    return marks


def _has_decorator(node: ast.AST, name: str) -> bool:
    return any(_decorator_name(dec) == name for dec in node.decorator_list)


def _adds_owns_pool_marker(func: ast.FunctionDef) -> bool:
    return any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_marker"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == OWNS_POOL
        for call in _calls(func)
    )


def _unmarked_pool_owners(tree: ast.Module, functions, path: Path) -> list[str]:
    creators = _pool_creators(functions)
    module_marked = OWNS_POOL in _module_marks(tree)
    found = []
    for func, classes in functions:
        if id(func) not in creators:
            continue
        qualname = ".".join([*(cls.name for cls in classes), func.name])
        marked = module_marked or any(_has_decorator(node, OWNS_POOL) for node in (func, *classes))
        if _has_decorator(func, "fixture") and not _adds_owns_pool_marker(func):
            found.append(
                f"{path.as_posix()}:{func.lineno}: fixture {qualname} creates a memory pool "
                f'but does not call request.node.add_marker("{OWNS_POOL}")'
            )
        elif func.name.startswith("test_") and not marked:
            found.append(
                f"{path.as_posix()}:{func.lineno}: {qualname} creates a memory pool but is not marked {OWNS_POOL}"
            )
    return found


def violations_in(path: Path) -> list[str]:
    """Return one message per hygiene violation in ``path``."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    functions = list(_functions(tree))
    found = _uncapped_pools(tree, lines, path)
    for func, _ in functions:
        found.extend(_unclosed_pools(func, lines, path))
    found.extend(_unmarked_pool_owners(tree, functions, path))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    print("error: memory-pool hygiene violations in tests:", file=sys.stderr)
    for violation in violations:
        print(f"  - {violation}", file=sys.stderr)
    print(
        f"Pools created by tests must set max_size (POOL_SIZE from cuda_core/tests/helpers/constants.py)\n"
        f"and must be closed in the function that creates them (.close() on the name, or a loop over the list).\n"
        f"Annotate a deliberate exception with a '# {UNCAPPED_MARKER}: <reason>' or\n"
        f"'# {UNCLOSED_MARKER}: <reason>' comment. A test that creates a pool carries\n"
        f'@pytest.mark.{OWNS_POOL}; a fixture that creates one calls request.node.add_marker("{OWNS_POOL}").\n'
        f"See cuda_core/tests/AGENTS.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
