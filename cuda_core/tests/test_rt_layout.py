# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Layout rules of cuda/core/_cpp/rt/, the C++ behind cuda.core._rt.

Source-tree properties only: no compiler, no GPU. Run with --noconftest, since
conftest.py imports the compiled package:

    pytest tests/test_rt_layout.py -v --noconftest

Why the rules exist: _rt.pxd names handles.hpp, and Cython compiles every
cimporting extension against a copy of that header placed in its build
directory, next to copies of the extension's `depends`. So handles.hpp may pull
in only what thirty-odd consumer extensions can safely compile: types,
templates and inline functions, through file-relative includes. Everything with
storage or a body lives behind rt.hpp, which only _rt.pyx names.
"""

import re
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parent.parent / "cuda" / "core"
RT = CORE / "_cpp" / "rt"
HEADERS = sorted(RT.glob("*.hpp"))
SOURCES = sorted(RT.glob("*.cpp"))
UMBRELLAS = {"rt.hpp", "handles.hpp"}
PYTHON_TOKEN = re.compile(r"\bPy[A-Z_]\w*|\bPyObject\b|Python\.h")
QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def read(path):
    return path.read_text(encoding="utf-8")


def quoted_includes(path):
    return QUOTED_INCLUDE.findall(read(path))


def include_closure(path):
    seen = []
    todo = [path]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.append(current)
        todo.extend(current.parent / name for name in quoted_includes(current))
    return sorted(seen)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_python_h_is_spelled_only_in_py_hpp():
    spellers = [p.name for p in HEADERS + SOURCES if "<Python.h>" in read(p)]
    assert spellers == ["py.hpp"]


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_neutral_files_do_not_name_python():
    """Every header except the seam and the umbrellas, and every source that does
    not include py.hpp, compiles without a Python include path."""
    neutral_sources = [p for p in SOURCES if "py.hpp" not in quoted_includes(p)]
    assert {p.name for p in neutral_sources} == {"driver_api.cpp", "error.cpp"}
    neutral = [p for p in HEADERS if p.name not in UMBRELLAS | {"py.hpp"}] + neutral_sources
    offenders = {p.name: PYTHON_TOKEN.findall(read(p)) for p in neutral}
    assert {name: hits for name, hits in offenders.items() if hits} == {}


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_in_tree_includes_are_bare_sibling_names_that_exist():
    for path in HEADERS + SOURCES:
        for name in quoted_includes(path):
            assert "/" not in name, f"{path.name} includes {name!r}; in-tree includes are bare sibling names"
            assert (RT / name).is_file(), f"{path.name} includes {name!r}, which does not exist"


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_umbrellas_are_named_only_by_their_cython_file():
    for path in HEADERS + SOURCES:
        assert UMBRELLAS.isdisjoint(quoted_includes(path)), f"{path.name} includes an umbrella"
    named = {}
    for path in sorted(list(CORE.rglob("*.pyx")) + list(CORE.rglob("*.pxd"))):
        for header in re.findall(r'cdef extern from "(_cpp/rt/[^"]+)"', read(path)):
            named.setdefault(header, set()).add(path.name)
    assert named == {"_cpp/rt/rt.hpp": {"_rt.pyx"}, "_cpp/rt/handles.hpp": {"_rt.pxd"}}


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_consumer_closure_is_types_and_the_python_seam():
    closure = {p.name for p in include_closure(RT / "handles.hpp")}
    assert closure == {"handles.hpp", "py.hpp", "types.hpp", "versions.hpp"}
    # Consumers are RTLD_LOCAL extensions that cannot link to _rt: nothing with storage.
    for name in sorted(closure):
        text = read(RT / name)
        assert not re.search(r'^extern (?!"C")', text, re.M), f"{name} declares an extern variable"
        assert not re.search(r"^(static|thread_local)\b", text, re.M), f"{name} defines storage"


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_cuda_version_is_named_only_in_versions_hpp():
    """The C++ may branch on CUDA_CORE_BUILD_MAJOR only. A `#if CUDA_VERSION >= 130x0`
    fence compiled a feature out of source builds against an older header while the
    run-time checks never noticed (https://github.com/NVIDIA/cuda-python/issues/2783);
    versions.hpp checks the header once and is the only file allowed to name it."""
    cpp = CORE / "_cpp"
    files = sorted(p for p in cpp.rglob("*") if p.suffix in (".hpp", ".h", ".cpp"))
    assert len(files) > 20
    spellers = sorted(p.relative_to(cpp).as_posix() for p in files if re.search(r"\bCUDA_VERSION\b", read(p)))
    assert spellers == ["rt/versions.hpp"]
    assert "CUDA_CORE_BUILD_MAJOR" in read(RT / "versions.hpp")


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_driver_function_table_matches_the_cuda_bindings_loader():
    """driver_api.hpp lists each driver function with the CUDA version cuda-bindings
    requests it at; that number decides which functions every supported driver
    must provide. Check it against the loader cuda-bindings generates."""
    loader = CORE.parents[2] / "cuda_bindings" / "cuda" / "bindings" / "_internal" / "driver_linux.pyx"
    if not loader.is_file():
        pytest.skip("cuda-bindings source is not next to cuda_core")
    entries = re.findall(r"^\s*X\((cu\w+), (\d+)\)", read(RT / "driver_api.hpp"), re.M)
    assert len(entries) >= 60
    assert len({name for name, _ in entries}) == len(entries), "duplicate table entry"
    requested = {}
    for name, version in re.findall(r"cuGetProcAddress_v2\('(\w+)', <void \*\*>&__\w+, (\d+)", read(loader)):
        requested.setdefault(name, set()).add(int(version))
    mismatched = {
        name: (int(introduced), sorted(requested.get(name, ())))
        for name, introduced in entries
        if int(introduced) not in requested.get(name, ())
    }
    assert mismatched == {}


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_driver_calls_go_through_the_table():
    """Every driver call uses DRIVER_CALL (or a pw_ wrapper), which resolves the
    table on first use and never dereferences null. The only raw p_ calls are
    the table's own machinery and the sites under ipc_import_mutex, where the
    table is resolved before the lock and marked `// raw:`."""
    machinery = {"driver_api.hpp", "driver_api.cpp", "py_driver_fns.cpp", "internal.hpp"}
    raw_call = re.compile(r"\bp_(cu|nv)\w+\(")
    offenders = []
    for path in HEADERS + SOURCES:
        if path.name in machinery:
            continue
        for number, line in enumerate(read(path).splitlines(), 1):
            if raw_call.search(line) and "// raw:" not in line and not line.lstrip().startswith("//"):
                offenders.append(f"{path.name}:{number}")
    assert offenders == []
    marked = [
        f"{path.name}:{number}"
        for path in SOURCES
        for number, line in enumerate(read(path).splitlines(), 1)
        if "// raw:" in line
    ]
    assert {name.split(":")[0] for name in marked} == {"memory.cpp"}, marked


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_pxd_functions_are_not_called_by_name_inside_the_module():
    """Cython emits a static prototype for each cdef function the .pxd declares, so
    calling one by that name from _rt.pyx clashes with the extern C++ declaration.
    The module calls through an alias with a different Cython name instead."""
    names = re.findall(r"^cdef\s+(?:[\w:.*& \[\]]+?\s)?(\w+)\(", read(CORE / "_rt.pxd"), re.M)
    assert len(names) > 90
    pyx = re.sub(r'""".*?"""', "", read(CORE / "_rt.pyx"), flags=re.S)
    pyx = re.sub(r"#[^\n]*", "", pyx)
    code = "\n".join(line for line in pyx.split("\n") if '"cuda_core::rt::' not in line)
    assert {name for name in names if re.search(rf"\b{name}\(", code)} == set()
