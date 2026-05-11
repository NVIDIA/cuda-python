# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import re
import rlcompleter
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from cuda.core.system import CUDA_BINDINGS_NVML_IS_COMPATIBLE

_CORE_ROOT = Path(__file__).resolve().parents[1] / "cuda" / "core"
_CDEF_CLASS_RE = re.compile(r"^cdef\s+class\s+([A-Za-z_]\w*)\b")
_DEF_RE = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(")
_PROPERTY_ASSIGN_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*python_property\(")


@dataclass(frozen=True)
class CythonProperty:
    module: str
    source: Path
    line: int
    class_name: str
    name: str
    decorator: str


def _module_for_source(source: Path) -> str:
    relative = source.relative_to(_CORE_ROOT)
    if source.suffix == ".pyx":
        return "cuda.core." + ".".join(relative.with_suffix("").parts)
    if relative.parts[0] == "system":
        return "cuda.core.system._device"
    raise ValueError(f"No module mapping for {source}")


def _iter_cython_properties():
    for source in sorted((*_CORE_ROOT.rglob("*.pyx"), *_CORE_ROOT.rglob("*.pxi"))):
        module = _module_for_source(source)
        class_name = None
        pending_decorator = None
        pending_line = None
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            class_match = _CDEF_CLASS_RE.match(line)
            if class_match is not None:
                class_name = class_match.group(1)
                pending_decorator = None
                pending_line = None
                continue

            if class_name is not None and line and not line[0].isspace() and line.strip() and not line.startswith("#"):
                class_name = None
                pending_decorator = None
                pending_line = None

            if class_name is None:
                continue

            stripped = line.strip()
            if stripped in {"@property", "@python_property"}:
                pending_decorator = stripped
                pending_line = line_number
                continue

            assignment_match = _PROPERTY_ASSIGN_RE.match(stripped)
            if assignment_match is not None:
                yield CythonProperty(
                    module=module,
                    source=source,
                    line=line_number,
                    class_name=class_name,
                    name=assignment_match.group(1),
                    decorator="python_property",
                )
                pending_decorator = None
                pending_line = None
                continue

            if pending_decorator is not None:
                def_match = _DEF_RE.match(stripped)
                if def_match is not None:
                    yield CythonProperty(
                        module=module,
                        source=source,
                        line=pending_line,
                        class_name=class_name,
                        name=def_match.group(1),
                        decorator=pending_decorator,
                    )
                pending_decorator = None
                pending_line = None


_CYTHON_PROPERTIES = tuple(_iter_cython_properties())


def _property_id(cython_property: CythonProperty) -> str:
    return f"{cython_property.module}.{cython_property.class_name}.{cython_property.name}"


@pytest.mark.no_cuda
def test_cython_cdef_class_getters_use_python_property_decorator():
    assert _CYTHON_PROPERTIES
    cython_properties = [
        f"{prop.source.relative_to(_CORE_ROOT)}:{prop.line}: {prop.class_name}.{prop.name}"
        for prop in _CYTHON_PROPERTIES
        if prop.decorator == "@property"
    ]

    assert not cython_properties


@pytest.mark.no_cuda
@pytest.mark.parametrize(
    "cython_property",
    [
        pytest.param(cython_property, id=_property_id(cython_property))
        for cython_property in _CYTHON_PROPERTIES
        if cython_property.decorator != "@property"
    ],
)
def test_cython_properties_are_python_properties(cython_property: CythonProperty):
    if cython_property.module.startswith("cuda.core.system.") and not CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        pytest.skip("cuda.core.system extension modules require NVML-compatible bindings")

    module = importlib.import_module(cython_property.module)
    cls = getattr(module, cython_property.class_name)
    descriptor = inspect.getattr_static(cls, cython_property.name)

    assert isinstance(descriptor, property)
    assert not isinstance(descriptor, types.GetSetDescriptorType)


def _rlcompleter_matches(obj):
    # rlcompleter suppresses names starting with "_" unless the query prefix
    # also starts with "_", so use the actual class name as the namespace key.
    namespace_key = type(obj).__name__
    completer = rlcompleter.Completer({namespace_key: obj})
    query = f"{namespace_key}."
    matches = []
    state = 0
    while True:
        match = completer.complete(query, state)
        if match is None:
            break
        matches.append(match)
        state += 1
    return matches


def _instance_device(init_cuda):
    return init_cuda


def _instance_device_memory_resource(init_cuda):
    from cuda.core import DeviceMemoryResource

    return DeviceMemoryResource(init_cuda)


def _instance_system_device(init_cuda):
    if not CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        pytest.skip("cuda.core.system.Device requires NVML-compatible bindings")
    import cuda.core.system

    return cuda.core.system.Device(index=0)


_TAB_COMPLETION_INSTANCE_FACTORIES = {
    "cuda.core.Device": _instance_device,
    "cuda.core.DeviceMemoryResource": _instance_device_memory_resource,
    "cuda.core.system.Device": _instance_system_device,
}


@pytest.mark.parametrize("factory_id", sorted(_TAB_COMPLETION_INSTANCE_FACTORIES))
def test_rlcompleter_returns_matches_for_cython_instances(init_cuda, factory_id):
    """rlcompleter must return completion candidates for cdef class instances.

    Guards against regressions that hide Cython-defined properties from
    interactive tab completion at the instance level (a failure mode that
    is distinct from class-level introspection).
    """
    obj = _TAB_COMPLETION_INSTANCE_FACTORIES[factory_id](init_cuda)
    matches = _rlcompleter_matches(obj)
    assert matches, f"rlcompleter returned no completions for {factory_id} instance"
