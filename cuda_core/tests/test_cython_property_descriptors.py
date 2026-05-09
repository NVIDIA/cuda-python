# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import pkgutil
import types

import pytest

import cuda.core
import cuda.core.graph
import cuda.core.system

pytestmark = pytest.mark.no_cuda


_NOT_ALLOWED_TO_IMPORT = {"cuda.core._tensor_bridge"}

_GETSET_FIELD_ALLOWLIST = {
    ("cuda.core._kernel_arg_handler", "ParamHolder", "ptr"),
    ("cuda.core._layout", "_StridedLayout", "itemsize"),
    ("cuda.core._layout", "_StridedLayout", "slice_offset"),
    ("cuda.core._memoryview", "StridedMemoryView", "device_id"),
    ("cuda.core._memoryview", "StridedMemoryView", "exporting_obj"),
    ("cuda.core._memoryview", "StridedMemoryView", "is_device_accessible"),
    ("cuda.core._memoryview", "StridedMemoryView", "ptr"),
    ("cuda.core._memoryview", "StridedMemoryView", "readonly"),
    ("cuda.core._memoryview", "_StridedMemoryViewProxy", "has_dlpack"),
    ("cuda.core._memoryview", "_StridedMemoryViewProxy", "obj"),
}


def _iter_cuda_core_modules():
    roots = (cuda.core, cuda.core.graph, cuda.core.system)
    module_names = set()
    for root in roots:
        for info in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
            module_names.add(info.name)

    module_names -= _NOT_ALLOWED_TO_IMPORT
    for module_name in sorted(module_names):
        yield importlib.import_module(module_name)


def _iter_cuda_core_classes():
    for module in _iter_cuda_core_modules():
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__:
                yield cls


def _is_allowed_getset_descriptor(cls, name, descriptor) -> bool:
    if name in {"__dict__", "__weakref__"}:
        return True
    # Typed cdef fields generate getset descriptors too, but their docs follow
    # this compact "field: type" form rather than a property docstring.
    doc = descriptor.__doc__
    if doc is not None and doc.startswith(f"{name}:"):
        return True
    return (cls.__module__, cls.__qualname__, name) in _GETSET_FIELD_ALLOWLIST


def test_cuda_core_classes_do_not_expose_cython_property_getset_descriptors():
    classes = tuple(_iter_cuda_core_classes())
    unexpected_getsets = [
        f"{cls.__module__}.{cls.__qualname__}.{name}"
        for cls in classes
        for name, descriptor in vars(cls).items()
        if isinstance(descriptor, types.GetSetDescriptorType)
        and not _is_allowed_getset_descriptor(cls, name, descriptor)
    ]

    assert classes
    assert not unexpected_getsets
