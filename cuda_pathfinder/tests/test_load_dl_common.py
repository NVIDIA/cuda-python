# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.pathfinder._dynamic_libs.descriptor_catalog import DescriptorSpec
from cuda.pathfinder._dynamic_libs.load_dl_common import (
    DynamicLibNotAvailableError,
    DynamicLibNotFoundError,
    DynamicLibUnknownError,
    LoadedDL,
    load_dependencies,
)


def _loaded(name: str) -> LoadedDL:
    return LoadedDL(f"/{name}", False, 1, "test")


@pytest.mark.agent_authored(model="gpt-5")
def test_load_dependencies_loads_required_then_optional_dependencies():
    desc = DescriptorSpec(
        name="subject",
        packaged_with="other",
        dependencies=("required",),
        optional_dependencies=("optional",),
    )
    calls = []

    def load_func(name):
        calls.append(name)
        return _loaded(name)

    load_dependencies(desc, load_func)

    assert calls == ["required", "optional"]


@pytest.mark.agent_authored(model="gpt-5")
def test_load_dependencies_continues_after_optional_dependency_is_absent():
    desc = DescriptorSpec(
        name="subject",
        packaged_with="other",
        optional_dependencies=("absent", "available"),
    )
    calls = []

    def load_func(name):
        calls.append(name)
        if name == "absent":
            raise DynamicLibNotFoundError(name)
        return _loaded(name)

    load_dependencies(desc, load_func)

    assert calls == ["absent", "available"]


@pytest.mark.parametrize("error_type", (DynamicLibUnknownError, DynamicLibNotAvailableError, RuntimeError))
@pytest.mark.agent_authored(model="gpt-5")
def test_load_dependencies_propagates_malformed_or_unloadable_optional_dependency(error_type):
    desc = DescriptorSpec(name="subject", packaged_with="other", optional_dependencies=("broken",))

    def load_func(name):
        raise error_type(name)

    with pytest.raises(error_type):
        load_dependencies(desc, load_func)


@pytest.mark.agent_authored(model="gpt-5")
def test_load_dependencies_keeps_required_dependencies_fail_fast():
    desc = DescriptorSpec(
        name="subject",
        packaged_with="other",
        dependencies=("required",),
        optional_dependencies=("optional",),
    )
    calls = []

    def load_func(name):
        calls.append(name)
        raise DynamicLibNotFoundError(name)

    with pytest.raises(DynamicLibNotFoundError):
        load_dependencies(desc, load_func)

    assert calls == ["required"]
