# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.core import WorkqueueResourceOptions
from cuda.core._utils.validators import check_str_enum, format_or_list
from cuda.core.typing import SourceCodeType


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param([], "", id="empty"),
        pytest.param(["a"], "'a'", id="one"),
        pytest.param(["a", "b"], "'a' or 'b'", id="two"),
        pytest.param(["a", "b", "c"], "'a', 'b' or 'c'", id="three"),
        pytest.param([None, "a"], "None or 'a'", id="none-first"),
    ],
)
def test_format_or_list(values, expected):
    assert format_or_list(values) == expected


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("value", ["c++", "ptx", "nvvm", SourceCodeType.CXX])
def test_check_str_enum_accepts_members_and_their_values(value):
    check_str_enum(value, SourceCodeType)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_check_str_enum_allow_none():
    check_str_enum(None, SourceCodeType, allow_none=True)

    with pytest.raises(ValueError, match="None is not a valid SourceCodeType"):
        check_str_enum(None, SourceCodeType)


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("fortran", id="str"),
        pytest.param(5, id="int"),
        # Unhashable arguments used to blow up inside the membership test with
        # "TypeError: unhashable type" before the ValueError was ever built.
        pytest.param(["c++"], id="list"),
        pytest.param({"code_type": "c++"}, id="dict"),
        pytest.param(bytearray(b"c++"), id="bytearray"),
    ],
)
def test_check_str_enum_rejects_invalid_values_with_value_error(value):
    with pytest.raises(ValueError, match="is not a valid SourceCodeType. Must be "):
        check_str_enum(value, SourceCodeType)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_options_reject_an_unhashable_scope_with_value_error():
    """A public entry point must report a bad option the same way for every
    kind of bad value. ``WorkqueueResourceOptions.__post_init__`` validates
    ``sharing_scope`` through ``check_str_enum``, so an unhashable argument
    used to surface as ``TypeError: unhashable type: 'list'`` while a plain
    string got the documented ValueError."""
    with pytest.raises(ValueError, match="is not a valid WorkqueueSharingScopeType. Must be "):
        WorkqueueResourceOptions(sharing_scope=["green_ctx_balanced"])
