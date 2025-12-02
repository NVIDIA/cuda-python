# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cuda.core
import pytest
from cuda.core import Device


def test_context_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Context objects cannot be instantiated directly\."):
        cuda.core.experimental._context.Context()  # Ensure back door is locked.


# ============================================================================
# Context Equality Tests
# ============================================================================


def test_context_equality_same_context(init_cuda):
    """Contexts from same device should be equal."""
    device = Device()

    s1 = device.create_stream()
    s2 = device.create_stream()

    ctx1 = s1.context
    ctx2 = s2.context

    # Same device, should have same context
    assert ctx1 == ctx2, "Streams on same device should share context"


def test_context_equality_reflexive(init_cuda):
    """Context should equal itself (reflexive property)."""
    device = Device()
    stream = device.create_stream()
    context = stream.context

    assert context == context, "Context should equal itself"


def test_context_type_safety(init_cuda):
    """Comparing Context with wrong type should return False."""
    device = Device()
    context = device.create_stream().context

    assert (context == "not a context") is False
    assert (context == 123) is False
    assert (context is None) is False


# ============================================================================
# Context Hash Tests
# ============================================================================


def test_context_hash_consistency(init_cuda):
    """Hash of same Context object should be consistent."""
    device = Device()
    stream = device.create_stream()
    context = stream.context

    hash1 = hash(context)
    hash2 = hash(context)
    assert hash1 == hash2, "Hash should be consistent for same object"


def test_context_hash_equality(init_cuda):
    """Contexts from same device should hash equal."""
    device = Device()

    s1 = device.create_stream()
    s2 = device.create_stream()

    ctx1 = s1.context
    ctx2 = s2.context

    # Same device, should have same context
    assert ctx1 == ctx2, "Streams on same device should share context"
    assert hash(ctx1) == hash(ctx2), "Same context should hash equal"


def test_context_dict_key(init_cuda):
    """Contexts should be usable as dictionary keys."""
    device = Device()
    stream = device.create_stream()
    context = stream.context

    ctx_cache = {context: "context_data"}
    assert ctx_cache[context] == "context_data"
