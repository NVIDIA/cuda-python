# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Reusable helpers to verify collections.abc protocol conformance."""

from collections.abc import MutableSet, Set

import pytest


def assert_mutable_set_interface(subject, items):
    """Exercise every MutableSet method on *subject* against a reference set.

    Parameters
    ----------
    subject : MutableSet
        An **empty** mutable-set-like object to test.
    items : sequence
        At least five distinct, hashable objects valid for insertion into
        *subject*.
    """
    assert len(items) >= 5
    a, b, c, d, e = items[:5]
    ref = set()

    # -- ABC conformance --
    assert isinstance(subject, Set)
    assert isinstance(subject, MutableSet)

    # -- empty state --
    assert len(subject) == 0
    assert subject == ref
    assert subject == set()
    assert list(subject) == []

    # -- add --
    subject.add(a)
    ref.add(a)
    assert subject == ref
    assert a in subject
    assert b not in subject
    assert len(subject) == 1

    subject.add(b)
    subject.add(c)
    ref.update({b, c})
    assert subject == ref
    assert len(subject) == 3

    # add duplicate is a no-op
    subject.add(a)
    assert subject == ref

    # -- discard --
    subject.discard(b)
    ref.discard(b)
    assert subject == ref

    # discard non-member is a no-op
    subject.discard(d)
    assert subject == ref

    # -- remove --
    subject.add(b)
    ref.add(b)
    subject.remove(b)
    ref.remove(b)
    assert subject == ref

    with pytest.raises(KeyError):
        subject.remove(d)

    # -- comparison with plain set --
    assert subject == {a, c}
    assert subject != {a, b}

    # -- isdisjoint --
    assert subject.isdisjoint({d, e})
    assert not subject.isdisjoint({a, d})

    # -- subset / superset --
    assert subject <= {a, c}
    assert subject <= {a, b, c}
    assert not (subject <= {a})
    assert subject < {a, b, c}
    assert not (subject < {a, c})
    assert {a, c} >= subject
    assert {a, b, c} > subject

    # -- binary operators --
    assert subject & {a, d} == {a}
    assert subject | {d} == {a, c, d}
    assert subject - {c} == {a}
    assert subject ^ {c, d} == {a, d}

    # -- in-place union (|=) --
    subject |= {d, e}
    ref |= {d, e}
    assert subject == ref

    # -- in-place intersection (&=) --
    subject &= {a, d, e}
    ref &= {a, d, e}
    assert subject == ref

    # -- in-place difference (-=) --
    subject -= {e}
    ref -= {e}
    assert subject == ref

    # -- in-place symmetric difference (^=) --
    subject ^= {a, b}
    ref ^= {a, b}
    assert subject == ref

    # -- pop --
    popped = subject.pop()
    ref.discard(popped)
    assert popped not in subject
    assert subject == ref

    # -- clear --
    subject.clear()
    ref.clear()
    assert subject == ref
    assert len(subject) == 0

    with pytest.raises(KeyError):
        subject.pop()

    # -- bulk add via |= --
    subject |= {a, b, c}
    ref.update({a, b, c})
    assert subject == ref

    # -- __iter__ --
    assert set(subject) == ref

    # -- __repr__ --
    r = repr(subject)
    assert isinstance(r, str)
    assert len(r) > 0
