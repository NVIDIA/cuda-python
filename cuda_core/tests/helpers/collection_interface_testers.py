# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable helpers to verify collections.abc protocol conformance."""

from collections.abc import MutableSet, Set

import pytest


def assert_mutable_set_interface(
    subject,
    items,
    *,
    non_members=None,
    support_multi_insert=True,
):
    """Exercise every MutableSet method on *subject* against a reference set.

    Two modes are supported:

    - ``support_multi_insert=True`` (default): the standard protocol pass that
      inserts up to five distinct elements simultaneously. Use this whenever
      the subject can hold an arbitrary number of items.
    - ``support_multi_insert=False``: a reduced pass for proxies whose backing
      store admits at most one insertable element at a time (e.g. a peer-access
      view on a 2-GPU system). The subject only ever holds ``{}`` or ``{a}``,
      and ``non_members`` supplies sentinel values used as the *other* side of
      comparisons, ``isdisjoint``, subset/superset, and binary/in-place
      operators. Every ``MutableSet`` method is still exercised at least once.

    Parameters
    ----------
    subject : MutableSet
        An **empty** mutable-set-like object to test.
    items : sequence
        Distinct, hashable objects valid for insertion into *subject*. With
        ``support_multi_insert=True`` at least five are required; with
        ``support_multi_insert=False`` exactly one is needed (extras ignored).
    non_members : sequence, optional
        Distinct, hashable values that compare equal across set semantics but
        are guaranteed *never* to be inserted into *subject* (typically because
        the backing store rejects them). Required and used only when
        ``support_multi_insert=False``; at least two are needed there.
    """
    if support_multi_insert:
        _assert_mutable_set_interface_multi(subject, items)
    else:
        if non_members is None:
            raise TypeError("non_members is required when support_multi_insert=False")
        _assert_mutable_set_interface_single(subject, items, non_members)


def _assert_mutable_set_interface_multi(subject, items):
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


def _assert_mutable_set_interface_single(subject, items, non_members):
    """Reduced protocol pass for subjects that admit at most one insertable element.

    Invariants:
    - ``a`` is the lone insertable element; the subject only ever holds ``{}``
      or ``{a}``.
    - ``x``, ``y`` are sentinels guaranteed *not* to be inserted; they appear
      only on the right-hand side of operators and comparisons. They must
      compare correctly under set semantics (i.e. equality and hashing).
    """
    assert len(items) >= 1
    assert len(non_members) >= 2
    a = items[0]
    x, y = non_members[:2]
    assert a != x and a != y and x != y, "items[0] and non_members[:2] must be distinct"

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
    assert x not in subject
    assert len(subject) == 1

    # add duplicate is a no-op
    subject.add(a)
    assert subject == ref
    assert len(subject) == 1

    # -- comparison with plain set --
    assert subject == {a}
    assert subject != {a, x}
    assert subject != set()

    # -- isdisjoint --
    assert subject.isdisjoint({x, y})
    assert not subject.isdisjoint({a, x})

    # -- subset / superset --
    assert subject <= {a}
    assert subject <= {a, x}
    assert not (subject <= set())
    assert subject < {a, x}
    assert not (subject < {a})
    assert {a, x} >= subject
    assert {a, x} > subject

    # -- binary operators (results are plain sets, never insert into subject) --
    assert subject & {a, x} == {a}
    assert subject & {x} == set()
    assert subject | {x} == {a, x}
    assert subject - {a} == set()
    assert subject - {x} == {a}
    assert subject ^ {x} == {a, x}
    assert subject ^ {a} == set()

    # -- discard non-member is a no-op --
    subject.discard(x)
    assert subject == ref

    # -- discard member --
    subject.discard(a)
    ref.discard(a)
    assert subject == ref

    # -- remove member --
    subject.add(a)
    ref.add(a)
    subject.remove(a)
    ref.remove(a)
    assert subject == ref

    # -- remove non-member raises --
    with pytest.raises(KeyError):
        subject.remove(x)

    # -- pop empty raises --
    with pytest.raises(KeyError):
        subject.pop()

    # -- pop populated --
    subject.add(a)
    ref.add(a)
    popped = subject.pop()
    ref.discard(popped)
    assert popped == a
    assert popped not in subject
    assert subject == ref

    # -- in-place union (|=) covers single insert via bulk path --
    subject |= {a}
    ref |= {a}
    assert subject == ref

    # -- in-place intersection (&=) keeps the lone member --
    subject &= {a, x}
    ref &= {a, x}
    assert subject == ref

    # -- in-place intersection (&=) drops the lone member --
    subject &= {x, y}
    ref &= {x, y}
    assert subject == ref

    # -- in-place union via bulk path again, ahead of -= and ^= --
    subject |= {a}
    ref |= {a}
    assert subject == ref

    # -- in-place difference (-=) on non-member is a no-op --
    subject -= {x}
    ref -= {x}
    assert subject == ref

    # -- in-place difference (-=) on member empties the subject --
    subject -= {a}
    ref -= {a}
    assert subject == ref

    # -- in-place symmetric difference (^=): toggle in then out --
    subject ^= {a}
    ref ^= {a}
    assert subject == ref
    subject ^= {a}
    ref ^= {a}
    assert subject == ref

    # -- clear --
    subject.add(a)
    ref.add(a)
    subject.clear()
    ref.clear()
    assert subject == ref
    assert len(subject) == 0

    # -- __iter__ on populated subject --
    subject.add(a)
    ref.add(a)
    assert set(subject) == ref

    # -- __repr__ --
    r = repr(subject)
    assert isinstance(r, str)
    assert len(r) > 0
