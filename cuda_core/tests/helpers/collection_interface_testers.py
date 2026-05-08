# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable helpers to verify collections.abc protocol conformance.

Two helpers are provided for ``MutableSet``-like subjects, picked by the
capacity of the backing store:

- :func:`assert_mutable_set_interface` is the standard pass; it requires at
  least five distinct insertable items so every method (including the
  multi-element bulk operators) can be exercised.
- :func:`assert_single_member_mutable_set_interface` is a focused pass for
  proxies whose backing store admits at most one insertable element at a time
  (for example, a peer-access view on a system with one valid peer device).
  It runs every ``MutableSet`` method at least once using a single member and
  one non-member sentinel.

The two helpers are intentionally separate rather than one helper with a
mode flag: a single-member proxy is a substantially different contract
("capacity one, by hardware") and naming it explicitly in the API keeps each
helper's signature small and its assertions linear.
"""

from collections.abc import MutableSet, Set

import pytest


def _assert_empty(subject):
    """Assertions that hold for any empty MutableSet-like subject."""
    assert isinstance(subject, Set)
    assert isinstance(subject, MutableSet)
    assert len(subject) == 0
    assert subject == set()
    assert list(subject) == []


def _assert_repr_nonempty(subject):
    """``__repr__`` produces a non-empty string."""
    r = repr(subject)
    assert isinstance(r, str)
    assert len(r) > 0


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

    _assert_empty(subject)

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

    _assert_repr_nonempty(subject)


def assert_single_member_mutable_set_interface(subject, member, non_member):
    """Exercise every MutableSet method on a subject with capacity one.

    Use this for proxies whose backing store admits at most one insertable
    element at a time (typically because the underlying resource is bounded
    by hardware, e.g. a peer-access view on a system with a single valid
    peer device). The subject only ever holds ``set()`` or ``{member}``;
    *non_member* supplies the right-hand side of comparisons, ``isdisjoint``,
    subset/superset, and binary/in-place operators so every ``MutableSet``
    method is exercised at least once.

    Parameters
    ----------
    subject : MutableSet
        An **empty** mutable-set-like object to test.
    member : hashable
        A distinct, hashable object valid for insertion into *subject*.
    non_member : hashable
        A distinct, hashable object that compares correctly under set
        semantics but is guaranteed never to be inserted into *subject*
        (typically because the backing store rejects it).
    """
    assert member != non_member, "member and non_member must be distinct"
    a = member
    x = non_member
    ref = set()

    _assert_empty(subject)

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
    assert subject.isdisjoint({x})
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
    subject &= {x}
    ref &= {x}
    assert subject == ref

    # -- in-place difference (-=) on non-member is a no-op --
    subject |= {a}
    ref |= {a}
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

    _assert_repr_nonempty(subject)
