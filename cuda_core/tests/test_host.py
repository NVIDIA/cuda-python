# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.core import Host


class TestHost:
    def test_default(self):
        h = Host()
        assert h.numa_id is None
        assert h.is_numa_current is False

    def test_numa(self):
        h = Host(numa_id=3)
        assert h.numa_id == 3
        assert h.is_numa_current is False

    def test_numa_current(self):
        h = Host.numa_current()
        assert h.is_numa_current is True
        assert h.numa_id is None

    def test_invalid_numa_id(self):
        with pytest.raises(ValueError, match="numa_id must be a non-negative int"):
            Host(numa_id=-1)

    def test_numa_id_rejects_bool(self):
        # bool is an int subclass; reject explicitly so Host(True) doesn't
        # alias Host(1) (and vice versa) in the singleton cache.
        with pytest.raises(ValueError, match="numa_id must be a non-negative int"):
            Host(numa_id=True)
        with pytest.raises(ValueError, match="numa_id must be a non-negative int"):
            Host(numa_id=False)

    @pytest.mark.agent_authored(model="claude-opus-5")
    @pytest.mark.parametrize("value", [1, 0, "yes", None, []], ids=["int-1", "int-0", "str", "none", "list"])
    def test_is_numa_current_rejects_non_bool(self, value):
        # Same hazard as test_numa_id_rejects_bool, from the other side: the
        # value lands in the singleton cache key and in the is_numa_current
        # property untouched. `1` and `True` hash and compare equal, so
        # Host(is_numa_current=1) would seed the numa_current singleton with
        # an instance whose is_numa_current is the int 1 -- process-wide, and
        # for whichever call happens to come first. Any other truthy value
        # would mint a second "numa_current" that is neither `is` nor `==` the
        # real one while sharing its repr.
        with pytest.raises(ValueError, match="is_numa_current must be a bool"):
            Host(is_numa_current=value)

    @pytest.mark.agent_authored(model="claude-opus-5")
    def test_numa_current_singleton_survives_a_rejected_construction(self):
        with pytest.raises(ValueError, match="is_numa_current must be a bool"):
            Host(is_numa_current=1)

        h = Host.numa_current()
        assert h.is_numa_current is True
        assert h is Host(is_numa_current=True)

    def test_numa_current_constructor_and_classmethod_agree(self):
        # Host(is_numa_current=True) and Host.numa_current() return the same singleton.
        assert Host(is_numa_current=True) is Host.numa_current()
        # numa_id and is_numa_current are mutually exclusive.
        with pytest.raises(ValueError, match="mutually exclusive"):
            Host(numa_id=0, is_numa_current=True)

    def test_immutable(self):
        h = Host(numa_id=2)
        with pytest.raises(AttributeError):
            h.numa_id = 3  # type: ignore[misc]

    def test_eq_hash(self):
        # Frozen dataclass equality is structural.
        assert Host() == Host()
        assert Host(numa_id=1) == Host(numa_id=1)
        assert Host() != Host(numa_id=0)
        assert Host.numa_current() != Host()
        assert hash(Host(numa_id=1)) == hash(Host(numa_id=1))

    def test_repr(self):
        assert repr(Host()) == "Host()"
        assert repr(Host(numa_id=2)) == "Host(numa_id=2)"
        assert repr(Host.numa_current()) == "Host.numa_current()"

    def test_pickle_roundtrip_preserves_singleton(self):
        # __reduce__ routes numa_current through _reconstruct_numa_current and
        # the others through Host(numa_id); both rebuild the same singleton.
        # copy.copy / copy.deepcopy share the same __reduce__ machinery.
        import copy
        import pickle

        for h in (Host(), Host(numa_id=4), Host.numa_current()):
            assert pickle.loads(pickle.dumps(h)) is h  # noqa: S301
            assert copy.copy(h) is h
