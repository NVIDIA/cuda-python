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
