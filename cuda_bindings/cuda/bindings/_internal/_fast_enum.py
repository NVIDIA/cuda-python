# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=8e5f9b5cdfa1966fc26d3fd6782a41f08bc8b6c24321b0fe5fd1ddd92393bea8
"""
This is a replacement for the stdlib enum.IntEnum.

Notably, it has much better import time performance, since it doesn't generate
and evaluate Python code at startup time.

It supports the most important subset of the IntEnum API.  See `test_enum` in
`cuda_bindings/tests/test_basics.py` for details.
"""

from typing import Any, Iterator


class FastEnumMetaclass(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)

        cls.__singletons__ = {}
        cls.__members__ = {}
        aliases = {}
        for name, value in cls.__dict__.items():
            if name.startswith("__") and name.endswith("__"):
                continue

            if isinstance(value, tuple):
                value, doc = value
            elif isinstance(value, int):
                doc = None
            else:
                continue

            # A name sharing a value with an already-processed member is an
            # alias (e.g. a deprecated name kept for backward compatibility):
            # it resolves to the same singleton, but isn't a distinct member
            # (excluded from __members__, iteration, and len()).
            if value in cls.__singletons__:
                aliases[name] = cls.__singletons__[value]
                continue

            singleton = int.__new__(cls, value)
            singleton.__doc__ = doc
            singleton._name = name
            cls.__singletons__[value] = singleton
            cls.__members__[name] = singleton

        for name, member in cls.__members__.items():
            setattr(cls, name, member)
        for name, member in aliases.items():
            setattr(cls, name, member)

    def __repr__(cls) -> str:
        return f"<enum '{cls.__name__}'>"

    def __len__(cls) -> int:
        return len(cls.__members__)

    def __iter__(cls) -> Iterator["FastEnum"]:
        return iter(cls.__members__.values())

    def __contains__(cls, item: Any) -> bool:
        return item in cls.__singletons__


class FastEnum(int, metaclass=FastEnumMetaclass):
    def __new__(cls, value: int) -> "FastEnum":
        singleton: FastEnum = cls.__singletons__.get(value)
        if singleton is None:
            raise ValueError(f"{value} is not a valid {cls.__name__}")
        return singleton

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self._name}: {int(self)}>"

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> int:
        return int(self)
