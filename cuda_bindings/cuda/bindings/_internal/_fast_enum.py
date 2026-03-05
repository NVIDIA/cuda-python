# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


# This code was automatically generated across versions from 12.9.1 to 13.1.1, generator version 0.3.1.dev1322+g646ce84ec. Do not modify it directly.


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
        for name, value in cls.__dict__.items():
            if name.startswith("__") and name.endswith("__"):
                continue

            if isinstance(value, tuple):
                value, doc = value
            elif isinstance(value, int):
                doc = None
            else:
                continue

            singleton = int.__new__(cls, value)
            singleton.__doc__ = doc
            singleton._name = name
            cls.__singletons__[value] = singleton
            cls.__members__[name] = singleton

        for name, member in cls.__members__.items():
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
