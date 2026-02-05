# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


from typing import Any


class _FastEnum(int):
    def __new__(cls, value):
        singleton = cls.__singletons__.get(value)
        if singleton is None:
            raise ValueError(f"{value} is not a valid {cls.__name__}")
        return singleton

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self._name}: {int(self)}>"

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return int(self)


class _FastEnumMetaclass(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)

        cls.__singletons__ = {}
        cls.__members__ = {}
        for name, value in cls.__dict__.items():
            if name.startswith("__") and name.endswith("__"):
                continue

            if isinstance(value, int):
                singleton = int.__new__(cls, value)
                singleton._name = name
                cls.__singletons__[value] = singleton
                cls.__members__[name] = singleton

        for name, member in cls.__members__.items():
            setattr(cls, name, member)

    def __repr__(cls):
        return f"<enum '{cls.__name__}'>"

    def __len__(cls):
        return len(cls.__members__)

    def __iter__(cls):
        return iter(cls.__members__.values())

    def __contains__(cls, item: Any) -> bool:
        return item in cls.__singletons__
