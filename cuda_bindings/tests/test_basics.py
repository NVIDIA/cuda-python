# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import enum

import pytest
from cuda.bindings.utils import _fast_enum

# Test both with the FastEnum implementation and the stdlib enum.IntEnum (even
# though we don't use the latter) to make sure that the two APIs are identical


class MyFastEnum(_fast_enum._FastEnum, metaclass=_fast_enum._FastEnumMetaclass):
    RED = 0
    GREEN = 1
    BLUE = 2


class MyIntEnum(enum.IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2


@pytest.mark.parametrize("MyEnum", [MyFastEnum, MyIntEnum])
def test_enum(MyEnum):
    container = MyEnum

    val = container.GREEN

    assert isinstance(val, MyEnum)

    assert container(1) is val

    with pytest.raises(ValueError):
        container(5)

    # Different Python versions raise different error types here from
    # stdlib.enum.IntEnum
    with pytest.raises((ValueError, TypeError)):
        container(1, 2, 3)

    with pytest.raises(TypeError):
        container(foo=1)

    assert val == 1
    assert val.value == 1
    assert isinstance(val.value, int)
    assert val.name == "GREEN"
    assert isinstance(val.name, str)

    assert container.GREEN | container.BLUE == 3
    assert container.GREEN.BLUE is container.BLUE

    assert repr(container) == f"<enum '{container.__name__}'>"
    assert repr(val) == f"<{container.__name__}.GREEN: 1>"

    assert len(container) == 3

    for item in container:
        assert isinstance(item, container)
        assert item in container
        assert item.value in container
        assert item.name in dir(container)
        for item2 in container:
            assert hasattr(item, item2.name)
            assert item2.name in dir(item)
            assert getattr(item, item2.name) is item2

    for name, val in container.__members__.items():
        assert isinstance(val, container)
        assert isinstance(name, str)
        assert name == val.name
