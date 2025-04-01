import os

import pytest

from cuda.bindings._path_finder_utils.sys_path_find_sub_dirs import _impl


@pytest.fixture
def test_tree(tmp_path):
    # Build:
    # tmp_path/
    #   sys1/nvidia/foo/lib
    #   sys1/nvidia/bar/lib
    #   sys2/nvidia/baz/nvvm/lib64
    base = tmp_path
    (base / "sys1" / "nvidia" / "foo" / "lib").mkdir(parents=True)
    (base / "sys1" / "nvidia" / "bar" / "lib").mkdir(parents=True)
    (base / "sys2" / "nvidia" / "baz" / "nvvm" / "lib64").mkdir(parents=True)

    return {
        "sys_path": (
            str(base / "sys1"),
            str(base / "sys2"),
            str(base / "nonexistent"),  # should be ignored
        ),
        "base": base,
    }


def test_exact_match(test_tree):
    sys_path = test_tree["sys_path"]
    base = test_tree["base"]
    result = _impl(sys_path, ("nvidia", "foo", "lib"))
    expected = [str(base / "sys1" / "nvidia" / "foo" / "lib")]
    assert result == expected


def test_single_wildcard(test_tree):
    sys_path = test_tree["sys_path"]
    base = test_tree["base"]
    result = _impl(sys_path, ("nvidia", "*", "lib"))
    expected = [
        str(base / "sys1" / "nvidia" / "bar" / "lib"),
        str(base / "sys1" / "nvidia" / "foo" / "lib"),
    ]
    assert sorted(result) == sorted(expected)


def test_double_wildcard(test_tree):
    sys_path = test_tree["sys_path"]
    base = test_tree["base"]
    result = _impl(sys_path, ("nvidia", "*", "nvvm", "lib64"))
    expected = [str(base / "sys2" / "nvidia" / "baz" / "nvvm" / "lib64")]
    assert result == expected


def test_no_match(test_tree):
    sys_path = test_tree["sys_path"]
    result = _impl(sys_path, ("nvidia", "nonexistent", "lib"))
    assert result == []


def test_empty_sys_path():
    result = _impl((), ("nvidia", "*", "lib"))
    assert result == []


def test_empty_sub_dirs(test_tree):
    sys_path = test_tree["sys_path"]
    result = _impl(sys_path, ())
    expected = [p for p in sys_path if os.path.isdir(p)]
    assert sorted(result) == sorted(expected)
