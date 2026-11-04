# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Site-package root priority must apply across alternative wheel layouts."""

from pathlib import Path

import pytest

from cuda.pathfinder import find_nvidia_binary_utility
from cuda.pathfinder._binaries import supported_nvidia_binaries
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.search_platform import LinuxSearchPlatform, WindowsSearchPlatform
from cuda.pathfinder._headers.find_nvidia_headers import find_in_site_packages
from cuda.pathfinder._headers.header_descriptor import HEADER_DESCRIPTORS
from cuda.pathfinder._static_libs.find_bitcode_lib import _FindBitcodeLib
from cuda.pathfinder._static_libs.find_static_lib import _FindStaticLib
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_cached
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


@pytest.fixture
def site_roots(monkeypatch, tmp_path):
    venv, user, system = (tmp_path / name for name in ("venv", "user", "system"))
    for root in (venv, user, system):
        root.mkdir()
    monkeypatch.setattr("site.getsitepackages", lambda: [str(venv), str(system)])
    monkeypatch.setattr("site.getusersitepackages", lambda: str(user))
    monkeypatch.setattr("site.ENABLE_USER_SITE", True)
    monkeypatch.setattr("sys.prefix", str(venv))
    monkeypatch.setattr("sys.base_prefix", str(system))
    for variable in ("CUDA_PATH", "CUDA_HOME", "CONDA_PREFIX"):
        monkeypatch.delenv(variable, raising=False)
    find_sub_dirs_cached.cache_clear()
    find_nvidia_binary_utility.cache_clear()
    yield venv, user, system
    find_sub_dirs_cached.cache_clear()
    find_nvidia_binary_utility.cache_clear()


def _case(kind):
    if kind == "bitcode":
        finder = _FindBitcodeLib("device")
        return finder.site_packages_dirs, finder.filename, finder.filename, finder.try_site_packages
    if kind == "static":
        finder = _FindStaticLib("cudadevrt")
        return finder.site_packages_dirs, finder.filename, finder.filename, finder.try_site_packages
    if kind == "header":
        desc = HEADER_DESCRIPTORS["cudart"]

        def lookup():
            result = find_in_site_packages(desc)
            return None if result is None else Path(result.abs_path) / desc.header_basename

        return desc.site_packages_dirs, desc.header_basename, desc.header_basename, lookup
    if kind == "binary":
        filename = "nvcc.exe" if IS_WINDOWS else "nvcc"
        return (
            supported_nvidia_binaries.SITE_PACKAGES_BINDIRS["nvcc"],
            filename,
            filename,
            lambda: find_nvidia_binary_utility("nvcc"),
        )
    desc = LIB_DESCRIPTORS["cudart"]
    if kind == "linux_dynamic":
        platform = LinuxSearchPlatform()
        newer, older = desc.linux_sonames
    else:
        platform = WindowsSearchPlatform(target_arch="x64")
        newer, older = desc.windows_dlls
    dirs = platform.site_packages_rel_dirs(desc)
    return dirs, newer, older, lambda: platform.find_in_site_packages(dirs, desc, [], [])


def _put(root, relative_dir, filename):
    path = root / relative_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("kind", ("bitcode", "static", "header", "binary", "linux_dynamic", "windows_dynamic"))
@pytest.mark.agent_authored(model="gpt-6")
def test_venv_layout_takes_priority_over_user_and_system(kind, site_roots):
    venv, user, system = site_roots
    layouts, newer, older, lookup = _case(kind)
    if len(layouts) < 2:
        pytest.skip("No alternative wheel layout for this platform")
    expected = _put(venv, layouts[-1], older)
    _put(user, layouts[0], newer)
    _put(system, layouts[0], newer)
    assert Path(lookup()) == expected


@pytest.mark.parametrize("kind", ("bitcode", "static", "header", "binary", "linux_dynamic", "windows_dynamic"))
@pytest.mark.agent_authored(model="gpt-6")
def test_layout_order_is_preserved_within_a_site_root(kind, site_roots):
    venv, _, _ = site_roots
    layouts, newer, older, lookup = _case(kind)
    if len(layouts) < 2:
        pytest.skip("No alternative wheel layout for this platform")
    expected = _put(venv, layouts[0], newer)
    _put(venv, layouts[-1], older)
    assert Path(lookup()) == expected


@pytest.mark.parametrize("kind", ("bitcode", "static", "header", "binary", "linux_dynamic", "windows_dynamic"))
@pytest.mark.agent_authored(model="gpt-6")
def test_lower_priority_root_is_used_when_higher_root_has_no_file(kind, site_roots):
    venv, user, _ = site_roots
    layouts, newer, _, lookup = _case(kind)
    if len(layouts) < 2:
        pytest.skip("No alternative wheel layout for this platform")
    (venv / layouts[-1]).mkdir(parents=True)
    expected = _put(user, layouts[0], newer)
    assert Path(lookup()) == expected


@pytest.mark.parametrize("in_venv", (True, False))
@pytest.mark.parametrize("kind", ("bitcode", "static", "header", "binary", "linux_dynamic", "windows_dynamic"))
@pytest.mark.agent_authored(model="gpt-6")
def test_user_layout_takes_priority_over_system(kind, in_venv, site_roots, monkeypatch):
    _, user, system = site_roots
    if not in_venv:
        monkeypatch.setattr("site.getsitepackages", lambda: [str(system)])
        monkeypatch.setattr("sys.prefix", str(system))
    layouts, newer, older, lookup = _case(kind)
    if len(layouts) < 2:
        pytest.skip("No alternative wheel layout for this platform")
    expected = _put(user, layouts[-1], older)
    _put(system, layouts[0], newer)
    assert Path(lookup()) == expected
