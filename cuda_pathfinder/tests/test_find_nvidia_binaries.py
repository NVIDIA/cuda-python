# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder import find_nvidia_binary_utility
from cuda.pathfinder._binaries import find_nvidia_binary_utility as binary_finder_module
from cuda.pathfinder._binaries.find_nvidia_binary_utility import UnsupportedBinaryError
from cuda.pathfinder._binaries.supported_nvidia_binaries import (
    SITE_PACKAGES_BINDIRS,
    SUPPORTED_BINARIES,
    SUPPORTED_BINARIES_ALL,
)


def test_unknown_utility_name():
    with pytest.raises(UnsupportedBinaryError, match=r"'unknown-utility' is not supported"):
        find_nvidia_binary_utility("unknown-utility")


@pytest.mark.parametrize("utility_name", SUPPORTED_BINARIES)
def test_find_binary_utilities(info_summary_append, utility_name):
    bin_path = find_nvidia_binary_utility(utility_name)
    info_summary_append(f"{bin_path=!r}")

    assert bin_path is None or os.path.isfile(bin_path)


def test_supported_binaries_consistency():
    assert set(SUPPORTED_BINARIES).issubset(SUPPORTED_BINARIES_ALL)
    assert set(SITE_PACKAGES_BINDIRS).issubset(SUPPORTED_BINARIES_ALL)


@pytest.fixture
def clear_find_binary_cache():
    find_nvidia_binary_utility.cache_clear()
    yield
    find_nvidia_binary_utility.cache_clear()


def _patch_exec_probe(mocker, existing=()):
    """Patch the executable-file probe and record probed candidates in order.

    ``existing`` is the set of candidate paths reported as present; every other
    candidate is treated as missing. Returns the list that accumulates probed
    candidates so tests can assert the deterministic search order.
    """
    existing = set(existing)
    checked: list[str] = []

    def fake_is_executable_file(path):
        checked.append(path)
        return path in existing

    mocker.patch.object(binary_finder_module, "_is_executable_file", side_effect=fake_is_executable_file)
    return checked


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_search_path_includes_site_packages_conda_cuda(monkeypatch, mocker):
    conda_prefix = os.path.join(os.sep, "conda")
    cuda_home = os.path.join(os.sep, "cuda")
    site_key = os.path.join("nvidia", "cuda_nvcc", "bin")
    site_dir = os.path.join("site-packages", "cuda_nvcc", "bin")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=False)
    mocker.patch.object(
        binary_finder_module.supported_nvidia_binaries,
        "SITE_PACKAGES_BINDIRS",
        {"nvcc": (site_key,)},
    )
    find_sub_dirs_mock = mocker.patch.object(
        binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[site_dir]
    )
    monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=cuda_home)
    mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary", return_value=None)
    expected_dirs = [
        site_dir,
        os.path.join(conda_prefix, "bin"),
        os.path.join(cuda_home, "bin"),
    ]
    checked = _patch_exec_probe(mocker)

    result = find_nvidia_binary_utility("nvcc")

    # No directory contains the binary, so every trusted dir is probed in order.
    assert result is None
    find_sub_dirs_mock.assert_called_once_with(site_key.split(os.sep))
    assert checked == [os.path.join(d, "nvcc") for d in expected_dirs]


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_windows_extension_and_search_dirs(monkeypatch, mocker):
    conda_prefix = os.path.join(os.sep, "conda")
    cuda_home = os.path.join(os.sep, "cuda")
    site_key = os.path.join("nvidia", "cuda_nvcc", "bin")
    site_dir = os.path.join("site-packages", "cuda_nvcc", "bin")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=True)
    mocker.patch.object(
        binary_finder_module.supported_nvidia_binaries,
        "SITE_PACKAGES_BINDIRS",
        {"nvcc": (site_key,)},
    )
    find_sub_dirs_mock = mocker.patch.object(
        binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[site_dir]
    )
    monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=cuda_home)
    mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary", return_value=None)
    expected_dirs = [
        site_dir,
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(cuda_home, "bin", "x64"),
        os.path.join(cuda_home, "bin", "x86_64"),
        os.path.join(cuda_home, "bin"),
    ]
    checked = _patch_exec_probe(mocker)

    result = find_nvidia_binary_utility("nvcc")

    # The .exe extension is appended and the Windows-specific dirs are probed in order.
    assert result is None
    find_sub_dirs_mock.assert_called_once_with(site_key.split(os.sep))
    assert checked == [os.path.join(d, "nvcc.exe") for d in expected_dirs]


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_first_matching_dir_wins(monkeypatch, mocker):
    conda_prefix = os.path.join(os.sep, "conda")
    cuda_home = os.path.join(os.sep, "cuda")
    site_key = os.path.join("nvidia", "cuda_nvcc", "bin")
    site_dir = os.path.join("site-packages", "cuda_nvcc", "bin")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=False)
    mocker.patch.object(
        binary_finder_module.supported_nvidia_binaries,
        "SITE_PACKAGES_BINDIRS",
        {"nvcc": (site_key,)},
    )
    mocker.patch.object(binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[site_dir])
    monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=cuda_home)
    mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary", return_value=None)
    conda_nvcc = os.path.join(conda_prefix, "bin", "nvcc")
    cuda_nvcc = os.path.join(cuda_home, "bin", "nvcc")
    checked = _patch_exec_probe(mocker, existing=[conda_nvcc, cuda_nvcc])

    result = find_nvidia_binary_utility("nvcc")

    # Conda comes before CUDA_HOME, so the Conda hit wins and CUDA_HOME is never probed.
    assert result == conda_nvcc
    assert checked == [os.path.join(site_dir, "nvcc"), conda_nvcc]


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_ctk_root_canary_fallback(monkeypatch, mocker):
    # When the explicit trusted dirs (wheels, conda, CUDA_HOME/PATH) all miss,
    # the cudart-canary-derived CTK root is searched last.
    ctk_root = os.path.join(os.sep, "opt", "cuda")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=False)
    mocker.patch.object(binary_finder_module.supported_nvidia_binaries, "SITE_PACKAGES_BINDIRS", {})
    mocker.patch.object(binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[])
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=None)
    canary_mock = mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary", return_value=ctk_root)
    ctk_nvcc = os.path.join(ctk_root, "bin", "nvcc")
    checked = _patch_exec_probe(mocker, existing=[ctk_nvcc])

    result = find_nvidia_binary_utility("nvcc")

    assert result == ctk_nvcc
    canary_mock.assert_called_once_with()
    # No earlier trusted dirs existed, so the only probe is the canary bin dir.
    assert checked == [ctk_nvcc]


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_canary_windows_bin_layout(monkeypatch, mocker):
    ctk_root = os.path.join("C:", os.sep, "cuda")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=True)
    mocker.patch.object(binary_finder_module.supported_nvidia_binaries, "SITE_PACKAGES_BINDIRS", {})
    mocker.patch.object(binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[])
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=None)
    mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary", return_value=ctk_root)
    expected_dirs = [
        os.path.join(ctk_root, "bin", "x64"),
        os.path.join(ctk_root, "bin", "x86_64"),
        os.path.join(ctk_root, "bin"),
    ]
    checked = _patch_exec_probe(mocker)

    result = find_nvidia_binary_utility("nvcc")

    assert result is None
    assert checked == [os.path.join(d, "nvcc.exe") for d in expected_dirs]


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_canary_not_consulted_when_found_earlier(monkeypatch, mocker):
    # An earlier trusted dir hit must short-circuit before the canary subprocess.
    conda_prefix = os.path.join(os.sep, "conda")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=False)
    mocker.patch.object(binary_finder_module.supported_nvidia_binaries, "SITE_PACKAGES_BINDIRS", {})
    mocker.patch.object(binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[])
    monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=None)
    canary_mock = mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary", return_value=None)
    conda_nvcc = os.path.join(conda_prefix, "bin", "nvcc")
    _patch_exec_probe(mocker, existing=[conda_nvcc])

    result = find_nvidia_binary_utility("nvcc")

    assert result == conda_nvcc
    canary_mock.assert_not_called()


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_returns_none_with_no_candidates(monkeypatch, mocker):
    site_key = os.path.join("nvidia", "cuda_nvcc", "bin")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=False)
    mocker.patch.object(
        binary_finder_module.supported_nvidia_binaries,
        "SITE_PACKAGES_BINDIRS",
        {"nvcc": (site_key,)},
    )
    find_sub_dirs_mock = mocker.patch.object(binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[])
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=None)
    mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary", return_value=None)
    checked = _patch_exec_probe(mocker)

    result = find_nvidia_binary_utility("nvcc")

    assert result is None
    find_sub_dirs_mock.assert_called_once_with(site_key.split(os.sep))
    # No trusted dirs were assembled, so nothing is probed at all.
    assert checked == []


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_without_site_packages_entry(monkeypatch, mocker):
    conda_prefix = os.path.join(os.sep, "conda")
    cuda_home = os.path.join(os.sep, "cuda")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=False)
    mocker.patch.object(binary_finder_module.supported_nvidia_binaries, "SITE_PACKAGES_BINDIRS", {})
    find_sub_dirs_mock = mocker.patch.object(binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[])
    monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=cuda_home)
    mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary", return_value=None)
    expected_dirs = [
        os.path.join(conda_prefix, "bin"),
        os.path.join(cuda_home, "bin"),
    ]
    checked = _patch_exec_probe(mocker)

    result = find_nvidia_binary_utility("nvcc")

    assert result is None
    find_sub_dirs_mock.assert_not_called()
    assert checked == [os.path.join(d, "nvcc") for d in expected_dirs]


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_find_binary_cache_negative_result(monkeypatch, mocker):
    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=False)
    mocker.patch.object(binary_finder_module.supported_nvidia_binaries, "SITE_PACKAGES_BINDIRS", {})
    mocker.patch.object(binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[])
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    cuda_home_mock = mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=None)
    canary_mock = mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary", return_value=None)
    _patch_exec_probe(mocker)

    first = find_nvidia_binary_utility("nvcc")
    second = find_nvidia_binary_utility("nvcc")

    assert first is None
    assert second is None
    # The second call is served from @functools.cache, so the body runs only
    # once, including the canary fallback.
    cuda_home_mock.assert_called_once_with()
    canary_mock.assert_called_once_with()


class TestResolveInTrustedDirs:
    """Unit tests for the deterministic resolver, including the #2119 contract."""

    @staticmethod
    def _make_executable(directory, name):
        path = os.path.join(str(directory), name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        os.chmod(path, 0o700)
        return path

    def test_cwd_is_not_searched(self, tmp_path, monkeypatch):
        # Regression for #2119: a binary in the process CWD must never shadow
        # the trusted directories.
        trusted = tmp_path / "trusted"
        trusted.mkdir()
        evil_cwd = tmp_path / "cwd"
        evil_cwd.mkdir()
        empty = tmp_path / "empty"
        empty.mkdir()
        trusted_nvcc = self._make_executable(trusted, "nvcc")
        self._make_executable(evil_cwd, "nvcc")  # the decoy that must be ignored
        monkeypatch.chdir(evil_cwd)

        # A trusted dir with no binary returns None, never the CWD copy.
        assert binary_finder_module._resolve_in_trusted_dirs("nvcc", [str(empty)]) is None
        # When a trusted dir holds it, that path wins regardless of CWD.
        assert binary_finder_module._resolve_in_trusted_dirs("nvcc", [str(empty), str(trusted)]) == trusted_nvcc

    def test_first_trusted_dir_wins(self, tmp_path):
        first = tmp_path / "a"
        first.mkdir()
        second = tmp_path / "b"
        second.mkdir()
        first_nvcc = self._make_executable(first, "nvcc")
        self._make_executable(second, "nvcc")
        assert binary_finder_module._resolve_in_trusted_dirs("nvcc", [str(first), str(second)]) == first_nvcc

    def test_duplicate_dirs_skipped(self, tmp_path):
        present = tmp_path / "p"
        present.mkdir()
        nvcc = self._make_executable(present, "nvcc")
        assert binary_finder_module._resolve_in_trusted_dirs("nvcc", [str(present), str(present)]) == nvcc
        assert binary_finder_module._resolve_in_trusted_dirs("nvcc", []) is None

    def test_empty_dir_asserts(self):
        with pytest.raises(AssertionError):
            binary_finder_module._resolve_in_trusted_dirs("nvcc", [""])

    @pytest.mark.skipif(binary_finder_module.IS_WINDOWS, reason="POSIX execute-bit semantics")
    def test_non_executable_file_rejected_on_posix(self, tmp_path):
        directory = tmp_path / "d"
        directory.mkdir()
        path = os.path.join(str(directory), "nvcc")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        os.chmod(path, 0o644)
        assert binary_finder_module._resolve_in_trusted_dirs("nvcc", [str(directory)]) is None
        os.chmod(path, 0o700)
        assert binary_finder_module._resolve_in_trusted_dirs("nvcc", [str(directory)]) == path


@pytest.mark.usefixtures("clear_find_binary_cache")
def test_caching_per_utility():
    """Verify that different utilities have independent cache entries."""
    nvdisasm1 = find_nvidia_binary_utility("nvdisasm")
    nvcc1 = find_nvidia_binary_utility("nvcc")
    nvdisasm2 = find_nvidia_binary_utility("nvdisasm")
    nvcc2 = find_nvidia_binary_utility("nvcc")

    # Same utility should return cached result
    assert nvdisasm1 is nvdisasm2
    assert nvcc1 is nvcc2

    # Different utilities should have different results (unless at least one of
    # them is None)
    if nvdisasm1 is not None and nvcc1 is not None:
        assert nvdisasm1 != nvcc1
