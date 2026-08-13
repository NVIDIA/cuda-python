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

    def fake_is_executable_candidate(path):
        checked.append(path)
        return path in existing

    mocker.patch.object(binary_finder_module, "_is_executable_candidate", side_effect=fake_is_executable_candidate)
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


@pytest.mark.parametrize(
    ("launcher_exists", "expected_rel", "checked_rels"),
    (
        (True, os.path.join("bin", "compute-sanitizer.bat"), (os.path.join("bin", "compute-sanitizer.bat"),)),
        (
            False,
            os.path.join("compute-sanitizer", "compute-sanitizer.exe"),
            (
                os.path.join("bin", "compute-sanitizer.bat"),
                os.path.join("compute-sanitizer", "compute-sanitizer.exe"),
            ),
        ),
    ),
)
@pytest.mark.usefixtures("clear_find_binary_cache")
@pytest.mark.agent_authored(model="gpt-5.6")
def test_find_compute_sanitizer_prefers_ctk_launcher_with_executable_fallback(
    monkeypatch, mocker, launcher_exists, expected_rel, checked_rels
):
    cuda_home = os.path.join(os.sep, "cuda")
    launcher = os.path.join(cuda_home, "bin", "compute-sanitizer.bat")
    executable = os.path.join(cuda_home, "compute-sanitizer", "compute-sanitizer.exe")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=True)
    mocker.patch.object(binary_finder_module.supported_nvidia_binaries, "SITE_PACKAGES_BINDIRS", {})
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=cuda_home)
    canary_mock = mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary")
    existing = [executable]
    if launcher_exists:
        existing.append(launcher)
    checked = _patch_exec_probe(mocker, existing=existing)

    assert find_nvidia_binary_utility("compute-sanitizer") == os.path.abspath(os.path.join(cuda_home, expected_rel))
    assert checked == [os.path.join(cuda_home, rel) for rel in checked_rels]
    canary_mock.assert_not_called()


@pytest.mark.usefixtures("clear_find_binary_cache")
@pytest.mark.agent_authored(model="gpt-5.6")
def test_find_compute_sanitizer_uses_canary_ctk_root(monkeypatch, mocker):
    ctk_root = os.path.join(os.sep, "cuda")
    launcher = os.path.join(ctk_root, "bin", "compute-sanitizer.bat")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=True)
    mocker.patch.object(binary_finder_module.supported_nvidia_binaries, "SITE_PACKAGES_BINDIRS", {})
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=None)
    canary = mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary", return_value=ctk_root)
    checked = _patch_exec_probe(mocker, existing=[launcher])

    assert find_nvidia_binary_utility("compute-sanitizer") == os.path.abspath(launcher)
    assert checked == [launcher]
    canary.assert_called_once_with()


@pytest.mark.parametrize(
    ("utility_name", "candidate_names"),
    (
        ("nsys", ("nsys.exe",)),
        ("ncu", ("ncu.bat", "ncu.exe")),
    ),
)
@pytest.mark.usefixtures("clear_find_binary_cache")
@pytest.mark.agent_authored(model="gpt-5.6")
def test_find_binary_windows_nsight_conda_precedes_registry(monkeypatch, mocker, utility_name, candidate_names):
    site_dir = os.path.join(os.sep, "site-packages", utility_name, "bin")
    conda_prefix = os.path.join(os.sep, "conda")
    conda_bin = os.path.join(conda_prefix, "Library", "bin")
    expected = os.path.join(conda_bin, candidate_names[0])

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=True)
    mocker.patch.object(binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[site_dir])
    monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    candidate_paths = mocker.patch.object(binary_finder_module.windows_nsight, f"{utility_name}_candidate_paths")
    get_cuda_path = mocker.patch.object(binary_finder_module, "get_cuda_path_or_home")
    canary = mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary")
    checked = _patch_exec_probe(mocker, existing=[expected])

    assert find_nvidia_binary_utility(utility_name) == os.path.abspath(expected)
    assert checked == [
        *(os.path.join(site_dir, name) for name in candidate_names),
        os.path.join(conda_bin, candidate_names[0]),
    ]
    candidate_paths.assert_not_called()
    get_cuda_path.assert_not_called()
    canary.assert_not_called()


@pytest.mark.parametrize(
    ("utility_name", "product", "machine_arch", "target_rel", "candidate_names"),
    (
        ("nsys", "Systems", "x64", os.path.join("target-windows-x64", "nsys.exe"), ("nsys.exe",)),
        ("nsys", "Systems", "arm64", os.path.join("target-windows-armv8", "nsys.exe"), ("nsys.exe",)),
        (
            "ncu",
            "Compute",
            "x64",
            os.path.join("target", "windows-desktop-win7-x64", "ncu.exe"),
            ("ncu.bat", "ncu.exe"),
        ),
        (
            "ncu",
            "Compute",
            "arm64",
            os.path.join("target", "windows-desktop-win10-t23x-a64", "ncu.exe"),
            ("ncu.bat", "ncu.exe"),
        ),
    ),
)
@pytest.mark.usefixtures("clear_find_binary_cache")
@pytest.mark.agent_authored(model="gpt-5.6")
def test_find_binary_windows_nsight_composes_registry_and_native_target(
    monkeypatch, mocker, utility_name, product, machine_arch, target_rel, candidate_names
):
    site_dir = os.path.join(os.sep, "site-packages", utility_name, "bin")
    conda_prefix = os.path.join(os.sep, "conda")
    conda_bin = os.path.join(conda_prefix, "Library", "bin")
    install_root = os.path.join(os.sep, "Program Files", utility_name)
    expected = os.path.join(install_root, target_rel)

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=True)
    mocker.patch.object(binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[site_dir])
    monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    registry_root = mocker.patch.object(
        binary_finder_module.windows_nsight, "_installed_product_root", return_value=install_root
    )
    machine_arch_mock = mocker.patch.object(
        binary_finder_module.windows_nsight, "windows_machine_arch", return_value=machine_arch
    )
    get_cuda_path = mocker.patch.object(binary_finder_module, "get_cuda_path_or_home")
    canary = mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary")
    checked = _patch_exec_probe(mocker, existing=[expected])

    assert find_nvidia_binary_utility(utility_name) == os.path.abspath(expected)
    assert checked == [
        *(os.path.join(directory, name) for directory in (site_dir, conda_bin) for name in candidate_names),
        *((os.path.join(install_root, "ncu.bat"),) if utility_name == "ncu" else ()),
        expected,
    ]
    registry_root.assert_called_once_with(product)
    machine_arch_mock.assert_called_once_with()
    get_cuda_path.assert_not_called()
    canary.assert_not_called()


@pytest.mark.usefixtures("clear_find_binary_cache")
@pytest.mark.agent_authored(model="gpt-5.6")
def test_find_binary_windows_ncu_launcher_hit_does_not_resolve_machine_arch(monkeypatch, mocker):
    install_root = os.path.join(os.sep, "Program Files", "Nsight Compute")
    launcher = os.path.join(install_root, "ncu.bat")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=True)
    mocker.patch.object(binary_finder_module.supported_nvidia_binaries, "SITE_PACKAGES_BINDIRS", {})
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    registry_root = mocker.patch.object(
        binary_finder_module.windows_nsight, "_installed_product_root", return_value=install_root
    )
    machine_arch = mocker.patch.object(binary_finder_module.windows_nsight, "windows_machine_arch")
    get_cuda_path = mocker.patch.object(binary_finder_module, "get_cuda_path_or_home")
    canary = mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary")
    checked = _patch_exec_probe(mocker, existing=[launcher])

    assert find_nvidia_binary_utility("ncu") == os.path.abspath(launcher)
    assert checked == [launcher]
    registry_root.assert_called_once_with("Compute")
    machine_arch.assert_not_called()
    get_cuda_path.assert_not_called()
    canary.assert_not_called()


@pytest.mark.parametrize(("utility_name", "product"), (("nsys", "Systems"), ("ncu", "Compute")))
@pytest.mark.usefixtures("clear_find_binary_cache")
@pytest.mark.agent_authored(model="gpt-5.6")
def test_find_binary_windows_nsight_registry_miss_is_terminal(monkeypatch, mocker, utility_name, product):
    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=True)
    mocker.patch.object(binary_finder_module.supported_nvidia_binaries, "SITE_PACKAGES_BINDIRS", {})
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    registry_root = mocker.patch.object(
        binary_finder_module.windows_nsight, "_installed_product_root", return_value=None
    )
    machine_arch = mocker.patch.object(binary_finder_module.windows_nsight, "windows_machine_arch")
    get_cuda_path = mocker.patch.object(binary_finder_module, "get_cuda_path_or_home")
    canary = mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary")

    assert find_nvidia_binary_utility(utility_name) is None
    registry_root.assert_called_once_with(product)
    machine_arch.assert_not_called()
    get_cuda_path.assert_not_called()
    canary.assert_not_called()


@pytest.mark.parametrize("utility_name", ("nsight-sys", "nsight-compute"))
@pytest.mark.usefixtures("clear_find_binary_cache")
@pytest.mark.agent_authored(model="gpt-5.6")
def test_find_windows_nsight_legacy_names_remain_literal_in_early_search(monkeypatch, mocker, utility_name):
    site_key = os.path.join("nvidia", utility_name, "bin")
    site_dir = os.path.join(os.sep, "site-packages", utility_name, "bin")
    conda_prefix = os.path.join(os.sep, "conda")
    conda_bin = os.path.join(conda_prefix, "Library", "bin")
    expected = os.path.join(conda_bin, f"{utility_name}.exe")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=True)
    mocker.patch.object(
        binary_finder_module.supported_nvidia_binaries,
        "SITE_PACKAGES_BINDIRS",
        {utility_name: (site_key,)},
    )
    find_sub_dirs = mocker.patch.object(binary_finder_module, "find_sub_dirs_all_sitepackages", return_value=[site_dir])
    monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    get_cuda_path = mocker.patch.object(binary_finder_module, "get_cuda_path_or_home")
    nsys_candidates = mocker.patch.object(binary_finder_module.windows_nsight, "nsys_candidate_paths")
    ncu_candidates = mocker.patch.object(binary_finder_module.windows_nsight, "ncu_candidate_paths")
    checked = _patch_exec_probe(mocker, existing=[expected])

    assert find_nvidia_binary_utility(utility_name) == os.path.abspath(expected)
    assert checked == [os.path.join(site_dir, f"{utility_name}.exe"), expected]
    find_sub_dirs.assert_called_once_with(site_key.split(os.sep))
    get_cuda_path.assert_not_called()
    nsys_candidates.assert_not_called()
    ncu_candidates.assert_not_called()


@pytest.mark.parametrize("utility_name", ("nsight-sys", "nsight-compute"))
@pytest.mark.usefixtures("clear_find_binary_cache")
@pytest.mark.agent_authored(model="gpt-5.6")
def test_find_windows_nsight_legacy_names_remain_literal_in_ctk(monkeypatch, mocker, utility_name):
    cuda_home = os.path.join(os.sep, "cuda")
    expected = os.path.join(cuda_home, "bin", f"{utility_name}.exe")

    mocker.patch.object(binary_finder_module, "IS_WINDOWS", new=True)
    mocker.patch.object(binary_finder_module.supported_nvidia_binaries, "SITE_PACKAGES_BINDIRS", {})
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    mocker.patch.object(binary_finder_module, "get_cuda_path_or_home", return_value=cuda_home)
    nsys_candidates = mocker.patch.object(binary_finder_module.windows_nsight, "nsys_candidate_paths")
    ncu_candidates = mocker.patch.object(binary_finder_module.windows_nsight, "ncu_candidate_paths")
    canary = mocker.patch.object(binary_finder_module, "_resolve_ctk_root_via_canary")
    checked = _patch_exec_probe(mocker, existing=[expected])

    assert find_nvidia_binary_utility(utility_name) == os.path.abspath(expected)
    assert checked == [
        os.path.join(cuda_home, "bin", "x64", f"{utility_name}.exe"),
        os.path.join(cuda_home, "bin", "x86_64", f"{utility_name}.exe"),
        expected,
    ]
    nsys_candidates.assert_not_called()
    ncu_candidates.assert_not_called()
    canary.assert_not_called()


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
    assert result == os.path.abspath(conda_nvcc)
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

    assert result == os.path.abspath(ctk_nvcc)
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

    assert result == os.path.abspath(conda_nvcc)
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
@pytest.mark.thread_unsafe(reason="functools.cache may replace entry.")
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


def test_resolve_in_trusted_dirs_returns_absolute_path(tmp_path, monkeypatch, mocker):
    """A match found under a relative search dir must be absolutized.

    ``find_nvidia_binary_utility`` documents an absolute, separator-resolved
    result. A relative search dir (e.g. a relative ``CUDA_HOME``) previously
    leaked a relative path that would re-resolve against a possibly different
    CWD at execution time.
    """
    rel_dir = os.path.join("some", "relative", "bin")
    candidate = os.path.join(rel_dir, "nvcc")
    mocker.patch.object(
        binary_finder_module,
        "_is_executable_candidate",
        side_effect=lambda path: path == candidate,
    )

    # Anchor CWD so os.path.abspath is deterministic for the assertion.
    monkeypatch.chdir(tmp_path)
    result = binary_finder_module._resolve_in_trusted_dirs("nvcc", [rel_dir])

    assert os.path.isabs(result)
    assert result == os.path.abspath(os.path.join(str(tmp_path), candidate))
