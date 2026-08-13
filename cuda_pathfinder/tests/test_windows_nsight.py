# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder._binaries import windows_nsight


def _patch_winreg(mocker):
    winreg = mocker.MagicMock()
    winreg.HKEY_LOCAL_MACHINE = object()
    winreg.KEY_READ = 0x20019
    winreg.KEY_WOW64_64KEY = 0x0100
    mocker.patch.object(windows_nsight.importlib, "import_module", return_value=winreg)
    return winreg


@pytest.mark.parametrize(
    ("machine_arch", "target_dir"),
    (
        ("x64", "target-windows-x64"),
        ("arm64", "target-windows-armv8"),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_nsys_candidate_paths_use_machine_arch(mocker, machine_arch, target_dir):
    install_root = os.path.join(os.sep, "Program Files", "Nsight Systems")
    expected = os.path.join(install_root, target_dir, "nsys.exe")
    mocker.patch.object(windows_nsight, "_installed_product_root", return_value=install_root)
    mocker.patch.object(windows_nsight, "windows_machine_arch", return_value=machine_arch)

    assert tuple(windows_nsight.nsys_candidate_paths()) == (expected,)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_nsys_candidate_paths_do_not_include_other_arch(mocker):
    install_root = os.path.join(os.sep, "Program Files", "Nsight Systems")
    arm64 = os.path.join(install_root, "target-windows-armv8", "nsys.exe")
    mocker.patch.object(windows_nsight, "_installed_product_root", return_value=install_root)
    mocker.patch.object(windows_nsight, "windows_machine_arch", return_value="arm64")

    assert tuple(windows_nsight.nsys_candidate_paths()) == (arm64,)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_ncu_candidate_paths_yield_launcher_before_resolving_machine_arch(mocker):
    install_root = os.path.join(os.sep, "Program Files", "Nsight Compute")
    launcher = os.path.join(install_root, "ncu.bat")
    mocker.patch.object(windows_nsight, "_installed_product_root", return_value=install_root)
    machine_arch = mocker.patch.object(windows_nsight, "windows_machine_arch")

    candidates = windows_nsight.ncu_candidate_paths()

    assert next(candidates) == launcher
    machine_arch.assert_not_called()


@pytest.mark.parametrize(
    ("machine_arch", "target_dir"),
    (
        ("x64", os.path.join("target", "windows-desktop-win7-x64")),
        ("arm64", os.path.join("target", "windows-desktop-win10-t23x-a64")),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_ncu_candidate_paths_fall_back_to_machine_binary(mocker, machine_arch, target_dir):
    install_root = os.path.join(os.sep, "Program Files", "Nsight Compute")
    launcher = os.path.join(install_root, "ncu.bat")
    expected = os.path.join(install_root, target_dir, "ncu.exe")
    mocker.patch.object(windows_nsight, "_installed_product_root", return_value=install_root)
    mocker.patch.object(windows_nsight, "windows_machine_arch", return_value=machine_arch)

    assert tuple(windows_nsight.ncu_candidate_paths()) == (launcher, expected)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_installed_product_root_reads_64_bit_registry(mocker):
    install_root = os.path.join(os.sep, "Program Files", "Nsight Systems")
    product_key = mocker.MagicMock()
    version_key = mocker.MagicMock()
    product_context = mocker.MagicMock()
    product_context.__enter__.return_value = product_key
    version_context = mocker.MagicMock()
    version_context.__enter__.return_value = version_key
    winreg = _patch_winreg(mocker)
    winreg.OpenKey.side_effect = (product_context, version_context)
    winreg.QueryValueEx.side_effect = (("2026.1.3", 1), (install_root, 1))

    assert windows_nsight._installed_product_root("Systems") == install_root
    access = winreg.KEY_READ | winreg.KEY_WOW64_64KEY
    winreg.OpenKey.assert_has_calls(
        (
            mocker.call(
                winreg.HKEY_LOCAL_MACHINE,
                rf"{windows_nsight._REGISTRY_ROOT}\Systems",
                0,
                access,
            ),
            mocker.call(product_key, "2026.1.3", 0, access),
        )
    )


@pytest.mark.agent_authored(model="gpt-5.6")
def test_installed_product_root_returns_none_when_product_key_is_absent(mocker):
    winreg = _patch_winreg(mocker)
    winreg.OpenKey.side_effect = FileNotFoundError("Nsight Systems is not installed")

    assert windows_nsight._installed_product_root("Systems") is None


@pytest.mark.agent_authored(model="gpt-5.6")
def test_installed_product_root_rejects_missing_current_version(mocker):
    product_context = mocker.MagicMock()
    product_context.__enter__.return_value = mocker.MagicMock()
    winreg = _patch_winreg(mocker)
    winreg.OpenKey.return_value = product_context
    winreg.QueryValueEx.side_effect = FileNotFoundError("CurrentVersion is missing")

    with pytest.raises(RuntimeError, match=r"Incomplete Nsight 'Systems' registry registration") as exc_info:
        windows_nsight._installed_product_root("Systems")

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("current_version", (None, "", "   ", 2026))
@pytest.mark.agent_authored(model="gpt-5.6")
def test_installed_product_root_rejects_invalid_current_version(mocker, current_version):
    product_context = mocker.MagicMock()
    product_context.__enter__.return_value = mocker.MagicMock()
    winreg = _patch_winreg(mocker)
    winreg.OpenKey.return_value = product_context
    winreg.QueryValueEx.return_value = (current_version, 1)

    with pytest.raises(RuntimeError, match=r"Invalid CurrentVersion value .*Nsight 'Systems' registry registration"):
        windows_nsight._installed_product_root("Systems")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_installed_product_root_rejects_missing_version_key(mocker):
    product_key = mocker.MagicMock()
    product_context = mocker.MagicMock()
    product_context.__enter__.return_value = product_key
    winreg = _patch_winreg(mocker)
    winreg.OpenKey.side_effect = (product_context, FileNotFoundError("Version key is missing"))
    winreg.QueryValueEx.return_value = ("2026.1.3", 1)

    with pytest.raises(RuntimeError, match=r"Incomplete Nsight 'Systems' registry registration") as exc_info:
        windows_nsight._installed_product_root("Systems")

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_installed_product_root_rejects_missing_installation_directory(mocker):
    product_context = mocker.MagicMock()
    product_context.__enter__.return_value = mocker.MagicMock()
    version_context = mocker.MagicMock()
    version_context.__enter__.return_value = mocker.MagicMock()
    winreg = _patch_winreg(mocker)
    winreg.OpenKey.side_effect = (product_context, version_context)
    winreg.QueryValueEx.side_effect = (("2026.1.3", 1), FileNotFoundError("Installation directory is missing"))

    with pytest.raises(RuntimeError, match=r"Incomplete Nsight 'Systems' registry registration") as exc_info:
        windows_nsight._installed_product_root("Systems")

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("install_root", (None, "", "   ", 2026))
@pytest.mark.agent_authored(model="gpt-5.6")
def test_installed_product_root_rejects_invalid_installation_directory(mocker, install_root):
    product_context = mocker.MagicMock()
    product_context.__enter__.return_value = mocker.MagicMock()
    version_context = mocker.MagicMock()
    version_context.__enter__.return_value = mocker.MagicMock()
    winreg = _patch_winreg(mocker)
    winreg.OpenKey.side_effect = (product_context, version_context)
    winreg.QueryValueEx.side_effect = (("2026.1.3", 1), (install_root, 1))

    with pytest.raises(
        RuntimeError,
        match=r"Invalid installation directory .*Nsight 'Systems' registry registration.*version '2026.1.3'",
    ):
        windows_nsight._installed_product_root("Systems")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_installed_product_root_propagates_access_errors(mocker):
    winreg = _patch_winreg(mocker)
    winreg.OpenKey.side_effect = PermissionError("Registry access denied")

    with pytest.raises(PermissionError, match="Registry access denied"):
        windows_nsight._installed_product_root("Systems")
