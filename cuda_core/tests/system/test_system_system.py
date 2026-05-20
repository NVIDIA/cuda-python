# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import contextlib
import locale
import os
import shutil
import subprocess
import sys

import pytest

try:
    from cuda.bindings import driver
except ImportError:
    from cuda import cuda as driver

from cuda.core import system
from cuda.core._utils.cuda_utils import handle_return

from .conftest import skip_if_nvml_unsupported


def _detect_wsl() -> bool:
    if any(var in os.environ for var in ("WSL_DISTRO_NAME", "WSL_INTEROP", "WSLENV")):
        return True
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            version = f.read().lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def _gather_wsl_host_info() -> list[str]:
    info: list[str] = ["=== Windows host (via WSL interop) ==="]

    cmd_exe = shutil.which("cmd.exe") or "/mnt/c/Windows/System32/cmd.exe"
    powershell = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

    # Active console code page via cmd.exe.
    try:
        result = subprocess.run(  # noqa: S603
            [cmd_exe, "/c", "chcp"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd="/",
        )
        info.append(f"cmd.exe chcp rc={result.returncode}")
        if result.stdout:
            info.append(f"  stdout: {result.stdout.strip()!r}")
        if result.stderr:
            info.append(f"  stderr: {result.stderr.strip()!r}")
    except (OSError, subprocess.SubprocessError) as e:
        info.append(f"cmd.exe chcp failed: {e!r}")

    # Richer culture / code page info via PowerShell.
    ps_script = ";".join(
        [
            "$OutputEncoding=[Text.Encoding]::UTF8",
            "Write-Output ('Culture=' + (Get-Culture).Name)",
            "Write-Output ('UICulture=' + (Get-UICulture).Name)",
            "Write-Output ('SystemLocale=' + (Get-WinSystemLocale).Name)",
            "Write-Output ('ConsoleOutputCP=' + [Console]::OutputEncoding.WebName)",
            "Write-Output ('ConsoleInputCP=' + [Console]::InputEncoding.WebName)",
            (
                "Write-Output ('ANSICodePage=' + "
                "[System.Globalization.CultureInfo]::CurrentCulture.TextInfo.ANSICodePage)"
            ),
            ("Write-Output ('OEMCodePage=' + [System.Globalization.CultureInfo]::CurrentCulture.TextInfo.OEMCodePage)"),
            "Write-Output ('OSVersion=' + [Environment]::OSVersion.VersionString)",
        ]
    )
    try:
        result = subprocess.run(  # noqa: S603
            [powershell, "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/",
        )
        info.append(f"powershell.exe rc={result.returncode}")
        for line in result.stdout.splitlines():
            info.append(f"  {line.strip()}")
        if result.stderr.strip():
            info.append(f"  stderr: {result.stderr.strip()!r}")
    except (OSError, subprocess.SubprocessError) as e:
        info.append(f"powershell.exe call failed: {e!r}")

    return info


def _print_locale_diagnostics() -> None:
    lines: list[str] = ["=== System / locale diagnostics ==="]
    lines.append(f"sys.platform: {sys.platform}")
    lines.append(f"sys.version: {sys.version.splitlines()[0]}")
    lines.append(f"sys.getdefaultencoding(): {sys.getdefaultencoding()!r}")
    lines.append(f"sys.getfilesystemencoding(): {sys.getfilesystemencoding()!r}")
    lines.append(f"sys.getfilesystemencodeerrors(): {sys.getfilesystemencodeerrors()!r}")
    lines.append(f"sys.stdout.encoding: {getattr(sys.stdout, 'encoding', None)!r}")
    lines.append(f"sys.stderr.encoding: {getattr(sys.stderr, 'encoding', None)!r}")
    lines.append(f"sys.flags.utf8_mode: {sys.flags.utf8_mode}")

    lines.append(f"locale.getpreferredencoding(False): {locale.getpreferredencoding(False)!r}")
    lines.append(f"locale.getpreferredencoding(True): {locale.getpreferredencoding(True)!r}")
    with contextlib.suppress(AttributeError):
        lines.append(f"locale.getencoding(): {locale.getencoding()!r}")
    try:
        lines.append(f"locale.getlocale(): {locale.getlocale()!r}")
    except (locale.Error, ValueError) as e:
        lines.append(f"locale.getlocale() error: {e!r}")

    for cat_name in (
        "LC_ALL",
        "LC_CTYPE",
        "LC_COLLATE",
        "LC_MESSAGES",
        "LC_MONETARY",
        "LC_NUMERIC",
        "LC_TIME",
    ):
        cat = getattr(locale, cat_name, None)
        if cat is None:
            continue
        try:
            lines.append(f"locale.getlocale({cat_name}): {locale.getlocale(cat)!r}")
        except (locale.Error, ValueError, TypeError) as e:
            lines.append(f"locale.getlocale({cat_name}) error: {e!r}")
        try:
            lines.append(f"locale.setlocale({cat_name}, None): {locale.setlocale(cat, None)!r}")
        except (locale.Error, ValueError, TypeError) as e:
            lines.append(f"locale.setlocale({cat_name}, None) error: {e!r}")

    for var in (
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_COLLATE",
        "LC_MESSAGES",
        "LC_MONETARY",
        "LC_NUMERIC",
        "LC_TIME",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
        "PYTHONLEGACYWINDOWSFSENCODING",
    ):
        lines.append(f"env {var}={os.environ.get(var, '<unset>')!r}")

    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            lines.append(f"/proc/version: {f.read().strip()!r}")
    except OSError as e:
        lines.append(f"/proc/version: <error: {e!r}>")

    for var in ("WSL_DISTRO_NAME", "WSL_INTEROP", "WSLENV"):
        if var in os.environ:
            lines.append(f"env {var}={os.environ[var]!r}")

    if _detect_wsl():
        lines.extend(_gather_wsl_host_info())
    else:
        lines.append("WSL not detected; skipping Windows host probe")

    lines.append("=== end diagnostics ===")
    print("\n".join(lines), flush=True)


def test_user_mode_driver_version():
    umd = system.get_user_mode_driver_version()
    assert isinstance(umd, tuple)
    assert len(umd) == 2
    version = handle_return(driver.cuDriverGetVersion())
    expected = (version // 1000, (version % 1000) // 10)
    assert umd == expected, "UMD driver version does not match expected value"


@skip_if_nvml_unsupported
def test_kernel_mode_driver_version():
    kmd = system.get_kernel_mode_driver_version()
    assert isinstance(kmd, tuple)
    assert len(kmd) in (2, 3)
    ver_maj, ver_min, *ver_patch = kmd
    assert 400 <= ver_maj < 1000
    assert ver_min >= 0
    if ver_patch:
        assert 0 <= ver_patch[0] <= 99


def test_kernel_mode_driver_version_requires_nvml():
    if system.CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        pytest.skip("NVML is available, cannot test the error path")
    with pytest.raises(RuntimeError, match="requires NVML support"):
        system.get_kernel_mode_driver_version()


@skip_if_nvml_unsupported
def test_nvml_version():
    nvml_version = system.get_nvml_version()
    assert isinstance(nvml_version, tuple)
    assert len(nvml_version) in (3, 4)

    (cuda_ver_maj, ver_maj, ver_min, *ver_patch) = nvml_version
    assert cuda_ver_maj >= 10
    assert 400 <= ver_maj < 1000
    assert ver_min >= 0
    if ver_patch:
        assert 0 <= ver_patch[0] <= 99


@skip_if_nvml_unsupported
def test_get_process_name():
    _print_locale_diagnostics()

    from cuda.bindings import nvml

    nvml.init_v2()
    print("BYTES:", repr(nvml.system_get_process_name_bytes(os.getpid())))

    try:
        process_name = system.get_process_name(os.getpid())
    except system.NotFoundError:
        pytest.skip("Process not found")

    print(f"process_name: {process_name!r}", flush=True)
    print(
        f"process_name.encode('utf-8', 'backslashreplace'): {process_name.encode('utf-8', 'backslashreplace')!r}",
        flush=True,
    )

    assert isinstance(process_name, str)
    assert "python" in process_name


def test_device_count():
    device_count = system.get_num_devices()
    assert isinstance(device_count, int)
    assert device_count >= 0


@skip_if_nvml_unsupported
def test_get_driver_branch():
    driver_branch = system.get_driver_branch()
    assert isinstance(driver_branch, str)
    assert len(driver_branch) > 0
