# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
import sysconfig

WINDOWS_PE_MACHINE_BY_ARCH = {
    "x64": 0x8664,
    "arm64": 0xAA64,
}

_WINDOWS_ARCH_BY_PE_MACHINE = {machine: arch for arch, machine in WINDOWS_PE_MACHINE_BY_ARCH.items()}


class UnsupportedArchError(RuntimeError):
    """Raised when Python reports an unsupported Windows architecture."""

    def __init__(self, platform_tag: str) -> None:
        self.platform_tag = platform_tag
        super().__init__(
            f"Unsupported Windows Python platform tag: {platform_tag!r}; expected 'win-amd64' or 'win-arm64'"
        )


def windows_python_arch() -> str:
    """Return the current Windows Python interpreter architecture."""
    raw_platform_tag = sysconfig.get_platform()
    platform_tag = raw_platform_tag.lower().replace("_", "-")

    if platform_tag == "win-arm64":
        return "arm64"

    if platform_tag == "win-amd64":
        return "x64"

    raise UnsupportedArchError(raw_platform_tag)


def _windows_machine_arch_from_platform() -> str:
    """Return the Windows architecture reported by Python's platform module."""
    raw_machine = platform.machine()
    machine = raw_machine.lower().replace("_", "-")

    if machine in ("amd64", "x86-64"):
        return "x64"

    if machine in ("arm64", "aarch64"):
        return "arm64"

    raise RuntimeError(f"Unsupported Windows machine architecture: {raw_machine!r}")


def _windows_native_machine() -> int | None:
    """Return the native Windows PE machine type, or None on older Windows."""
    import ctypes
    from ctypes import wintypes

    try:
        # These ctypes attributes are absent from the type stubs on non-Windows hosts.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined, unused-ignore]
    except OSError as exc:
        raise RuntimeError("Failed to load kernel32 while detecting the native Windows architecture") from exc

    get_current_process = kernel32.GetCurrentProcess
    try:
        is_wow64_process2 = kernel32.IsWow64Process2
    except AttributeError:
        return None

    get_current_process.argtypes = ()
    get_current_process.restype = wintypes.HANDLE
    is_wow64_process2.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.USHORT),
        ctypes.POINTER(wintypes.USHORT),
    )
    is_wow64_process2.restype = wintypes.BOOL

    process_machine = wintypes.USHORT()
    native_machine = wintypes.USHORT()
    if not is_wow64_process2(
        get_current_process(),
        ctypes.byref(process_machine),
        ctypes.byref(native_machine),
    ):
        error_code = ctypes.get_last_error()  # type: ignore[attr-defined, unused-ignore]
        error = ctypes.WinError(error_code)  # type: ignore[attr-defined, unused-ignore]
        raise RuntimeError(
            f"IsWow64Process2 failed while detecting the native Windows architecture "
            f"(Windows error {error_code}): {error}"
        ) from error
    return native_machine.value


def windows_machine_arch() -> str:
    """Return the native Windows machine architecture, ignoring process emulation."""
    native_machine = _windows_native_machine()
    if native_machine is None:
        # IsWow64Process2 predates x64-on-Arm emulation, so this fallback is only
        # needed on older Windows versions where platform.machine() is sufficient.
        return _windows_machine_arch_from_platform()

    try:
        return _WINDOWS_ARCH_BY_PE_MACHINE[native_machine]
    except KeyError:
        raise RuntimeError(f"Unsupported native Windows PE machine type: 0x{native_machine:04x}") from None


def windows_pe_matches_arch(path: str, target_arch: str) -> bool:
    """Return whether a Windows Portable Executable (PE) targets the requested architecture.

    PE is the file format used for Windows executables and DLLs. This reads the
    PE/COFF header's machine field to distinguish x64 images from Arm64 images.
    """
    expected_machine = WINDOWS_PE_MACHINE_BY_ARCH.get(target_arch)
    if expected_machine is None:
        raise ValueError(f"Unsupported Windows target architecture: {target_arch!r}")

    try:
        with open(path, "rb") as stream:
            if stream.read(2) != b"MZ":
                return False
            stream.seek(0x3C)
            pe_offset_bytes = stream.read(4)
            if len(pe_offset_bytes) != 4:
                return False
            stream.seek(int.from_bytes(pe_offset_bytes, "little"))
            if stream.read(4) != b"PE\0\0":
                return False
            machine_bytes = stream.read(2)
            if len(machine_bytes) != 2:
                return False
    except OSError:
        return False

    return int.from_bytes(machine_bytes, "little") == expected_machine
