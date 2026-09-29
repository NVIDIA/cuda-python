# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
from collections.abc import Iterator
from typing import Any

from cuda.pathfinder._utils.windows_arch import windows_machine_arch

_REGISTRY_ROOT = r"SOFTWARE\NVIDIA Corporation\Installed Products\Nsight"

_NSYS_TARGET_DIR_BY_ARCH = {
    "x64": "target-windows-x64",
    "arm64": "target-windows-armv8",
}

_NCU_TARGET_DIR_BY_ARCH = {
    "x64": os.path.join("target", "windows-desktop-win7-x64"),
    "arm64": os.path.join("target", "windows-desktop-win10-t23x-a64"),
}


def _installed_product_root(product: str) -> str | None:
    """Return the active Nsight product installation recorded by its MSI."""
    # ``winreg`` attributes are absent from the type stubs on non-Windows hosts.
    winreg: Any = importlib.import_module("winreg")

    access = winreg.KEY_READ | winreg.KEY_WOW64_64KEY
    product_key_path = rf"{_REGISTRY_ROOT}\{product}"
    try:
        product_context = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, product_key_path, 0, access)
    except FileNotFoundError:
        return None

    try:
        with product_context as product_key:
            current_version, _ = winreg.QueryValueEx(product_key, "CurrentVersion")
            if not isinstance(current_version, str) or not current_version.strip():
                raise RuntimeError(
                    f"Invalid CurrentVersion value {current_version!r} in "
                    f"Nsight {product!r} registry registration at {product_key_path!r}"
                )
            with winreg.OpenKey(product_key, current_version, 0, access) as version_key:
                install_root, _ = winreg.QueryValueEx(version_key, None)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Incomplete Nsight {product!r} registry registration at {product_key_path!r}") from exc

    if not isinstance(install_root, str) or not install_root.strip():
        raise RuntimeError(
            f"Invalid installation directory {install_root!r} in Nsight {product!r} "
            f"registry registration at {product_key_path!r} version {current_version!r}"
        )
    return install_root


def nsys_candidate_paths() -> Iterator[str]:
    install_root = _installed_product_root("Systems")
    if install_root is None:
        return

    target_dir = _NSYS_TARGET_DIR_BY_ARCH[windows_machine_arch()]
    yield os.path.join(install_root, target_dir, "nsys.exe")


def ncu_candidate_paths() -> Iterator[str]:
    install_root = _installed_product_root("Compute")
    if install_root is None:
        return

    yield os.path.join(install_root, "ncu.bat")

    target_dir = _NCU_TARGET_DIR_BY_ARCH[windows_machine_arch()]
    yield os.path.join(install_root, target_dir, "ncu.exe")
