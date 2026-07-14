# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the relocatable-device-code fixtures used by cuda.core tests."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def _run(command: list[str]) -> None:
    print(f"+ {subprocess.list2cmdline(command)}")
    result = subprocess.run(command)  # noqa: S603
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    source_path = script_dir / "saxpy.cu"
    final_object_path = script_dir / "saxpy.o"
    final_library_path = script_dir / ("saxpy.lib" if os.name == "nt" else "saxpy.a")

    nvcc_extra_flags = ["-std=c++17"]
    if os.name == "nt":
        nvcc_extra_flags.extend(["-Xcompiler", "/Zc:preprocessor"])

    with tempfile.TemporaryDirectory(prefix="build_test_binaries-", dir=script_dir) as temp_dir:
        temp_dir_path = Path(temp_dir)
        object_path = temp_dir_path / final_object_path.name
        library_path = temp_dir_path / final_library_path.name

        _run(
            [
                "nvcc",
                "-dc",
                *nvcc_extra_flags,
                "-arch=all",
                "-o",
                str(object_path),
                str(source_path),
            ]
        )
        _run(["nvcc", "-lib", "-o", str(library_path), str(object_path)])

        object_path.replace(final_object_path)
        library_path.replace(final_library_path)

    for path in (final_object_path, final_library_path):
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
