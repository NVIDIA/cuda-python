# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the relocatable-device-code fixtures used by cuda.core tests."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(command: list[str]) -> None:
    print(f"+ {subprocess.list2cmdline(command)}")
    result = subprocess.run(command)  # noqa: S603
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _prepare_output(script_dir: Path, value: str | None) -> Path:
    if value is None:
        return script_dir
    project_root = script_dir.parents[1]
    output_root = project_root / ".moon-out"
    requested = Path(value)
    output = Path(os.path.abspath(requested if requested.is_absolute() else project_root.parent / requested))
    if output_root not in output.parents:
        raise ValueError(f"output must be below {output_root}: {output}")
    current = output
    while current != project_root:
        if current.is_symlink():
            raise ValueError(f"output path must not traverse a symlink: {current}")
        current = current.parent
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"refusing to replace non-directory output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    source_path = script_dir / "saxpy.cu"
    output = _prepare_output(script_dir, args.output_dir)
    final_object_path = output / "saxpy.o"
    final_library_path = output / ("saxpy.lib" if os.name == "nt" else "saxpy.a")

    nvcc_extra_flags = ["-std=c++17"]
    if os.name == "nt":
        nvcc_extra_flags.extend(["-Xcompiler", "/Zc:preprocessor"])

    with tempfile.TemporaryDirectory(prefix="build_test_binaries-", dir=output) as temp_dir:
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
