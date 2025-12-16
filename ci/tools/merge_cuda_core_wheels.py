#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to merge CUDA-specific wheels into a single multi-CUDA wheel.

This script takes wheels built for different CUDA versions (cu12, cu13) and merges them
into a single wheel that supports both CUDA versions.

In particular, each wheel contains a CUDA-specific build of the `cuda.core` library
and the associated bindings. This script merges these directories into a single wheel
that supports both CUDA versions, i.e., containing both `cuda/core/cu12`
and `cuda/core/cu13`. At runtime, the code in `cuda/core/__init__.py`
is used to import the appropriate CUDA-specific bindings.

This script is based on the one in NVIDIA/CCCL.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


def run_command(cmd: List[str], cwd: Path = None, env: dict = os.environ) -> subprocess.CompletedProcess:
    """Run a command with error handling."""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"  Working directory: {cwd}")

    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)  # noqa: S603

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        result.check_returncode()

    return result


def merge_wheels(wheels: List[Path], output_dir: Path) -> Path:
    """Merge multiple wheels into a single wheel with version-specific binaries."""
    print("\n=== Merging wheels ===", file=sys.stderr)
    print(f"Input wheels: {[w.name for w in wheels]}", file=sys.stderr)

    if len(wheels) == 1:
        raise RuntimeError("only one wheel is provided, nothing to merge")

    # Extract all wheels to temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extracted_wheels = []

        for i, wheel in enumerate(wheels):
            print(f"Extracting wheel {i + 1}/{len(wheels)}: {wheel.name}", file=sys.stderr)
            # Extract wheel - wheel unpack creates the directory itself
            run_command(
                [
                    sys.executable,
                    "-m",
                    "wheel",
                    "unpack",
                    str(wheel),
                    "--dest",
                    str(temp_path),
                ]
            )

            # Find the extracted directory (wheel unpack creates a subdirectory)
            extract_dir = None
            for item in temp_path.iterdir():
                if item.is_dir() and item.name.startswith("cuda_core"):
                    extract_dir = item
                    break

            if not extract_dir:
                raise RuntimeError(f"Could not find extracted wheel directory for {wheel.name}")

            # Rename to our expected name
            expected_name = temp_path / f"wheel_{i}"
            extract_dir.rename(expected_name)
            extract_dir = expected_name

            extracted_wheels.append(extract_dir)

        # Use the first wheel as the base and merge binaries from others
        base_wheel = extracted_wheels[0]

        # now copy the version-specific directory from other wheels
        # into the appropriate place in the base wheel
        for i, wheel_dir in enumerate(extracted_wheels):
            cuda_version = wheels[i].name.split(".cu")[1].split(".")[0]
            base_dir = Path("cuda") / "core"
            # Copy from other wheels
            print(f"  Copying {wheel_dir} to {base_wheel}", file=sys.stderr)
            shutil.copytree(wheel_dir / base_dir, base_wheel / base_dir / f"cu{cuda_version}")

            # Overwrite the __init__.py in versioned dirs
            os.truncate(base_wheel / base_dir / f"cu{cuda_version}" / "__init__.py", 0)

        # The base dir should only contain __init__.py, the include dir, and the versioned dirs
        files_to_remove = os.scandir(base_wheel / base_dir)
        for f in files_to_remove:
            f_abspath = f.path
            if f.name not in ("__init__.py", "cu12", "cu13", "include"):
                if f.is_dir():
                    shutil.rmtree(f_abspath)
                else:
                    os.remove(f_abspath)

        # Repack the merged wheel
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a clean wheel name without CUDA version suffixes
        base_wheel_name = wheels[0].with_suffix(".whl").name

        print(f"Repacking merged wheel as: {base_wheel_name}", file=sys.stderr)
        run_command(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(base_wheel),
                "--dest-dir",
                str(output_dir),
            ]
        )

        # Find the output wheel
        output_wheels = list(output_dir.glob("*.whl"))
        if not output_wheels:
            raise RuntimeError("Failed to create merged wheel")

        merged_wheel = output_wheels[0]
        print(f"Successfully merged wheel: {merged_wheel}", file=sys.stderr)
        return merged_wheel


def main():
    """Main merge script."""
    parser = argparse.ArgumentParser(description="Merge CUDA-specific wheels into a single multi-CUDA wheel")
    parser.add_argument("wheels", nargs="+", help="Paths to the CUDA-specific wheels to merge")
    parser.add_argument("--output-dir", "-o", default="dist", help="Output directory for merged wheel")

    args = parser.parse_args()

    print("cuda.core Wheel Merger", file=sys.stderr)
    print("======================", file=sys.stderr)

    # Convert wheel paths to Path objects and validate
    wheels = []
    for wheel_path in args.wheels:
        wheel = Path(wheel_path)
        if not wheel.exists():
            print(f"Error: Wheel not found: {wheel}", file=sys.stderr)
            sys.exit(1)
        if not wheel.name.endswith(".whl"):
            print(f"Error: Not a wheel file: {wheel}", file=sys.stderr)
            sys.exit(1)
        wheels.append(wheel)

    if not wheels:
        print("Error: No wheels provided", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Check that we have wheel tool available
    try:
        import wheel
    except ImportError:
        print("Error: wheel package not available. Install with: pip install wheel", file=sys.stderr)
        sys.exit(1)

    # Merge the wheels
    merged_wheel = merge_wheels(wheels, output_dir)
    print(f"\nMerge complete! Output: {merged_wheel}")


if __name__ == "__main__":
    main()
