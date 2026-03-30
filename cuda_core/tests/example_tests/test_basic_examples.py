# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# If we have subcategories of examples in the future, this file can be split along those lines

import glob
import importlib.metadata
import os
import platform
import re
import subprocess
import sys

import pytest

from cuda.core import Device, system

# Each example in cuda_core/examples is tested in two ways:
#
# 1) Directly running the example in the same environment as the test suite.
#    This gives access to the current development version of cuda_core.
# 2) Running the example in a subprocess with "uv run" to verify that the PEP
#    723 metadata works correctly and that the example can be run in isolation from
#    the test suite.


def has_compute_capability_9_or_higher() -> bool:
    return Device().compute_capability >= (9, 0)


def has_multiple_devices() -> bool:
    return system.get_num_devices() >= 2


def has_display() -> bool:
    # We assume that we don't want to open any windows during testing,
    # so we always return False
    return False


def is_not_windows() -> bool:
    return sys.platform != "win32"


def is_x86_64() -> bool:
    return platform.machine() == "x86_64"


def uv_installed() -> bool:
    try:
        subprocess.run(["uv", "--version"], check=True)  # noqa: S607
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


def has_cuda_path() -> bool:
    return os.environ.get("CUDA_PATH", os.environ.get("CUDA_HOME")) is not None


# Specific system requirements for each of the examples.


SYSTEM_REQUIREMENTS = {
    "gl_interop_plasma.py": has_display,
    "pytorch_example.py": lambda: (
        has_compute_capability_9_or_higher() and is_x86_64()
    ),  # PyTorch only provides CUDA support for x86_64
    "saxpy.py": has_compute_capability_9_or_higher,
    "simple_multi_gpu_example.py": has_multiple_devices,
    "strided_memory_view_cpu.py": is_not_windows,
    "thread_block_cluster.py": lambda: has_compute_capability_9_or_higher() and has_cuda_path(),
    "tma_tensor_map.py": has_cuda_path,
}


samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
sample_files = [os.path.basename(x) for x in glob.glob(samples_path + "**/*.py", recursive=True)]


def has_package_requirements_or_skip(example):
    with open(example, encoding="utf-8") as f:
        content = f.read()

    # The canonical regex as defined in PEP 723
    pep723 = re.search(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$", content)
    if not pep723:
        return

    metadata = {}
    for line in pep723.group("content").splitlines():
        line = line.lstrip("# ").rstrip()
        if not line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        metadata[key] = value

    if "dependencies" in metadata:
        dependencies = eval(metadata["dependencies"])  # noqa: S307
        for dependency in dependencies:
            name = re.match("[a-zA-Z0-9_-]+", dependency)
            try:
                importlib.metadata.distribution(name.string)
            except importlib.metadata.PackageNotFoundError:
                pytest.skip(f"Skipping {example} due to missing package requirement: {name}")


@pytest.mark.parametrize("example", sample_files)
def test_example(example):
    example_path = os.path.join(samples_path, example)
    has_package_requirements_or_skip(example_path)

    system_requirement = SYSTEM_REQUIREMENTS.get(example, lambda: True)
    if not system_requirement():
        pytest.skip(f"Skipping {example} due to unmet system requirement")

    process = subprocess.run([sys.executable, example_path], capture_output=True)  # noqa: S603
    if process.returncode != 0:
        if process.stdout:
            print(process.stdout.decode())
        if process.stderr:
            print(process.stderr.decode(), file=sys.stderr)
        raise AssertionError(f"`{example}` failed ({process.returncode})")
