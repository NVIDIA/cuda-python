# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# If we have subcategories of examples in the future, this file can be split along those lines

import glob
import os
import platform
import subprocess
import sys

import pytest

from cuda.core import Device, system

try:
    from cuda.bindings._test_helpers.pep723 import has_package_requirements_or_skip
except ImportError:
    # If the import fails, we define a dummy function that will cause all tests to be skipped.
    def has_package_requirements_or_skip(example):
        pytest.skip("PEP 723 test helper is not available")


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
            print(process.stdout.decode(errors="replace"))
        if process.stderr:
            print(process.stderr.decode(errors="replace"), file=sys.stderr)
        raise AssertionError(f"`{example}` failed ({process.returncode})")
