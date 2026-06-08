# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared cuda.core setup for the latency benchmarks.

Holds the one Device/stream/ObjectCode instance that every bench module
reuses. No state is shared with the cuda_bindings suite — each suite
runs standalone in its own process.
"""

import atexit

from cuda.core import Device, Program, ProgramOptions

_device: Device | None = None
_modules: list = []


def ensure_device() -> Device:
    """Return the primary Device, initializing it on first call."""
    global _device
    if _device is not None:
        return _device
    dev = Device()
    dev.set_current()
    _device = dev
    return dev


def register_module(module) -> object:
    """Keep a reference to an ObjectCode so its kernels stay alive."""
    _modules.append(module)
    return module


def compile_module(kernel_source: str, name_expressions: tuple[str, ...]) -> object:
    """Compile a CUDA C++ source with NVRTC via cuda.core.Program.

    name_expressions must list every __global__ function the caller
    intends to fetch via ObjectCode.get_kernel().
    """
    dev = ensure_device()
    options = ProgramOptions(arch=f"sm_{dev.arch}", fma=False)
    prog = Program(kernel_source, code_type="c++", options=options)
    return register_module(prog.compile("cubin", name_expressions=name_expressions))


def cleanup() -> None:
    _modules.clear()


atexit.register(cleanup)
