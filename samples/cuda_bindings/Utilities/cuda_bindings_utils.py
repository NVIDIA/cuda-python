# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Common helpers for cuda-bindings-flavored samples.

Provides small utilities used by samples that program directly against
``cuda.bindings.driver`` / ``cuda.bindings.runtime`` / ``cuda.bindings.nvrtc``
/ ``cuda.bindings.nvml``, so each sample stays focused on the concept it
demonstrates instead of the boilerplate:

  * ``check_cuda_errors(result)`` -- unwrap the ``(status, *values)`` tuple
    returned by any ``cuda.bindings`` call, raising on error.
  * ``KernelHelper`` -- compile a CUDA C++ source string with NVRTC and load
    the resulting cubin/PTX as a module.
  * ``find_cuda_device()`` / ``find_cuda_device_drv()`` -- pick a CUDA device,
    honoring an optional ``--device=<id>`` CLI flag.
  * ``check_cmd_line_flag(flag)`` / ``get_cmd_line_argument_int(flag)`` --
    minimal CLI flag helpers used by the upstream cuda-samples style.
  * ``requirement_not_met(msg)`` -- print ``msg`` to stderr and exit with the
    orchestrator-recognized WAIVED status (exit code 2).
  * ``check_compute_capability_too_low(dev_id, (major, minor))`` -- waive when
    the current device is below a required compute capability.

These helpers were adapted from the private example-helpers module that
used to ship with ``cuda.bindings`` before the examples migration, with
one difference: ``requirement_not_met`` exits with code 2 (WAIVED,
recognized by the ``samples/`` orchestrator) rather than the historic
code 1.
"""

import sys

import numpy as np

from cuda import pathfinder
from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc
from cuda.bindings import runtime as cudart

EXIT_WAIVED = 2


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_cmd_line_option(option: str) -> tuple[str, str | None]:
    """Return an option name and its optional value, ignoring leading dashes."""
    name, separator, value = option.lstrip("-").partition("=")
    return name, value if separator else None


def check_cmd_line_flag(flag: str) -> bool:
    """Return whether ``flag`` is present, with or without leading dashes.

    A trailing ``=`` denotes a value-taking option. Matching the normalized
    option name exactly prevents flags such as ``--device-name`` from being
    mistaken for ``--device``.
    """
    normalized_flag = flag.lstrip("-")
    expects_value = normalized_flag.endswith("=")
    flag_name = normalized_flag.removesuffix("=")

    for arg in sys.argv[1:]:
        arg_name, value = _parse_cmd_line_option(arg)
        if arg_name == flag_name and (value is not None) == expects_value:
            return True
    return False


def get_cmd_line_argument_int(flag: str) -> int:
    """Return the integer following ``flag=`` in argv, or 0 if invalid or absent."""
    flag_name = flag.lstrip("-").removesuffix("=")
    for arg in sys.argv[1:]:
        arg_name, value = _parse_cmd_line_option(arg)
        if arg_name == flag_name and value is not None:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


# ---------------------------------------------------------------------------
# Waive helpers
# ---------------------------------------------------------------------------


def requirement_not_met(message: str) -> None:
    """Print ``message`` to stderr and exit with WAIVED status (exit code 2)."""
    print(message, file=sys.stderr)
    sys.exit(EXIT_WAIVED)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _cuda_get_error_enum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    if isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    if isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    raise RuntimeError(f"Unknown error type: {error}")


def check_cuda_errors(result):
    """Unwrap ``result = (status, *values)`` from a ``cuda.bindings`` call.

    Raises ``RuntimeError`` when ``status`` indicates a failure; otherwise
    returns the single value (for two-tuples), all trailing values (for
    three-or-more-tuples), or ``None`` (for one-tuples).
    """
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}({_cuda_get_error_enum(result[0])})")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def find_cuda_device() -> int:
    """Runtime-API device selection. Honors ``--device=<id>`` on argv."""
    dev_id = 0
    if check_cmd_line_flag("device="):
        dev_id = get_cmd_line_argument_int("device=")
    check_cuda_errors(cudart.cudaSetDevice(dev_id))
    return dev_id


def find_cuda_device_drv():
    """Driver-API device selection. Honors ``--device=<id>`` on argv."""
    dev_id = 0
    if check_cmd_line_flag("device="):
        dev_id = get_cmd_line_argument_int("device=")
    check_cuda_errors(cuda.cuInit(0))
    return check_cuda_errors(cuda.cuDeviceGet(dev_id))


def check_compute_capability_too_low(dev_id: int, required_cc_major_minor) -> None:
    """Waive if the current device is below the required compute capability."""
    cc_major = check_cuda_errors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev_id)
    )
    cc_minor = check_cuda_errors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, dev_id)
    )
    have = (cc_major, cc_minor)
    if have < tuple(required_cc_major_minor):
        requirement_not_met(
            f"CUDA device compute capability too low: have={have!r}, required={tuple(required_cc_major_minor)!r}"
        )


# ---------------------------------------------------------------------------
# NVRTC compile + module load helper
# ---------------------------------------------------------------------------


class KernelHelper:
    """Compile a CUDA C++ source string via NVRTC and load it as a module.

    On construction the source is compiled to a cubin (or PTX on older NVRTC),
    the resulting module is loaded, and ``get_function(name)`` returns the
    ``CUfunction`` for the given kernel entry point.

    The NVRTC include path is populated via ``cuda.pathfinder`` so kernels
    that ``#include`` CUDA headers (e.g. ``cooperative_groups.h``) resolve
    against the currently installed toolkit.
    """

    def __init__(self, code: str, dev_id: int) -> None:
        include_dirs = []
        for libname in ("cudart", "cccl"):
            hdr_dir = pathfinder.find_nvidia_header_directory(libname)
            if hdr_dir is None:
                requirement_not_met(f'pathfinder.find_nvidia_header_directory("{libname}") returned None')
            include_dirs.append(hdr_dir)

        prog = check_cuda_errors(nvrtc.nvrtcCreateProgram(str.encode(code), b"sourceCode.cu", 0, None, None))

        # Initialize CUDA runtime (needed on the very first device call).
        check_cuda_errors(cudart.cudaFree(0))

        major = check_cuda_errors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev_id)
        )
        minor = check_cuda_errors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, dev_id)
        )
        _, nvrtc_minor = check_cuda_errors(nvrtc.nvrtcVersion())
        use_cubin = nvrtc_minor >= 1
        prefix = "sm" if use_cubin else "compute"
        arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

        opts = [
            b"--fmad=true",
            arch_arg,
            b"--std=c++17",
            b"-default-device",
        ]
        for inc_dir in include_dirs:
            opts.append(f"--include-path={inc_dir}".encode())

        try:
            check_cuda_errors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except RuntimeError as err:
            log_size = check_cuda_errors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * log_size
            check_cuda_errors(nvrtc.nvrtcGetProgramLog(prog, log))
            print(log.decode(), file=sys.stderr)
            print(err, file=sys.stderr)
            sys.exit(1)

        if use_cubin:
            data_size = check_cuda_errors(nvrtc.nvrtcGetCUBINSize(prog))
            data = b" " * data_size
            check_cuda_errors(nvrtc.nvrtcGetCUBIN(prog, data))
        else:
            data_size = check_cuda_errors(nvrtc.nvrtcGetPTXSize(prog))
            data = b" " * data_size
            check_cuda_errors(nvrtc.nvrtcGetPTX(prog, data))

        self.module = check_cuda_errors(cuda.cuModuleLoadData(np.char.array(data)))

    def get_function(self, name: bytes):
        return check_cuda_errors(cuda.cuModuleGetFunction(self.module, name))
