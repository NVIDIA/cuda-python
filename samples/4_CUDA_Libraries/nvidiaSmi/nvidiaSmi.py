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

# /// script
# dependencies = ["cuda-python>=13.0.0"]
# ///

"""
Mini nvidia-smi with cuda.bindings.nvml

A Python subset of the ``nvidia-smi`` command-line tool implemented against
the raw ``cuda.bindings.nvml`` module. Prints a compact table with:

  * driver version, CUDA driver version
  * per-GPU: index, name, persistence mode, PCI bus id, display state,
    ECC mode, fan speed, temperature, performance state,
    power usage / cap, memory used / total, GPU utilization, compute mode

This sample is the canonical low-level demo for
``cuda.bindings.nvml``. The high-level counterpart lives in
[`samples/systemInfo/`](../../systemInfo/), which wraps NVML through
``cuda.core.system``.

Note: fields like fan speed, power draw, ECC state and display state come
back as ``NvmlError`` on GPUs that don't report them (server SKUs, headless
setups). The sample tolerates those with per-field ``try/except`` and
prints ``N/A``.
"""

import sys

try:
    from cuda.bindings import nvml
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


##################################################################################
# FORMATTING HELPERS
# Utilities to help format the output table. See below for NVML usage.
##################################################################################


def format_size(bytes_val: int) -> str:
    """Format bytes as MiB, matching nvidia-smi's memory column."""
    return f"{bytes_val / (1024 * 1024):.0f}MiB"


LINES = [[[4, 27, 6], [18, 3], [20]], [[4, 6, 13, 13], [22], [9, 10]]]


class TableFormatter:
    def __init__(self, lines):
        self.formats, self.sizes, self.counts = zip(*[self._create_line_format(line) for line in lines])

    def _create_line_format(self, descriptor):
        parts = []
        sizes = []
        for section in descriptor:
            parts.append("| ")
            sizes.append(1)
            for i, align in enumerate(section):
                direct = ">" if i == len(section) - 1 else "<"
                parts.append(f"{{:{direct}{align}}} ")
                sizes[-1] += align + 1
        parts.append("|")
        return "".join(parts), sizes, sum(len(x) for x in descriptor)

    def print_line(self, char="-"):
        parts = ["+"]
        for size in self.sizes[0]:
            parts.append(char * size)
            parts.append("+")
        print("".join(parts))

    def print_values(self, *args):
        for line_format, count in zip(self.formats, self.counts):
            print(line_format.format(*args[:count]))
            args = args[count:]


def print_table(metadata, devices):
    formatter = TableFormatter(LINES)

    print("+-----------------------------------------------------------------------------------------+")
    print(
        f"| NVIDIA-MINI-SMI {metadata['driver_version']:<16} Driver Version: {metadata['driver_version']:<15} CUDA Version: {metadata['cuda_version']:<9}|"
    )
    formatter.print_line()
    print("| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |")
    print("| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |")
    formatter.print_line("=")

    for device in devices:
        formatter.print_values(
            str(device["index"]),
            device["name"],
            device["persistence"],
            device["bus_id"],
            device["display_active"],
            device["ecc_mode"],
            device["fan_speed"],
            device["temperature"],
            device["performance_state"],
            device["power"],
            device["memory"],
            device["utilization"],
            device["compute_mode"],
        )
        formatter.print_line()


##################################################################################
# NVML USAGE
##################################################################################


def collect_info():
    metadata = {}
    metadata["driver_version"] = nvml.system_get_driver_version()
    cuda_version_int = nvml.system_get_cuda_driver_version()
    cuda_major = cuda_version_int // 1000
    cuda_minor = (cuda_version_int % 1000) // 10
    metadata["cuda_version"] = f"{cuda_major}.{cuda_minor}"

    devices = []
    device_count = nvml.device_get_count_v2()

    for i in range(device_count):
        device = {"index": i}
        handle = nvml.device_get_handle_by_index_v2(i)
        device["name"] = nvml.device_get_name(handle)

        try:
            persistence = nvml.device_get_persistence_mode(handle)
            persistence_str = "On" if persistence == nvml.EnableState.FEATURE_ENABLED else "Off"
        except nvml.NvmlError:
            persistence_str = "N/A"
        device["persistence"] = persistence_str

        try:
            pci_info = nvml.device_get_pci_info_v3(handle)
            bus_id = pci_info.bus_id
        except nvml.NvmlError:
            bus_id = "N/A"
        device["bus_id"] = bus_id

        try:
            display_active = nvml.device_get_display_active(handle)
            disp_str = "On" if display_active == nvml.EnableState.FEATURE_ENABLED else "Off"
        except nvml.NvmlError:
            disp_str = "N/A"
        device["display_active"] = disp_str

        try:
            current, _ = nvml.device_get_ecc_mode(handle)
            ecc_str = "On" if current == nvml.EnableState.FEATURE_ENABLED else "Off"
        except nvml.NvmlError:
            ecc_str = "N/A"
        device["ecc_mode"] = ecc_str

        try:
            fan = nvml.device_get_fan_speed(handle)
            fan_str = f"{fan: >3}%"
        except nvml.NvmlError:
            fan_str = "N/A"
        device["fan_speed"] = fan_str

        try:
            temp = nvml.device_get_temperature_v(handle, nvml.TemperatureSensors.TEMPERATURE_GPU)
            temp_str = f"{temp}C"
        except nvml.NvmlError:
            temp_str = "N/A"
        device["temperature"] = temp_str

        try:
            perf_state = nvml.device_get_performance_state(handle)
            perf_str = f"P{perf_state}"
        except nvml.NvmlError:
            perf_str = "N/A"
        device["performance_state"] = perf_str

        try:
            power_usage = nvml.device_get_power_usage(handle)  # mW
            usage_str = f"{power_usage // 1000}W"
        except nvml.NvmlError:
            usage_str = "N/A"
        try:
            power_cap = nvml.device_get_power_management_limit(handle)  # mW
            cap_str = f"{power_cap // 1000}W"
        except nvml.NvmlError:
            cap_str = "N/A"
        device["power"] = f"{usage_str} / {cap_str}"

        try:
            mem_info = nvml.device_get_memory_info_v2(handle)
        except nvml.NvmlError:
            device["memory"] = "N/A"
        else:
            device["memory"] = f"{format_size(mem_info.used)} / {format_size(mem_info.total)}"

        try:
            util_rates = nvml.device_get_utilization_rates(handle)
        except nvml.NvmlError:
            device["utilization"] = "N/A"
        else:
            device["utilization"] = f"{util_rates.gpu: >3}%"

        try:
            compute_mode = nvml.device_get_compute_mode(handle)
        except nvml.NvmlError:
            comp_str = "N/A"
        else:
            if compute_mode == nvml.ComputeMode.COMPUTEMODE_DEFAULT:
                comp_str = "Default"
            elif compute_mode == nvml.ComputeMode.COMPUTEMODE_EXCLUSIVE_PROCESS:
                comp_str = "E. Process"
            elif compute_mode == nvml.ComputeMode.COMPUTEMODE_PROHIBITED:
                comp_str = "Prohibited"
            else:
                comp_str = "Unknown"
        device["compute_mode"] = comp_str

        devices.append(device)

    return metadata, devices


def main():
    try:
        nvml.init_v2()
    except nvml.NvmlError as e:
        print(f"Failed to initialize NVML: {e}", file=sys.stderr)
        return 1

    try:
        metadata, devices = collect_info()
        print_table(metadata, devices)
    finally:
        nvml.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
