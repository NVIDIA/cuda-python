# Copyright 2026 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


# ################################################################################
#
# This example demonstrates the core cuda.bindings.nvml functionality by
# implementing a subset of the NVIDIA System Management Interface (nvidia-smi)
# command line tool in Python.
#
# ################################################################################


# /// script
# dependencies = ["cuda_bindings>13.2.1"]
# ///

import sys

from cuda.bindings import nvml

##################################################################################
# FORMATTING HELPERS

# Utilities to help format the output table. See below for NVML usage.


def format_size(bytes_val: int) -> str:
    """Formats bytes to MiB."""
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
                if i == len(section) - 1:
                    direct = ">"
                else:
                    direct = "<"
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
# NVML USAGE EXAMPLES


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
        device = {}
        device["index"] = i

        handle = nvml.device_get_handle_by_index_v2(i)

        name = nvml.device_get_name(handle)
        device["name"] = name

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

        pwr_str = f"{usage_str} / {cap_str}"
        device["power"] = pwr_str

        try:
            mem_info = nvml.device_get_memory_info_v2(handle)
        except nvml.NvmlError:
            mem_str = "N/A"
        else:
            mem_used = format_size(mem_info.used)
            mem_total = format_size(mem_info.total)
            mem_str = f"{mem_used} / {mem_total}"

        device["memory"] = mem_str

        try:
            util_rates = nvml.device_get_utilization_rates(handle)
        except nvml.NvmlError:
            util_str = "N/A"
        else:
            gpu_util = util_rates.gpu
            util_str = f"{gpu_util: >3}%"

        device["utilization"] = util_str

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
        print(f"Failed to initialize NVML: {e}")
        sys.exit(1)

    try:
        metadata, devices = collect_info()
        print_table(metadata, devices)
    finally:
        nvml.shutdown()


if __name__ == "__main__":
    main()
