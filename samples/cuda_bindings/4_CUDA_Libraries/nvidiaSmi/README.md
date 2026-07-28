# Sample: Mini nvidia-smi via NVML (Python)

## Description

A Python subset of the ``nvidia-smi`` command-line tool, implemented against
the raw ``cuda.bindings.nvml`` module. Prints a compact table with:

- Driver version and CUDA driver version
- Per-GPU: index, name, persistence mode, PCI bus id, display state, ECC
  mode, fan speed, temperature, performance state, power usage / cap,
  memory used / total, GPU utilization, compute mode

This is the canonical low-level demo for ``cuda.bindings.nvml``. The
high-level counterpart is [`samples/cuda_core/systemInfo/`](../../../cuda_core/systemInfo/), which
uses ``cuda.core.system`` (which itself wraps NVML).

Fields that aren't supported on a particular GPU (e.g. fan speed on server
SKUs, display state on headless nodes) are caught with ``NvmlError`` and
printed as ``N/A`` rather than failing the sample.

## What You'll Learn

- Initializing and shutting down NVML via ``nvml.init_v2()`` / ``nvml.shutdown()``
- Enumerating devices with ``device_get_count_v2`` /
  ``device_get_handle_by_index_v2``
- Querying every user-visible field the C ``nvidia-smi`` tool prints in its
  default output
- Gracefully tolerating fields that a given GPU does not expose via NVML

## Key Libraries

- [`cuda.bindings.nvml`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/nvml.html) - raw NVML bindings

## Key APIs

### From `cuda.bindings.nvml`

- `init_v2` / `shutdown`
- `system_get_driver_version` / `system_get_cuda_driver_version`
- `device_get_count_v2` / `device_get_handle_by_index_v2`
- `device_get_name`
- `device_get_persistence_mode` (+ `EnableState`)
- `device_get_pci_info_v3`
- `device_get_display_active`
- `device_get_ecc_mode`
- `device_get_fan_speed`
- `device_get_temperature_v` (+ `TemperatureSensors`)
- `device_get_performance_state`
- `device_get_power_usage` / `device_get_power_management_limit`
- `device_get_memory_info_v2`
- `device_get_utilization_rates`
- `device_get_compute_mode` (+ `ComputeMode`)

## Requirements

### Hardware

- Any NVIDIA GPU visible to NVML

### Software

- CUDA Toolkit 13.0 or newer (only for the accompanying NVML library; no
  CUDA runtime is used)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python nvidiaSmi.py
```

## Expected Output

Content depends on the system.

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-MINI-SMI 560.35.03      Driver Version: 560.35.03      CUDA Version: 12.6      |
+-----------------------------------------------------------------------------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
+============================================================================================+
|   0  NVIDIA GeForce RTX 4090           Off | 0000:01:00.0             Off |                Off |
| N/A  35C  P8                    12W / 450W |     0MiB / 24564MiB |    0%     Default |
+-----------------------------------------------------------------------------------------+
```

## Files

- `nvidiaSmi.py` - Python implementation using `cuda.bindings.nvml`
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.bindings` NVML API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/nvml.html)
- [`samples/cuda_core/systemInfo/`](../../../cuda_core/systemInfo/) - the high-level `cuda.core.system` equivalent
- [NVML Reference](https://docs.nvidia.com/deploy/nvml-api/index.html)
