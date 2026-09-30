# CUDA Python Samples

The samples are grouped by the primary API they demonstrate:

- [`cuda_bindings/`](cuda_bindings/) contains low-level examples that call
  `cuda.bindings` driver, runtime, NVRTC, or NVML APIs directly.
- [`cuda_core/`](cuda_core/) contains high-level examples built primarily with
  `cuda.core` and `cuda.compute`.

Some `cuda.core` samples use individual `cuda.bindings` calls where the
high-level API does not yet expose the required operation. They remain in the
`cuda_core` collection because that is the API driving the sample workflow.

## Running a Sample

Each sample has its own `README.md` and `requirements.txt`. Run its install and
launch commands from that sample's directory. From the repository root, for
example:

```bash
cd samples/cuda_core/vectorAdd
pip install -r requirements.txt
python vectorAdd.py
```

Many samples require a supported NVIDIA GPU and CUDA Toolkit. See the
individual README for hardware, toolkit, and optional dependency requirements.

## Shared Utilities

Utilities are private implementation helpers for their respective sample
collections:

- [`cuda_bindings/Utilities/`](cuda_bindings/Utilities/) supports low-level
  bindings samples.
- [`cuda_core/Utilities/`](cuda_core/Utilities/) supports high-level core
  samples.

They are not installed as part of the CUDA Python packages and should not be
treated as public APIs.
