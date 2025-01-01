# Installation

## Runtime Requirements

`cuda.core` is supported on all platforms that CUDA is supported. Specific
dependencies are as follows:

|                   | CUDA 11      | CUDA 12     |
|------------------ | ------------ | ----------- |
| CUDA Toolkit [^1] | 11.2 - 11.8  | 12.0 - 12.6 |
| Driver            | 450.80.02+ (Linux), 452.39+ (Windows) | 525.60.13+ (Linux), 527.41+ (Windows) |

[^1]: Including `cuda-python`.

`cuda.core` supports Python 3.9 - 3.12, on Linux (x86-64, arm64) and Windows (x86-64).

## Installing from PyPI

`cuda.core` works with `cuda.bindings` (part of `cuda-python`) 11 or 12. For example with CUDA 12:
```console
$ pip install cuda-core[cu12]
```
and likewise use `[cu11]` for CUDA 11.

Note that using `cuda.core` with NVRTC or nvJitLink installed from PyPI via `pip install` is currently
not supported. This will be fixed in a future release.

## Installing from Source

```console
$ git clone https://github.com/NVIDIA/cuda-python
$ cd cuda-python/cuda_core
$ pip install .
```
For now `cuda-python` (`cuda-bindings` later) 11.x or 12.x is a required dependency.
