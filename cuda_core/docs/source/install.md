# Installation

## Runtime Requirements

`cuda.core` is supported on all platforms that CUDA is supported. Specific
dependencies are as follows:

|                   | CUDA 11      | CUDA 12     |
|------------------ | ------------ | ----------- |
| CUDA Toolkit [^1] | 11.2 - 11.8  | 12.0 - 12.6 |
| Driver            | 450.80.02+ (Linux), 452.39+ (Windows) | 525.60.13+ (Linux), 527.41+ (Windows) |

[^1]: Including `cuda-python`.


## Installing from Source

```console
$ git clone https://github.com/NVIDIA/cuda-python
$ cd cuda-python/cuda_core
$ pip install .
```
For now `cuda-python` (`cuda-bindings` later) 11.x or 12.x is a required dependency.
