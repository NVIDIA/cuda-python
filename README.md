# CUDA-Python

CUDA Python is the home for accessing NVIDIA’s CUDA platform from Python. It consists of multiple components:

* [cuda.core](https://nvidia.github.io/cuda-python/cuda-core/latest): Pythonic access to CUDA Runtime and other core functionalities
* [cuda.bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest): Low-level Python bindings to CUDA C APIs
* [cuda.cooperative](https://nvidia.github.io/cccl/cuda_cooperative/): Pythonic exposure of CUB cooperative algorithms
* [cuda.parallel](https://nvidia.github.io/cccl/cuda_parallel/): Pythonic exposure of Thrust parallel algorithms

For access to NVIDIA Math Libraries, please refer to [nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest).

CUDA Python is currently undergoing an overhaul to improve existing and bring up new components. All of the previously available functionalities from the cuda-python package will continue to be available, please refer to the [cuda.bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest) documentation for installation guide and further detail.

## CUDA-Python as a metapackage

CUDA-Python is structured to become a metapackage that contains a collection of subpackages. Each subpackage is versioned independently, allowing installation of each component as needed.

### Subpackage: `cuda.core`

The `cuda.core` package offers idiomatic, pythonic access to CUDA Runtime and other functionalities.

The goals are to

1. Provide **idiomatic (“pythonic”)** access to CUDA Driver/Runtime
2. Focus on **developer productivity** by ensuring end-to-end CUDA development can be performed quickly and entirely in Python
3. **Avoid homegrown** Python abstractions for CUDA for new Python GPU libraries starting from scratch
4. **Ease** developer **burden of maintaining** and catching up with latest CUDA features
5. **Flatten the learning curve** for current and future generations of CUDA developers

### Subpackage: `cuda.bindings`

The `cuda.bindings` package is a standard set of low-level interfaces, providing full coverage of and access to the CUDA host APIs from Python.

The list of available interfaces are:

* CUDA Driver
* CUDA Runtime
* NVRTC
* nvJitLink
