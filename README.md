# cuda-python

CUDA Python is the home for accessing NVIDIA’s CUDA platform from Python. It consists of multiple components:

* [cuda.core](https://nvidia.github.io/cuda-python/cuda-core/latest): Pythonic access to CUDA Runtime and other core functionality
* [cuda.bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest): Low-level Python bindings to CUDA C APIs
* [cuda.pathfinder](https://nvidia.github.io/cuda-python/cuda-pathfinder/latest): Utilities for locating CUDA components installed in the user's Python environment
* [cuda.coop](https://nvidia.github.io/cccl/python/coop): A Python module providing CCCL's reusable block-wide and warp-wide *device* primitives for use within Numba CUDA kernels
* [cuda.compute](https://nvidia.github.io/cccl/python/compute): A Python module for easy access to CCCL's highly efficient and customizable parallel algorithms, like `sort`, `scan`, `reduce`, `transform`, etc. that are callable on the *host*
* [numba.cuda](https://nvidia.github.io/numba-cuda/): A Python DSL that exposes CUDA **SIMT** programming model and compiles a restricted subset of Python code into CUDA kernels and device functions
* [cuda.tile](https://docs.nvidia.com/cuda/cutile-python/): A new Python DSL that exposes CUDA **Tile** programming model and allows users to write NumPy-like code in CUDA kernels
* [nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest): Pythonic access to NVIDIA CPU & GPU Math Libraries, with [*host*](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html#host-apis), [*device*](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html#device-apis), and [*distributed*](https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html) APIs. It also provides low-level Python bindings to host C APIs ([nvmath.bindings](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html)).
* [nvshmem4py](https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/index.html): Pythonic interface to the NVSHMEM library, enabling Python applications to leverage NVSHMEM's high-performance PGAS (Partitioned Global Address Space) programming model for GPU-accelerated computing
* [Nsight Python](https://docs.nvidia.com/nsight-python/index.html): Python kernel profiling interface that automates performance analysis across multiple kernel configurations using NVIDIA Nsight Tools
* [CUPTI Python](https://docs.nvidia.com/cupti-python/): Python APIs for creation of profiling tools that target CUDA Python applications via the CUDA Profiling Tools Interface (CUPTI)
* [Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub): Open-source learning materials related to GPU computing. You will find user guides, tutorials, and other works freely available for all learners interested in GPU computing.

CUDA Python is currently undergoing an overhaul to improve existing and introduce new components. All of the previously available functionality from the `cuda-python` package will continue to be available, please refer to the [cuda.bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest) documentation for installation guide and further detail.

## cuda-python as a metapackage

`cuda-python` is being restructured to become a metapackage that contains a collection of subpackages. Each subpackage is versioned independently, allowing installation of each component as needed.

### Subpackage: `cuda.core`

The `cuda.core` package offers idiomatic, Pythonic access to CUDA Runtime and other functionalities.

The goals are to

1. Provide **idiomatic ("Pythonic")** access to CUDA Driver, Runtime, and JIT compiler toolchain
2. Focus on **developer productivity** by ensuring end-to-end CUDA development can be performed quickly and entirely in Python
3. **Avoid homegrown** Python abstractions for CUDA for new Python GPU libraries starting from scratch
4. **Ease** developer **burden of maintaining** and catching up with latest CUDA features
5. **Flatten the learning curve** for current and future generations of CUDA developers

### Subpackage: `cuda.bindings`

The `cuda.bindings` package is a standard set of low-level interfaces, providing full coverage of and access to the CUDA host APIs from Python.

The list of available interfaces is:

* CUDA Driver
* CUDA Runtime
* NVRTC
* nvJitLink
* NVVM
* cuFile
