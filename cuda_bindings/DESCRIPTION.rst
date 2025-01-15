*******************************************************
cuda.bindings: Low-level CUDA interfaces
*******************************************************

`cuda.bindings` is a standard set of low-level interfaces, providing full coverage of and access to the CUDA host APIs from Python. Checkout the [Overview](https://nvidia.github.io/cuda-python/cuda-bindings/latest/overview.html) for the workflow and performance results.

Installation
============

`cuda.bindings` can be installed from:

* PyPI
* Conda (conda-forge/nvidia channels)
* Source builds

Differences between these options are described in [Installation](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html) documentation. Each package guarantees minor version compatibility.

Runtime Dependencies
====================

`cuda.bindings` is supported on all the same platforms as CUDA. Specific dependencies are as follows:

* Driver: Linux (450.80.02 or later) Windows (456.38 or later)
* CUDA Toolkit 12.x

Only the NVRTC and nvJitLink redistributable components are required from the CUDA Toolkit, which can be obtained via PyPI, Conda, or local installers (as described in the CUDA Toolkit [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) Installation Guides).
