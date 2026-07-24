.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Examples
========

The canonical, runnable examples for ``cuda.core`` live under
:samples:`samples/cuda_core/ </>` in the cuda-python repository. Each sample is
a self-contained directory with its own ``README.md``, ``requirements.txt``,
and PEP 723 dependency block, and every sample is exercised as part of the
``cuda.core`` test suite.

Getting started
---------------

- :sample:`vectorAdd <vectorAdd/>`
  compiles and launches a templated vector-add kernel and demonstrates
  ``Device.allocate()`` and ``name_expressions``.
- :sample:`deviceQuery <deviceQuery/>`
  enumerates every CUDA device with an ``nvidia-smi``-style summary and an
  optional ``--verbose`` mode for the long-tail property list.
- :sample:`systemInfo <systemInfo/>`
  reports system-wide CUDA information (driver / runtime / NVML).

Kernels and launch configuration
--------------------------------

- :sample:`launchConfigTuning <launchConfigTuning/>`
  explores how launch-configuration choices affect kernel performance.
- :sample:`threadBlockCluster <threadBlockCluster/>`
  demonstrates ``LaunchConfig(cluster=...)`` on Hopper-class GPUs.
- :sample:`tmaTensorMap <tmaTensorMap/>`
  uses Tensor Memory Accelerator descriptors for bulk data movement.
- :sample:`greenContext <greenContext/>`
  partitions SMs across kernels with green contexts.
- :sample:`kernelNsysProfile <kernelNsysProfile/>`
  annotates kernel launches with NVTX markers for Nsight Systems.

Memory management and interop
-----------------------------

- :sample:`memoryResources <memoryResources/>`
  covers ``DeviceMemoryResource``, ``PinnedMemoryResource``,
  ``ManagedMemoryResource``, and ``GraphMemoryResource``, plus configurable
  resource options.
- :sample:`copyImageArraytoGPU <copyImageArraytoGPU/>`
  stages host-to-device copies through ``PinnedMemoryResource``.
- :sample:`blurImageUnifiedMemory <blurImageUnifiedMemory/>`
  applies a stencil kernel over unified memory.
- :sample:`stridedMemoryViewConstructors <stridedMemoryViewConstructors/>`
  walks through the four ``StridedMemoryView.from_*`` constructors.
- :sample:`stridedMemoryViewCpu <stridedMemoryViewCpu/>`
  dispatches to a JIT-compiled CPU function via ``cffi``.
- :sample:`stridedMemoryViewGpu <stridedMemoryViewGpu/>`
  dispatches to an NVRTC-compiled GPU kernel through the same decorator.
- :sample:`ipcMemoryPool <ipcMemoryPool/>`
  shares an IPC-enabled ``DeviceMemoryResource`` across processes.

CUDA graphs and linking
-----------------------

- :sample:`cudaGraphs <cudaGraphs/>`
  captures and replays a multi-kernel graph, then reuses it via ``Graph.update()``.
- :sample:`jitLtoLinking <jitLtoLinking/>`
  links two device modules with PTX vs LTO and swaps in a runtime plug-in.

Framework interop and compute algorithms
----------------------------------------

- :sample:`customPyTorchKernel <customPyTorchKernel/>`
  launches a cuda.core kernel from a PyTorch autograd function.
- :sample:`customTensorFlowKernel <customTensorFlowKernel/>`
  wires a cuda.core kernel into a TensorFlow custom op.
- :sample:`numpyVsCupy <numpyVsCupy/>`
  compares NumPy and CuPy execution paths side-by-side.
- :sample:`fftSignalAnalysis <fftSignalAnalysis/>`
  runs a CuPy FFT-based signal analysis pipeline.
- :sample:`binarySearch <binarySearch/>`,
  :sample:`prefixSum <prefixSum/>`,
  :sample:`reduction <reduction/>`,
  :sample:`parallelReduction <parallelReduction/>`,
  :sample:`reductionMultiBlockCG <reductionMultiBlockCG/>`,
  :sample:`parallelHistogram <parallelHistogram/>`,
  :sample:`blockwiseSum <blockwiseSum/>`,
  :sample:`matrixMulSharedMem <matrixMulSharedMem/>`,
  :sample:`cudaComputeLambdas <cudaComputeLambdas/>`,
  :sample:`pageRank <pageRank/>` -- classic parallel-algorithm building blocks.

Multi-GPU and streams
---------------------

- :sample:`simpleMultiGpu <simpleMultiGpu/>`
  runs independent kernels on two GPUs in the same process.
- :sample:`simpleP2P <simpleP2P/>`
  demonstrates peer-to-peer memory access between GPUs.
- :sample:`multiGPUGradientAverage <multiGPUGradientAverage/>`
  averages gradients across GPUs with MPI.
- :sample:`streamingCopyComputeOverlap <streamingCopyComputeOverlap/>`
  overlaps copies and compute across multiple streams.
- :sample:`simpleZeroCopy <simpleZeroCopy/>`
  runs kernels against zero-copy mapped host memory.
- :sample:`processCheckpoint <processCheckpoint/>`
  checkpoints and restores CUDA process state.

Graphics interop
----------------

- :sample:`glInteropPlasma <glInteropPlasma/>`
  writes CUDA-generated pixels into an OpenGL PBO with zero CPU round-trip.

Simple utilities
----------------

- :sample:`simplePrint <simplePrint/>`
  minimal kernel that prints from the device.
