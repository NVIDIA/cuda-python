# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the samples owned by cuda.core as part of its test suite."""

from __future__ import annotations

import pytest
from cuda_python_test_helpers.sample_runner import SampleSuite
from run_samples import DEFAULT_CONFIG, DEFAULT_SAMPLES_DIR, SAMPLE_NAMESPACE

_SUITE = SampleSuite(DEFAULT_SAMPLES_DIR, SAMPLE_NAMESPACE, DEFAULT_CONFIG)


# Samples launch full GPU workloads in their own subprocess, so keep them
# serialized when the suite is run under pytest-run-parallel.
# For sample_key values: cuda_core/binarySearch, cuda_core/cudaComputeLambdas, cuda_core/customPyTorchKernel,
# cuda_core/customTensorFlowKernel, cuda_core/fftSignalAnalysis, cuda_core/glInteropFluidNumbaCudaMlir,
# cuda_core/glInteropFluid, cuda_core/glInteropMipmapLod, cuda_core/glInteropPlasma, cuda_core/greenContext,
# cuda_core/matrixMulSharedMem, cuda_core/multiGPUGradientAverage, cuda_core/numpyVsCupy, cuda_core/pageRank,
# cuda_core/parallelReduction, cuda_core/prefixSum, cuda_core/reductionMultiBlockCG,
# cuda_core/simplePrint, cuda_core/tmaTensorMap:
# This test is not currently run in any CI job
@pytest.mark.parallel_threads_limit(1)
@pytest.mark.parametrize("sample_key", _SUITE.keys)
@pytest.mark.samples
@pytest.mark.agent_authored(model="gpt-5")
def test_sample(sample_key: str) -> None:
    _SUITE.check(sample_key)
