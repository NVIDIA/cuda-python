# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the cuda.pathfinder filesystem benchmark suite."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUDA_BINDINGS_SUITE = HERE.parent / "cuda_bindings"

if str(CUDA_BINDINGS_SUITE) not in sys.path:
    sys.path.append(str(CUDA_BINDINGS_SUITE))

from runner.main import main

if __name__ == "__main__":
    main(
        bench_dir=HERE / "benchmarks",
        default_output=HERE / "results-python.json",
        module_name_prefix="cuda_pathfinder_bench",
        bench_filter_env_var="CUDA_PATHFINDER_BENCH_FILTER",
    )
