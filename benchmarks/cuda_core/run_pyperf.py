# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the cuda.core latency benchmark suite.

The pyperf runner lives in the cuda_bindings suite. Reuse it by putting
that directory on sys.path, then call main() with this suite's paths.
pyperf workers re-execute this script, so the sys.path tweak is done
before the worker can import anything.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUDA_BINDINGS_SUITE = HERE.parent / "cuda_bindings"

# Share the runner with cuda_bindings; keep cuda_core's own modules
# (benchmarks/, runtime.py) resolvable via the script's own directory.
if str(CUDA_BINDINGS_SUITE) not in sys.path:
    sys.path.append(str(CUDA_BINDINGS_SUITE))

from runner.main import main

if __name__ == "__main__":
    main(
        bench_dir=HERE / "benchmarks",
        default_output=HERE / "results-python.json",
        module_name_prefix="cuda_core_bench",
        bench_filter_env_var="CUDA_CORE_BENCH_FILTER",
    )
