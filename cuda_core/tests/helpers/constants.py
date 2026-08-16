# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants shared across the cuda_core test suite."""

# Cap for memory pools created by tests. A pool created without an explicit
# max_size instead reserves a system-dependent window that scales with
# installed device memory -- hundreds of GiB on large-memory GPUs. The
# per-process virtual address budget is bounded (~1 TB on Windows MCDM), and a
# reservation is not returned until the pool is torn down and its
# stream-ordered frees retire, so oversized windows accumulate across a session
# and eventually starve later pool creations with CUDA_ERROR_OUT_OF_MEMORY
# (issue #2381). See AGENTS.md in the tests directory.
POOL_SIZE = 2097152  # 2 MiB
