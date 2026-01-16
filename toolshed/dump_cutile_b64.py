#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""
Embeds a sample cuTile kernel, executes it with CUDA_TILE_DUMP_BYTECODE=.,
loads the resulting .cutile file, and prints its base64-encoded content.
"""

import base64
import glob
import os
import sys

import cuda.tile as ct
import cupy


def _run_sample_cutile_kernel() -> None:
    # Import after env var setup so CUDA_TILE_DUMP_BYTECODE is honored.
    TILE_SIZE = 16

    @ct.kernel
    def vector_add_kernel(a, b, result):
        block_id = ct.bid(0)
        a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
        b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))

        result_tile = a_tile + b_tile
        ct.store(result, index=(block_id,), tile=result_tile)

    a = cupy.arange(128, dtype="float32")
    b = cupy.arange(128, dtype="float32")
    result = cupy.zeros_like(a)

    grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
    ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))

    cupy.cuda.get_current_stream().synchronize()

    assert result[-1] == 254


def main():
    # CUDA_TILE_DUMP_BYTECODE=. means dump to current directory
    os.environ["CUDA_TILE_DUMP_BYTECODE"] = "."

    try:
        _run_sample_cutile_kernel()
    except Exception as e:
        print(f"Sample kernel execution failed: {e}", file=sys.stderr)
        raise

    # Find the .cutile file in current directory
    cutile_files = glob.glob("./*.cutile")
    if not cutile_files:
        print("No .cutile file found in current directory", file=sys.stderr)
        sys.exit(1)

    # Use the most recently modified one if multiple exist
    cutile_path = max(cutile_files, key=os.path.getmtime)

    # Read the binary content
    with open(cutile_path, "rb") as f:
        binary_content = f.read()

    # Encode with base64 in ASCII mode
    b64_encoded = base64.b64encode(binary_content).decode("ascii")

    # Print with lines less than 79 characters, wrapped with quotes
    line_width = 76  # 78 - 2 for the quotes on both sides
    for i in range(0, len(b64_encoded), line_width):
        chunk = b64_encoded[i : i + line_width]
        print(f'"{chunk}"')


if __name__ == "__main__":
    main()
