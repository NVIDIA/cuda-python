# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates building a 2D CUDA OpaqueArray, binding it as a
# bindless TextureObject, and sampling it from a kernel with both POINT-exact
# and LINEAR-interpolated coordinates.
#
# Texture coordinate convention (non-normalized): each texel (i, j) is centered
# at (i + 0.5, j + 0.5). So tex2D(tex, 0.5, 0.5) returns texel (0, 0) exactly,
# while tex2D(tex, 1.0, 0.5) returns the linear blend of texels (0, 0) and (1, 0).
# All test coordinates below are chosen with that half-pixel offset in mind.
#
# ################################################################################

# /// script
# dependencies = ["cuda-python>=13.0.0", "cuda-core>=1.0.0", "numpy>=1.24"]
# ///

import numpy as np

from cuda.core import (
    Device,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)
from cuda.core.texture import (
    OpaqueArrayOptions,
    ResourceDescriptor,
    TextureObjectOptions,
)
from cuda.core.typing import (
    AddressModeType,
    ArrayFormatType,
    FilterModeType,
    ReadModeType,
)

# Kernel reads N (x, y) coordinates from `coords` (interleaved float pairs) and
# writes tex2D<float>(tex, x, y) to out[i]. Compiled as C++ so the templated
# tex2D<float> overload resolves.
code = r"""
extern "C" __global__
void sample_texture(cudaTextureObject_t tex,
                    float *out,
                    const float *coords,
                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = coords[2 * i + 0];
    float y = coords[2 * i + 1];
    out[i] = tex2D<float>(tex, x, y);
}
"""


def main():
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()

    pinned_mr = LegacyPinnedMemoryResource()
    try:
        # Allocate a 2D OpaqueArray: shape=(W, H), single-channel float32.
        # Note: create_opaque_array takes shape=(width, height), so the host
        # buffer fed into copy_from must be laid out as H rows of W elements
        # (row-major), i.e. host_pattern.shape == (H, W).
        width, height = 16, 16
        with Device().create_opaque_array(
            OpaqueArrayOptions(
                shape=(width, height),
                format=ArrayFormatType.FLOAT32,
                num_channels=1,
            )
        ) as arr:
            # Plant a known pattern: pattern[y, x] = x + 100*y.
            # Cast to float32 so the byte count matches the array's storage.
            ys, xs = np.meshgrid(
                np.arange(height, dtype=np.float32),
                np.arange(width, dtype=np.float32),
                indexing="ij",
            )
            pattern = (xs + 100.0 * ys).astype(np.float32)
            assert pattern.shape == (height, width)
            arr.copy_from(pattern, stream=stream)

            # Build a linear-filtering, clamped, non-normalized texture.
            res_desc = ResourceDescriptor.from_opaque_array(arr)
            tex_desc = TextureObjectOptions(
                address_mode=AddressModeType.CLAMP,
                filter_mode=FilterModeType.LINEAR,
                read_mode=ReadModeType.ELEMENT_TYPE,
                normalized_coords=False,
            )
            with Device().create_texture_object(resource=res_desc, options=tex_desc) as tex:
                _run_kernel_and_verify(dev, stream, tex, pattern, width, height, pinned_mr)
    finally:
        stream.close()


def _run_kernel_and_verify(dev, stream, tex, pattern, width, height, pinned_mr):
    """Kernel launch + correctness check, isolated so the with-blocks in main()
    stay readable. Owns its own pinned-buffer cleanup."""
    coords_buf = None
    out_buf = None
    try:
        # Build the test coordinate list:
        # - Texel-center samples should return the exact planted value.
        # - Half-integer samples land between texels and exercise LINEAR
        #   filtering -- they should equal the average of the surrounding
        #   texels.
        center_samples = [
            (0.5, 0.5),  # -> pattern[0, 0] = 0
            (3.5, 0.5),  # -> pattern[0, 3] = 3
            (0.5, 4.5),  # -> pattern[4, 0] = 400
            (7.5, 9.5),  # -> pattern[9, 7] = 907
            (15.5, 15.5),  # -> pattern[15, 15] = 1515
        ]
        half_samples = [
            # (1.0, 0.5): blend of texels (0, 0) and (1, 0) -> 0.5
            (1.0, 0.5),
            # (0.5, 1.0): blend of texels (0, 0) and (0, 1) -> 50.0
            (0.5, 1.0),
            # (1.0, 1.0): blend of the 2x2 block at (0..1, 0..1) -> 50.5
            (1.0, 1.0),
            # (4.0, 5.0): blend of the 2x2 block at (3..4, 4..5) -> 453.5
            (4.0, 5.0),
        ]
        coords = np.array(center_samples + half_samples, dtype=np.float32)
        n = coords.shape[0]
        coords_flat = coords.reshape(-1)
        coords_nbytes = int(coords_flat.nbytes)
        out_nbytes = n * np.dtype(np.float32).itemsize

        # Use pinned host memory for inputs and outputs. Pinned allocations are
        # GPU-accessible (zero-copy), so the kernel can read coords directly
        # and we can read results without a separate device->host copy.
        coords_buf = pinned_mr.allocate(coords_nbytes)
        out_buf = pinned_mr.allocate(out_nbytes)
        coords_view = np.from_dlpack(coords_buf).view(dtype=np.float32)
        out_view = np.from_dlpack(out_buf).view(dtype=np.float32)
        coords_view[:] = coords_flat
        out_view[:] = 0.0

        # Compile the kernel as C++ (templated tex2D<float> requires this).
        program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
        prog = Program(code, code_type="c++", options=program_options)
        mod = prog.compile("cubin", name_expressions=("sample_texture",))
        kernel = mod.get_kernel("sample_texture")

        block = 64
        grid = (n + block - 1) // block
        config = LaunchConfig(grid=grid, block=block)
        # cudaTextureObject_t is a 64-bit handle; pass it as uint64 to be
        # unambiguous (a bare Python int would also work since intptr_t is
        # 8 bytes on 64-bit platforms).
        launch(
            stream,
            config,
            kernel,
            np.uint64(tex.handle),
            out_buf,
            coords_buf,
            np.int32(n),
        )
        stream.sync()
        results = np.asarray(out_view)

        # Verify texel-center samples (POINT-exact regardless of filter mode).
        n_center = len(center_samples)
        for i, (x, y) in enumerate(center_samples):
            expected = (x - 0.5) + 100.0 * (y - 0.5)
            got = float(results[i])
            assert np.isclose(got, expected, atol=1e-4), (
                f"center sample {i} at ({x}, {y}): expected {expected}, got {got}"
            )

        # Verify half-integer samples against the analytic mean of the 4
        # surrounding texels. Allow a small tolerance for the 1/256 fixed-point
        # weight quantization that hardware filtering performs.
        for j, (x, y) in enumerate(half_samples):
            idx = n_center + j
            # Surrounding integer texel coordinates: (xi, yi), (xi+1, yi),
            # (xi, yi+1), (xi+1, yi+1). With x = xi + 1, y = yi + 1 (e.g.
            # (1.0, 1.0)) the four neighbors are (0,0)..(1,1).
            xi = int(np.floor(x - 0.5))
            yi = int(np.floor(y - 0.5))
            tx = (x - 0.5) - xi
            ty = (y - 0.5) - yi
            corners = []
            for dy in (0, 1):
                for dx in (0, 1):
                    xv = min(max(xi + dx, 0), width - 1)
                    yv = min(max(yi + dy, 0), height - 1)
                    corners.append(pattern[yv, xv])
            v00, v10, v01, v11 = corners
            expected = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
            got = float(results[idx])
            assert np.isclose(got, expected, atol=1e-2), (
                f"half sample {j} at ({x}, {y}): expected {expected}, got {got}"
            )

        print("Texture sampling example completed successfully.")
        print(f"  texel-center samples verified: {n_center}")
        print(f"  half-integer samples verified: {len(half_samples)}")
    finally:
        if coords_buf is not None:
            coords_buf.close()
        if out_buf is not None:
            out_buf.close()


if __name__ == "__main__":
    main()
