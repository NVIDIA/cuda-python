# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from cuda.bindings import runtime as cudart


def test_graphics_api_smoketest():
    pyglet = pytest.importorskip("pyglet")

    tex = pyglet.image.Texture.create(512, 512)

    err, gfx_resource = cudart.cudaGraphicsGLRegisterImage(
        tex.id, tex.target, cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
    )
    error_name = cudart.cudaGetErrorName(err)[1].decode()
    if error_name == "cudaSuccess":
        assert int(gfx_resource) != 0
    else:
        assert error_name in ("cudaErrorInvalidValue", "cudaErrorUnknown")


def test_cuda_register_image_invalid():
    """Exercise cudaGraphicsGLRegisterImage with dummy handle only using CUDA runtime API."""
    fake_gl_texture_id = 1
    fake_gl_target = 0x0DE1
    flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard

    err, resource = cudart.cudaGraphicsGLRegisterImage(fake_gl_texture_id, fake_gl_target, flags)
    err_name = cudart.cudaGetErrorName(err)[1].decode()
    err_str = cudart.cudaGetErrorString(err)[1].decode()

    if err == 0:
        cudart.cudaGraphicsUnregisterResource(resource)
        raise AssertionError("Expected error from invalid GL texture ID")
