# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest
from cuda.bindings import runtime as cudart


def test_graphics_api_smoketest():
    _ = pytest.importorskip("PySide6")
    from PySide6 import QtGui, QtOpenGL

    class GLWidget(QtOpenGL.QOpenGLWindow):
        def initializeGL(self):
            self.m_texture = QtOpenGL.QOpenGLTexture(QtOpenGL.QOpenGLTexture.Target.Target2D)
            self.m_texture.setFormat(QtOpenGL.QOpenGLTexture.TextureFormat.RGBA8_UNorm)
            self.m_texture.setSize(512, 512)
            self.m_texture.allocateStorage()

            err, self.gfx_resource = cudart.cudaGraphicsGLRegisterImage(
                self.m_texture.textureId(),
                self.m_texture.target().value,
                cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
            )
            error_name = cudart.cudaGetErrorName(err)[1].decode()

            # We either have everything set up correctly and we get a gfx_resource,
            # or we get an error.  Either way, we know the API actually did something,
            # which is enough for this basic smoketest.
            if error_name == "cudaSuccess":
                assert int(self.gfx_resource) != 0
            else:
                assert error_name == "cudaErrorInvalidValue"

    app = QtGui.QGuiApplication([])
    win = GLWidget()
    win.initializeGL()
    win.show()
