# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental import Stream


class StreamWrapper:
    """
    A wrapper around Stream for testing IsStreamT conversions.
    """

    def __init__(self, stream: Stream):
        self._stream = stream

    def __cuda_stream__(self):
        return self._stream.__cuda_stream__()

    def close(self):
        self._stream.close()

    @property
    def handle(self):
        return self._stream.handle

    def sync(self):
        return self._stream.sync()
