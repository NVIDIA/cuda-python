# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

class StreamWrapper:
    """
    A wrapper around Stream for testing IsStreamT conversions.
    """

    def __init__(self, stream):
        self._stream = stream

    def __cuda_stream__(self):
        return self._stream.__cuda_stream__()

