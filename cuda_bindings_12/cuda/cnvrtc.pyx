# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cynvrtc cimport *
from cuda.bindings import cynvrtc
__pyx_capi__ = cynvrtc.__pyx_capi__
del cynvrtc
