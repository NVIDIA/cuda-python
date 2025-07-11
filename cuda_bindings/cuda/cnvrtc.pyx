# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.bindings.cynvrtc cimport *
from cuda.bindings import cynvrtc
__pyx_capi__ = cynvrtc.__pyx_capi__
del cynvrtc
