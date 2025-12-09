# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module exists to compile _cpp/resource_handles.cpp into a shared library.
# The helper functions (native, intptr, py) are implemented as inline C++ functions
# in _cpp/resource_handles.hpp and declared as extern in _resource_handles.pxd.
