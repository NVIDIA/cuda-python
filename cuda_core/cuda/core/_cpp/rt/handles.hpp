// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Consumer umbrella, named only by _rt.pxd: the handle types, the inline
// accessors and the Python seam. Everything else reaches consumers through
// __pyx_capi__, never through a header.

#include "py.hpp"
#include "types.hpp"
