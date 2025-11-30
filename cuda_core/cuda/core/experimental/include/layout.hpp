// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

#ifndef CUDA_CORE_LAYOUT_HPP
#define CUDA_CORE_LAYOUT_HPP

#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>


#define STRIDED_LAYOUT_MAX_NDIM 64
#define AXIS_MASK_ALL 0xFFFFFFFFFFFFFFFEULL

inline int64_t _c_abs(int64_t x)
{
    return std::abs(x);
}

template <typename T>
void _swap(T &a, T &b) noexcept
{
    std::swap(a, b);
}

inline void _order_from_strides(std::vector<int>& indices, const int64_t* shape, const int64_t* strides, int ndim)
{
    indices.resize(ndim);
    std::iota(indices.begin(), indices.end(), 0);
    if (!strides) {
        return;
    }
    std::sort(indices.begin(), indices.end(),
              [&strides, &shape](int i, int j)
              {
                  int64_t stride_i = _c_abs(strides[i]);
                  int64_t stride_j = _c_abs(strides[j]);
                  if (stride_i != stride_j)
                  {
                      return stride_i > stride_j;
                  }
                  int64_t shape_i = shape[i];
                  int64_t shape_j = shape[j];
                  if (shape_i != shape_j)
                  {
                      return shape_i > shape_j;
                  }
                  return i < j;
              });
}

#endif // CUDA_CORE_LAYOUT_HPP
