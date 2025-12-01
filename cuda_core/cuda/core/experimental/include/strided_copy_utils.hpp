// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

#ifndef CUDA_CORE_STRIDED_COPY_UTILS_HPP
#define CUDA_CORE_STRIDED_COPY_UTILS_HPP

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "layout.hpp"
#include "strided_copy/args.h"

template <int N>
void _get_strided_copy_args_ndim(
    std::unique_ptr<void, std::function<void(void *)>> &args,
    void *dst_ptr, const void *src_ptr,
    int dst_ndim, int src_ndim,
    int64_t *dst_shape, int64_t *src_shape,
    int64_t *dst_strides, int64_t *src_strides,
    int64_t grid_arg)
{
    using uptr_t = std::unique_ptr<cuda_core::KernelArgs<N>, std::function<void(void *)>>;
    uptr_t ptr{new cuda_core::KernelArgs<N>, [](void *p)
               { delete (static_cast<cuda_core::KernelArgs<N> *>(p)); }};
    ptr->dst_ptr = dst_ptr;
    ptr->src_ptr = src_ptr;
    for (int i = 0; i < dst_ndim; i++)
    {
        ptr->dst_shape[i] = dst_shape[i];
        ptr->dst_strides[i] = dst_strides[i];
    }
    for (int i = 0; i < src_ndim; i++)
    {
        ptr->src_shape[i] = src_shape[i];
        ptr->src_strides[i] = src_strides[i];
    }
    ptr->grid_arg = grid_arg;
    args = std::move(ptr);
}

template <typename Cb, int i = 1, int max_ndim = STRIDED_LAYOUT_MAX_NDIM>
void _call_with_static_ndim(int ndim, Cb &&cb)
{
    if constexpr (i > max_ndim)
    {
        throw std::runtime_error("unsupported ndim");
    }
    else if constexpr (i <= max_ndim)
    {
        if (ndim == i)
        {
            cb(std::integral_constant<int, i>());
        }
        else
        {
            _call_with_static_ndim<Cb, i + 1, max_ndim>(ndim, std::move(cb));
        }
    }
}

void inline _get_strided_copy_args(
    std::unique_ptr<void, std::function<void(void *)>> &args,
    void *dst_ptr, const void *src_ptr,
    int dst_ndim, int src_ndim,
    int64_t *dst_shape, int64_t *src_shape,
    int64_t *dst_strides, int64_t *src_strides,
    int64_t grid_arg)
{
    int max_ndim = dst_ndim >= src_ndim ? dst_ndim : src_ndim;
    _call_with_static_ndim(max_ndim, [&](auto static_ndim_holder)
                           {
        constexpr int static_ndim = decltype(static_ndim_holder)::value;
        _get_strided_copy_args_ndim<static_ndim>(
            args, dst_ptr, src_ptr,
            dst_ndim, src_ndim,
            dst_shape, src_shape,
            dst_strides, src_strides,
            grid_arg); });
}

#endif // CUDA_CORE_STRIDED_COPY_UTILS_HPP
