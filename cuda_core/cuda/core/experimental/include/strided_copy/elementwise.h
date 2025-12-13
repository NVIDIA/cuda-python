// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

#ifndef CUDA_CORE_STRIDED_COPY_ELEMENTWISE_H
#define CUDA_CORE_STRIDED_COPY_ELEMENTWISE_H

#include "args.h"
#include "impl/array_view.h"
#include "impl/elementwise.h"
#include "impl/type_utils.h"
#include "impl/utils.h"
#include "impl/vec.h"

#define ELEMENTWISE_KERNEL(stride_t, dst_ndim, src_ndim, itemsize,                 \
                           needs_grid_stride_loop)                                 \
    extern "C"                                                                     \
    {                                                                              \
        constexpr int max_ndim = dst_ndim > src_ndim ? dst_ndim : src_ndim;        \
        void __global__ execute(const cuda_core::KernelArgs<max_ndim> args)        \
        {                                                                          \
            cuda_core::elementwise_copy<cuda_core::##stride_t, dst_ndim, src_ndim, \
                                        itemsize, needs_grid_stride_loop>          \
                kernel;                                                            \
            kernel(args);                                                          \
        }                                                                          \
    }

namespace cuda_core
{

template <typename stride_t, int dst_ndim, int src_ndim, int itemsize,
          int needs_grid_stride_loop>
struct elementwise_copy
{
    using dtype_t = opaque_t<itemsize>;
    using dst_coords_t = vec<dst_ndim, stride_t>;
    using src_coords_t = vec<src_ndim, stride_t>;
    using dst_array_view_t = array_view<dtype_t, dst_coords_t>;
    using src_array_view_t = array_view<const dtype_t, src_coords_t>;
    using grid_indexer_t = element_indexer<stride_t, needs_grid_stride_loop>;
    constexpr static bool has_equal_shapes = dst_ndim == src_ndim;
    constexpr static int ndim = dst_ndim > src_ndim ? dst_ndim : src_ndim;

    DEV void operator()(const KernelArgs<ndim> args) const
    {
        dst_array_view_t dst_array_view{static_cast<dtype_t *>(args.dst_ptr),
                                        dst_coords_t{args.dst_shape},
                                        dst_coords_t{args.dst_strides}};
        src_array_view_t src_array_view{static_cast<const dtype_t *>(args.src_ptr),
                                        src_coords_t{args.src_shape},
                                        src_coords_t{args.src_strides}};
        auto kernel = elementwise_copy_impl<has_equal_shapes, dst_array_view_t,
                                            src_array_view_t, grid_indexer_t>{};
        kernel(std::move(dst_array_view), std::move(src_array_view),
               grid_indexer_t{static_cast<stride_t>(args.grid_arg)});
    }
};

} // namespace cuda_core

#endif // CUDA_CORE_STRIDED_COPY_ELEMENTWISE_H
