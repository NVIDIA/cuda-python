// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

#ifndef CUDA_CORE_STRIDED_COPY_IMPL_ELEMENTWISE_H
#define CUDA_CORE_STRIDED_COPY_IMPL_ELEMENTWISE_H

#include "array_view.h"
#include "type_utils.h"
#include "utils.h"
#include "vec.h"

namespace cuda_core
{

namespace detail
{

template <typename coords_t, typename stride_t = typename coords_t::type>
DEV coords_t unravel_idx(const stride_t flat_idx, const coords_t shape)
{
    constexpr int ndim = coords_t::ndim;
    if constexpr (ndim <= 0)
    {
        return {};
    }
    else if constexpr (ndim == 1)
    {
        return {flat_idx};
    }
    else if constexpr (ndim > 1)
    {

        // the extents cannot be negative and the arithmetic on unsigned integer
        // is noticeably faster
        using u_stride_t = typename type_traits::unsign<stride_t>::type;
        u_stride_t u_flat_idx = flat_idx;
        coords_t unraveled_coords;
#pragma unroll
        for (int i = ndim - 1; i >= 1; i--)
        {
            u_stride_t extent = shape[i];
            if (extent & (extent - 1))
            {
                u_stride_t next_flat_idx = u_flat_idx / extent;
                unraveled_coords[i] = u_flat_idx - next_flat_idx * extent;
                u_flat_idx = next_flat_idx;
            }
            else
            {
                unraveled_coords[i] = u_flat_idx & (extent - 1);
                u_flat_idx >>= ffs(extent) - 1;
            }
        }
        unraveled_coords[0] = u_flat_idx;
        return unraveled_coords;
    }
}

} // namespace detail

template <typename stride_t, bool _needs_grid_stride_loop>
struct element_indexer
{
    // stride_t can be 32-bit integer for tensor_volume and gridDim * blockDim up
    // to INT_MAX, this way unsigned x < INT_MAX; x += INT_MAX cannot overflow
    using ustride_t = typename type_traits::unsign<stride_t>::type;
    static constexpr bool needs_grid_stride_loop = _needs_grid_stride_loop;

    constexpr HOST_DEV element_indexer(const stride_t tensor_volume)
        : tensor_volume(tensor_volume) {}

    template <typename Cb>
    DEV void with_grid_stride_loop(Cb &&cb) const
    {
        // early cast the special indexing variables to the desired integer width
        // type to avoid arithmetic on 32-bit integers when 64-bit stride_t is used
        const ustride_t thread_idx = threadIdx.x;
        const ustride_t block_idx = blockIdx.x;
        const ustride_t block_dim = blockDim.x;
        if constexpr (!needs_grid_stride_loop)
        {
            const ustride_t x = block_idx * block_dim + thread_idx;
            if (x < tensor_volume)
            {
                cb(x);
            }
        }
        else if constexpr (needs_grid_stride_loop)
        {
            const ustride_t grid_dim = gridDim.x;
            const ustride_t grid_size = grid_dim * block_dim;
            for (ustride_t x = block_idx * block_dim + thread_idx; x < tensor_volume;
                 x += grid_size)
            {
                cb(x);
            }
        }
    }

    ustride_t tensor_volume;
};

template <bool has_equal_shapes, typename dst_array_view_t,
          typename src_array_view_t, typename grid_indexer_t>
struct elementwise_copy_impl
{
    using stride_t = typename dst_array_view_t::stride_t;

    DEV void operator()(const dst_array_view_t &&dst_view,
                        const src_array_view_t &&src_view,
                        const grid_indexer_t &&grid_helper)
    {
        grid_helper.with_grid_stride_loop([=](const stride_t flat_element_idx)
                                          {
      const auto dst_coords =
          detail::unravel_idx(flat_element_idx, dst_view.shape());
      const auto src_coords =
          cond_val(bconst<has_equal_shapes>(), dst_coords,
                   detail::unravel_idx(flat_element_idx, src_view.shape()));
      dst_view[dst_coords] = src_view[src_coords]; });
    }
};

} // namespace cuda_core

#endif // CUDA_CORE_STRIDED_COPY_IMPL_ELEMENTWISE_H
