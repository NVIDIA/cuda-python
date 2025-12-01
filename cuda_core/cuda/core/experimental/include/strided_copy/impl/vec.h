// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

#ifndef CUDA_CORE_STRIDED_COPY_IMPL_VEC_H_
#define CUDA_CORE_STRIDED_COPY_IMPL_VEC_H_

#include "utils.h"

namespace cuda_core
{

template <typename T, int N>
struct vec_base
{
    T v[N];

    template <typename... Components,
              typename = typename type_traits::enable_if<
                  sizeof...(Components) <= N &&
                  (type_traits::is_32_or_64_int<Components>::value && ...)>::type>
    HOST_DEV constexpr vec_base(Components... components) : v{T(components)...} {}

    template <typename U, typename = typename type_traits::enable_if<
                              type_traits::is_32_or_64_int<U>::value>::type>
    HOST_DEV constexpr vec_base(const U *ptr)
    {
        for (int i = 0; i < N; i++)
        {
            v[i] = ptr[i];
        }
    }

    HOST_DEV constexpr T &operator[](int i) { return v[i]; }
    HOST_DEV constexpr const T &operator[](int i) const { return v[i]; }
};

template <typename T>
struct vec_base<T, 0>
{
};

template <int N, typename T>
struct vec : vec_base<T, N>
{
    using base_t = vec_base<T, N>;
    using type = T;
    constexpr static int ndim = N;

    constexpr vec() = default;

    template <typename... Components>
    HOST_DEV constexpr vec(Components... components) : base_t{components...} {}

    template <typename U>
    HOST_DEV constexpr vec(const U *ptr, int ndim) : base_t(ptr, ndim) {}

    HOST_DEV constexpr int size() const { return ndim; }

    template <int K>
    HOST_DEV constexpr auto last(const iconst<K>) const
    {
        static_assert(K <= ndim);
        return slice(iconst<ndim - K>(), iconst<ndim>());
    }

    template <int K>
    HOST_DEV constexpr auto first(const iconst<K>) const
    {
        static_assert(K <= ndim);
        return slice(iconst<0>(), iconst<K>());
    }

    template <int start, int end>
    HOST_DEV constexpr vec<end - start, T> slice(const iconst<start>,
                                                 const iconst<end>) const
    {
        static_assert(start >= 0 && end <= ndim);
        constexpr int slice_ndim = end - start;
        static_assert(slice_ndim >= 0);
        if constexpr (slice_ndim != 0)
        {
            vec<slice_ndim, T> result;
#pragma unroll
            for (int i = 0; i < slice_ndim; i++)
            {
                result[i] = this->operator[](start + i);
            }
            return result;
        }
        return {};
    }
};

template <int N, int M, typename T>
HOST_DEV constexpr vec<N + M, T> cat(const vec<N, T> a, const vec<M, T> b)
{
    constexpr int ndim = N + M;
    if constexpr (ndim != 0)
    {
        vec<ndim, T> result;
        if constexpr (N > 0)
        {
#pragma unroll
            for (int i = 0; i < N; i++)
            {
                result[i] = a[i];
            }
        }
        if constexpr (M > 0)
        {
#pragma unroll
            for (int i = 0; i < M; i++)
            {
                result[N + i] = b[i];
            }
        }
        return result;
    }
    return {};
}

template <int N, typename T, typename Op>
HOST_DEV constexpr auto vector_bin_op(const vec<N, T> a, const vec<N, T> b,
                                      Op &&op)
{
    static_assert(N > 0, "N must be positive");
    using result_t = decltype(op(a[0], b[0]));
    vec<N, result_t> result;
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        result[i] = op(a[i], b[i]);
    }
    return result;
}

template <int N, typename T>
HOST_DEV constexpr auto operator+(const vec<N, T> a, const vec<N, T> b)
{
    return vector_bin_op(a, b, [](T a, T b)
                         { return a + b; });
}

template <int N, typename T>
HOST_DEV constexpr auto operator*(const vec<N, T> a, const vec<N, T> b)
{
    return vector_bin_op(a, b, [](T a, T b)
                         { return a * b; });
}

template <int N, typename T>
HOST_DEV constexpr auto operator-(const vec<N, T> a, const vec<N, T> b)
{
    return vector_bin_op(a, b, [](T a, T b)
                         { return a - b; });
}

template <int N, typename T>
HOST_DEV constexpr auto operator/(const vec<N, T> a, const vec<N, T> b)
{
    return vector_bin_op(a, b, [](T a, T b)
                         { return a / b; });
}

template <int N, typename Pred, typename T, typename... Ts>
HOST_DEV constexpr bool any(Pred &&pred, const vec<N, T> a,
                            const vec<N, Ts>... vs)
{
    for (int i = 0; i < N; i++)
    {
        if (pred(a[i], vs[i]...))
            return true;
    }
    return false;
}

template <int N, typename Pred, typename T, typename... Ts>
HOST_DEV constexpr bool all(Pred &&pred, const vec<N, T> a,
                            const vec<N, Ts>... vs)
{
    for (int i = 0; i < N; i++)
    {
        if (!pred(a[i], vs[i]...))
            return false;
    }
    return true;
}

template <int N, typename T>
HOST_DEV constexpr T dot(const vec<N, T> a, const vec<N, T> b)
{
    if constexpr (N == 0)
    {
        return 0;
    }
    else if constexpr (N != 0)
    {
        T sum = a[0] * b[0];
#pragma unroll
        for (int i = 1; i < N; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }
}

} // namespace cuda_core

#endif // CUDA_CORE_STRIDED_COPY_IMPL_VEC_H_
