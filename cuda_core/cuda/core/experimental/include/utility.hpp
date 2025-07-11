// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

#pragma once

#include <type_traits>

// In cuda.bindings 12.8, the private member name was renamed from "_ptr" to "_pvt_ptr".
// We want to have the C++ layer supporting all past 12.x versions, so some tricks are needed.
// Since there's no std::has_member<T, member_name> so we use SFINAE to create the same effect.

template <typename T,
          std::enable_if_t<std::is_pointer_v<decltype(std::remove_pointer_t<T>::_pvt_ptr)>, int> = 0>
inline auto& get_cuda_native_handle(const T& obj) {
    return *(obj->_pvt_ptr);
}

template <typename T,
          std::enable_if_t<std::is_pointer_v<decltype(std::remove_pointer_t<T>::_ptr)>, int> = 0>
inline auto& get_cuda_native_handle(const T& obj) {
    return *(obj->_ptr);
}
