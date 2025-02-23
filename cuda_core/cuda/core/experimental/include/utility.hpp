#include <type_traits>

// In cuda.bindings 12.8, the private member name was renamed from "_ptr" to "_pvt_ptr".
// We want to have the C++ layer supporting all past 12.x versions, so some tricks are needed.
// Since there's no std::has_member<T, member_name> so we use SFINAE to create the same effect.

template <typename T,
          std::enable_if_t<std::is_pointer_v<decltype(std::remove_pointer_t<T>::_pvt_ptr)>, bool> = true>
inline auto& get_cuda_native_handle(const T& obj) {
    return *(obj->_pvt_ptr);
}

template <typename T,
          std::enable_if_t<std::is_pointer_v<decltype(std::remove_pointer_t<T>::_ptr)>, bool> = true>
inline auto& get_cuda_native_handle(const T& obj) {
    return *(obj->_ptr);
}
