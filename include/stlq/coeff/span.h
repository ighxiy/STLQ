#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace stlq {

// Minimal span (C++17-compatible) to keep the coefficient codec header-only friendly.
template <typename T>
class Span {
public:
  using element_type = T;

  Span() : data_(nullptr), size_(0) {}
  Span(T* data, std::size_t size) : data_(data), size_(size) {}

  template <typename U,
            typename = std::enable_if_t<std::is_const<T>::value &&
                                        std::is_same<std::remove_const_t<T>, U>::value>>
  Span(const Span<U>& other) : data_(other.data()), size_(other.size()) {}

  T* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T& operator[](std::size_t i) const { return data_[i]; }

  T* begin() const { return data_; }
  T* end() const { return data_ + size_; }

private:
  T* data_;
  std::size_t size_;
};

}  // namespace stlq

