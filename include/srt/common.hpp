#pragma once
#include <array>
#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <type_traits>

namespace srt {

// ---------------------------------------------
// Helpers / traits
// ---------------------------------------------
template<class V>
concept Indexable4 = requires(V v) {
    { v[0] };
    { v[1] };
    { v[2] };
    { v[3] };
};

template<class M>
concept Matrix4Indexable = requires(M m) {
    { m(0, 0) };
    { m(3, 3) };
};

template<class T>
constexpr const T& conj_if_needed(const T& x) noexcept {
    return x;
}

template<class R>
constexpr std::complex<R> conj_if_needed(const std::complex<R>& x) noexcept {
    using std::conj;
    return conj(x);
}

template<class T>
using uncvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

inline constexpr std::complex<double> imag_unit{0.0, 1.0};

// Strided random-access iterator
template<class U>
class StridedIterator {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::remove_cv_t<U>;
    using difference_type = std::ptrdiff_t;
    using reference = U&;
    using pointer = U*;

    constexpr StridedIterator(pointer base, std::ptrdiff_t stride, std::ptrdiff_t idx) noexcept
        : base_(base), stride_(stride), idx_(idx) {}

    constexpr reference operator*() const noexcept {
        return *(base_ + idx_ * stride_);
    }
    constexpr pointer operator->() const noexcept {
        return (base_ + idx_ * stride_);
    }

    // index access
    constexpr reference operator[](difference_type n) const noexcept {
        return *(base_ + (idx_ + n) * stride_);
    }

    // increment/decrement
    constexpr StridedIterator& operator++() noexcept {
        ++idx_;
        return *this;
    }
    constexpr StridedIterator operator++(int) noexcept {
        StridedIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    constexpr StridedIterator& operator--() noexcept {
        --idx_;
        return *this;
    }
    constexpr StridedIterator operator--(int) noexcept {
        StridedIterator tmp = *this;
        --(*this);
        return tmp;
    }

    // arithmetic
    constexpr StridedIterator& operator+=(difference_type n) noexcept {
        idx_ += n;
        return *this;
    }
    constexpr StridedIterator& operator-=(difference_type n) noexcept {
        idx_ -= n;
        return *this;
    }
    [[nodiscard]] constexpr StridedIterator operator+(difference_type n) const noexcept {
        auto t = *this;
        t += n;
        return t;
    }
    [[nodiscard]] constexpr StridedIterator operator-(difference_type n) const noexcept {
        auto t = *this;
        t -= n;
        return t;
    }
    [[nodiscard]] constexpr difference_type operator-(const StridedIterator& o) const noexcept {
        return idx_ - o.idx_;
    }

    // comparisons
    constexpr bool operator==(const StridedIterator& o) const noexcept {
        return idx_ == o.idx_ && base_ == o.base_ && stride_ == o.stride_;
    }
    constexpr bool operator!=(const StridedIterator& o) const noexcept {
        return !(*this == o);
    }
    constexpr bool operator<(const StridedIterator& o) const noexcept {
        return idx_ < o.idx_;
    }
    constexpr bool operator>(const StridedIterator& o) const noexcept {
        return idx_ > o.idx_;
    }
    constexpr bool operator<=(const StridedIterator& o) const noexcept {
        return idx_ <= o.idx_;
    }
    constexpr bool operator>=(const StridedIterator& o) const noexcept {
        return idx_ >= o.idx_;
    }

   private:
    pointer base_;
    std::ptrdiff_t stride_;
    std::ptrdiff_t idx_;
};

}  // namespace srt
