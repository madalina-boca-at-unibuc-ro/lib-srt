

#pragma once
#include "common.hpp"
#include "fwd.hpp"

namespace srt {

// ---------------------------------------------
/* FourVecView (non-owning) with StridedIterator from common.hpp */
// ---------------------------------------------
template<typename T>
class FourVecView {
   public:
    using value_type = T;

    // Use StridedIterator from common.hpp
    using iterator = StridedIterator<T>;
    using const_iterator = StridedIterator<const T>;

    // state
    // Constructors will be updated to use the new StridedIterator.
    constexpr FourVecView(T *ptr, std::ptrdiff_t stride = 1) noexcept : ptr_(ptr), stride_(stride) {}

    // Copy/move from a non-const view to a const view
    constexpr FourVecView(const FourVecView<std::remove_const_t<T>> &other) noexcept
        requires(std::is_const_v<T>)
        : ptr_(other.ptr_), stride_(other.stride_) {}

    static constexpr std::size_t size() noexcept {
        return 4;
    }

    // access
    [[nodiscard]] constexpr T *data() noexcept {
        return ptr_;
    }
    [[nodiscard]] constexpr const T *data() const noexcept {
        return ptr_;
    }
    [[nodiscard]] constexpr std::ptrdiff_t stride() const noexcept {
        return stride_;
    }

    [[nodiscard]] constexpr T &operator[](std::size_t i) noexcept {
        assert(i < 4 && "FourVecView index out of range");
        return *(ptr_ + i * stride_);
    }
    [[nodiscard]] constexpr const T &operator[](std::size_t i) const noexcept {
        assert(i < 4 && "FourVecView index out of range");
        return *(ptr_ + i * stride_);
    }

    // iterators over 4 elements
    [[nodiscard]] constexpr iterator begin() noexcept {
        return iterator(ptr_, stride_, 0);
    }
    [[nodiscard]] constexpr iterator end() noexcept {
        return iterator(ptr_, stride_, 4);
    }
    [[nodiscard]] constexpr const_iterator begin() const noexcept {
        return const_iterator(ptr_, stride_, 0);
    }
    [[nodiscard]] constexpr const_iterator end() const noexcept {
        return const_iterator(ptr_, stride_, 4);
    }
    [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
        return begin();
    }
    [[nodiscard]] constexpr const_iterator cend() const noexcept {
        return end();
    }

    // assignment from anything indexable (out of place, writes into this)
    template<Indexable4 RHS>
        requires(!std::is_const_v<T>) && std::convertible_to<decltype(std::declval<const RHS &>()[0]), T> &&
                std::convertible_to<decltype(std::declval<const RHS &>()[1]), T> &&
                std::convertible_to<decltype(std::declval<const RHS &>()[2]), T> &&
                std::convertible_to<decltype(std::declval<const RHS &>()[3]), T>
    constexpr FourVecView &operator=(const RHS &r) & noexcept(
        std::is_nothrow_convertible_v<decltype(std::declval<const RHS &>()[0]), T> &&
        std::is_nothrow_convertible_v<decltype(std::declval<const RHS &>()[1]), T> &&
        std::is_nothrow_convertible_v<decltype(std::declval<const RHS &>()[2]), T> &&
        std::is_nothrow_convertible_v<decltype(std::declval<const RHS &>()[3]), T>) {
        (*this)[0] = static_cast<T>(r[0]);
        (*this)[1] = static_cast<T>(r[1]);
        (*this)[2] = static_cast<T>(r[2]);
        (*this)[3] = static_cast<T>(r[3]);
        return *this;
    }

    [[nodiscard]] constexpr std::array<T, 4> to_array() const noexcept {
        return {(*this)[0], (*this)[1], (*this)[2], (*this)[3]};
    }

    // comparison ==
    template<Indexable4 RHS>
        requires std::equality_comparable_with<T, decltype(std::declval<const RHS &>()[0])>
    [[nodiscard]] constexpr bool operator==(const RHS &r) const
        noexcept(noexcept((*this)[0] == r[0] && (*this)[1] == r[1] && (*this)[2] == r[2] && (*this)[3] == r[3])) {
        return (*this)[0] == r[0] && (*this)[1] == r[1] && (*this)[2] == r[2] && (*this)[3] == r[3];
    }
    //  comparison !=
    template<Indexable4 RHS>
    [[nodiscard]] constexpr bool operator!=(const RHS &r) const noexcept(noexcept(*this == r)) {
        return !(*this == r);
    }

    // vector +/- indexable
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto element_wise_sum(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r[0])>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R((*this)[0]) + R(r[0]), R((*this)[1]) + R(r[1]), R((*this)[2]) + R(r[2]),
                          R((*this)[3]) + R(r[3])};
    }
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto operator+(const RHS &r) const noexcept {
        return element_wise_sum(r);
    }

    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto element_wise_diff(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r[0])>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R((*this)[0]) - R(r[0]), R((*this)[1]) - R(r[1]), R((*this)[2]) - R(r[2]),
                          R((*this)[3]) - R(r[3])};
    }
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto operator-(const RHS &r) const noexcept {
        return element_wise_diff(r);
    }

    // in-place +/- only if T is non-const and no promotion
    template<Indexable4 RHS>
    constexpr FourVecView &element_wise_sum_inplace(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>)
    {
        (*this)[0] += r[0];
        (*this)[1] += r[1];
        (*this)[2] += r[2];
        (*this)[3] += r[3];
        return *this;
    }
    template<Indexable4 RHS>
    constexpr FourVecView &operator+=(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>)
    {
        return element_wise_sum_inplace(r);
    }

    template<Indexable4 RHS>
    constexpr FourVecView &element_wise_diff_inplace(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>)
    {
        (*this)[0] -= r[0];
        (*this)[1] -= r[1];
        (*this)[2] -= r[2];
        (*this)[3] -= r[3];
        return *this;
    }
    template<Indexable4 RHS>
    constexpr FourVecView &operator-=(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>)
    {
        return element_wise_diff_inplace(r);
    }

    // scalar multiply/divide
    template<class U>
    [[nodiscard]] constexpr auto scalar_mul(const U &s) const noexcept {
        using R = std::common_type_t<T, U>;
        return FourVec<R>{R((*this)[0]) * s, R((*this)[1]) * s, R((*this)[2]) * s, R((*this)[3]) * s};
    }
    template<class U>
    [[nodiscard]] constexpr auto operator*(const U &s) const noexcept {
        return scalar_mul(s);
    }

    template<class U>
    [[nodiscard]] constexpr auto scalar_div(const U &s) const noexcept {
        using R = std::common_type_t<T, U>;
        return FourVec<R>{R((*this)[0]) / s, R((*this)[1]) / s, R((*this)[2]) / s, R((*this)[3]) / s};
    }
    template<class U>
    [[nodiscard]] constexpr auto operator/(const U &s) const noexcept {
        return scalar_div(s);
    }

    // scalar multiply (commutative)
    template<class U>
    [[nodiscard]] friend constexpr auto operator*(const U &s, const FourVecView &v) noexcept {
        return v * s;
    }

    // in-place scalar mul/div
    template<class U>
    constexpr FourVecView &scalar_mul_inplace(const U &s) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, U>>)
    {
        (*this)[0] = T((*this)[0] * s);
        (*this)[1] = T((*this)[1] * s);
        (*this)[2] = T((*this)[2] * s);
        (*this)[3] = T((*this)[3] * s);
        return *this;
    }
    template<class U>
    constexpr FourVecView &operator*=(const U &s) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, U>>)
    {
        return scalar_mul_inplace(s);
    }

    template<class U>
    constexpr FourVecView &scalar_div_inplace(const U &s) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, U>>)
    {
        (*this)[0] = T((*this)[0] / s);
        (*this)[1] = T((*this)[1] / s);
        (*this)[2] = T((*this)[2] / s);
        (*this)[3] = T((*this)[3] / s);
        return *this;
    }
    template<class U>
    constexpr FourVecView &operator/=(const U &s) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, U>>)
    {
        return scalar_div_inplace(s);
    }

    // Hadamard
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto element_wise_mul(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r[0])>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R((*this)[0]) * R(r[0]), R((*this)[1]) * R(r[1]), R((*this)[2]) * R(r[2]),
                          R((*this)[3]) * R(r[3])};
    }
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto operator%(const RHS &r) const noexcept {
        return element_wise_mul(r);
    }

    template<Indexable4 RHS>
    constexpr FourVecView &element_wise_mul_inplace(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>)
    {
        (*this)[0] *= r[0];
        (*this)[1] *= r[1];
        (*this)[2] *= r[2];
        (*this)[3] *= r[3];
        return *this;
    }
    template<Indexable4 RHS>
    constexpr FourVecView &operator%=(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>)
    {
        return element_wise_mul_inplace(r);
    }

    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto minkowski_dot(const RHS &b) const noexcept {
        using R0 = uncvref_t<decltype(b[0])>;
        using R = std::common_type_t<T, R0>;
        return R((*this)[0]) * b[0] - (R((*this)[1]) * b[1] + R((*this)[2]) * b[2] + R((*this)[3]) * b[3]);
    }

    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto dot(const RHS &b) const noexcept {
        using R0 = uncvref_t<decltype(b[0])>;
        using R = std::common_type_t<T, R0>;
        return R((*this)[1]) * b[1] + R((*this)[2]) * b[2] + R((*this)[3]) * b[3];
    }

    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto dot_four(const RHS &b) const noexcept {
        using R0 = uncvref_t<decltype(b[0])>;
        using R = std::common_type_t<T, R0>;
        return R((*this)[0]) * b[0] + R((*this)[1]) * b[1] + R((*this)[2]) * b[2] + R((*this)[3]) * b[3];
    }

    // Vector-Matrix multiplication (this_vec . matrix), out-of-place.
    // Returns a new FourVec with common type promotion.
    template<Matrix4Indexable RHS_Mat>
    [[nodiscard]] constexpr auto dot_four_left(const RHS_Mat &m) const noexcept {
        using R0 = uncvref_t<decltype(m(0, 0))>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{
            R((*this)[0]) * m(0, 0) + R((*this)[1]) * m(1, 0) + R((*this)[2]) * m(2, 0) + R((*this)[3]) * m(3, 0),
            R((*this)[0]) * m(0, 1) + R((*this)[1]) * m(1, 1) + R((*this)[2]) * m(2, 1) + R((*this)[3]) * m(3, 1),
            R((*this)[0]) * m(0, 2) + R((*this)[1]) * m(1, 2) + R((*this)[2]) * m(2, 2) + R((*this)[3]) * m(3, 2),
            R((*this)[0]) * m(0, 3) + R((*this)[1]) * m(1, 3) + R((*this)[2]) * m(2, 3) + R((*this)[3]) * m(3, 3)};
    }

    // In-place Vector-Matrix multiplication (this_vec . matrix).
    // Requires the view's underlying type T to be non-const.
    template<Matrix4Indexable RHS_Mat>
    constexpr FourVecView &dot_four_left_inplace(const RHS_Mat &m) & noexcept
        requires(!std::is_const_v<T>) && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(m(0, 0))>>>
    {
        // Calculate the result using the out-of-place dot and assign back to the view.
        *this = (*this).dot_four_left(m);
        return *this;
    }

    // Matrix-Vector multiplication (matrix . this_vec), out-of-place.
    // Returns a new FourVec with common type promotion.
    template<Matrix4Indexable RHS_Mat>
    [[nodiscard]] constexpr auto dot_four_right(const RHS_Mat &m) const noexcept {
        using R0 = uncvref_t<decltype(m(0, 0))>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{
            R((*this)[0]) * m(0, 0) + R((*this)[1]) * m(0, 1) + R((*this)[2]) * m(0, 2) + R((*this)[3]) * m(0, 3),
            R((*this)[0]) * m(1, 0) + R((*this)[1]) * m(1, 1) + R((*this)[2]) * m(1, 2) + R((*this)[3]) * m(1, 3),
            R((*this)[0]) * m(2, 0) + R((*this)[1]) * m(2, 1) + R((*this)[2]) * m(2, 2) + R((*this)[3]) * m(2, 3),
            R((*this)[0]) * m(3, 0) + R((*this)[1]) * m(3, 1) + R((*this)[2]) * m(3, 2) + R((*this)[3]) * m(3, 3)};
    }

    // In-place Matrix-Vector multiplication (matrix . this_vec).
    // Requires the view's underlying type T to be non-const.
    template<Matrix4Indexable RHS_Mat>
    constexpr FourVecView &dot_four_right_inplace(const RHS_Mat &m) & noexcept
        requires(!std::is_const_v<T>) && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(m(0, 0))>>>
    {
        // Calculate the result using the out-of-place dot and assign back to the view.
        *this = (*this).dot_four_right(m);
        return *this;
    }

    // Minkowski Vector-Matrix multiplication (this_vec . matrix), out-of-place.
    // Returns a new FourVec with common type promotion.
    template<Matrix4Indexable RHS_Mat>
    [[nodiscard]] constexpr auto minkowski_dot_four_left(const RHS_Mat &m) const noexcept {
        using R0 = uncvref_t<decltype(m(0, 0))>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{
            R((*this)[0]) * m(0, 0) - R((*this)[1]) * m(1, 0) - R((*this)[2]) * m(2, 0) - R((*this)[3]) * m(3, 0),
            R((*this)[0]) * m(0, 1) - R((*this)[1]) * m(1, 1) - R((*this)[2]) * m(2, 1) - R((*this)[3]) * m(3, 1),
            R((*this)[0]) * m(0, 2) - R((*this)[1]) * m(1, 2) - R((*this)[2]) * m(2, 2) - R((*this)[3]) * m(3, 2),
            R((*this)[0]) * m(0, 3) - R((*this)[1]) * m(1, 3) - R((*this)[2]) * m(2, 3) - R((*this)[3]) * m(3, 3)};
    }

    // In-place Minkowski Vector-Matrix multiplication (this_vec . matrix).
    // Requires the view's underlying type T to be non-const.
    template<Matrix4Indexable RHS_Mat>
    constexpr FourVecView &minkowski_dot_four_left_inplace(const RHS_Mat &m) & noexcept
        requires(!std::is_const_v<T>) && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(m(0, 0))>>>
    {
        // Calculate the result using the out-of-place dot and assign back to the view.
        *this = (*this).minkowski_dot_four_left(m);
        return *this;
    }

    // Minkowski Matrix-Vector multiplication (matrix . this_vec), out-of-place.
    // Returns a new FourVec with common type promotion.
    template<Matrix4Indexable RHS_Mat>
    [[nodiscard]] constexpr auto minkowski_dot_four_right(const RHS_Mat &m) const noexcept {
        using R0 = uncvref_t<decltype(m(0, 0))>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{
            R((*this)[0]) * m(0, 0) - R((*this)[1]) * m(0, 1) - R((*this)[2]) * m(0, 2) - R((*this)[3]) * m(0, 3),
            R((*this)[0]) * m(1, 0) - R((*this)[1]) * m(1, 1) - R((*this)[2]) * m(1, 2) - R((*this)[3]) * m(1, 3),
            R((*this)[0]) * m(2, 0) - R((*this)[1]) * m(2, 1) - R((*this)[2]) * m(2, 2) - R((*this)[3]) * m(2, 3),
            R((*this)[0]) * m(3, 0) - R((*this)[1]) * m(3, 1) - R((*this)[2]) * m(3, 2) - R((*this)[3]) * m(3, 3)};
    }

    // In-place Matrix-Vector multiplication (matrix . this_vec).
    // Requires the view's underlying type T to be non-const.
    template<Matrix4Indexable RHS_Mat>
    constexpr FourVecView &minkowski_dot_four_right_inplace(const RHS_Mat &m) & noexcept
        requires(!std::is_const_v<T>) && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(m(0, 0))>>>
    {
        // Calculate the result using the out-of-place dot and assign back to the view.
        *this = (*this).minkowski_dot_four_right(m);
        return *this;
    }

    // conjugation (uses conj_if_needed)
    constexpr FourVecView &conjugate_inplace() noexcept
        requires(!std::is_const_v<T>)
    {
        (*this)[0] = conj_if_needed((*this)[0]);
        (*this)[1] = conj_if_needed((*this)[1]);
        (*this)[2] = conj_if_needed((*this)[2]);
        (*this)[3] = conj_if_needed((*this)[3]);
        return *this;
    }
    [[nodiscard]] constexpr FourVec<T> conjugate() const noexcept {
        FourVec<T> out{(*this)[0], (*this)[1], (*this)[2], (*this)[3]};
        return out.conjugate_inplace();
    }

    // real and imaginary parts (out-of-place)
    [[nodiscard]] constexpr auto real() const noexcept {
        using RealType = uncvref_t<decltype(std::real((*this)[0]))>;
        return FourVec<RealType>{std::real((*this)[0]), std::real((*this)[1]), std::real((*this)[2]),
                                 std::real((*this)[3])};
    }

    [[nodiscard]] constexpr auto imag() const noexcept {
        using RealType = uncvref_t<decltype(std::imag((*this)[0]))>;
        return FourVec<RealType>{std::imag((*this)[0]), std::imag((*this)[1]), std::imag((*this)[2]),
                                 std::imag((*this)[3])};
    }

    // covariant lowering with eta = diag(+,-,-,-)
    constexpr FourVecView &covariant_inplace() noexcept
        requires(!std::is_const_v<T>)
    {
        (*this)[1] = -(*this)[1];
        (*this)[2] = -(*this)[2];
        (*this)[3] = -(*this)[3];
        return *this;
    }
    [[nodiscard]] constexpr FourVec<T> covariant() const noexcept {
        return FourVec<T>{(*this)[0], -(*this)[1], -(*this)[2], -(*this)[3]};
    }

    // cross product with anything indexable; the first element (X0) becomes 0 and the rest are the cross product
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto cross(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r[0])>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R(0), R((*this)[2]) * R(r[3]) - R((*this)[3]) * R(r[2]),
                          R((*this)[3]) * R(r[1]) - R((*this)[1]) * R(r[3]),
                          R((*this)[1]) * R(r[2]) - R((*this)[2]) * R(r[1])};
    }

    // Minkowski norm squared: (+,-,-,-) metric
    [[nodiscard]] constexpr auto minkowski_norm2() const noexcept {
        return minkowski_dot(*this);
    }
    // Euclidean norm squared: (+,+,+,+) metric
    [[nodiscard]] constexpr auto euclidean4_norm2() const noexcept {
        return dot_four(*this);
    }
    // 3-vector norm squared: (+,+,+) metric
    [[nodiscard]] constexpr auto euclidean3_norm2() const noexcept {
        return dot(*this);
    }

    // utilities
    void print(const std::string &text = "", std::ostream &os = std::cout, int width = 10, int precision = 4) const {
        if (!text.empty()) os << text << "\n";
        auto f = os.flags();
        auto p = os.precision();
        os.setf(std::ios::scientific, std::ios::floatfield);
        os.precision(precision);
        os << std::setw(width) << (*this)[0] << "\n"
           << std::setw(width) << (*this)[1] << "\n"
           << std::setw(width) << (*this)[2] << "\n"
           << std::setw(width) << (*this)[3] << "\n";
        os.flags(f);
        os.precision(p);
    }

   private:
    T *ptr_;
    std::ptrdiff_t stride_;
};

}  // namespace srt
