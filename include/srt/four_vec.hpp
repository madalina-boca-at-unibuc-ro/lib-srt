#pragma once

#include "common.hpp"
#include "fwd.hpp"

namespace srt {

// ---------------------------------------------
/* FourVec (owning) */
// ---------------------------------------------
template<typename T>
class FourVec {
   public:
    using value_type = T;

    T x0, x1, x2, x3;

    // ctors
    constexpr FourVec() noexcept : x0(0), x1(0), x2(0), x3(0) {}
    constexpr FourVec(const T &a0, const T &a1, const T &a2, const T &a3) noexcept : x0(a0), x1(a1), x2(a2), x3(a3) {}
    constexpr explicit FourVec(const std::array<T, 4> &a) noexcept : x0(a[0]), x1(a[1]), x2(a[2]), x3(a[3]) {}

    constexpr FourVec(const FourVec &) noexcept = default;
    constexpr FourVec(FourVec &&) noexcept = default;
    constexpr FourVec &operator=(const FourVec &) noexcept = default;
    constexpr FourVec &operator=(FourVec &&) noexcept = default;

    // assignment from anything indexable (out of place, returns common_type)
    template<Indexable4 RHS>
        requires
        // don’t hijack same-type assignment
        (!std::same_as<std::remove_cvref_t<RHS>, FourVec>) &&
        // each element must be convertible to T
        std::convertible_to<decltype(std::declval<const RHS &>()[0]), T> &&
        std::convertible_to<decltype(std::declval<const RHS &>()[1]), T> &&
        std::convertible_to<decltype(std::declval<const RHS &>()[2]), T> &&
        std::convertible_to<decltype(std::declval<const RHS &>()[3]), T>
        constexpr FourVec &operator=(const RHS &r) noexcept(
            std::is_nothrow_convertible_v<decltype(std::declval<const RHS &>()[0]), T> &&
            std::is_nothrow_convertible_v<decltype(std::declval<const RHS &>()[1]), T> &&
            std::is_nothrow_convertible_v<decltype(std::declval<const RHS &>()[2]), T> &&
            std::is_nothrow_convertible_v<decltype(std::declval<const RHS &>()[3]), T>) {
        // Stronger exception safety: convert first, then commit.
        T a0 = static_cast<T>(r[0]);
        T a1 = static_cast<T>(r[1]);
        T a2 = static_cast<T>(r[2]);
        T a3 = static_cast<T>(r[3]);

        x0 = a0;
        x1 = a1;
        x2 = a2;
        x3 = a3;
        return *this;
    }

    // access
    constexpr std::size_t size() noexcept {
        return 4;
    }
    [[nodiscard]] constexpr T *data() noexcept {
        return &x0;
    }
    [[nodiscard]] constexpr const T *data() const noexcept {
        return &x0;
    }

    constexpr T &operator[](std::size_t i) noexcept {
        assert(i < 4 && "FourVec index out of range");
        return *(data() + i);
    }
    constexpr const T &operator[](std::size_t i) const noexcept {
        assert(i < 4 && "FourVec index out of range");
        return *(data() + i);
    }

    [[nodiscard]] constexpr T *begin() noexcept {
        return data();
    }
    [[nodiscard]] constexpr T *end() noexcept {
        return data() + 4;
    }
    [[nodiscard]] constexpr const T *begin() const noexcept {
        return data();
    }
    [[nodiscard]] constexpr const T *end() const noexcept {
        return data() + 4;
    }

    [[nodiscard]] constexpr const T *cbegin() const noexcept {
        return begin();
    }
    [[nodiscard]] constexpr const T *cend() const noexcept {
        return end();
    }

    // to_array, returns array of 4 elements
    [[nodiscard]] constexpr std::array<T, 4> to_array() const noexcept {
        return {x0, x1, x2, x3};
    }
    // unary + (in place, returns self)
    [[nodiscard]] constexpr FourVec operator+() const noexcept {
        return *this;
    }
    // unary - (out of place, returns negated)
    [[nodiscard]] constexpr FourVec operator-() const noexcept {
        return FourVec{-x0, -x1, -x2, -x3};
    }

    template<Indexable4 RHS>
        requires std::equality_comparable_with<T, decltype(std::declval<const RHS &>()[0])>
    [[nodiscard]] constexpr bool operator==(const RHS &r) const
        noexcept(noexcept(x0 == r[0] && x1 == r[1] && x2 == r[2] && x3 == r[3])) {
        return x0 == r[0] && x1 == r[1] && x2 == r[2] && x3 == r[3];
    }

    template<Indexable4 RHS>
    [[nodiscard]] constexpr bool operator!=(const RHS &r) const noexcept(noexcept(*this == r)) {
        return !(*this == r);
    }

    // vector + anything indexable (out of place, returns common_type)
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto element_wise_sum(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r[0])>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R(x0) + R(r[0]), R(x1) + R(r[1]), R(x2) + R(r[2]), R(x3) + R(r[3])};
    }
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto operator+(const RHS &r) const noexcept {
        return element_wise_sum(r);
    }

    // vector - anything indexable (out of place, returns common_type)
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto element_wise_diff(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r[0])>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R(x0) - R(r[0]), R(x1) - R(r[1]), R(x2) - R(r[2]), R(x3) - R(r[3])};
    }
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto operator-(const RHS &r) const noexcept {
        return element_wise_diff(r);
    }

    // vector += with anything indexable: only if common_type == T (no promotion)
    template<Indexable4 RHS>
    constexpr FourVec &element_wise_sum_inplace(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>
    {
        x0 += r[0];
        x1 += r[1];
        x2 += r[2];
        x3 += r[3];
        return *this;
    }
    // vector += with anything indexable: only if common_type == T (no promotion)
    template<Indexable4 RHS>
    constexpr FourVec &operator+=(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>
    {
        return element_wise_sum_inplace(r);
    }
    // vector -= with anything indexable: only if common_type == T (no promotion)
    template<Indexable4 RHS>
    constexpr FourVec &element_wise_diff_inplace(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>
    {
        x0 -= r[0];
        x1 -= r[1];
        x2 -= r[2];
        x3 -= r[3];
        return *this;
    }
    // vector -= with anything indexable: only if common_type == T (no promotion)
    template<Indexable4 RHS>
    constexpr FourVec &operator-=(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>
    {
        return element_wise_diff_inplace(r);
    }

    // scalar multiply (templated; out of place, returns common_type)
    template<class U>
    [[nodiscard]] constexpr auto scalar_mul(const U &s) const noexcept {
        using R = std::common_type_t<T, U>;
        return FourVec<R>{R(x0) * s, R(x1) * s, R(x2) * s, R(x3) * s};
    }
    template<class U>
    [[nodiscard]] constexpr auto operator*(const U &s) const noexcept {
        return scalar_mul(s);
    }
    // scalar divide (templated; out of place, returns common_type)
    template<class U>
    [[nodiscard]] constexpr auto scalar_div(const U &s) const noexcept {
        using R = std::common_type_t<T, U>;
        return FourVec<R>{R(x0) / s, R(x1) / s, R(x2) / s, R(x3) / s};
    }
    template<class U>
    [[nodiscard]] constexpr auto operator/(const U &s) const noexcept {
        using R = std::common_type_t<T, U>;
        return FourVec<R>{R(x0) / s, R(x1) / s, R(x2) / s, R(x3) / s};
    }

    // scalar multiply (commutative): out of place (templated; returns s * v;
    // allows s * v = v * s)
    template<class U>
    [[nodiscard]] friend constexpr auto operator*(const U &s, const FourVec &v) noexcept {
        return v * s;
    }

    // scalar multiply *= (templated; in place, only if common_type == T)
    template<class U>
    constexpr FourVec &scalar_mul_inplace(const U &s) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, U>>
    {
        x0 = T(x0 * s);
        x1 = T(x1 * s);
        x2 = T(x2 * s);
        x3 = T(x3 * s);
        return *this;
    }
    template<class U>
    constexpr FourVec &operator*=(const U &s) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, U>>
    {
        return scalar_mul_inplace(s);
    }

    // scalar divide /= (templated; in place, only if common_type == T)
    template<class U>
    constexpr FourVec &scalar_div_inplace(const U &s) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, U>>
    {
        x0 = T(x0 / s);
        x1 = T(x1 / s);
        x2 = T(x2 / s);
        x3 = T(x3 / s);
        return *this;
    }
    // scalar divide /= (templated; in place, only if common_type == T)
    template<class U>
    constexpr FourVec &operator/=(const U &s) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, U>>
    {
        return scalar_div_inplace(s);
    }

    // element-wise (Hadamard) multiply with anything indexable (out of place,
    // returns common_type)
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto element_wise_mul(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r[0])>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R(x0) * R(r[0]), R(x1) * R(r[1]), R(x2) * R(r[2]), R(x3) * R(r[3])};
    }
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto operator%(const RHS &r) const noexcept {
        return element_wise_mul(r);
    }

    // element-wise (Hadamard) multiply with anything indexable (in place, only if common_type == T)
    template<Indexable4 RHS>
    constexpr FourVec &element_wise_mul_inplace(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>
    {
        x0 *= r[0];
        x1 *= r[1];
        x2 *= r[2];
        x3 *= r[3];
        return *this;
    }

    template<Indexable4 RHS>
    constexpr FourVec &operator%=(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r[0])>>>
    {
        return element_wise_mul_inplace(r);
    }
    // Minkowski dot (+,-,-,-): plain (no conjugation)
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto minkowski_dot(const RHS &b) const noexcept {
        using R0 = uncvref_t<decltype(b[0])>;
        using R = std::common_type_t<T, R0>;
        return R(x0) * b[0] - (R(x1) * b[1] + R(x2) * b[2] + R(x3) * b[3]);
    }
    // 3-vector dot (+,+,+): plain (no conjugation), ignores the first element
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto dot(const RHS &b) const noexcept {
        using R0 = uncvref_t<decltype(b[0])>;
        using R = std::common_type_t<T, R0>;
        return R(x1) * b[1] + R(x2) * b[2] + R(x3) * b[3];
    }
    // 4-vector dot (+,+,+,+) metric: plain (no conjugation)
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto dot_four(const RHS &b) const noexcept {
        using R0 = uncvref_t<decltype(b[0])>;
        using R = std::common_type_t<T, R0>;
        return R(x0) * b[0] + R(x1) * b[1] + R(x2) * b[2] + R(x3) * b[3];
    }

    // Vector-Matrix multiplication (this_vec . matrix), out-of-place.
    // Returns a new FourVec with common type promotion.
    template<Matrix4Indexable RHS_Mat>
    [[nodiscard]] constexpr auto dot_four_left(const RHS_Mat &m) const noexcept {
        using R0 = uncvref_t<decltype(m(0, 0))>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R(x0) * m(0, 0) + R(x1) * m(1, 0) + R(x2) * m(2, 0) + R(x3) * m(3, 0),
                          R(x0) * m(0, 1) + R(x1) * m(1, 1) + R(x2) * m(2, 1) + R(x3) * m(3, 1),
                          R(x0) * m(0, 2) + R(x1) * m(1, 2) + R(x2) * m(2, 2) + R(x3) * m(3, 2),
                          R(x0) * m(0, 3) + R(x1) * m(1, 3) + R(x2) * m(2, 3) + R(x3) * m(3, 3)};
    }

    // In-place Vector-Matrix multiplication (this_vec . matrix).
    // Requires the vector's underlying type T to be non-const.
    template<Matrix4Indexable RHS_Mat>
    constexpr FourVec &dot_four_left_inplace(const RHS_Mat &m) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(m(0, 0))>>>
    {
        // Calculate the result using the out-of-place dot and assign back.
        *this = (*this).dot_four_left(m);
        return *this;
    }

    // Vector-Matrix multiplication (matrix . this_vec), out-of-place.
    // Returns a new FourVec with common type promotion.
    template<Matrix4Indexable RHS_Mat>
    [[nodiscard]] constexpr auto dot_four_right(const RHS_Mat &m) const noexcept {
        using R0 = uncvref_t<decltype(m(0, 0))>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R(x0) * m(0, 0) + R(x1) * m(0, 1) + R(x2) * m(0, 2) + R(x3) * m(0, 3),
                          R(x0) * m(1, 0) + R(x1) * m(1, 1) + R(x2) * m(1, 2) + R(x3) * m(1, 3),
                          R(x0) * m(2, 0) + R(x1) * m(2, 1) + R(x2) * m(2, 2) + R(x3) * m(2, 3),
                          R(x0) * m(3, 0) + R(x1) * m(3, 1) + R(x2) * m(3, 2) + R(x3) * m(3, 3)};
    }

    // In-place Vector-Matrix multiplication (matrix . this_vec).
    // Requires the vector's underlying type T to be non-const.
    template<Matrix4Indexable RHS_Mat>
    constexpr FourVec &dot_four_right_inplace(const RHS_Mat &m) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(m(0, 0))>>>
    {
        // Calculate the result using the out-of-place dot and assign back.
        *this = (*this).dot_four_right(m);
        return *this;
    }

    // Minkowski Vector-Matrix multiplication (this_vec . matrix), out-of-place.
    // Returns a new FourVec with common type promotion.
    template<Matrix4Indexable RHS_Mat>
    [[nodiscard]] constexpr auto minkowski_dot_four_left(const RHS_Mat &m) const noexcept {
        using R0 = uncvref_t<decltype(m(0, 0))>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R(x0) * m(0, 0) - R(x1) * m(1, 0) - R(x2) * m(2, 0) - R(x3) * m(3, 0),
                          R(x0) * m(0, 1) - R(x1) * m(1, 1) - R(x2) * m(2, 1) - R(x3) * m(3, 1),
                          R(x0) * m(0, 2) - R(x1) * m(1, 2) - R(x2) * m(2, 2) - R(x3) * m(3, 2),
                          R(x0) * m(0, 3) - R(x1) * m(1, 3) - R(x2) * m(2, 3) - R(x3) * m(3, 3)};
    }

    // In-place Minkowski Vector-Matrix multiplication (this_vec . matrix).
    // Requires the vector's underlying type T to be non-const.
    template<Matrix4Indexable RHS_Mat>
    constexpr FourVec &minkowski_dot_four_left_inplace(const RHS_Mat &m) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(m(0, 0))>>>
    {
        // Calculate the result using the out-of-place dot and assign back.
        *this = (*this).minkowski_dot_four_left(m);
        return *this;
    }

    // Minkowski Matrix-Vector multiplication (matrix . this_vec), out-of-place.
    // Returns a new FourVec with common type promotion.
    template<Matrix4Indexable RHS_Mat>
    [[nodiscard]] constexpr auto minkowski_dot_four_right(const RHS_Mat &m) const noexcept {
        using R0 = uncvref_t<decltype(m(0, 0))>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R(x0) * m(0, 0) - R(x1) * m(0, 1) - R(x2) * m(0, 2) - R(x3) * m(0, 3),
                          R(x0) * m(1, 0) - R(x1) * m(1, 1) - R(x2) * m(1, 2) - R(x3) * m(1, 3),
                          R(x0) * m(2, 0) - R(x1) * m(2, 1) - R(x2) * m(2, 2) - R(x3) * m(2, 3),
                          R(x0) * m(3, 0) - R(x1) * m(3, 1) - R(x2) * m(3, 2) - R(x3) * m(3, 3)};
    }

    // In-place Minkowski Matrix-Vector multiplication (matrix . this_vec).
    // Requires the vector's underlying type T to be non-const.
    template<Matrix4Indexable RHS_Mat>
    constexpr FourVec &minkowski_dot_four_right_inplace(const RHS_Mat &m) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(m(0, 0))>>>
    {
        // Calculate the result using the out-of-place dot and assign back.
        *this = (*this).minkowski_dot_four_right(m);
        return *this;
    }

    // conjugation (uses conj_if_needed)
    constexpr FourVec &conjugate_inplace() noexcept {
        x0 = conj_if_needed(x0);
        x1 = conj_if_needed(x1);
        x2 = conj_if_needed(x2);
        x3 = conj_if_needed(x3);
        return *this;
    }
    [[nodiscard]] constexpr FourVec conjugate() const noexcept {
        FourVec tmp = *this;
        return tmp.conjugate_inplace();
    }

    // the real part, only out of place real()
    [[nodiscard]] constexpr auto real() const noexcept -> FourVec<uncvref_t<decltype(std::real(std::declval<T>()))>> {
        using R = uncvref_t<decltype(std::real(std::declval<T>()))>;
        return FourVec<R>{static_cast<R>(std::real(x0)), static_cast<R>(std::real(x1)), static_cast<R>(std::real(x2)),
                          static_cast<R>(std::real(x3))};
    }

    // the imaginary part, only out of place imag()
    [[nodiscard]] constexpr auto imag() const noexcept -> FourVec<uncvref_t<decltype(std::imag(std::declval<T>()))>> {
        using R = uncvref_t<decltype(std::imag(std::declval<T>()))>;
        return FourVec<R>{static_cast<R>(std::imag(x0)), static_cast<R>(std::imag(x1)), static_cast<R>(std::imag(x2)),
                          static_cast<R>(std::imag(x3))};
    }

    // covariant lowering with eta = diag(+,-,-,-)
    constexpr FourVec &covariant_inplace() noexcept {
        x1 = -x1;
        x2 = -x2;
        x3 = -x3;  // x0 unchanged
        return *this;
    }
    [[nodiscard]] constexpr FourVec covariant() const noexcept {
        FourVec tmp = *this;
        return tmp.covariant_inplace();
    }

    // 3-vector cross product with anything indexable; the first element (X0)
    // becomes 0 and the rest are the cross product; templated; out of place,
    // returns common_type returns self x r (this order)
    template<Indexable4 RHS>
    [[nodiscard]] constexpr auto cross(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r[0])>;
        using R = std::common_type_t<T, R0>;
        return FourVec<R>{R(0), R(x2) * R(r[3]) - R(x3) * R(r[2]), R(x3) * R(r[1]) - R(x1) * R(r[3]),
                          R(x1) * R(r[2]) - R(x2) * R(r[1])};
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
        os << std::setw(width) << x0 << "\n"
           << std::setw(width) << x1 << "\n"
           << std::setw(width) << x2 << "\n"
           << std::setw(width) << x3 << "\n";
        os.flags(f);
        os.precision(p);
    }

    template<class U>
    [[nodiscard]] constexpr FourVec<U> cast() const noexcept {
        return {U(x0), U(x1), U(x2), U(x3)};
    }
};

}  // namespace srt
