#pragma once

#include "common.hpp"
#include "fwd.hpp"

namespace srt {

// ---------------------------------------------
/* FourMat (owning) */
// ---------------------------------------------
template<typename T>
class FourMat {
   public:
    using value_type = T;

    // Data stored in row-major order for cache efficiency and linear iteration.
    T m_elems[16];

    // Default constructor: Initializes all elements to their default (e.g., 0 for arithmetic types).
    constexpr FourMat() noexcept : m_elems{} {}

    // Constructor with all 16 elements provided.
    constexpr FourMat(const T &m00, const T &m01, const T &m02, const T &m03, const T &m10, const T &m11, const T &m12,
                      const T &m13, const T &m20, const T &m21, const T &m22, const T &m23, const T &m30, const T &m31,
                      const T &m32, const T &m33) noexcept
        : m_elems{m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33} {}

    // Constructor from std::array<T, 16>.
    constexpr explicit FourMat(const std::array<T, 16> &a) noexcept {
        for (std::size_t i = 0; i < 16; ++i) {
            m_elems[i] = a[i];
        }
    }

    // Default copy/move constructors and assignment operators for efficiency.
    constexpr FourMat(const FourMat &) noexcept = default;
    constexpr FourMat(FourMat &&) noexcept = default;
    constexpr FourMat &operator=(const FourMat &) noexcept = default;
    constexpr FourMat &operator=(FourMat &&) noexcept = default;

    // Templated assignment operator from any type satisfying Matrix4Indexable.
    // It converts elements to type T and copies them into this matrix.
    template<Matrix4Indexable RHS>
        requires
        // Prevent hijacking same-type assignment.
        (!std::same_as<std::remove_cvref_t<RHS>, FourMat>) &&
        // Ensure each element from RHS is convertible to T.
        std::convertible_to<decltype(std::declval<const RHS &>()(0, 0)), T>
        constexpr FourMat &operator=(const RHS &r) noexcept(
            std::is_nothrow_convertible_v<decltype(std::declval<const RHS &>()(0, 0)), T>) {
        // Use a temporary array for stronger exception safety (copy-then-swap or similar principle).
        T tmp_elems[16];
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                tmp_elems[i * 4 + j] = static_cast<T>(r(i, j));
            }
        }
        // Commit the changes after successful conversion of all elements.
        for (std::size_t i = 0; i < 16; ++i) {
            m_elems[i] = tmp_elems[i];
        }
        return *this;
    }

    // Returns the total number of elements in the matrix (4x4 = 16).
    static constexpr std::size_t size() noexcept {
        return 16;
    }
    // Returns a pointer to the raw underlying data.
    [[nodiscard]] constexpr T *data() noexcept {
        return m_elems;
    }
    // Returns a const pointer to the raw underlying data.
    [[nodiscard]] constexpr const T *data() const noexcept {
        return m_elems;
    }

    // Element access operator using (row, col) indexing.
    // Asserts if indices are out of bounds in debug builds.
    constexpr T &operator()(std::size_t row, std::size_t col) noexcept {
        assert(row < 4 && col < 4 && "FourMat index out of range");
        return m_elems[row * 4 + col];  // Row-major access
    }
    // Const element access operator.
    constexpr const T &operator()(std::size_t row, std::size_t col) const noexcept {
        assert(row < 4 && col < 4 && "FourMat index out of range");
        return m_elems[row * 4 + col];  // Row-major access
    }

    // Standard library compliant iterators for linear traversal over all 16 elements.
    [[nodiscard]] constexpr T *begin() noexcept {
        return data();
    }
    [[nodiscard]] constexpr T *end() noexcept {
        return data() + 16;
    }
    [[nodiscard]] constexpr const T *begin() const noexcept {
        return data();
    }
    [[nodiscard]] constexpr const T *end() const noexcept {
        return data() + 16;
    }
    // Const iterators for when the matrix itself is const.
    [[nodiscard]] constexpr const T *cbegin() const noexcept {
        return begin();
    }
    [[nodiscard]] constexpr const T *cend() const noexcept {
        return end();
    }

    // Converts the FourMat to a std::array<T, 16>.
    [[nodiscard]] constexpr std::array<T, 16> to_array() const noexcept {
        std::array<T, 16> arr;
        for (std::size_t i = 0; i < 16; ++i) {
            arr[i] = m_elems[i];
        }
        return arr;
    }

    // Unary plus operator: returns a copy of itself.
    [[nodiscard]] constexpr FourMat operator+() const noexcept {
        return *this;
    }
    // Unary minus operator: returns a new matrix with all elements negated.
    [[nodiscard]] constexpr FourMat operator-() const noexcept {
        FourMat res;
        for (std::size_t i = 0; i < 16; ++i) {
            res.m_elems[i] = -m_elems[i];
        }
        return res;
    }

    // Equality comparison operator: compares all elements with another Matrix4Indexable type.
    template<Matrix4Indexable RHS>
        requires std::equality_comparable_with<T, decltype(std::declval<const RHS &>()(0, 0))>
    [[nodiscard]] constexpr bool operator==(const RHS &r) const
        noexcept(noexcept(std::declval<T>() == std::declval<decltype(std::declval<const RHS &>()(0, 0))>())) {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                if ((*this)(i, j) != r(i, j)) return false;
            }
        }
        return true;
    }

    // Inequality comparison operator: uses the equality operator.
    template<Matrix4Indexable RHS>
    [[nodiscard]] constexpr bool operator!=(const RHS &r) const noexcept(noexcept(*this == r)) {
        return !(*this == r);
    }

    // Element-wise sum with another Matrix4Indexable type (out-of-place).
    // Returns a new FourMat with common type promotion.
    template<Matrix4Indexable RHS>
    [[nodiscard]] constexpr auto element_wise_sum(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r(0, 0))>;
        using R = std::common_type_t<T, R0>;
        FourMat<R> res;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                res(i, j) = R((*this)(i, j)) + R(r(i, j));
            }
        }
        return res;
    }
    // Operator+ overload for element-wise sum.
    template<Matrix4Indexable RHS>
    [[nodiscard]] constexpr auto operator+(const RHS &r) const noexcept {
        return element_wise_sum(r);
    }

    // Element-wise difference with another Matrix4Indexable type (out-of-place).
    // Returns a new FourMat with common type promotion.
    template<Matrix4Indexable RHS>
    [[nodiscard]] constexpr auto element_wise_diff(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r(0, 0))>;
        using R = std::common_type_t<T, R0>;
        FourMat<R> res;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                res(i, j) = R((*this)(i, j)) - R(r(i, j));
            }
        }
        return res;
    }
    // Operator- overload for element-wise difference.
    template<Matrix4Indexable RHS>
    [[nodiscard]] constexpr auto operator-(const RHS &r) const noexcept {
        return element_wise_diff(r);
    }

    // In-place element-wise sum: only if common_type is T (no implicit promotion).
    template<Matrix4Indexable RHS>
    constexpr FourMat &element_wise_sum_inplace(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>
    {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                (*this)(i, j) += r(i, j);
            }
        }
        return *this;
    }
    // Operator+= overload for in-place element-wise sum.
    template<Matrix4Indexable RHS>
    constexpr FourMat &operator+=(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>
    {
        return element_wise_sum_inplace(r);
    }

    // In-place element-wise difference: only if common_type is T.
    template<Matrix4Indexable RHS>
    constexpr FourMat &element_wise_diff_inplace(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>
    {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                (*this)(i, j) -= r(i, j);
            }
        }
        return *this;
    }
    // Operator-= overload for in-place element-wise difference.
    template<Matrix4Indexable RHS>
    constexpr FourMat &operator-=(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>
    {
        return element_wise_diff_inplace(r);
    }

    // Scalar multiplication (out-of-place): returns a new matrix with common type promotion.
    template<class U>
    [[nodiscard]] constexpr auto scalar_mul(const U &s) const noexcept {
        using R = std::common_type_t<T, U>;
        FourMat<R> res;
        for (std::size_t i = 0; i < 16; ++i) {
            res.m_elems[i] = R(m_elems[i]) * s;
        }
        return res;
    }
    // Operator* overload for scalar multiplication (matrix * scalar).
    template<class U>
    [[nodiscard]] constexpr auto operator*(const U &s) const noexcept {
        return scalar_mul(s);
    }

    // Scalar division (out-of-place): returns a new matrix with common type promotion.
    template<class U>
    [[nodiscard]] constexpr auto scalar_div(const U &s) const noexcept {
        using R = std::common_type_t<T, U>;
        FourMat<R> res;
        for (std::size_t i = 0; i < 16; ++i) {
            res.m_elems[i] = R(m_elems[i]) / s;
        }
        return res;
    }
    // Operator/ overload for scalar division (matrix / scalar).
    template<class U>
    [[nodiscard]] constexpr auto operator/(const U &s) const noexcept {
        return scalar_div(s);
    }

    // Commutative scalar multiplication (friend function for scalar * matrix).
    template<class U>
    [[nodiscard]] friend constexpr FourMat<std::common_type_t<T, U>> operator*(const U &s, const FourMat &v) noexcept {
        return v * s;
    }

    // In-place scalar multiplication: only if common_type is T.
    template<class U>
    constexpr FourMat &scalar_mul_inplace(const U &s) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, U>>
    {
        for (std::size_t i = 0; i < 16; ++i) {
            m_elems[i] = T(m_elems[i] * s);
        }
        return *this;
    }
    // Operator*= overload for in-place scalar multiplication.
    template<class U>
    constexpr FourMat &operator*=(const U &s) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, U>>
    {
        return scalar_mul_inplace(s);
    }

    // In-place scalar division: only if common_type is T.
    template<class U>
    constexpr FourMat &scalar_div_inplace(const U &s) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, U>>
    {
        for (std::size_t i = 0; i < 16; ++i) {
            m_elems[i] = T(m_elems[i] / s);
        }
        return *this;
    }
    // Operator/= overload for in-place scalar division.
    template<class U>
    constexpr FourMat &operator/=(const U &s) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, U>>
    {
        return scalar_div_inplace(s);
    }

    // Element-wise (Hadamard) multiplication with another Matrix4Indexable type (out-of-place).
    template<Matrix4Indexable RHS>
    [[nodiscard]] constexpr auto element_wise_mul(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r(0, 0))>;
        using R = std::common_type_t<T, R0>;
        FourMat<R> res;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                res(i, j) = R((*this)(i, j)) * R(r(i, j));
            }
        }
        return res;
    }
    // Operator% overload for element-wise multiplication (Hadamard product).
    template<Matrix4Indexable RHS>
    [[nodiscard]] constexpr auto operator%(const RHS &r) const noexcept {
        return element_wise_mul(r);
    }

    // In-place element-wise (Hadamard) multiplication: only if common_type is T.
    template<Matrix4Indexable RHS>
    constexpr FourMat &element_wise_mul_inplace(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>
    {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                (*this)(i, j) *= r(i, j);
            }
        }
        return *this;
    }
    // Operator%= overload for in-place element-wise multiplication.
    template<Matrix4Indexable RHS>
    constexpr FourMat &operator%=(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>
    {
        return element_wise_mul_inplace(r);
    }

    // Lower the second index: A^{\mu\nu} -> A^{\mu}_{\nu} = A^{\mu\sigma} \eta_{\sigma\nu}
    // This negates spatial columns (indices 1, 2, 3) for each row.
    constexpr FourMat &lower_second_index_inplace() noexcept {
        for (std::size_t i = 0; i < 4; ++i) {  // Iterate over rows (mu)
            (*this)(i, 1) = -(*this)(i, 1);    // Negate column 1
            (*this)(i, 2) = -(*this)(i, 2);    // Negate column 2
            (*this)(i, 3) = -(*this)(i, 3);    // Negate column 3
        }
        return *this;
    }

    [[nodiscard]] constexpr FourMat lower_second_index() const noexcept {
        FourMat tmp = *this;
        return tmp.lower_second_index_inplace();
    }

    // Lower the first index: A^{\mu\nu} -> A_{\mu}^{\nu} = \eta_{\mu\sigma} A^{\sigma\nu}
    // This negates spatial rows (indices 1, 2, 3) for each column.
    constexpr FourMat &lower_first_index_inplace() noexcept {
        for (std::size_t j = 0; j < 4; ++j) {  // Iterate over columns (nu)
            (*this)(1, j) = -(*this)(1, j);    // Negate row 1
            (*this)(2, j) = -(*this)(2, j);    // Negate row 2
            (*this)(3, j) = -(*this)(3, j);    // Negate row 3
        }
        return *this;
    }

    [[nodiscard]] constexpr FourMat lower_first_index() const noexcept {
        FourMat tmp = *this;
        return tmp.lower_first_index_inplace();
    }

    // Standard matrix multiplication (this * r), out-of-place. Define the dot product of two matrices; use the
    // function, do not overload the operator* for consistency with the FourVec class. Returns a new FourMat with common
    // type promotion.
    template<Matrix4Indexable RHS>
    [[nodiscard]] constexpr auto dot(const RHS &r) const noexcept {
        using R0 = uncvref_t<decltype(r(0, 0))>;
        using R = std::common_type_t<T, R0>;
        FourMat<R> res;
        for (std::size_t i = 0; i < 4; ++i) {          // Iterate over rows of 'this'
            for (std::size_t j = 0; j < 4; ++j) {      // Iterate over columns of 'r'
                R sum{};                               // Initialize sum for dot product of row i and column j
                for (std::size_t k = 0; k < 4; ++k) {  // Iterate over common dimension
                    sum += R((*this)(i, k)) * R(r(k, j));
                }
                res(i, j) = sum;
            }
        }
        return res;
    }

    // In-place matrix multiplication (this *= r).
    // Performs the multiplication using the out-of-place dot function and then assigns back. Do not overload the
    // operator* for consistency with the FourVec class.
    template<Matrix4Indexable RHS>
    constexpr FourMat &dot_inplace(const RHS &r) & noexcept
        requires std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>
    {
        *this = (*this).dot(r);  // Safe because operator* returns a new object.
        return *this;
    }

    // In-place transpose: swaps elements across the main diagonal.
    constexpr FourMat &transpose_inplace() noexcept {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = i + 1; j < 4; ++j) {  // Only iterate upper triangle to avoid double swapping
                T temp = (*this)(i, j);
                (*this)(i, j) = (*this)(j, i);
                (*this)(j, i) = temp;
            }
        }
        return *this;
    }
    // Out-of-place transpose: returns a new transposed matrix.
    [[nodiscard]] constexpr FourMat transpose() const noexcept {
        FourMat tmp = *this;
        return tmp.transpose_inplace();
    }

    // In-place conjugation: applies conj_if_needed to each element.
    constexpr FourMat &conjugate_inplace() noexcept {
        for (std::size_t i = 0; i < 16; ++i) {
            m_elems[i] = conj_if_needed(m_elems[i]);
        }
        return *this;
    }
    // Out-of-place conjugation: returns a new conjugated matrix.
    [[nodiscard]] constexpr FourMat conjugate() const noexcept {
        FourMat tmp = *this;
        return tmp.conjugate_inplace();
    }

    // Returns a new FourMat containing the real part of each element.
    [[nodiscard]] constexpr auto real() const noexcept -> FourMat<uncvref_t<decltype(std::real(std::declval<T>()))>> {
        using R = uncvref_t<decltype(std::real(std::declval<T>()))>;
        FourMat<R> res;
        for (std::size_t i = 0; i < 16; ++i) {
            res.m_elems[i] = static_cast<R>(std::real(m_elems[i]));
        }
        return res;
    }

    // Returns a new FourMat containing the imaginary part of each element.
    [[nodiscard]] constexpr auto imag() const noexcept -> FourMat<uncvref_t<decltype(std::imag(std::declval<T>()))>> {
        using R = uncvref_t<decltype(std::imag(std::declval<T>()))>;
        FourMat<R> res;
        for (std::size_t i = 0; i < 16; ++i) {
            res.m_elems[i] = static_cast<R>(std::imag(m_elems[i]));
        }
        return res;
    }

    // Returns a copy of the specified row as a FourVec.
    [[nodiscard]] constexpr FourVec<T> row(std::size_t row_idx) const noexcept {
        assert(row_idx < 4 && "FourMat row index out of range");
        return FourVec<T>((*this)(row_idx, 0), (*this)(row_idx, 1), (*this)(row_idx, 2), (*this)(row_idx, 3));
    }

    // Returns a copy of the specified column as a FourVec.
    [[nodiscard]] constexpr FourVec<T> col(std::size_t col_idx) const noexcept {
        assert(col_idx < 4 && "FourMat col index out of range");
        return FourVec<T>((*this)(0, col_idx), (*this)(1, col_idx), (*this)(2, col_idx), (*this)(3, col_idx));
    }

    // Returns a mutable FourVecView of the specified row.
    [[nodiscard]] constexpr FourVecView<T> row_view(std::size_t row_idx) noexcept {
        assert(row_idx < 4 && "FourMat row index out of range for row_view");
        // For a row in a row-major matrix, elements are contiguous (stride 1)
        return FourVecView<T>(&m_elems[row_idx * 4], 1);
    }
    // Returns a const FourVecView of the specified row.
    [[nodiscard]] constexpr FourVecView<const T> row_view(std::size_t row_idx) const noexcept {
        assert(row_idx < 4 && "FourMat row index out of range for row_view");
        // For a row in a row-major matrix, elements are contiguous (stride 1)
        return FourVecView<const T>(&m_elems[row_idx * 4], 1);
    }

    // Returns a mutable FourVecView of the specified column.
    [[nodiscard]] constexpr FourVecView<T> col_view(std::size_t col_idx) noexcept {
        assert(col_idx < 4 && "FourMat col index out of range for col_view");
        // For a column in a row-major matrix, elements are strided by 4 (row stride)
        return FourVecView<T>(&m_elems[col_idx], 4);
    }
    // Returns a const FourVecView of the specified column.
    [[nodiscard]] constexpr FourVecView<const T> col_view(std::size_t col_idx) const noexcept {
        assert(col_idx < 4 && "FourMat col index out of range for col_view");
        // For a column in a row-major matrix, elements are strided by 4 (row stride)
        return FourVecView<const T>(&m_elems[col_idx], 4);
    }

    // Utility function to print the matrix to an output stream.
    void print(const std::string &text = "", std::ostream &os = std::cout, int width = 10, int precision = 4) const {
        if (!text.empty()) os << text << "\n";
        auto f = os.flags();
        auto p = os.precision();
        os.setf(std::ios::scientific, std::ios::floatfield);  // Use scientific notation for consistent width
        os.precision(precision);
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                os << std::setw(width) << (*this)(i, j)
                   << (j == 3 ? "" : " ");  // Space between columns, newline after row
            }
            os << "\n";
        }
        os.flags(f);      // Restore original flags
        os.precision(p);  // Restore original precision
    }

    // Casts the FourMat to a new FourMat with a different underlying type U.
    template<class U>
    [[nodiscard]] constexpr FourMat<U> cast() const noexcept {
        FourMat<U> res;
        for (std::size_t i = 0; i < 16; ++i) {
            res.m_elems[i] = U(m_elems[i]);  // Static_cast equivalent for numeric types
        }
        return res;
    }
};

}  // namespace srt
