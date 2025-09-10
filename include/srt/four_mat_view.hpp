#pragma once
#include "common.hpp"
#include "fwd.hpp"

namespace srt {

// ---------------------------------------------
/* FourMatView (non-owning) with StridedIterator from common.hpp */
// ---------------------------------------------
template<typename T>
class FourMatView {
   public:
    using value_type = T;

    // StridedIterator for row and column traversal.
    using iterator = StridedIterator<T>;
    using const_iterator = StridedIterator<const T>;

    // Constructor: Takes a pointer to the start of the data, row stride, and optional column stride.
    // Default col_stride of 1 assumes contiguous elements within a row.
    constexpr FourMatView(T *ptr, std::ptrdiff_t row_stride, std::ptrdiff_t col_stride = 1) noexcept
        : ptr_(ptr), row_stride_(row_stride), col_stride_(col_stride) {}

    // Copy constructor for converting a non-const view to a const view.
    constexpr FourMatView(const FourMatView<std::remove_const_t<T>> &other) noexcept
        requires(std::is_const_v<T>)  // This constructor is only enabled if T is const.
        : ptr_(other.ptr_), row_stride_(other.row_stride_), col_stride_(other.col_stride_) {}

    // Returns the number of rows.
    static constexpr std::size_t rows() noexcept {
        return 4;
    }
    // Returns the number of columns.
    static constexpr std::size_t cols() noexcept {
        return 4;
    }
    // Returns the total number of elements (4x4 = 16).
    static constexpr std::size_t size() noexcept {
        return 16;
    }

    // Returns a pointer to the raw underlying data.
    [[nodiscard]] constexpr T *data() noexcept {
        return ptr_;
    }
    // Returns a const pointer to the raw underlying data.
    [[nodiscard]] constexpr const T *data() const noexcept {
        return ptr_;
    }
    // Returns the row stride.
    [[nodiscard]] constexpr std::ptrdiff_t row_stride() const noexcept {
        return row_stride_;
    }
    // Returns the column stride.
    [[nodiscard]] constexpr std::ptrdiff_t col_stride() const noexcept {
        return col_stride_;
    }

    // Element access operator using (row, col) indexing.
    // Asserts if indices are out of bounds in debug builds.
    [[nodiscard]] constexpr T &operator()(std::size_t row, std::size_t col) noexcept {
        assert(row < 4 && col < 4 && "FourMatView index out of range");
        return *(ptr_ + row * row_stride_ + col * col_stride_);
    }
    // Const element access operator.
    [[nodiscard]] constexpr const T &operator()(std::size_t row, std::size_t col) const noexcept {
        assert(row < 4 && col < 4 && "FourMatView index out of range");
        return *(ptr_ + row * row_stride_ + col * col_stride_);
    }

    // Returns an iterator to the beginning of the specified row.
    [[nodiscard]] constexpr iterator row_begin(std::size_t row_idx) noexcept {
        assert(row_idx < 4 && "FourMatView row index out of range for row_begin");
        return iterator(ptr_ + row_idx * row_stride_, col_stride_, 0);
    }
    // Returns an iterator to the end of the specified row.
    [[nodiscard]] constexpr iterator row_end(std::size_t row_idx) noexcept {
        assert(row_idx < 4 && "FourMatView row index out of range for row_end");
        return iterator(ptr_ + row_idx * row_stride_, col_stride_, 4);
    }
    // Returns a const iterator to the beginning of the specified row.
    [[nodiscard]] constexpr const_iterator row_cbegin(std::size_t row_idx) const noexcept {
        assert(row_idx < 4 && "FourMatView row index out of range for row_cbegin");
        return const_iterator(ptr_ + row_idx * row_stride_, col_stride_, 0);
    }
    // Returns a const iterator to the end of the specified row.
    [[nodiscard]] constexpr const_iterator row_cend(std::size_t row_idx) const noexcept {
        assert(row_idx < 4 && "FourMatView row index out of range for row_cend");
        return const_iterator(ptr_ + row_idx * row_stride_, col_stride_, 4);
    }

    // Returns an iterator to the beginning of the specified column.
    // Note: Iterating columns uses row_stride as the 'stride' for the StridedIterator.
    [[nodiscard]] constexpr iterator col_begin(std::size_t col_idx) noexcept {
        assert(col_idx < 4 && "FourMatView col index out of range for col_begin");
        return iterator(ptr_ + col_idx * col_stride_, row_stride_, 0);
    }
    // Returns an iterator to the end of the specified column.
    [[nodiscard]] constexpr iterator col_end(std::size_t col_idx) noexcept {
        assert(col_idx < 4 && "FourMatView col index out of range for col_end");
        return iterator(ptr_ + col_idx * col_stride_, row_stride_, 4);
    }
    // Returns a const iterator to the beginning of the specified column.
    [[nodiscard]] constexpr const_iterator col_cbegin(std::size_t col_idx) const noexcept {
        assert(col_idx < 4 && "FourMatView col index out of range for col_cbegin");
        return const_iterator(ptr_ + col_idx * col_stride_, row_stride_, 0);
    }
    // Returns a const iterator to the end of the specified column.
    [[nodiscard]] constexpr const_iterator col_cend(std::size_t col_idx) const noexcept {
        assert(col_idx < 4 && "FourMatView col index out of range for col_cend");
        return const_iterator(ptr_ + col_idx * col_stride_, row_stride_, 4);
    }

    // Templated assignment operator from any type satisfying Matrix4Indexable.
    // Requires the view's underlying type T to be non-const.
    template<Matrix4Indexable RHS>
        requires(!std::is_const_v<T>) && std::convertible_to<decltype(std::declval<const RHS &>()(0, 0)), T>
    constexpr FourMatView &operator=(const RHS &r) & noexcept(
        std::is_nothrow_convertible_v<decltype(std::declval<const RHS &>()(0, 0)), T>) {
        // Use a temporary array for stronger exception safety (convert all, then write).
        T tmp_elems[16];
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                tmp_elems[i * 4 + j] = static_cast<T>(r(i, j));
            }
        }
        // Commit the changes to the view's memory.
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                (*this)(i, j) = tmp_elems[i * 4 + j];
            }
        }
        return *this;
    }

    // Converts the FourMatView's content to a std::array<T, 16>.
    [[nodiscard]] constexpr std::array<T, 16> to_array() const noexcept {
        std::array<T, 16> arr;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                arr[i * 4 + j] = (*this)(i, j);
            }
        }
        return arr;
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

    // In-place element-wise sum: only if T is non-const and common_type is T (no promotion).
    template<Matrix4Indexable RHS>
    constexpr FourMatView &element_wise_sum_inplace(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>)
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
    constexpr FourMatView &operator+=(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>)
    {
        return element_wise_sum_inplace(r);
    }

    // In-place element-wise difference: only if T is non-const and common_type is T.
    template<Matrix4Indexable RHS>
    constexpr FourMatView &element_wise_diff_inplace(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>)
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
    constexpr FourMatView &operator-=(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>)
    {
        return element_wise_diff_inplace(r);
    }

    // Scalar multiplication (out-of-place): returns a new FourMat with common type promotion.
    template<class U>
    [[nodiscard]] constexpr auto scalar_mul(const U &s) const noexcept {
        using R = std::common_type_t<T, U>;
        FourMat<R> res;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                res(i, j) = R((*this)(i, j)) * s;
            }
        }
        return res;
    }
    // Operator* overload for scalar multiplication (matrix * scalar).
    template<class U>
    [[nodiscard]] constexpr auto operator*(const U &s) const noexcept {
        return scalar_mul(s);
    }

    // Scalar division (out-of-place): returns a new FourMat with common type promotion.
    template<class U>
    [[nodiscard]] constexpr auto scalar_div(const U &s) const noexcept {
        using R = std::common_type_t<T, U>;
        FourMat<R> res;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                res(i, j) = R((*this)(i, j)) / s;
            }
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
    [[nodiscard]] friend constexpr FourMat<std::common_type_t<T, U>> operator*(const U &s,
                                                                               const FourMatView &v) noexcept {
        return v * s;
    }

    // In-place scalar multiplication: only if T is non-const and common_type is T.
    template<class U>
    constexpr FourMatView &scalar_mul_inplace(const U &s) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, U>>)
    {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                (*this)(i, j) = T((*this)(i, j) * s);
            }
        }
        return *this;
    }
    // Operator*= overload for in-place scalar multiplication.
    template<class U>
    constexpr FourMatView &operator*=(const U &s) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, U>>)
    {
        return scalar_mul_inplace(s);
    }

    // In-place scalar division: only if T is non-const and common_type is T.
    template<class U>
    constexpr FourMatView &scalar_div_inplace(const U &s) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, U>>)
    {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                (*this)(i, j) = T((*this)(i, j) / s);
            }
        }
        return *this;
    }
    // Operator/= overload for in-place scalar division.
    template<class U>
    constexpr FourMatView &operator/=(const U &s) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, U>>)
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

    // In-place element-wise (Hadamard) multiplication: only if T is non-const and common_type is T.
    template<Matrix4Indexable RHS>
    constexpr FourMatView &element_wise_mul_inplace(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>)
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
    constexpr FourMatView &operator%=(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>)
    {
        return element_wise_mul_inplace(r);
    }

    // Standard matrix multiplication (this * r), out-of-place.
    // Returns a new FourMat with common type promotion. Uses the function dot, do not overload the operator* for
    // consistency with the FourVec class.
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
    // Requires the view's underlying type T to be non-const. Uses the function dot_inplace, do not overload the
    // operator* for consistency with the FourVec class.
    template<Matrix4Indexable RHS>
    constexpr FourMatView &dot_inplace(const RHS &r) & noexcept
        requires(!std::is_const_v<T> && std::is_same_v<T, std::common_type_t<T, uncvref_t<decltype(r(0, 0))>>>)
    {
        // Perform out-of-place multiplication and assign the result back to the view.
        // This is safe and handles general strides for the view.
        *this = (*this).dot(r);
        return *this;
    }

    // In-place transpose: modifies the view's underlying data.
    // Requires the view's underlying type T to be non-const.
    // Note: For general strides, this involves a temporary copy for correctness.
    constexpr FourMatView &transpose_inplace() noexcept
        requires(!std::is_const_v<T>)
    {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = i + 1; j < 4; ++j) {  // Only iterate upper triangle to avoid double swapping
                T temp = (*this)(i, j);
                (*this)(i, j) = (*this)(j, i);
                (*this)(j, i) = temp;
            }
        }
        return *this;
    }
    // Out-of-place transpose: returns a new transposed FourMat.
    [[nodiscard]] constexpr FourMat<T> transpose() const noexcept {
        FourMat<T> out;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                out(i, j) = (*this)(j, i);  // Swap row and column indices
            }
        }
        return out;
    }

    // In-place lower_second_index: negates spatial columns (indices 1, 2, 3) for each row.
    // Requires the view's underlying type T to be non-const.
    constexpr FourMatView &lower_second_index_inplace() noexcept
        requires(!std::is_const_v<T>)
    {
        for (std::size_t i = 0; i < 4; ++i) {  // Iterate over rows (mu)
            (*this)(i, 1) = -(*this)(i, 1);    // Negate column 1
            (*this)(i, 2) = -(*this)(i, 2);    // Negate column 2
            (*this)(i, 3) = -(*this)(i, 3);    // Negate column 3
        }
        return *this;
    }

    // Out-of-place lower_second_index: returns a new FourMat.
    [[nodiscard]] constexpr FourMat<T> lower_second_index() const noexcept {
        FourMat<T> out;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                out(i, j) = (*this)(i, j);
            }
        }
        return out.lower_second_index_inplace();
    }

    // In-place lower_first_index: negates spatial rows (indices 1, 2, 3) for each column.
    // Requires the view's underlying type T to be non-const.
    constexpr FourMatView &lower_first_index_inplace() noexcept
        requires(!std::is_const_v<T>)
    {
        for (std::size_t j = 0; j < 4; ++j) {  // Iterate over columns (nu)
            (*this)(1, j) = -(*this)(1, j);    // Negate row 1
            (*this)(2, j) = -(*this)(2, j);    // Negate row 2
            (*this)(3, j) = -(*this)(3, j);    // Negate row 3
        }
        return *this;
    }

    // Out-of-place lower_first_index: returns a new FourMat.
    [[nodiscard]] constexpr FourMat<T> lower_first_index() const noexcept {
        FourMat<T> out;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                out(i, j) = (*this)(i, j);
            }
        }
        return out.lower_first_index_inplace();
    }

    // In-place conjugation: applies conj_if_needed to each element.
    // Requires the view's underlying type T to be non-const.
    constexpr FourMatView &conjugate_inplace() noexcept
        requires(!std::is_const_v<T>)
    {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                (*this)(i, j) = conj_if_needed((*this)(i, j));
            }
        }
        return *this;
    }
    // Out-of-place conjugation: returns a new conjugated FourMat.
    [[nodiscard]] constexpr FourMat<T> conjugate() const noexcept {
        FourMat<T> out;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                out(i, j) = conj_if_needed((*this)(i, j));
            }
        }
        return out;
    }

    // Returns a new FourMat containing the real part of each element.
    [[nodiscard]] constexpr auto real() const noexcept {
        using RealType = uncvref_t<decltype(std::real((*this)(0, 0)))>;
        FourMat<RealType> res;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                res(i, j) = std::real((*this)(i, j));
            }
        }
        return res;
    }

    // Returns a new FourMat containing the imaginary part of each element.
    [[nodiscard]] constexpr auto imag() const noexcept {
        using RealType = uncvref_t<decltype(std::imag((*this)(0, 0)))>;
        FourMat<RealType> res;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                res(i, j) = std::imag((*this)(i, j));
            }
        }
        return res;
    }

    // Returns a copy of the specified row as a FourVec.
    [[nodiscard]] constexpr FourVec<T> row(std::size_t row_idx) const noexcept {
        assert(row_idx < 4 && "FourMatView row index out of range");
        return FourVec<T>((*this)(row_idx, 0), (*this)(row_idx, 1), (*this)(row_idx, 2), (*this)(row_idx, 3));
    }

    // Returns a copy of the specified column as a FourVec.
    [[nodiscard]] constexpr FourVec<T> col(std::size_t col_idx) const noexcept {
        assert(col_idx < 4 && "FourMatView col index out of range");
        return FourVec<T>((*this)(0, col_idx), (*this)(1, col_idx), (*this)(2, col_idx), (*this)(3, col_idx));
    }

    // Returns a mutable FourVecView of the specified row.
    [[nodiscard]] constexpr FourVecView<T> row_view(std::size_t row_idx) noexcept {
        assert(row_idx < 4 && "FourMatView row index out of range for row_view");
        // Pointer to the start of the row, with stride equal to col_stride_ of this FourMatView.
        return FourVecView<T>(ptr_ + row_idx * row_stride_, col_stride_);
    }
    // Returns a const FourVecView of the specified row.
    [[nodiscard]] constexpr FourVecView<const T> row_view(std::size_t row_idx) const noexcept {
        assert(row_idx < 4 && "FourMatView row index out of range for row_view");
        // Pointer to the start of the row, with stride equal to col_stride_ of this FourMatView.
        return FourVecView<const T>(ptr_ + row_idx * row_stride_, col_stride_);
    }

    // Returns a mutable FourVecView of the specified column.
    [[nodiscard]] constexpr FourVecView<T> col_view(std::size_t col_idx) noexcept {
        assert(col_idx < 4 && "FourMatView col index out of range for col_view");
        // Pointer to the start of the column, with stride equal to row_stride_ of this FourMatView.
        return FourVecView<T>(ptr_ + col_idx * col_stride_, row_stride_);
    }
    // Returns a const FourVecView of the specified column.
    [[nodiscard]] constexpr FourVecView<const T> col_view(std::size_t col_idx) const noexcept {
        assert(col_idx < 4 && "FourMatView col index out of range for col_view");
        // Pointer to the start of the column, with stride equal to row_stride_ of this FourMatView.
        return FourVecView<const T>(ptr_ + col_idx * col_stride_, row_stride_);
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

    // Casts the FourMatView's content to a new FourMat with a different underlying type U.
    template<class U>
    [[nodiscard]] constexpr FourMat<U> cast() const noexcept {
        FourMat<U> res;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                res(i, j) = U((*this)(i, j));  // Static_cast equivalent for numeric types
            }
        }
        return res;
    }

   private:
    T *ptr_;                     // Pointer to the first element of the view.
    std::ptrdiff_t row_stride_;  // Stride (in number of elements) to move to the next row.
    std::ptrdiff_t col_stride_;  // Stride (in number of elements) to move to the next column.
};

}  // namespace srt
