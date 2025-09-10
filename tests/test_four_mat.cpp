// tests/test_four_mat.cpp
#include "test_utils.hpp"

// Function to run all FourMat tests
bool testFourMat() {
    std::cout << "--- Testing FourMat ---\n";

    // Constructor tests
    srt::FourMat<double> m1;  // Default constructor, all zeros
    TEST_ASSERT(m1(0, 0) == 0 && m1(3, 3) == 0, "FourMat default constructor failed");

    srt::FourMat<double> m2(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    TEST_ASSERT(m2(0, 0) == 1 && m2(1, 1) == 6 && m2(3, 3) == 16, "FourMat param constructor failed");

    std::array<double, 16> arr_mat;
    for (int i = 0; i < 16; ++i) arr_mat[i] = i + 1.0;
    srt::FourMat<double> m3(arr_mat);
    TEST_ASSERT(m3(0, 0) == 1 && m3(1, 1) == 6 && m3(3, 3) == 16, "FourMat array constructor failed");

    // Assignment
    srt::FourMat<double> m4 = m2;
    TEST_ASSERT(m4 == m2, "FourMat assignment failed");

    // Arithmetic
    srt::FourMat<double> m_sum = m2 + m3;  // Element-wise sum
    TEST_ASSERT(m_sum(0, 0) == 2 && m_sum(3, 3) == 32, "FourMat addition failed");

    // Create a copy for in-place sum test as m2's original value is needed for comparison
    srt::FourMat<double> m2_copy_for_inplace = m2;
    m2_copy_for_inplace += m3;  // In-place sum
    TEST_ASSERT(m2_copy_for_inplace == m_sum, "FourMat in-place addition failed");

    srt::FourMat<double> m_scalar_mul = m3 * 2.0;
    TEST_ASSERT(m_scalar_mul(0, 0) == 2 && m_scalar_mul(3, 3) == 32, "FourMat scalar multiplication failed");

    // Matrix multiplication tests (using dot function)
    srt::FourMat<double> mat_A(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);  // Identity matrix
    srt::FourMat<double> mat_B(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);

    srt::FourMat<double> mat_prod = mat_A.dot(mat_B);  // Should be mat_B
    TEST_ASSERT(mat_prod == mat_B, "FourMat matrix dot product by identity failed");

    srt::FourMat<double> mat_C(1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    srt::FourMat<double> mat_D(5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    srt::FourMat<double> expected_CD;
    expected_CD(0, 0) = 1 * 5 + 2 * 7;  // 5 + 14 = 19
    expected_CD(0, 1) = 1 * 6 + 2 * 8;  // 6 + 16 = 22
    expected_CD(0, 2) = 0;
    expected_CD(0, 3) = 0;
    expected_CD(1, 0) = 3 * 5 + 4 * 7;  // 15 + 28 = 43
    expected_CD(1, 1) = 3 * 6 + 4 * 8;  // 18 + 32 = 50
    expected_CD(1, 2) = 0;
    expected_CD(1, 3) = 0;
    expected_CD(2, 0) = 0;
    expected_CD(2, 1) = 0;
    expected_CD(2, 2) = 1;
    expected_CD(2, 3) = 0;
    expected_CD(3, 0) = 0;
    expected_CD(3, 1) = 0;
    expected_CD(3, 2) = 0;
    expected_CD(3, 3) = 1;

    srt::FourMat<double> mat_CD = mat_C.dot(mat_D);
    TEST_ASSERT(mat_CD == expected_CD, "FourMat general matrix dot product failed");

    // In-place matrix multiplication (using dot_inplace function)
    srt::FourMat<double> mat_C_inplace_test = mat_C;
    mat_C_inplace_test.dot_inplace(mat_D);
    TEST_ASSERT(mat_C_inplace_test == expected_CD, "FourMat in-place dot product failed");

    // Transpose
    srt::FourMat<double> m_trans_orig(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    srt::FourMat<double> m_trans_expected(1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16);
    m_trans_orig.transpose_inplace();
    TEST_ASSERT(m_trans_orig == m_trans_expected, "FourMat transpose (inplace) failed");

    srt::FourMat<double> m_trans_orig2(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    srt::FourMat<double> m_trans_out = m_trans_orig2.transpose();
    TEST_ASSERT(m_trans_out == m_trans_expected, "FourMat transpose failed");

    // Complex conjugation
    srt::FourMat<std::complex<double>> mc({{1, 1},
                                           {2, -2},
                                           {3, 3},
                                           {4, -4},
                                           {5, 5},
                                           {6, -6},
                                           {7, 7},
                                           {8, -8},
                                           {9, 9},
                                           {10, -10},
                                           {11, 11},
                                           {12, -12},
                                           {13, 13},
                                           {14, -14},
                                           {15, 15},
                                           {16, -16}});
    mc.conjugate_inplace();
    TEST_ASSERT(mc(0, 0) == std::complex<double>(1, -1) && mc(1, 1) == std::complex<double>(6, 6) &&
                    mc(3, 3) == std::complex<double>(16, 16),
                "FourMat complex conjugate (inplace) failed");

    // Real and Imaginary parts
    srt::FourMat<std::complex<double>> mc_original({{1, 1},
                                                    {2, -2},
                                                    {3, 3},
                                                    {4, -4},
                                                    {5, 5},
                                                    {6, -6},
                                                    {7, 7},
                                                    {8, -8},
                                                    {9, 9},
                                                    {10, -10},
                                                    {11, 11},
                                                    {12, -12},
                                                    {13, 13},
                                                    {14, -14},
                                                    {15, 15},
                                                    {16, -16}});
    srt::FourMat<double> mc_real = mc_original.real();
    TEST_ASSERT(mc_real(0, 0) == 1.0 && mc_real(1, 1) == 6.0 && mc_real(3, 3) == 16.0, "FourMat real part failed");

    srt::FourMat<double> mc_imag = mc_original.imag();
    TEST_ASSERT(mc_imag(0, 0) == 1.0 && mc_imag(1, 1) == -6.0 && mc_imag(3, 3) == -16.0,
                "FourMat imaginary part failed");

    // Tests for Index Lowering
    srt::FourMat<double> m_lower_test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    // Test lower_second_index_inplace()
    srt::FourMat<double> m_lower_second_expected(1, -2, -3, -4, 5, -6, -7, -8, 9, -10, -11, -12, 13, -14, -15, -16);
    srt::FourMat<double> m_lower_second_inplace_copy = m_lower_test;
    m_lower_second_inplace_copy.lower_second_index_inplace();
    TEST_ASSERT(m_lower_second_inplace_copy == m_lower_second_expected, "FourMat lower_second_index_inplace() failed");

    // Test lower_second_index()
    srt::FourMat<double> m_lower_second_out = m_lower_test.lower_second_index();
    TEST_ASSERT(m_lower_second_out == m_lower_second_expected, "FourMat lower_second_index() failed");

    // Test lower_first_index_inplace()
    srt::FourMat<double> m_lower_first_expected(1, 2, 3, 4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16);
    srt::FourMat<double> m_lower_first_inplace_copy = m_lower_test;
    m_lower_first_inplace_copy.lower_first_index_inplace();
    TEST_ASSERT(m_lower_first_inplace_copy == m_lower_first_expected, "FourMat lower_first_index_inplace() failed");

    // Test lower_first_index()
    srt::FourMat<double> m_lower_first_out = m_lower_test.lower_first_index();
    TEST_ASSERT(m_lower_first_out == m_lower_first_expected, "FourMat lower_first_index() failed");

    // --- New Tests for Row/Column Accessors ---
    srt::FourMat<double> m_accessor_test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    // row()
    srt::FourVec<double> expected_row0(1, 2, 3, 4);
    TEST_ASSERT(m_accessor_test.row(0) == expected_row0, "FourMat row(0) failed");
    srt::FourVec<double> expected_row2(9, 10, 11, 12);
    TEST_ASSERT(m_accessor_test.row(2) == expected_row2, "FourMat row(2) failed");

    // col()
    srt::FourVec<double> expected_col0(1, 5, 9, 13);
    TEST_ASSERT(m_accessor_test.col(0) == expected_col0, "FourMat col(0) failed");
    srt::FourVec<double> expected_col3(4, 8, 12, 16);
    TEST_ASSERT(m_accessor_test.col(3) == expected_col3, "FourMat col(3) failed");

    // row_view()
    srt::FourVecView<double> row0_view = m_accessor_test.row_view(0);
    TEST_ASSERT(row0_view == expected_row0, "FourMat row_view(0) failed");
    // Test modification through row_view
    row0_view[0] = 100.0;
    TEST_ASSERT(m_accessor_test(0, 0) == 100.0, "FourMat row_view modification failed");
    m_accessor_test(0, 0) = 1.0;  // Reset for next tests

    // col_view()
    srt::FourVecView<double> col1_view = m_accessor_test.col_view(1);
    srt::FourVec<double> expected_col1(2, 6, 10, 14);
    TEST_ASSERT(col1_view == expected_col1, "FourMat col_view(1) failed");
    // Test modification through col_view
    col1_view[0] = 200.0;  // Modifies m_accessor_test(0,1)
    TEST_ASSERT(m_accessor_test(0, 1) == 200.0, "FourMat col_view modification failed");
    m_accessor_test(0, 1) = 2.0;  // Reset

    // const row_view() and col_view()
    const srt::FourMat<double> const_m_accessor_test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    srt::FourVecView<const double> const_row0_view = const_m_accessor_test.row_view(0);
    TEST_ASSERT(const_row0_view == expected_row0, "FourMat const row_view(0) failed");
    srt::FourVecView<const double> const_col0_view = const_m_accessor_test.col_view(0);
    TEST_ASSERT(const_col0_view == expected_col0, "FourMat const col_view(0) failed");

    std::cout << "--- FourMat tests PASSED ---\n\n";
    return true;
}

bool testFourMat_II() {
    std::cout << "--- Testing FourMat_II ---\n";
    srt::FourMat<double> M{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    // element access and data layout (row-major)
    TEST_ASSERT(M(0, 0) == 1, "element access failed");
    TEST_ASSERT(M(3, 3) == 16, "element access failed");
    M(2, 1) = 42;
    TEST_ASSERT(M(2, 1) == 42, "element access failed");

    // row()/col() copies
    auto r1 = M.row(2);
    TEST_ASSERT((r1.to_array() == std::array<double, 4>{9, 42, 11, 12}), "row() failed");
    auto c1 = M.col(1);
    TEST_ASSERT((c1.to_array() == std::array<double, 4>{2, 6, 42, 14}), "col() failed");

    // row_view (mutable) and const row_view
    auto rv = M.row_view(1);
    rv[2] = -7;
    TEST_ASSERT(M(1, 2) == -7, "row_view() failed");
    const auto &Mc = M;
    auto rvc = Mc.row_view(1);
    TEST_ASSERT(rvc[2] == -7, "row_view() failed");

    // lower index operations
    srt::FourMat<double> A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto A_low2 = A.lower_second_index();  // negate spatial columns
    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT(A_low2(i, 0) == A(i, 0), "lower_second_index() failed");
        TEST_ASSERT(A_low2(i, 1) == -A(i, 1), "lower_second_index() failed");
        TEST_ASSERT(A_low2(i, 2) == -A(i, 2), "lower_second_index() failed");
        TEST_ASSERT(A_low2(i, 3) == -A(i, 3), "lower_second_index() failed");
    }
    auto A_low1 = A.lower_first_index();  // negate spatial rows
    for (int j = 0; j < 4; ++j) {
        TEST_ASSERT(A_low1(0, j) == A(0, j), "lower_first_index() failed");
        TEST_ASSERT(A_low1(1, j) == -A(1, j), "lower_first_index() failed");
        TEST_ASSERT(A_low1(2, j) == -A(2, j), "lower_first_index() failed");
        TEST_ASSERT(A_low1(3, j) == -A(3, j), "lower_first_index() failed");
    }

    // Hadamard product and dot (matrix multiply)
    srt::FourMat<int> B{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    auto H = A % B;  // element-wise product
    TEST_ASSERT(H(2, 3) == A(2, 3) * B(2, 3), "Hadamard product failed");

    auto C = A.dot(B);  // matrix product
    // check (0,0): sum_k A(0,k)*B(k,0) -> 1*1 + 2*2 + 3*3 + 4*4 = 30
    TEST_ASSERT(C(0, 0) == 30, "matrix product failed");

    // mat-vec multiply (right)
    srt::FourVec<double> v{1, 2, 3, 4};
    auto r = v.dot_four_right(A);
    // r[i] = sum_j A(i,j)*v[j]
    TEST_ASSERT(r[0] == 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4, "matrix-vector product failed");
    assert(r[3] == 13 * 1 + 14 * 2 + 15 * 3 + 16 * 4);

    // cast
    auto Ai = A.cast<int>();
    static_assert(std::is_same_v<decltype(Ai)::value_type, int>);

    std::cout << "--- FourMat_II tests PASSED ---\n\n";
    return true;
}