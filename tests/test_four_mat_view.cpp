// tests/test_four_mat_view.cpp
#include "test_utils.hpp"

// Function to run all FourMatView tests
bool testFourMatView() {
    std::cout << "--- Testing FourMatView ---\n";

    std::vector<double> data(20);  // More than 16 elements
    for (int i = 0; i < 20; ++i) data[i] = i + 1.0;

    // Contiguous view: A 4x4 matrix starting at data[0]
    // row_stride = 4 (elements to jump to next row)
    // col_stride = 1 (elements to jump to next column within a row)
    srt::FourMatView<double> mv1(&data[0], 4, 1);
    TEST_ASSERT(mv1(0, 0) == 1 && mv1(0, 1) == 2 && mv1(1, 0) == 5 && mv1(3, 3) == 16,
                "FourMatView contiguous construction failed");

    // Modify via view
    mv1(0, 0) = 100.0;
    TEST_ASSERT(data[0] == 100.0, "FourMatView modification failed");

    // Test matrix assignment from FourMat
    srt::FourMat<double> source_mat(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160);
    mv1 = source_mat;  // Assign to the view
    // Verify changes in underlying data through the view
    TEST_ASSERT(data[0] == 10 && data[1] == 20 && data[4] == 50 && data[15] == 160,
                "FourMatView assignment from FourMat failed");

    // Strided view example: Viewing elements of a column from a larger matrix
    // This isn't a 4x4 matrix view, but demonstrates striding
    std::vector<double> big_mat(5 * 5);  // A 5x5 matrix for example
    for (int i = 0; i < 25; ++i) big_mat[i] = i + 1.0;
    // View a column (e.g., column 1, which is elements 2, 7, 12, 17 in a 1-indexed view)
    // The `FourVecView` is more appropriate for viewing a single column as a vector.
    srt::FourVecView<double> col_view_as_vec(&big_mat[1], 5);  // Start at big_mat[1], stride 5
    TEST_ASSERT(
        col_view_as_vec[0] == 2 && col_view_as_vec[1] == 7 && col_view_as_vec[2] == 12 && col_view_as_vec[3] == 17,
        "FourVecView from column failed (used to show striding)");

    // Matrix multiplication tests (using dot function)
    srt::FourMat<double> mat_view_A_data(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);  // Identity matrix
    srt::FourMat<double> mat_view_B_data(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
    srt::FourMatView<double> mv_A(&mat_view_A_data.m_elems[0], 4, 1);
    srt::FourMatView<double> mv_B(&mat_view_B_data.m_elems[0], 4, 1);

    srt::FourMat<double> mv_prod = mv_A.dot(mv_B);  // Should result in B
    TEST_ASSERT(mv_prod == mat_view_B_data, "FourMatView matrix dot product by identity failed");

    // In-place matrix multiplication (using dot_inplace function)
    std::vector<double> in_place_data(16);
    for (int i = 0; i < 16; ++i) in_place_data[i] = i + 1.0;
    srt::FourMatView<double> mv_in_place(&in_place_data[0], 4, 1);
    srt::FourMat<double> multiplier(  // Simple diagonal multiplier
        2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2);
    mv_in_place.dot_inplace(multiplier);  // All elements in 'in_place_data' should be doubled
    TEST_ASSERT(in_place_data[0] == 2 && in_place_data[1] == 4 && in_place_data[15] == 32,
                "FourMatView in-place matrix dot product failed");

    // Transpose (inplace)
    std::vector<double> trans_data(16);
    for (int i = 0; i < 16; ++i) trans_data[i] = i + 1.0;
    srt::FourMatView<double> mv_trans_inplace(&trans_data[0], 4, 1);
    srt::FourMat<double> trans_expected(1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16);
    mv_trans_inplace.transpose_inplace();
    TEST_ASSERT(mv_trans_inplace == trans_expected, "FourMatView transpose (inplace) failed");

    // Transpose (out-of-place)
    std::vector<double> trans_data_orig(16);
    for (int i = 0; i < 16; ++i) trans_data_orig[i] = i + 1.0;
    srt::FourMatView<double> mv_trans_orig_view(&trans_data_orig[0], 4, 1);
    srt::FourMat<double> mv_trans_out = mv_trans_orig_view.transpose();
    TEST_ASSERT(mv_trans_out == trans_expected, "FourMatView transpose (out-of-place) failed");

    // Complex conjugation (inplace)
    std::complex<double> complex_mat_data[16];
    for (int i = 0; i < 16; ++i) complex_mat_data[i] = std::complex<double>(i + 1, -(i + 1));
    srt::FourMatView<std::complex<double>> mc_view_inplace(complex_mat_data, 4, 1);
    mc_view_inplace.conjugate_inplace();
    TEST_ASSERT(complex_mat_data[0] == std::complex<double>(1, 1) &&
                    complex_mat_data[5] == std::complex<double>(6, 6) &&
                    complex_mat_data[15] == std::complex<double>(16, 16),
                "FourMatView complex conjugate (inplace) failed");

    // Complex conjugation (out-of-place)
    std::complex<double> complex_mat_data_orig[16];
    for (int i = 0; i < 16; ++i) complex_mat_data_orig[i] = std::complex<double>(i + 1, -(i + 1));
    srt::FourMatView<std::complex<double>> mc_view_orig(complex_mat_data_orig, 4, 1);
    srt::FourMat<std::complex<double>> mc_conj_out = mc_view_orig.conjugate();
    TEST_ASSERT(mc_conj_out(0, 0) == std::complex<double>(1, 1) && mc_conj_out(1, 1) == std::complex<double>(6, 6) &&
                    mc_conj_out(3, 3) == std::complex<double>(16, 16),
                "FourMatView complex conjugate (out-of-place) failed");

    // Real and Imaginary parts
    srt::FourMat<double> real_part_view = mc_view_orig.real();
    TEST_ASSERT(real_part_view(0, 0) == 1.0 && real_part_view(1, 1) == 6.0 && real_part_view(3, 3) == 16.0,
                "FourMatView real part failed");
    srt::FourMat<double> imag_part_view = mc_view_orig.imag();
    TEST_ASSERT(imag_part_view(0, 0) == -1.0 && imag_part_view(1, 1) == -6.0 && imag_part_view(3, 3) == -16.0,
                "FourMatView imaginary part failed");

    // Tests for Index Lowering
    std::vector<double> lower_test_data(16);
    for (int i = 0; i < 16; ++i) lower_test_data[i] = i + 1.0;
    srt::FourMatView<double> mv_lower_test(&lower_test_data[0], 4, 1);

    // Test lower_second_index_inplace()
    srt::FourMat<double> mv_lower_second_expected(1, -2, -3, -4, 5, -6, -7, -8, 9, -10, -11, -12, 13, -14, -15, -16);
    std::vector<double> lower_second_inplace_data = lower_test_data;  // Copy data for inplace test
    srt::FourMatView<double> mv_lower_second_inplace_view(&lower_second_inplace_data[0], 4, 1);
    mv_lower_second_inplace_view.lower_second_index_inplace();
    TEST_ASSERT(mv_lower_second_inplace_view == mv_lower_second_expected,
                "FourMatView lower_second_index_inplace() failed");

    // Test lower_second_index()
    srt::FourMat<double> mv_lower_second_out = mv_lower_test.lower_second_index();
    TEST_ASSERT(mv_lower_second_out == mv_lower_second_expected, "FourMatView lower_second_index() failed");

    // Test lower_first_index_inplace()
    srt::FourMat<double> mv_lower_first_expected(1, 2, 3, 4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16);
    std::vector<double> lower_first_inplace_data = lower_test_data;  // Copy data for inplace test
    srt::FourMatView<double> mv_lower_first_inplace_view(&lower_first_inplace_data[0], 4, 1);
    mv_lower_first_inplace_view.lower_first_index_inplace();
    TEST_ASSERT(mv_lower_first_inplace_view == mv_lower_first_expected,
                "FourMatView lower_first_index_inplace() failed");

    // Test lower_first_index()
    srt::FourMat<double> mv_lower_first_out = mv_lower_test.lower_first_index();
    TEST_ASSERT(mv_lower_first_out == mv_lower_first_expected, "FourMatView lower_first_index() failed");

    // --- New Tests for Row/Column Accessors (FourMatView) ---
    std::vector<double> mv_accessor_test_data(16);
    for (int i = 0; i < 16; ++i) mv_accessor_test_data[i] = i + 1.0;
    srt::FourMatView<double> mv_accessor_test_view(&mv_accessor_test_data[0], 4, 1);

    // row()
    srt::FourVec<double> expected_row0(1, 2, 3, 4);
    TEST_ASSERT(mv_accessor_test_view.row(0) == expected_row0, "FourMatView row(0) failed");
    srt::FourVec<double> expected_row2(9, 10, 11, 12);
    TEST_ASSERT(mv_accessor_test_view.row(2) == expected_row2, "FourMatView row(2) failed");

    // col()
    srt::FourVec<double> expected_col0(1, 5, 9, 13);
    TEST_ASSERT(mv_accessor_test_view.col(0) == expected_col0, "FourMatView col(0) failed");
    srt::FourVec<double> expected_col3(4, 8, 12, 16);
    TEST_ASSERT(mv_accessor_test_view.col(3) == expected_col3, "FourMatView col(3) failed");

    // row_view()
    srt::FourVecView<double> row0_view = mv_accessor_test_view.row_view(0);
    TEST_ASSERT(row0_view == expected_row0, "FourMatView row_view(0) failed");
    // Test modification through row_view
    row0_view[0] = 100.0;
    TEST_ASSERT(mv_accessor_test_view(0, 0) == 100.0, "FourMatView row_view modification failed");
    mv_accessor_test_view(0, 0) = 1.0;  // Reset for next tests

    // col_view()
    srt::FourVecView<double> col1_view = mv_accessor_test_view.col_view(1);
    srt::FourVec<double> expected_col1(2, 6, 10, 14);
    TEST_ASSERT(col1_view == expected_col1, "FourMatView col_view(1) failed");
    // Test modification through col_view
    col1_view[0] = 200.0;  // Modifies mv_accessor_test_view(0,1)
    TEST_ASSERT(mv_accessor_test_view(0, 1) == 200.0, "FourMatView col_view modification failed");
    mv_accessor_test_view(0, 1) = 2.0;  // Reset

    // const row_view() and col_view()
    const srt::FourMatView<double> const_mv_accessor_test_view(&mv_accessor_test_data[0], 4, 1);
    srt::FourVecView<const double> const_row0_view = const_mv_accessor_test_view.row_view(0);
    TEST_ASSERT(const_row0_view == expected_row0, "FourMatView const row_view(0) failed");
    srt::FourVecView<const double> const_col0_view = const_mv_accessor_test_view.col_view(0);
    TEST_ASSERT(const_col0_view == expected_col0, "FourMatView const col_view(0) failed");

    std::cout << "--- FourMatView tests PASSED ---\n\n";
    return true;
}

bool testFourMatView_II() {
    std::cout << "--- Testing FourMatView_II ---\n";
    // Build a 4x4 matrix in a 1D buffer with row-major layout
    double buf[16];
    for (int i = 0; i < 16; ++i) buf[i] = i + 1;

    // Row stride = 4, col stride = 1 (contiguous rows)
    srt::FourMatView<double> V(buf, /*row_stride*/ 4, /*col_stride*/ 1);
    TEST_ASSERT(V(0, 0) == 1, "FourMatView construction failed");
    TEST_ASSERT(V(3, 3) == 16, "FourMatView construction failed");

    // mutate via view
    V(2, 1) = 123.0;
    TEST_ASSERT(buf[2 * 4 + 1] == 123.0, "FourMatView mutation failed");

    // row_view and col_view as FourVecView
    auto rv = V.row_view(2);
    TEST_ASSERT(rv[1] == 123.0, "FourMatView row_view failed");
    rv[3] = -5.0;
    TEST_ASSERT(V(2, 3) == -5.0, "FourMatView row_view modification failed");

    auto cv = V.col_view(1);
    // elements at (0,1),(1,1),(2,1),(3,1)
    TEST_ASSERT(cv[0] == 2.0 && cv[1] == 6.0 && cv[2] == 123.0 && cv[3] == 14.0, "FourMatView col_view failed");

    // element-wise ops
    srt::FourMat<double> A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    V += A;
    TEST_ASSERT(V(0, 0) == 2.0 && V(2, 1) == 133.0, "FourMatView element-wise ops failed");

    V -= A;
    TEST_ASSERT(V(0, 0) == 1.0 && V(2, 1) == 123.0, "FourMatView element-wise ops failed");

    V %= A;
    TEST_ASSERT(V(0, 0) == 1.0 && V(2, 1) == 123.0 * 10.0, "FourMatView element-wise ops failed");

    // matrix multiply via dot()
    auto R = V.dot(A);
    // R(0,0) = sum_k V(0,k) * A(k,0) = 1*1*1 + 2*2*5 + 3*3*9 + 4*4*13 = 310
    TEST_ASSERT(R(0, 0) == 310.0, "FourMatView matrix multiply via dot() failed");

    // right-multiply by vector
    srt::FourVec<double> x{1, 2, 3, 4};
    auto y = x.dot_four_right(V);
    TEST_ASSERT(y[0] == 1 * 1 * 1 + 2 * 2 * 2 + 3 * 3 * 3 + 4 * 4 * 4,
                "FourMatView matrix multiply via dot_four_right() failed");

    // lowering operations
    auto L1 = V.lower_first_index();
    auto L2 = V.lower_second_index();
    // spot checks (sign flips on spatial rows/cols)
    TEST_ASSERT(L1(1, 0) == -V(1, 0) && L1(0, 0) == V(0, 0), "FourMatView lowering operations failed");
    TEST_ASSERT(L2(0, 1) == -V(0, 1) && L2(0, 0) == V(0, 0), "FourMatView lowering operations failed");

    std::cout << "test_four_mat_view: OK\n";

    return true;
}