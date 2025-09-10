// tests/test_four_vec_view.cpp
#include "test_utils.hpp"

// Function to run all FourVecView tests
bool testFourVecView() {
    std::cout << "--- Testing FourVecView ---\n";

    std::vector<double> data = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0};  // Some extra data
    // Contiguous view starting at data[0]
    srt::FourVecView<double> v_view(&data[0], 1);

    TEST_ASSERT(v_view[0] == 10.0 && v_view[1] == 11.0 && v_view[2] == 12.0 && v_view[3] == 13.0,
                "FourVecView contiguous construction failed");

    // Modify via view
    v_view[0] = 100.0;
    TEST_ASSERT(data[0] == 100.0, "FourVecView modification failed");

    // Assignment to view from an owning FourVec
    srt::FourVec<double> v_owner(1.0, 2.0, 3.0, 4.0);
    v_view = v_owner;  // Assigns (1,2,3,4) to data[0..3]
    TEST_ASSERT(v_view[0] == 1.0 && v_view[1] == 2.0 && v_view[2] == 3.0 && v_view[3] == 4.0 && data[0] == 1.0 &&
                    data[1] == 2.0 && data[2] == 3.0 && data[3] == 4.0,
                "FourVecView assignment from FourVec failed");

    // Test with stride: view elements at indices 0, 2, 4, 6 in strided_data
    std::vector<double> strided_data = {1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0};
    srt::FourVecView<double> v_strided_view(&strided_data[0], 2);  // Stride 2: views (1, 2, 3, 4)
    TEST_ASSERT(
        v_strided_view[0] == 1.0 && v_strided_view[1] == 2.0 && v_strided_view[2] == 3.0 && v_strided_view[3] == 4.0,
        "Strided FourVecView construction failed");

    // Arithmetic operations: FourVec + FourVecView -> FourVec
    srt::FourVec<double> v_add_result = v_owner + v_strided_view;  // (1+1, 2+2, 3+3, 4+4) = (2,4,6,8)
    TEST_ASSERT(v_add_result[0] == 2.0 && v_add_result[1] == 4.0 && v_add_result[2] == 6.0 && v_add_result[3] == 8.0,
                "FourVecView addition (out-of-place) failed");

    // In-place addition: FourVecView += FourVec
    // The underlying data pointed to by v_strided_view will change.
    v_strided_view += v_owner;  // strided_data[0] is now 1+1=2, strided_data[2] is 2+2=4, etc.
    TEST_ASSERT(
        v_strided_view[0] == 2.0 && v_strided_view[1] == 4.0 && v_strided_view[2] == 6.0 && v_strided_view[3] == 8.0,
        "FourVecView in-place addition failed");

    // Test comparison
    srt::FourVec<double> v_compare(2.0, 4.0, 6.0, 8.0);
    TEST_ASSERT(v_strided_view == v_compare, "FourVecView comparison failed");

    // Scalar multiplication (out-of-place)
    srt::FourVec<double> v_scalar_mul_view = v_strided_view * 2.0;  // (4, 8, 12, 16)
    TEST_ASSERT(v_scalar_mul_view[0] == 4.0 && v_scalar_mul_view[1] == 8.0 && v_scalar_mul_view[2] == 12.0 &&
                    v_scalar_mul_view[3] == 16.0,
                "FourVecView scalar multiplication failed");

    // In-place scalar multiplication
    v_strided_view *= 0.5;  // (1, 2, 3, 4) again
    TEST_ASSERT(
        v_strided_view[0] == 1.0 && v_strided_view[1] == 2.0 && v_strided_view[2] == 3.0 && v_strided_view[3] == 4.0,
        "FourVecView in-place scalar multiplication failed");

    // Conjugation for complex view (inplace only if T is non-const)
    // Make sure the underlying array is mutable
    std::complex<double> complex_data_arr[4] = {std::complex<double>(1.0, 1.0), std::complex<double>(2.0, -2.0),
                                                std::complex<double>(3.0, 3.0), std::complex<double>(4.0, -4.0)};
    srt::FourVecView<std::complex<double>> vc_view(complex_data_arr, 1);
    vc_view.conjugate_inplace();
    TEST_ASSERT(complex_data_arr[0] == std::complex<double>(1.0, -1.0) &&
                    complex_data_arr[1] == std::complex<double>(2.0, 2.0) &&
                    complex_data_arr[2] == std::complex<double>(3.0, -3.0) &&
                    complex_data_arr[3] == std::complex<double>(4.0, 4.0),
                "FourVecView complex conjugation (inplace) failed");

    // Out-of-place conjugation
    std::complex<double> complex_data_orig[4] = {std::complex<double>(1.0, 1.0), std::complex<double>(2.0, -2.0),
                                                 std::complex<double>(3.0, 3.0), std::complex<double>(4.0, -4.0)};
    srt::FourVecView<std::complex<double>> vc_view_orig(complex_data_orig, 1);
    srt::FourVec<std::complex<double>> vc_conj_out = vc_view_orig.conjugate();
    TEST_ASSERT(vc_conj_out[0] == std::complex<double>(1.0, -1.0) && vc_conj_out[1] == std::complex<double>(2.0, 2.0) &&
                    vc_conj_out[2] == std::complex<double>(3.0, -3.0) &&
                    vc_conj_out[3] == std::complex<double>(4.0, 4.0),
                "FourVecView complex conjugation (out-of-place) failed");

    // Real and Imaginary parts
    srt::FourVec<double> real_part_view = vc_view_orig.real();
    TEST_ASSERT(
        real_part_view[0] == 1.0 && real_part_view[1] == 2.0 && real_part_view[2] == 3.0 && real_part_view[3] == 4.0,
        "FourVecView real part failed");
    srt::FourVec<double> imag_part_view = vc_view_orig.imag();
    TEST_ASSERT(
        imag_part_view[0] == 1.0 && imag_part_view[1] == -2.0 && imag_part_view[2] == 3.0 && imag_part_view[3] == -4.0,
        "FourVecView imaginary part failed");

    // --- New Tests for Vector-Matrix Multiplication (FourVecView) ---
    std::vector<double> v_test_data = {1.0, 2.0, 3.0, 4.0};
    srt::FourVecView<double> v_test_view(&v_test_data[0], 1);
    srt::FourMat<double> m_test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    // dot_four_left (Euclidean: row vector * matrix)
    srt::FourVec<double> expected_dot_four_left(90.0, 100.0, 110.0, 120.0);
    srt::FourVec<double> res_dot_four_left = v_test_view.dot_four_left(m_test);
    TEST_ASSERT(res_dot_four_left == expected_dot_four_left, "dot_four_left (FourVecView) failed");

    // dot_four_left_inplace
    std::vector<double> v_test_inplace_left_data = {1.0, 2.0, 3.0, 4.0};
    srt::FourVecView<double> v_test_inplace_left_view(&v_test_inplace_left_data[0], 1);
    v_test_inplace_left_view.dot_four_left_inplace(m_test);
    TEST_ASSERT(v_test_inplace_left_view == expected_dot_four_left, "dot_four_left_inplace (FourVecView) failed");

    // dot_four_right (Euclidean: matrix * column vector)
    srt::FourVec<double> expected_dot_four_right(30.0, 70.0, 110.0, 150.0);
    srt::FourVec<double> res_dot_four_right = v_test_view.dot_four_right(m_test);
    TEST_ASSERT(res_dot_four_right == expected_dot_four_right, "dot_four_right (FourVecView) failed");

    // dot_four_right_inplace
    std::vector<double> v_test_inplace_right_data = {1.0, 2.0, 3.0, 4.0};
    srt::FourVecView<double> v_test_inplace_right_view(&v_test_inplace_right_data[0], 1);
    v_test_inplace_right_view.dot_four_right_inplace(m_test);
    TEST_ASSERT(v_test_inplace_right_view == expected_dot_four_right, "dot_four_right_inplace (FourVecView) failed");

    // minkowski_dot_four_left (Minkowski: row vector * matrix)
    srt::FourVec<double> expected_mink_dot_four_left(-88.0, -96.0, -104.0, -112.0);
    srt::FourVec<double> res_mink_dot_four_left = v_test_view.minkowski_dot_four_left(m_test);
    TEST_ASSERT(res_mink_dot_four_left == expected_mink_dot_four_left, "minkowski_dot_four_left (FourVecView) failed");

    // minkowski_dot_four_left_inplace
    std::vector<double> v_test_mink_inplace_left_data = {1.0, 2.0, 3.0, 4.0};
    srt::FourVecView<double> v_test_mink_inplace_left_view(&v_test_mink_inplace_left_data[0], 1);
    v_test_mink_inplace_left_view.minkowski_dot_four_left_inplace(m_test);
    TEST_ASSERT(v_test_mink_inplace_left_view == expected_mink_dot_four_left,
                "minkowski_dot_four_left_inplace (FourVecView) failed");

    // minkowski_dot_four_right (Minkowski: matrix * column vector)
    srt::FourVec<double> expected_mink_dot_four_right(-28.0, -60.0, -92.0, -124.0);
    srt::FourVec<double> res_mink_dot_four_right = v_test_view.minkowski_dot_four_right(m_test);
    TEST_ASSERT(res_mink_dot_four_right == expected_mink_dot_four_right,
                "minkowski_dot_four_right (FourVecView) failed");

    // minkowski_dot_four_right_inplace
    std::vector<double> v_test_mink_inplace_right_data = {1.0, 2.0, 3.0, 4.0};
    srt::FourVecView<double> v_test_mink_inplace_right_view(&v_test_mink_inplace_right_data[0], 1);
    v_test_mink_inplace_right_view.minkowski_dot_four_right_inplace(m_test);
    TEST_ASSERT(v_test_mink_inplace_right_view == expected_mink_dot_four_right,
                "minkowski_dot_four_right_inplace (FourVecView) failed");

    std::cout << "--- FourVecView tests PASSED ---\n\n";
    return true;
}

bool testFourVecView_II() {
    std::cout << "--- Testing FourVecView_II ---\n";
    double raw[8] = {10, 20, 30, 40, 1, 2, 3, 4};

    // contiguous view over the last four elements
    srt::FourVecView<double> v(&raw[4], 1);
    TEST_ASSERT(v[0] == 1 && v[3] == 4, "FourVecView_II construction failed");

    // non-contiguous (stride 2) view over even-indexed first 4 values: 10,30,  (we need four items -> make another
    // layout)
    double buf[8] = {1, 10, 2, 20, 3, 30, 4, 40};
    srt::FourVecView<double> vs(&buf[0], 2);  // elements: 1,2,3,4 (taking every 2nd)
    TEST_ASSERT(vs[0] == 1 && vs[1] == 2 && vs[2] == 3 && vs[3] == 4, "FourVecView_II construction failed");

    // iteration uses stride
    double sum = 0;
    for (auto it = vs.begin(); it != vs.end(); ++it) sum += *it;
    TEST_ASSERT(sum == 10, "FourVecView_II iteration failed");

    // write-through (mutability)
    vs[2] = 99;
    TEST_ASSERT(buf[4] == 99, "FourVecView_II write-through failed");

    // assignment from FourVec into view (zero-copy write)
    srt::FourVec<int> src{7, 8, 9, 10};
    vs = src;
    TEST_ASSERT(buf[0] == 7 && buf[2] == 8 && buf[4] == 9 && buf[6] == 10, "FourVecView_II assignment failed");

    // element-wise ops on view (+=, -=, %=)
    srt::FourVec<double> ones{1, 1, 1, 1};
    vs += ones;
    TEST_ASSERT((buf[0] == 8 && buf[2] == 9 && buf[4] == 10 && buf[6] == 11), "FourVecView_II += failed");
    vs -= ones;
    TEST_ASSERT((buf[0] == 7 && buf[2] == 8 && buf[4] == 9 && buf[6] == 10), "FourVecView_II -= failed");
    vs %= srt::FourVec<double>{2, 3, 4, 5};
    TEST_ASSERT((buf[0] == 14 && buf[2] == 24 && buf[4] == 36 && buf[6] == 50), "FourVecView_II %= failed");

    // to_array & equality
    auto arr = vs.to_array();
    TEST_ASSERT(arr[0] == 14 && arr[3] == 50, "FourVecView_II to_array failed");
    TEST_ASSERT(vs == srt::FourVec<double>(14, 24, 36, 50), "FourVecView_II equality failed");

    std::cout << "--- FourVecView_II tests PASSED ---\n\n";
    return true;
}