// tests/test_four_vec.cpp
#include "test_utils.hpp"

// Function to run all FourVec tests
bool testFourVec() {
    std::cout << "--- Testing FourVec ---\n";

    // Constructor tests
    srt::FourVec<double> v1;
    TEST_ASSERT(v1.x0 == 0 && v1.x1 == 0 && v1.x2 == 0 && v1.x3 == 0, "Default constructor failed");

    srt::FourVec<double> v2(1.0, 2.0, 3.0, 4.0);
    TEST_ASSERT(v2[0] == 1.0 && v2[1] == 2.0 && v2[2] == 3.0 && v2[3] == 4.0, "Param constructor failed");

    std::array<double, 4> arr = {5.0, 6.0, 7.0, 8.0};
    srt::FourVec<double> v3(arr);
    TEST_ASSERT(v3.x0 == 5.0 && v3.x1 == 6.0 && v3.x2 == 7.0 && v3.x3 == 8.0, "Array constructor failed");

    // Assignment tests
    srt::FourVec<double> v4;
    v4 = v2;
    TEST_ASSERT(v4 == v2, "Assignment operator failed");

    v4 = arr;  // Assign from std::array (Indexable4 concept)
    TEST_ASSERT(v4 == v3, "Templated assignment from array failed");

    // Arithmetic operations
    srt::FourVec<double> v_sum = v2 + v3;  // (1+5, 2+6, 3+7, 4+8) = (6,8,10,12)
    TEST_ASSERT(v_sum[0] == 6.0 && v_sum[1] == 8.0 && v_sum[2] == 10.0 && v_sum[3] == 12.0, "Vector addition failed");

    srt::FourVec<double> v_diff = v3 - v2;  // (5-1, 6-2, 7-3, 8-4) = (4,4,4,4)
    TEST_ASSERT(v_diff[0] == 4.0 && v_diff[1] == 4.0 && v_diff[2] == 4.0 && v_diff[3] == 4.0,
                "Vector subtraction failed");

    srt::FourVec<double> v2_copy = v2;  // Make a copy for += test
    v2_copy += v3;                      // v2_copy is now (6,8,10,12)
    TEST_ASSERT(v2_copy == v_sum, "In-place vector addition failed");

    srt::FourVec<double> v_scalar_mul = v3 * 2.0;  // (10,12,14,16)
    TEST_ASSERT(
        v_scalar_mul[0] == 10.0 && v_scalar_mul[1] == 12.0 && v_scalar_mul[2] == 14.0 && v_scalar_mul[3] == 16.0,
        "Scalar multiplication failed");

    double scalar = 3.0;
    srt::FourVec<double> v_scalar_mul_comm = scalar * v3;
    TEST_ASSERT(v_scalar_mul_comm[0] == 15.0 && v_scalar_mul_comm[1] == 18.0 && v_scalar_mul_comm[2] == 21.0 &&
                    v_scalar_mul_comm[3] == 24.0,
                "Commutative scalar multiplication failed");

    srt::FourVec<double> v3_copy = v3;  // Make a copy for *= test
    v3_copy *= 2.0;                     // v3_copy is now (10,12,14,16)
    TEST_ASSERT(v3_copy == v_scalar_mul, "In-place scalar multiplication failed");

    srt::FourVec<double> v_elem_mul = v3_copy % v_scalar_mul_comm;  // (10*15, 12*18, 14*21, 16*24)
    TEST_ASSERT(v_elem_mul[0] == 150.0 && v_elem_mul[1] == 216.0 && v_elem_mul[2] == 294.0 && v_elem_mul[3] == 384.0,
                "Element-wise multiplication failed");

    // Dot products and norms
    srt::FourVec<double> v_dot_a(1.0, 2.0, 3.0, 4.0);
    srt::FourVec<double> v_dot_b(5.0, 6.0, 7.0, 8.0);

    double minkowski_norm2 = v_dot_a.minkowski_norm2();  // 1*1 - (2*2 + 3*3 + 4*4) = 1 - (4 + 9 + 16) = 1 - 29 = -28
    TEST_ASSERT(minkowski_norm2 == -28.0, "Minkowski norm squared failed");

    double euclidean4_norm2 = v_dot_a.euclidean4_norm2();  // 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
    TEST_ASSERT(euclidean4_norm2 == 30.0, "Euclidean 4D norm squared failed");

    double euclidean3_norm2 = v_dot_a.euclidean3_norm2();  // 2*2 + 3*3 + 4*4 = 4 + 9 + 16 = 29
    TEST_ASSERT(euclidean3_norm2 == 29.0, "Euclidean 3D norm squared failed");

    double minkowski_dot_prod =
        v_dot_a.minkowski_dot(v_dot_b);  // 1*5 - (2*6 + 3*7 + 4*8) = 5 - (12 + 21 + 32) = 5 - 65 = -60
    TEST_ASSERT(minkowski_dot_prod == -60.0, "Minkowski dot product failed");

    double dot_prod_3d = v_dot_a.dot(v_dot_b);  // 2*6 + 3*7 + 4*8 = 12 + 21 + 32 = 65
    TEST_ASSERT(dot_prod_3d == 65.0, "3D dot product failed");

    double dot_prod_4d = v_dot_a.dot_four(v_dot_b);  // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    TEST_ASSERT(dot_prod_4d == 70.0, "4D dot product failed");

    // Conjugation for complex numbers
    srt::FourVec<std::complex<double>> vc(std::complex<double>(1.0, 1.0), std::complex<double>(2.0, -2.0),
                                          std::complex<double>(3.0, 3.0), std::complex<double>(4.0, -4.0));
    vc.conjugate_inplace();
    TEST_ASSERT(vc[0] == std::complex<double>(1.0, -1.0) && vc[1] == std::complex<double>(2.0, 2.0) &&
                    vc[2] == std::complex<double>(3.0, -3.0) && vc[3] == std::complex<double>(4.0, 4.0),
                "Complex conjugation (inplace) failed");

    srt::FourVec<std::complex<double>> vc_original(std::complex<double>(1.0, 1.0), std::complex<double>(2.0, -2.0),
                                                   std::complex<double>(3.0, 3.0), std::complex<double>(4.0, -4.0));
    srt::FourVec<std::complex<double>> vc_conj = vc_original.conjugate();
    TEST_ASSERT(vc_conj[0] == std::complex<double>(1.0, -1.0) && vc_conj[1] == std::complex<double>(2.0, 2.0) &&
                    vc_conj[2] == std::complex<double>(3.0, -3.0) && vc_conj[3] == std::complex<double>(4.0, 4.0),
                "Complex conjugation failed");

    // Real/Imaginary parts
    srt::FourVec<double> real_part = vc_original.real();
    TEST_ASSERT(real_part[0] == 1.0 && real_part[1] == 2.0 && real_part[2] == 3.0 && real_part[3] == 4.0,
                "Real part failed");
    srt::FourVec<double> imag_part = vc_original.imag();
    TEST_ASSERT(imag_part[0] == 1.0 && imag_part[1] == -2.0 && imag_part[2] == 3.0 && imag_part[3] == -4.0,
                "Imaginary part failed");

    // Covariant
    srt::FourVec<double> v_cov(1.0, 2.0, 3.0, 4.0);
    v_cov.covariant_inplace();
    TEST_ASSERT(v_cov[0] == 1.0 && v_cov[1] == -2.0 && v_cov[2] == -3.0 && v_cov[3] == -4.0,
                "Covariant (inplace) failed");
    srt::FourVec<double> v_cov_orig(1.0, 2.0, 3.0, 4.0);
    srt::FourVec<double> v_cov_out = v_cov_orig.covariant();
    TEST_ASSERT(v_cov_out[0] == 1.0 && v_cov_out[1] == -2.0 && v_cov_out[2] == -3.0 && v_cov_out[3] == -4.0,
                "Covariant failed");

    // Cross product
    srt::FourVec<double> va_cross(0.0, 1.0, 0.0, 0.0);
    srt::FourVec<double> vb_cross(0.0, 0.0, 1.0, 0.0);
    srt::FourVec<double> v_res_cross = va_cross.cross(vb_cross);  // (0, 0, 0, 1) expected (i x j = k)
    TEST_ASSERT(v_res_cross[0] == 0.0 && v_res_cross[1] == 0.0 && v_res_cross[2] == 0.0 && v_res_cross[3] == 1.0,
                "Cross product failed (i x j)");

    // --- New Tests for Vector-Matrix Multiplication ---
    srt::FourVec<double> v_test(1.0, 2.0, 3.0, 4.0);
    srt::FourMat<double> m_test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    // dot_four_left (Euclidean: row vector * matrix)
    // Result: (1*1+2*5+3*9+4*13, 1*2+2*6+3*10+4*14, 1*3+2*7+3*11+4*15, 1*4+2*8+3*12+4*16)
    // = (1+10+27+52, 2+12+30+56, 3+14+33+60, 4+16+36+64)
    // = (90, 100, 110, 120)
    srt::FourVec<double> expected_dot_four_left(90.0, 100.0, 110.0, 120.0);
    srt::FourVec<double> res_dot_four_left = v_test.dot_four_left(m_test);
    TEST_ASSERT(res_dot_four_left == expected_dot_four_left, "dot_four_left (FourVec) failed");

    // dot_four_left_inplace
    srt::FourVec<double> v_test_inplace_left = v_test;
    v_test_inplace_left.dot_four_left_inplace(m_test);
    TEST_ASSERT(v_test_inplace_left == expected_dot_four_left, "dot_four_left_inplace (FourVec) failed");

    // dot_four_right (Euclidean: matrix * column vector)
    // Result: (1*1+2*2+3*3+4*4, 5*1+6*2+7*3+8*4, 9*1+10*2+11*3+12*4, 13*1+14*2+15*3+16*4)
    // = (1+4+9+16, 5+12+21+32, 9+20+33+48, 13+28+45+64)
    // = (30, 70, 110, 150)
    srt::FourVec<double> expected_dot_four_right(30.0, 70.0, 110.0, 150.0);
    srt::FourVec<double> res_dot_four_right = v_test.dot_four_right(m_test);
    TEST_ASSERT(res_dot_four_right == expected_dot_four_right, "dot_four_right (FourVec) failed");

    // dot_four_right_inplace
    srt::FourVec<double> v_test_inplace_right = v_test;
    v_test_inplace_right.dot_four_right_inplace(m_test);
    TEST_ASSERT(v_test_inplace_right == expected_dot_four_right, "dot_four_right_inplace (FourVec) failed");

    // minkowski_dot_four_left (Minkowski: row vector * matrix)
    // v = (x0, x1, x2, x3)
    // m = [[m00, m01, m02, m03],
    //      [m10, m11, m12, m13],
    //      [m20, m21, m22, m23],
    //      [m30, m31, m32, m33]]
    // result_i = x0*m0i - x1*m1i - x2*m2i - x3*m3i
    // v_test = (1, 2, 3, 4)
    // m_test is the 1-16 matrix
    // Result_0 = 1*1 - 2*5 - 3*9 - 4*13 = 1 - 10 - 27 - 52 = 1 - 89 = -88
    // Result_1 = 1*2 - 2*6 - 3*10 - 4*14 = 2 - 12 - 30 - 56 = 2 - 98 = -96
    // Result_2 = 1*3 - 2*7 - 3*11 - 4*15 = 3 - 14 - 33 - 60 = 3 - 107 = -104
    // Result_3 = 1*4 - 2*8 - 3*12 - 4*16 = 4 - 16 - 36 - 64 = 4 - 116 = -112
    srt::FourVec<double> expected_mink_dot_four_left(-88.0, -96.0, -104.0, -112.0);
    srt::FourVec<double> res_mink_dot_four_left = v_test.minkowski_dot_four_left(m_test);
    TEST_ASSERT(res_mink_dot_four_left == expected_mink_dot_four_left, "minkowski_dot_four_left (FourVec) failed");

    // minkowski_dot_four_left_inplace
    srt::FourVec<double> v_test_mink_inplace_left = v_test;
    v_test_mink_inplace_left.minkowski_dot_four_left_inplace(m_test);
    TEST_ASSERT(v_test_mink_inplace_left == expected_mink_dot_four_left,
                "minkowski_dot_four_left_inplace (FourVec) failed");

    // minkowski_dot_four_right (Minkowski: matrix * column vector)
    // result_i = m_i0*x0 - m_i1*x1 - m_i2*x2 - m_i3*x3
    // v_test = (1, 2, 3, 4)
    // m_test is the 1-16 matrix
    // Result_0 = 1*1 - 2*2 - 3*3 - 4*4 = 1 - 4 - 9 - 16 = -28
    // Result_1 = 5*1 - 6*2 - 7*3 - 8*4 = 5 - 12 - 21 - 32 = -60
    // Result_2 = 9*1 - 10*2 - 11*3 - 12*4 = 9 - 20 - 33 - 48 = -92
    // Result_3 = 13*1 - 14*2 - 15*3 - 16*4 = 13 - 28 - 45 - 64 = -124
    srt::FourVec<double> expected_mink_dot_four_right(-28.0, -60.0, -92.0, -124.0);
    srt::FourVec<double> res_mink_dot_four_right = v_test.minkowski_dot_four_right(m_test);
    TEST_ASSERT(res_mink_dot_four_right == expected_mink_dot_four_right, "minkowski_dot_four_right (FourVec) failed");

    // minkowski_dot_four_right_inplace
    srt::FourVec<double> v_test_mink_inplace_right = v_test;
    v_test_mink_inplace_right.minkowski_dot_four_right_inplace(m_test);
    TEST_ASSERT(v_test_mink_inplace_right == expected_mink_dot_four_right,
                "minkowski_dot_four_right_inplace (FourVec) failed");

    std::cout << "--- FourVec tests PASSED ---\n\n";
    return true;
}

bool testFourVec_II() {
    std::cout << "--- Testing FourVec_II ---\n";

    // construction & access
    srt::FourVec<double> a{1.0, 2.0, 3.0, 4.0};
    TEST_ASSERT(a.x0 == 1.0 && a.x1 == 2.0 && a.x2 == 3.0 && a.x3 == 4.0, "Construction failed");
    TEST_ASSERT(a[0] == 1.0 && a[3] == 4.0, "Access failed");

    // to_array & iteration
    auto arr = a.to_array();
    TEST_ASSERT((arr == std::array<double, 4>{1.0, 2.0, 3.0, 4.0}), "to_array failed");
    double sum_iter = 0.0;
    for (auto v : a) sum_iter += v;
    TEST_ASSERT(sum_iter == 10.0, "Iteration failed");

    // copy/assign from Indexable4 (std::array)
    srt::FourVec<double> b;
    std::array<int, 4> ai{1, 2, 3, 4};
    b = ai;
    TEST_ASSERT(b == a, "Assignment failed");

    // equality/inequality with different types
    srt::FourVec<int> ai_vec{1, 2, 3, 4};
    TEST_ASSERT(a == ai_vec, "Equality failed");
    TEST_ASSERT(!(a != ai_vec), "Inequality failed");

    // unary + / -
    auto neg = -a;
    TEST_ASSERT(neg[0] == -1.0 && neg[3] == -4.0, "Unary - failed");
    auto pos = +a;
    TEST_ASSERT(pos == a, "Unary + failed");

    // element-wise ops (Hadamard, +=, -=, %=)
    srt::FourVec<double> c{2.0, 3.0, 4.0, 5.0};
    srt::FourVec<double> d = a % c;
    TEST_ASSERT((d.to_array() == std::array<double, 4>{2.0, 6.0, 12.0, 20.0}), "Element-wise multiplication failed");
    c %= a;
    TEST_ASSERT((c.to_array() == std::array<double, 4>{2.0, 6.0, 12.0, 20.0}),
                "In-place element-wise multiplication failed");
    srt::FourVec<double> e{1.0, 1.0, 1.0, 1.0};
    e += a;  // in-place sum
    TEST_ASSERT((e.to_array() == std::array<double, 4>{2.0, 3.0, 4.0, 5.0}), "In-place sum failed");
    e -= srt::FourVec<int>{1, 1, 1, 1};
    TEST_ASSERT((e.to_array() == std::array<double, 4>{1.0, 2.0, 3.0, 4.0}), "In-place subtraction failed");

    // scalar multiply/divide and commutative multiply
    auto s1 = a * 2.0;
    auto s2 = 2.0 * a;
    auto s3 = a / 2.0;
    TEST_ASSERT((s1.to_array() == std::array<double, 4>{2.0, 4.0, 6.0, 8.0}), "Scalar multiplication failed");
    TEST_ASSERT((s2.to_array() == std::array<double, 4>{2.0, 4.0, 6.0, 8.0}), "Scalar multiplication failed");
    TEST_ASSERT((s3.to_array() == std::array<double, 4>{0.5, 1.0, 1.5, 2.0}), "Scalar division failed");

    // 3-vector dot, 4-vector dot (+,+,+,+), Minkowski (+,-,-,-)
    srt::FourVec<double> v{1.0, 1.0, 2.0, 3.0};
    srt::FourVec<double> w{2.0, 4.0, 5.0, 6.0};
    auto dot3 = v.dot(w);            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    auto dot4 = v.dot_four(w);       // 1*2 + 1*4 + 2*5 + 3*6 = 2 + 4 + 10 + 18 = 34
    auto mdot = v.minkowski_dot(w);  // 1*2 - (1*4 + 2*5 + 3*6) = 2 - (4+10+18) = -30
    TEST_ASSERT(dot3 == 32.0, "3D dot product failed");
    TEST_ASSERT(dot4 == 34.0, "4D dot product failed");
    TEST_ASSERT(mdot == -30.0, "Minkowski dot product failed");

    // vec-mat left dot (row vector times matrix)
    srt::FourMat<double> M{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto vm = v.dot_four_left(M);
    // Manual check: vm[j] = sum_i v[i]*M(i,j)
    TEST_ASSERT(vm[0] == 1 * 1 + 1 * 5 + 2 * 9 + 3 * 13, "vec-mat left dot failed");
    TEST_ASSERT(vm[1] == 1 * 2 + 1 * 6 + 2 * 10 + 3 * 14, "vec-mat left dot failed");
    TEST_ASSERT(vm[2] == 1 * 3 + 1 * 7 + 2 * 11 + 3 * 15, "vec-mat left dot failed");
    TEST_ASSERT(vm[3] == 1 * 4 + 1 * 8 + 2 * 12 + 3 * 16, "vec-mat left dot failed");

    // casting
    auto v_int = v.cast<int>();
    static_assert(std::is_same_v<decltype(v_int)::value_type, int>);

    // complex compatibility via common_type
    srt::FourVec<std::complex<double>> z{{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
    auto mix = v % z;  // element-wise multiply with promotion
    TEST_ASSERT(mix[2] == std::complex<double>(2.0 * 3.0, 0.0), "Element-wise multiplication with promotion failed");

    std::cout << "--- FourVec_II tests PASSED ---\n\n";
    return true;
}
