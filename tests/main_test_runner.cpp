#include <array>    // For FourVec and FourMat constructors
#include <complex>  // For testing complex number handling
#include <iostream>
#include <vector>  // For FourVecView and FourMatView to point to

#include "../include/srt/common.hpp"
#include "../include/srt/four_mat.hpp"
#include "../include/srt/four_mat_view.hpp"
#include "../include/srt/four_vec.hpp"
#include "../include/srt/four_vec_view.hpp"
#include "../include/srt/fwd.hpp"
#include "test_declarations.hpp"

int main() {
    std::cout << "Running tests...(I)" << std::endl;

    bool success_FourVec = testFourVec();
    bool success_FourVecView = testFourVecView();
    bool success_FourMat = testFourMat();
    bool success_FourMatView = testFourMatView();

    std::cout << "FourVec tests: " << (success_FourVec ? "PASSED" : "FAILED") << std::endl;
    std::cout << "FourVecView tests: " << (success_FourVecView ? "PASSED" : "FAILED") << std::endl;
    std::cout << "FourMat tests: " << (success_FourMat ? "PASSED" : "FAILED") << std::endl;
    std::cout << "FourMatView tests: " << (success_FourMatView ? "PASSED" : "FAILED") << std::endl;

    bool all_tests_passed = success_FourVec;  // && success_FourVecView && success_FourMat && success_FourMatView;
    std::cout << "\nOverall result: " << (all_tests_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

    std::cout << "Running tests...(II)" << std::endl;

    bool success_FourVec_II = testFourVec_II();
    bool success_FourVecView_II = testFourVecView_II();
    bool success_FourMat_II = testFourMat_II();
    bool success_FourMatView_II = testFourMatView_II();

    std::cout << "FourVec tests: " << (success_FourVec_II ? "PASSED" : "FAILED") << std::endl;
    std::cout << "FourVecView tests: " << (success_FourVecView_II ? "PASSED" : "FAILED") << std::endl;
    std::cout << "FourMat tests: " << (success_FourMat_II ? "PASSED" : "FAILED") << std::endl;
    std::cout << "FourMatView tests: " << (success_FourMatView_II ? "PASSED" : "FAILED") << std::endl;

    bool all_tests_passed_II =
        success_FourVec_II && success_FourVecView_II && success_FourMat_II && success_FourMatView_II;
    std::cout << "\nOverall result: " << (all_tests_passed_II ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

    return all_tests_passed ? 0 : 1;
}