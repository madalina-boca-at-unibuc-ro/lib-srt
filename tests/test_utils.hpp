#pragma once

#include <array>    // For FourMat constructions in tests
#include <complex>  // For testing complex number handling
#include <complex>
#include <iostream>
#include <iostream>               // Required for std::cerr and std::endl
#include <srt/common.hpp>         // Your library's common definitions
#include <srt/four_mat.hpp>       // The class being tested
#include <srt/four_mat_view.hpp>  // The class being tested
#include <srt/four_vec.hpp>       // The class being tested
#include <srt/four_vec_view.hpp>  // The class being tested
#include <vector>                 // For FourMatView to point to
#include <vector>                 // For FourVecView to point to

// Simple macro for assertion testing
// This macro prints a FAILED message with file and line information
// and returns false from the current function (assumed to be a bool test function).
#define TEST_ASSERT(condition, message)                                                           \
    if (!(condition)) {                                                                           \
        std::cerr << "FAILED: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false;                                                                             \
    }
