cmake_minimum_required(VERSION 3.10)
project(lib-srt VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

# Include directories
include_directories(include)

# Create header-only library target
add_library(srt INTERFACE)
target_include_directories(srt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set target properties
set_target_properties(srt PROPERTIES
    VERSION ${PROJECT_VERSION}
)

# Example executable
add_executable(example src/main.cpp)
target_link_libraries(example srt)

# Test executable
add_executable(test_runner 
    tests/main_test_runner.cpp
    tests/test_four_vec.cpp
    tests/test_four_vec_view.cpp
    tests/test_four_mat.cpp
    tests/test_four_mat_view.cpp
)
target_link_libraries(test_runner srt)

# Enable testing
enable_testing()
add_test(NAME lib_srt_tests COMMAND test_runner)

# Installation rules
install(TARGETS srt
    EXPORT srtTargets
)

install(DIRECTORY include/ DESTINATION include)

# Install the export targets
install(EXPORT srtTargets
    FILE srtTargets.cmake
    NAMESPACE srt::
    DESTINATION lib/cmake/srt
)

# Create a config file for find_package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    srtConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/srtConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/srtConfig.cmake
    INSTALL_DESTINATION lib/cmake/srt
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/srtConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/srtConfigVersion.cmake
    DESTINATION lib/cmake/srt
)

# Create cmake directory and config template
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/srtConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/srtConfig.cmake
    @ONLY
)
