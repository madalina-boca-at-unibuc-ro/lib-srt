cmake_minimum_required(VERSION 3.10)
project(example_usage LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the srt library
find_package(srt REQUIRED)

# Create an executable that uses the library
add_executable(usage_example main.cpp)
target_link_libraries(usage_example srt::srt)
