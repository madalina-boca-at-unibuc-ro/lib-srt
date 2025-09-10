#!/bin/bash

# Build script for lib-srt
set -e

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build . --parallel

echo "Build completed successfully!"
echo "Run './example' to run the example program"
echo "Run './test_runner' to run the tests"
