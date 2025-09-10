# lib-srt

A header-only C++ library for 4-vector and 4-matrix operations.

## Building

### Using the build script (recommended)
```bash
./build.sh
```

### Using CMake manually
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### Using Make (simple alternative)
```bash
make
make test  # Run tests
```

## Running

After building, you can run:

- **Example program**: `./build/example`
- **Tests**: `./build/test_runner`

## Compiling the Example Usage

The `example_usage/` directory shows how to use the library as a dependency. Here are two ways to compile it:

### Option 1: Direct Compilation (Simplest)
```bash
g++ -std=c++20 -Iinclude -o example_usage_simple example_usage/main.cpp
./example_usage_simple
```

### Option 2: Using CMake (Recommended for larger projects)
```bash
# First install the library
cd build
cmake --install . --prefix /tmp/srt_local/

# Then compile the example
cd ../example_usage
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/tmp/srt_local/
make
./usage_example
```

## Using as a dependency

This library can be used as a CMake dependency:

```cmake
find_package(srt REQUIRED)
target_link_libraries(your_target srt)
```

Then include the headers:
```cpp
#include "srt/all.hpp"
// or include specific headers:
#include "srt/four_vec.hpp"
#include "srt/four_mat.hpp"
```

## Installation

To install the library system-wide:

```bash
cd build
cmake --install . --prefix /usr/local
```

## Project Structure

- `include/srt/` - Header files
- `src/` - Example usage
- `tests/` - Test suite
- `cmake/` - CMake configuration files
- `example_usage/` - Example of using the library as a dependency

## Example Usage as Dependency

To use this library in your own CMake project:

```cmake
find_package(srt REQUIRED)
target_link_libraries(your_target srt::srt)
```

See `example_usage/` directory for a complete example.
