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
