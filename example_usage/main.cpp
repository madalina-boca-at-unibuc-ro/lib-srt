#include "srt/all.hpp"
#include <complex>
#include <iostream>

int main() {
  using namespace srt;

  // Example usage of the library
  std::complex<double> data[4] = {
      std::complex<double>(1, 0), std::complex<double>(0, 1),
      std::complex<double>(1, 1), std::complex<double>(2, 3)};

  FourVecView<std::complex<double>> vec(data);

  std::cout << "Four-vector: " << vec << std::endl;
  std::cout << "Magnitude squared: " << vec.dot(vec) << std::endl;

  return 0;
}
