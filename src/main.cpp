#include <complex>
#include <cstdint>
#include <iostream>
// #include "srt/all.hpp"
#include "srt/four_vec.hpp"
#include "srt/four_vec_view.hpp"
#include "srt/io.hpp"

int main() {
    using namespace srt;

    std::complex<double> a[4] = {std::complex<double>(1, 2), std::complex<double>(3, 4), std::complex<double>(5, 6),
                                 std::complex<double>(7, 8)};
    double b[4] = {1, 2, 3, 4};

    FourVecView<std::complex<double>> c = FourVecView(a);
    FourVecView<double> d = FourVecView(b);

    std::cout << c << std::endl;
    std::cout << d << std::endl;

    c += d;
    std::cout << c << std::endl;

    c -= d;
    std::cout << c << std::endl;

    c *= 2.0;
    std::cout << c << std::endl;

    c /= 2.0;
    std::cout << c << std::endl;

    c %= d;
    std::cout << c << std::endl;
}
