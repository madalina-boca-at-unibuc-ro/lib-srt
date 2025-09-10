#pragma once
#include "four_mat.hpp"

namespace srt {

template<class T>
inline constexpr FourMat<T> eta() noexcept {
    FourMat<T> g{};
    for(int r=0;r<4;++r) for(int c=0;c<4;++c) g(r,c)=T(0);
    g(0,0)=T(+1); g(1,1)=T(-1); g(2,2)=T(-1); g(3,3)=T(-1);
    return g;
}

template<class T>
inline constexpr FourMat<T> Eta = eta<T>();

} // namespace srt
