#pragma once
#include <ostream>
#include <iomanip>
#include "four_vec.hpp"
#include "four_vec_view.hpp"
#include "four_mat.hpp"
#include "four_mat_view.hpp"

namespace srt {

template<class T>
std::ostream& operator<<(std::ostream& os,const FourVec<T>& v){
    auto f=os.flags(); auto p=os.precision();
    os.setf(std::ios::scientific); os.precision(6);
    os << "[ "<<v[0]<<" "<<v[1]<<" "<<v[2]<<" "<<v[3]<<" ]";
    os.flags(f); os.precision(p); return os;
}
template<class T>
std::ostream& operator<<(std::ostream& os,const FourVecView<T>& v){
    auto f=os.flags(); auto p=os.precision();
    os.setf(std::ios::scientific); os.precision(6);
    os << "[ "<<v[0]<<" "<<v[1]<<" "<<v[2]<<" "<<v[3]<<" ]";
    os.flags(f); os.precision(p); return os;
}
template<class T>
std::ostream& operator<<(std::ostream& os,const FourMat<T>& M){
    auto f=os.flags(); auto p=os.precision();
    os.setf(std::ios::scientific); os.precision(6); int w=12;
    for(int r=0;r<4;++r){ os<<"[ "; for(int c=0;c<4;++c){ if(c) os<<' '; os<<std::setw(w)<<M(r,c);} os<<" ]"; if(r!=3) os<<'\n'; }
    os.flags(f); os.precision(p); return os;
}
template<class T>
std::ostream& operator<<(std::ostream& os,const FourMatView<T>& M){
    auto f=os.flags(); auto p=os.precision();
    os.setf(std::ios::scientific); os.precision(6); int w=12;
    for(int r=0;r<4;++r){ os<<"[ "; for(int c=0;c<4;++c){ if(c) os<<' '; os<<std::setw(w)<<M(r,c);} os<<" ]"; if(r!=3) os<<'\n'; }
    os.flags(f); os.precision(p); return os;
}



} // namespace srt