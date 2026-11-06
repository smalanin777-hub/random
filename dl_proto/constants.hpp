#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

template<int x, int n>
struct Power{
  static const unsigned int val = x * Power<x, n - 1>::val;
};
template<int x>
struct Power<x, 0>{
  static const unsigned int val = 1;
};

inline sycl::queue global_q(sycl::gpu_selector_v);

template<int n>
constexpr unsigned int pow2_v = Power<2, n>::val;
template<int x, int n>
constexpr unsigned int pow_v = Power<x, n>::val;

inline const unsigned int THREADSNUM = pow2_v<16>;
inline const unsigned int TILE_DIM = pow2_v<4>;


#endif
