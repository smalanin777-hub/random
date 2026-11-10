#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include"memory.hpp"
#include<sycl/sycl.hpp>

//helper zero-cost metafunction

template<int x, int n>
struct Power{
  static const unsigned int val = x * Power<x, n - 1>::val;
};
template<int x>
struct Power<x, 0>{
  static const unsigned int val = 1;
};

template<int n>
constexpr unsigned int pow2_v = Power<2, n>::val;
template<int x, int n>
constexpr unsigned int pow_v = Power<x, n>::val;

//numeric csontants

inline const unsigned int THREADSNUM = pow2_v<16>;
inline const unsigned int TILE_DIM = pow2_v<4>;

//memory constants

inline sycl::queue global_q(sycl::gpu_selector_v);

inline mem::ArenaAlloc global_alloc{pow2_v<24>, global_q};
inline mem::ArenaAlloc temp_alloc{pow2_v<26>, global_q};

inline mem::mem_chunk global_chunk{global_alloc};
inline mem::mem_chunk temp_chunk{temp_alloc};

#endif
