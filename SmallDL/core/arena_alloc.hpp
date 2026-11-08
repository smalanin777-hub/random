#ifndef ARENA_ALLOC_HPP
#define ARENA_ALLOC_HPP

#include<sycl/sycl.hpp>
#include<iostream>

namespace mem{

struct ArenaAlloc{
private:
  char* begin;
  char* current;
  unsigned long size;
public:
  sycl::queue q;
  ArenaAlloc(unsigned long size, sycl::queue q);
  ArenaAlloc(ArenaAlloc&& other) noexcept;
  ArenaAlloc& operator=(ArenaAlloc&& other) noexcept;
  ArenaAlloc(const ArenaAlloc& other) = delete;
  ArenaAlloc& operator=(const ArenaAlloc& other) = delete;
  ~ArenaAlloc();
  float* alloc(unsigned long count);
  void reset();
  
};

}

#endif
