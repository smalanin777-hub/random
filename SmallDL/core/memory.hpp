#ifndef MEMORY_HPP
#define MEMORY_HPP

#include "arena_alloc.hpp"

namespace mem{

struct mem_chunk{
  mem::ArenaAlloc* alloc_ptr;
  sycl::queue* q;
  mem_chunk(ArenaAlloc& alloc) : alloc_ptr(&alloc), q(&(alloc.q))
  {
  };
};

}
#endif
