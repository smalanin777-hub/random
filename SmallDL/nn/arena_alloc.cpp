#include "arena_alloc.hpp"
#include "constants.hpp"
#include <iostream>

namespace mem {

ArenaAlloc::ArenaAlloc(unsigned long size, sycl::queue q) : size(size), q(q) {
  begin = (char *)(sycl::malloc_device(size, q));
  current = begin;
}
ArenaAlloc::ArenaAlloc(ArenaAlloc &&other) noexcept : begin(other.begin), current(other.current), size(other.size), q(other.q) {
  other.begin = nullptr;
  other.current = nullptr;
  other.size = 0;
}
ArenaAlloc &ArenaAlloc::operator=(ArenaAlloc &&other) noexcept {
  if (this != &other) {
    if (begin)
      sycl::free(begin, q);
    begin = other.begin;
    current = other.current;
    size = other.size;
    q = other.q;
    other.begin = nullptr;
    other.current = nullptr;
    other.size = 0;
  }
  return *this;
}
float *ArenaAlloc::alloc(unsigned long count) {
  unsigned long real_size = (count * sizeof(float) + 63) & ~63;
  long offset = current - begin;
  if (offset + real_size > size) {
    std::cout << "Out of memory (need " << real_size << ", have " << size - offset << ")\n";
    std::abort();
  }
  float *allocated = reinterpret_cast<float *>(current);
  current += real_size;
  return allocated;
}
void ArenaAlloc::reset() {
  for (sycl::event ev : events)
    ev.wait();
  events.clear();
  current = begin;
}
ArenaAlloc::~ArenaAlloc() {
  if (begin)
    sycl::free(begin, q);
}

} // namespace mem
