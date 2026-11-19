#ifndef ARENA_ALLOC_HPP
#define ARENA_ALLOC_HPP

#include <sycl/sycl.hpp>
#include <vector>

namespace mem {

struct ArenaAlloc {
private:
  char *begin;
  char *current;
  unsigned long size;

public:
  std::vector<sycl::event> events;
  sycl::queue q;
  ArenaAlloc(unsigned long size, sycl::queue q);
  ArenaAlloc(ArenaAlloc &&other) noexcept;
  ArenaAlloc &operator=(ArenaAlloc &&other) noexcept;
  ArenaAlloc(const ArenaAlloc &other) = delete;
  ArenaAlloc &operator=(const ArenaAlloc &other) = delete;
  float *alloc(unsigned long count);
  void reset();
  unsigned long get_offset() const;
  void reset_to(unsigned long offset);
  ~ArenaAlloc();
};

} // namespace mem

#endif
