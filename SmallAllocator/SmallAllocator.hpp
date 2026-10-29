#ifndef __SMALLALLOC__
#define __SMALLALLOC__

struct alignas(64) MetaData {
  bool is_busy;
  unsigned long long size;
};

class SmallAllocator {
private:
  unsigned long long size;
  char *begin;
  char *end;

public:
  SmallAllocator(unsigned long long pool_sz = 1048576);
  ~SmallAllocator();
  void *Alloc(unsigned int Size);
  void *ReAlloc(void *Pointer, unsigned int Size);
  void Free(void *Pointer);
};

#endif
