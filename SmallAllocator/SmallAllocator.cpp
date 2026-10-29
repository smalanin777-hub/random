#include "SmallAllocator.hpp"
#include <cerrno>
#include <cstring>
#include <sys/mman.h>

SmallAllocator::SmallAllocator(unsigned long long pool_sz) : size(pool_sz) {
  begin = reinterpret_cast<char *>(mmap(nullptr, pool_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (begin == MAP_FAILED) {
    begin = nullptr;
    end = nullptr;
  }
  end = begin + pool_sz;
  *(reinterpret_cast<MetaData *>(begin)) = {0, pool_sz - 2 * sizeof(MetaData)};
  *(reinterpret_cast<MetaData *>(end - sizeof(MetaData))) = {0, pool_sz - 2 * sizeof(MetaData)};
}
SmallAllocator::~SmallAllocator() {
  if (begin)
    munmap(begin, size);
}
void *SmallAllocator::Alloc(unsigned int Size) {
  void *allocated = nullptr;
  int i = 0;
  if (!begin) {
    errno = ENOMEM;
    return allocated;
  }
  Size = (Size + 63) & ~(63);
  while (i + sizeof(MetaData) < size) {
    if (reinterpret_cast<MetaData *>(begin + i)->is_busy) {
      i += reinterpret_cast<MetaData *>(begin + i)->size;
      continue;
    } else if (!(reinterpret_cast<MetaData *>(begin + i)->is_busy) &&
               reinterpret_cast<MetaData *>(begin + i)->size >= Size + 2 * sizeof(MetaData)) {
      MetaData *free_mem = reinterpret_cast<MetaData *>(begin + i);
      unsigned int old_free_size = free_mem->size;
      if (reinterpret_cast<MetaData *>(begin + i)->size > Size + 2 * sizeof(MetaData)) {
        *(reinterpret_cast<MetaData *>(begin + i)) = {1, (unsigned int)(Size + 2 * sizeof(MetaData))};
        i += sizeof(MetaData);
        allocated = static_cast<void *>(begin + i);
        i += Size;
        *(reinterpret_cast<MetaData *>(begin + i)) = {1, (unsigned int)(Size + 2 * sizeof(MetaData))};
        i += sizeof(MetaData);
        *(reinterpret_cast<MetaData *>(begin + i)) = {0, old_free_size - (unsigned int)(Size + 2 * sizeof(MetaData))};
        reinterpret_cast<MetaData *>((char *)free_mem + old_free_size - sizeof(MetaData))->size =
            old_free_size - (Size + 2 * sizeof(MetaData));
        break;
      } else {
        *(reinterpret_cast<MetaData *>(begin + i)) = {1, (unsigned int)(Size + 2 * sizeof(MetaData))};
        allocated = static_cast<void *>(begin + i + sizeof(MetaData));
        reinterpret_cast<MetaData *>((char *)free_mem + free_mem->size - sizeof(MetaData))->is_busy = 1;
        break;
      }
    } else {
      i += reinterpret_cast<MetaData *>(begin + i)->size;
    }
  }
  if (!allocated)
    errno = ENOMEM;
  return allocated;
}
void *SmallAllocator::ReAlloc(void *Pointer, unsigned int Size) {
  if (!Pointer)
    return Alloc(Size);
  if (((char *)Pointer - sizeof(MetaData)) >= begin && ((char *)Pointer - sizeof(MetaData)) < end) {
    MetaData *self_mem = reinterpret_cast<MetaData *>((char *)Pointer - sizeof(MetaData));
    if (self_mem->size >= Size + 2 * sizeof(MetaData)) {
      if (self_mem->size > Size + 2 * sizeof(MetaData)) {
        unsigned int old_size = self_mem->size;
        self_mem->size = Size + 2 * sizeof(MetaData);
        *(reinterpret_cast<MetaData *>((char *)self_mem + self_mem->size - sizeof(MetaData))) = {1, self_mem->size};
        MetaData *second_mem = reinterpret_cast<MetaData *>((char *)self_mem + old_size);
        if (!(second_mem->is_busy)) {
          if (second_mem->size > 0) {
            *(reinterpret_cast<MetaData *>((char *)self_mem + self_mem->size)) = {0, second_mem->size + (old_size - self_mem->size)};
            reinterpret_cast<MetaData *>((char *)second_mem + second_mem->size - sizeof(MetaData))->size =
                second_mem->size + (old_size - self_mem->size);
          }
        } else if (second_mem->is_busy) {
          if (old_size - self_mem->size > 2 * sizeof(MetaData)) {
            *(reinterpret_cast<MetaData *>((char *)self_mem + self_mem->size - sizeof(MetaData))) = {1, self_mem->size};
            *(reinterpret_cast<MetaData *>((char *)self_mem + self_mem->size)) = {0, old_size - self_mem->size};
            *(reinterpret_cast<MetaData *>((char *)self_mem + old_size - sizeof(MetaData))) = {0, old_size - self_mem->size};
          }
        }
      }
      return Pointer;
    } else if (self_mem->size < Size + 2 * sizeof(MetaData)) {
      void *new_ptr = this->Alloc(Size);
      if (new_ptr) {
        std::memcpy(new_ptr, Pointer, self_mem->size - 2 * sizeof(MetaData));
        this->Free(Pointer);
        return new_ptr;
      }
    }
  }
  return Pointer;
}
void SmallAllocator::Free(void *Pointer) {
  if (!Pointer || (char *)Pointer < begin || (char *)Pointer >= end)
    return;
  MetaData *self_mem = reinterpret_cast<MetaData *>((char *)Pointer - sizeof(MetaData));
  self_mem->is_busy = 0;
  MetaData *self_footer = reinterpret_cast<MetaData *>((char *)self_mem + self_mem->size - sizeof(MetaData));
  self_footer->is_busy = 0;
  MetaData *next_mem = reinterpret_cast<MetaData *>((char *)self_mem + self_mem->size);
  if ((char *)next_mem < end && !next_mem->is_busy && next_mem->size > 0) {
    self_mem->size += next_mem->size;
    reinterpret_cast<MetaData *>((char *)self_mem + self_mem->size - sizeof(MetaData))->size = self_mem->size;
  }
  MetaData *prev_footer = reinterpret_cast<MetaData *>((char *)self_mem - sizeof(MetaData));
  if ((char *)prev_footer >= begin && !prev_footer->is_busy && prev_footer->size > 0) {
    MetaData *prev_header = reinterpret_cast<MetaData *>((char *)self_mem - prev_footer->size);
    prev_header->size += self_mem->size;
    reinterpret_cast<MetaData *>((char *)prev_header + prev_header->size - sizeof(MetaData))->size = prev_header->size;
  }
}
