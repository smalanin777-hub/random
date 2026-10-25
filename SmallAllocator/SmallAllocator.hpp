#ifndef __SMALLALLOC__ 
#define __SMALLALLOC__

struct MetaData {
    bool is_busy;
    unsigned int size;
};

class SmallAllocator {
private:
    char Memory[1048576];
    char* begin;
    char* end;

public:
    SmallAllocator();
    void* Alloc(unsigned int Size);
    void* ReAlloc(void* Pointer, unsigned int Size);
    void Free(void* Pointer);
};

#endif
